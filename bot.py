"""ScaleYourJunk Multi-Tenant AI Phone Agent — Pipecat Pipeline.

Builds and runs the full voice agent pipeline per call:
  Twilio audio in → GPT Realtime 2 (tools + text output) → Cartesia TTS → Twilio audio out

Each call gets its own pipeline with the client's config (agent name, voice,
company name, hours, etc.) loaded dynamically from the dashboard.

Handles:
  - Pre-generated greeting for zero-latency first utterance
  - Tool registration with cancel_on_interruption settings
  - Call duration timer (10 min)
  - Post-call summary
  - Context window management
"""

import asyncio
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import openai
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMMessagesAppendFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    InputAudioTranscription,
    PCMAudioFormat,
    SemanticTurnDetection,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.sentence import SentenceAggregator

from agent.prosody import CartesiaContinuationTTS, ProsodyProcessor

from config import (
    OPENAI_API_KEY,
    CARTESIA_API_KEY,
    DEFAULT_CARTESIA_VOICE_ID,
    CARTESIA_MODEL,
    REALTIME_MODEL,
    UTILITY_MODEL,
    MAX_CALL_DURATION_SECONDS,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
)
from agent.prompt import build_system_prompt
from agent.handlers import (
    handle_create_booking,
    handle_check_container_availability,
    handle_check_available_slots,
    handle_lookup_appointment,
    handle_reschedule_appointment,
    handle_cancel_appointment,
    handle_schedule_callback,
    handle_transfer_to_human,
    handle_validate_promo_code,
    handle_record_sms_consent,
    handle_verify_address,
    set_call_context,
    set_pipeline_task,
    clear_call_context,
    is_booking_complete,
    was_transfer_complete,
    get_transfer_state,
    get_booking_log_state,
    has_sms_consent,
    send_automated_followup,
    log_call_to_dashboard,
    get_inflight_tasks,
    update_pending_transfer_snapshot,
    get_pending_transfer_state,
    clear_pending_transfer_if_terminal,
    _current_call_sid,
)
from agent.context import should_compress, compress_context


# ── Automated Follow-Up SMS ─────────────────────────────

FOLLOWUP_DELAY_SECONDS = 300  # 5 minutes
INFLIGHT_TASK_WAIT_SECONDS = 17  # Covers 15s dashboard tool calls plus small scheduling overhead
CALL_LOG_WAIT_SECONDS = 12

async def _delayed_followup_sms(caller_number: str, config: dict, delay_seconds: int = FOLLOWUP_DELAY_SECONDS):
    """Wait, then send automated follow-up SMS to callers who didn't book."""
    try:
        company = config.get("companyName", "unknown")
        logger.info(f"Scheduled follow-up SMS for {caller_number} ({company}) in {delay_seconds}s")
        await asyncio.sleep(delay_seconds)
        sent = await send_automated_followup(caller_number, config, sms_consented=True)
        if sent:
            logger.info(f"Follow-up SMS delivered to {caller_number} ({company})")
        else:
            logger.info(f"Follow-up SMS skipped for {caller_number} ({company}) — gates not met")
    except asyncio.CancelledError:
        logger.debug(f"Follow-up SMS cancelled for {caller_number}")
    except Exception as e:
        logger.error(f"Follow-up SMS error for {caller_number}: {e}")


class RealtimeTranscriptTracker(FrameProcessor):
    """Capture Realtime transcripts/text for call summaries and silence timers."""

    def __init__(self, *, on_user_transcript, on_assistant_text, **kwargs):
        super().__init__(**kwargs)
        self._on_user_transcript = on_user_transcript
        self._on_assistant_text = on_assistant_text
        self._assistant_parts: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            self._on_user_transcript(frame.text)
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._assistant_parts = []
        elif isinstance(frame, LLMTextFrame) and frame.text:
            self._assistant_parts.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame) and self._assistant_parts:
            text = "".join(self._assistant_parts).strip()
            self._assistant_parts = []
            if text:
                self._on_assistant_text(text)

        await self.push_frame(frame, direction)


async def _queue_context_instruction(task: PipelineTask, context: LLMContext, content: str) -> None:
    """Append a system instruction and trigger one LLM response.

    Realtime sessions already maintain their own context, so append only the new
    instruction instead of resending the full local transcript.
    """
    message = {"role": "system", "content": content}
    context.add_message(message)
    await task.queue_frames([LLMMessagesAppendFrame([message], run_llm=True)])


# ── Tool Definitions ────────────────────────────────────

tools = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="create_booking",
        description="Book a junk removal pickup, dumpster rental, or dumpster swap. ONLY call this after reading back all details and receiving verbal confirmation from the caller.",
        properties={
            "name": {"type": "string", "description": "Customer full name"},
            "phone": {"type": "string", "description": "Customer phone number"},
            "address": {"type": "string", "description": "Service address"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "time": {"type": "string", "description": "Time slot: use the start-end format from check_available_slots (e.g. '08:00-11:00'), or labels: morning, midday, afternoon"},
            "description": {"type": "string", "description": "Items for removal, project description for dumpster, or swap reason"},
            "type": {"type": "string", "enum": ["pickup", "dumpster_rental", "dumpster_swap"], "description": "Type of appointment. Use 'pickup' for all junk removal jobs (including when a caller asks for an estimate or someone to come look — every job is an on-site assessment + removal). Use 'dumpster_rental' for new deliveries, 'dumpster_swap' for swapping a full container for an empty one."},
            "container_size": {"type": "string", "enum": ["10", "15", "20", "30", "40"], "description": "Container size in cubic yards (for dumpster_rental or dumpster_swap)"},
            "rental_duration_days": {"type": "integer", "description": "Rental duration in days, default 7 (only for dumpster_rental)"},
            "promo_code": {"type": "string", "description": "Validated promo code to apply discount (optional)"},
        },
        required=["name", "phone", "address", "date", "description", "type"],
    ),
    FunctionSchema(
        name="check_container_availability",
        description="Check if a dumpster container of the requested size is available for a specific delivery date and get live pricing. Call this AFTER collecting the customer's preferred delivery date and rental duration.",
        properties={
            "size": {"type": "string", "enum": ["10", "15", "20", "30", "40"],
                     "description": "Container size in cubic yards"},
            "date": {"type": "string",
                     "description": "Requested delivery date in YYYY-MM-DD format"},
            "days": {"type": "integer",
                     "description": "Rental duration in days (default 7)"},
        },
        required=["size"],
    ),
    FunctionSchema(
        name="lookup_appointment",
        description="Find existing appointment by phone number",
        properties={
            "phone": {"type": "string", "description": "Phone number to search"},
        },
        required=["phone"],
    ),
    FunctionSchema(
        name="reschedule_appointment",
        description="Reschedule an existing appointment. Call check_available_slots first to find open times, then read back new details and confirm.",
        properties={
            "phone": {"type": "string", "description": "Customer phone to find booking"},
            "new_date": {"type": "string", "description": "YYYY-MM-DD"},
            "new_time": {"type": "string", "description": "Time slot: use start-end format from check_available_slots (e.g. '08:00-11:00'), or labels: morning, midday, afternoon"},
            "job_id": {"type": "string", "description": "Job ID from lookup_appointment to target a specific booking (required when customer has multiple bookings)"},
        },
        required=["phone", "new_date", "new_time"],
    ),
    FunctionSchema(
        name="cancel_appointment",
        description="Cancel an existing appointment",
        properties={
            "phone": {"type": "string", "description": "Customer phone to find booking"},
            "reason": {"type": "string", "description": "Reason for cancellation"},
            "job_id": {"type": "string", "description": "Job ID from lookup_appointment to target a specific booking (required when customer has multiple bookings)"},
        },
        required=["phone"],
    ),
    FunctionSchema(
        name="schedule_callback",
        description="Schedule a human follow-up callback at a specific date and time. Use only after the caller asks for a callback and confirms the exact callback time.",
        properties={
            "requested_time": {"type": "string", "description": "Exact callback date and time in YYYY-MM-DDTHH:MM:SS format, resolved from relative wording in the client's timezone"},
            "reason": {"type": "string", "description": "Why the caller wants a callback"},
            "caller_phone": {"type": "string", "description": "Best phone number for the callback. Optional if the caller confirms the number they called from is best."},
            "caller_name": {"type": "string", "description": "Caller name, if known"},
        },
        required=["requested_time", "reason"],
    ),
    FunctionSchema(
        name="transfer_to_human",
        description="Transfer the live call to a human team member. Use when the caller explicitly asks to speak to a real person, has a complaint or billing dispute, or after repeated tool failures.",
        properties={
            "reason": {"type": "string", "description": "Why the caller wants to be transferred"},
        },
        required=["reason"],
    ),
    FunctionSchema(
        name="validate_promo_code",
        description="Validate a promo or referral code. Call when the caller mentions having a discount code.",
        properties={
            "code": {"type": "string", "description": "The promo code to validate"},
        },
        required=["code"],
    ),
    FunctionSchema(
        name="check_available_slots",
        description="Check which business-hour booking windows are available for a specific date. Call this BEFORE offering times to the caller. This is not a junk-removal job capacity check.",
        properties={
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format to check availability for"},
        },
        required=["date"],
    ),
    FunctionSchema(
        name="record_sms_consent",
        description="Record whether the caller consented to receiving text messages. Call this IMMEDIATELY after asking for SMS consent and receiving a clear yes or no answer. You MUST call this before any text-related action.",
        properties={
            "consented": {"type": "boolean", "description": "True if caller said yes to receiving texts, false if they declined"},
        },
        required=["consented"],
    ),
    FunctionSchema(
        name="verify_address",
        description="Verify and normalize a service address using Google Maps. Call this after the caller gives their address to confirm it's correct. Read back the verified address and ask the caller to confirm.",
        properties={
            "address": {"type": "string", "description": "The address as spoken by the caller, including street, city, and state if provided"},
        },
        required=["address"],
    ),
])


async def run_bot(
    websocket_client,
    stream_id: str,
    call_id: str,
    caller_number: str,
    client_config: dict[str, Any],
):
    """Build and run the Pipecat pipeline for a single call."""

    # Set the call_sid context var so all handlers know which call they belong to
    _current_call_sid.set(call_id)

    # Extract client-specific values
    agent_name = client_config.get("agentName", "Sarah")
    company_name = client_config.get("companyName", "the company")
    voice_id = client_config.get("voiceId") or DEFAULT_CARTESIA_VOICE_ID
    timezone = client_config.get("timezone", "America/Chicago")

    # ── Tracking ────────────────────────────────────────
    call_start_time = datetime.now(ZoneInfo(timezone))
    tool_calls_made = []
    last_caller_speech_time = datetime.now(ZoneInfo(timezone))

    # ── Greeting ────────────────────────────────────────
    greeting = f"Thanks for calling {company_name}, this is {agent_name}, how can I help you?"

    # ── Transport ───────────────────────────────────────
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=24000,
            audio_in_passthrough=True,
            audio_out_enabled=True,
            audio_out_sample_rate=8000,
            serializer=TwilioFrameSerializer(
                stream_sid=stream_id,
                call_sid=call_id,
                account_sid=TWILIO_ACCOUNT_SID,
                auth_token=TWILIO_AUTH_TOKEN,
                params=TwilioFrameSerializer.InputParams(sample_rate=24000),
            ),
        ),
    )

    # ── Realtime LLM (audio understanding + reasoning + tools) ───────
    llm = OpenAIRealtimeLLMService(
        api_key=OPENAI_API_KEY,
        settings=OpenAIRealtimeLLMService.Settings(
            model=REALTIME_MODEL,
            system_instruction=build_system_prompt(client_config),
            session_properties=SessionProperties(
                output_modalities=["text"],
                audio=AudioConfiguration(
                    input=AudioInput(
                        format=PCMAudioFormat(),
                        transcription=InputAudioTranscription(
                            model="gpt-4o-transcribe",
                            language="en",
                        ),
                        turn_detection=SemanticTurnDetection(
                            eagerness="high",
                            create_response=True,
                            interrupt_response=True,
                        ),
                    ),
                ),
                tools=tools,
                tool_choice="auto",
                max_output_tokens=512,
            ),
        ),
    )

    # ── TTS ─────────────────────────────────────────────
    tts = CartesiaContinuationTTS(
        api_key=CARTESIA_API_KEY,
        voice_id=voice_id,
        model=CARTESIA_MODEL,
        sample_rate=8000,
        max_buffer_delay_ms=150,
        aggregate_sentences=False,
        params=CartesiaTTSService.InputParams(
            speed="slow",
        ),
    )

    # ── Context ─────────────────────────────────────────
    system_prompt = build_system_prompt(client_config)
    messages = [{"role": "system", "content": system_prompt}]

    context = LLMContext(messages=messages, tools=tools)

    async def maybe_compress_context():
        if should_compress(context.messages):
            try:
                summary_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
                context.set_messages(await compress_context(context.messages, summary_client))
            except Exception as e:
                logger.warning(f"Context compression failed (non-fatal): {e}")

    def record_user_transcript(text: str):
        nonlocal last_caller_speech_time
        last_caller_speech_time = datetime.now(ZoneInfo(timezone))
        context.add_message({"role": "user", "content": text})
        asyncio.create_task(maybe_compress_context())

    def record_assistant_text(text: str):
        context.add_message({"role": "assistant", "content": text})
        asyncio.create_task(maybe_compress_context())

    user_transcript_tracker = RealtimeTranscriptTracker(
        name="RealtimeUserTranscriptTracker",
        on_user_transcript=record_user_transcript,
        on_assistant_text=lambda _text: None,
    )
    assistant_text_tracker = RealtimeTranscriptTracker(
        name="RealtimeAssistantTextTracker",
        on_user_transcript=lambda _text: None,
        on_assistant_text=record_assistant_text,
    )

    # Set call context for handlers (passes client_config through)
    set_call_context(call_id, caller_number, client_config)

    # ── Register Tool Handlers ──────────────────────────

    llm.register_function(
        "create_booking", handle_create_booking, cancel_on_interruption=False
    )
    llm.register_function(
        "check_container_availability", handle_check_container_availability
    )
    llm.register_function("lookup_appointment", handle_lookup_appointment)
    llm.register_function(
        "reschedule_appointment", handle_reschedule_appointment, cancel_on_interruption=False
    )
    llm.register_function(
        "cancel_appointment", handle_cancel_appointment, cancel_on_interruption=False
    )
    llm.register_function(
        "schedule_callback", handle_schedule_callback, cancel_on_interruption=False
    )
    llm.register_function(
        "transfer_to_human", handle_transfer_to_human, cancel_on_interruption=False
    )
    llm.register_function("validate_promo_code", handle_validate_promo_code)
    llm.register_function("check_available_slots", handle_check_available_slots)
    llm.register_function("record_sms_consent", handle_record_sms_consent)
    llm.register_function("verify_address", handle_verify_address)

    # ── Pipeline ────────────────────────────────────────
    sentence_aggregator = SentenceAggregator()
    prosody = ProsodyProcessor()

    pipeline = Pipeline([
        transport.input(),
        user_transcript_tracker,
        llm,
        assistant_text_tracker,
        sentence_aggregator,
        prosody,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=24000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Give handlers access to the task for pipeline cancellation (e.g. after transfer)
    set_pipeline_task(call_id, task)

    # ── Event Handlers ──────────────────────────────────

    # Track background watcher tasks so we can cancel them on disconnect
    _timeout_task = None
    _booking_watcher_task = None
    _pre_silence_watcher_task = None

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        nonlocal _timeout_task, _booking_watcher_task, _pre_silence_watcher_task
        logger.info(f"Call connected: {call_id} from {caller_number} → {company_name}")

        # Play greeting immediately via TTS frame
        await task.queue_frames([TTSSpeakFrame(text=greeting)])

        # Start call duration timer, post-booking silence watcher, and pre-booking silence watcher
        _timeout_task = asyncio.create_task(_call_timeout_watcher(task, context))
        _booking_watcher_task = asyncio.create_task(_post_booking_watcher(task, context, call_id))
        _pre_silence_watcher_task = asyncio.create_task(_pre_booking_silence_watcher(task, context, call_id))

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info(f"Call disconnected: {call_id}")

        # Cancel background watcher tasks
        if _timeout_task and not _timeout_task.done():
            _timeout_task.cancel()
        if _booking_watcher_task and not _booking_watcher_task.done():
            _booking_watcher_task.cancel()
        if _pre_silence_watcher_task and not _pre_silence_watcher_task.done():
            _pre_silence_watcher_task.cancel()

        # Wait for any in-flight tool tasks (e.g. create_booking API request
        # in mid-flight) before snapshotting state. Prevents "ghost bookings"
        # where the dashboard creates the booking but the call log records
        # outcome="info_only" because mark_booking_complete didn't fire in time.
        inflight = get_inflight_tasks(call_id)
        if inflight:
            logger.info(f"Waiting up to {INFLIGHT_TASK_WAIT_SECONDS}s for {len(inflight)} in-flight tool task(s) before snapshotting {call_id}")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*inflight, return_exceptions=True),
                    timeout=INFLIGHT_TASK_WAIT_SECONDS
                )
                logger.info(f"In-flight tasks completed for {call_id}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for in-flight tasks on {call_id} — snapshotting anyway")

        # Keep per-call context until after summary/logging so a Twilio
        # transfer-status callback can still update the state if it arrives fast.
        saved_caller = caller_number
        saved_config = dict(client_config)  # shallow copy
        saved_twilio_number = client_config.get("twilioNumber", "")

        # Generate post-call summary
        call_end_time = datetime.now(ZoneInfo(timezone))
        duration_s = int((call_end_time - call_start_time).total_seconds())

        transcript_messages = [
            m for m in context.messages
            if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
        ]

        try:
            if not transcript_messages:
                summary = "No caller conversation captured. Caller disconnected before a real exchange was recorded."
            else:
                summary_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
                summary_resp = await summary_client.chat.completions.create(
                    model=UTILITY_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "Summarize this phone call in one sentence using only the caller and agent transcript provided. Do not infer that a booking, cancellation, reschedule, or transfer happened unless it is explicitly present in the transcript.",
                        },
                        *transcript_messages[-10:],
                    ],
                    max_tokens=100,
                )
                summary = summary_resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Fallback: build a minimal transcript from the last 10 real turns so
            # the operator has SOME context instead of "Summary unavailable".
            try:
                transcript_parts = []
                for m in transcript_messages[-10:]:
                    role = m.get("role", "")
                    content = str(m.get("content", "")).strip()[:80]
                    if content:
                        label = "Caller" if role == "user" else "Agent"
                        transcript_parts.append(f"{label}: {content}")
                transcript = " | ".join(transcript_parts)[:500]
                summary = (
                    f"AI summary unavailable. Recent transcript: {transcript}"
                    if transcript
                    else "Summary unavailable"
                )
            except Exception as fallback_err:
                logger.error(f"Transcript fallback also failed: {fallback_err}")
                summary = "Summary unavailable"

        logger.info(f"Call summary [{call_id}] ({duration_s}s): {summary}")

        # Capture state after summary generation, immediately before logging.
        booked = is_booking_complete(call_id)
        transferred = was_transfer_complete(call_id)
        transfer_state = get_transfer_state(call_id)
        booking_log_state = get_booking_log_state(call_id)
        sms_consented = has_sms_consent(call_id)
        transfer_status = transfer_state.get("transfer_status")
        callback_requested = bool(transfer_state.get("callback_requested"))
        pending_transfer = get_pending_transfer_state(call_id)
        if pending_transfer.get("terminal_logged"):
            transfer_status = pending_transfer.get("transfer_status") or transfer_status
            callback_requested = bool(pending_transfer.get("callback_requested"))
            if pending_transfer.get("summary"):
                summary = pending_transfer["summary"]

        # Determine call outcome
        if booked:
            outcome = "booked"
        elif callback_requested or transfer_status in ("failed", "unavailable_no_forwarding_phone"):
            outcome = "callback_requested"
        elif transferred or transfer_status in ("requested", "dialing", "completed"):
            outcome = "transferred"
        elif duration_s < 10:
            outcome = "voicemail"
        else:
            outcome = "info_only"

        caller_name = str(booking_log_state.get("caller_name") or "")
        appointment_date = str(booking_log_state.get("appointment_date") or "")

        update_pending_transfer_snapshot(
            call_id,
            duration=duration_s,
            summary=summary or "",
            from_number=saved_caller or "",
            to_number=saved_twilio_number,
            config=saved_config,
            sms_consent=sms_consented,
            callback_requested=callback_requested,
            callback_due_at=transfer_state.get("callback_due_at"),
            caller_name=caller_name,
            appointment_date=appointment_date,
        )

        latest_pending_transfer = get_pending_transfer_state(call_id)
        if latest_pending_transfer.get("terminal_logged"):
            transfer_status = latest_pending_transfer.get("transfer_status") or transfer_status
            callback_requested = bool(latest_pending_transfer.get("callback_requested"))
            if latest_pending_transfer.get("summary"):
                summary = latest_pending_transfer["summary"]
            outcome = "callback_requested" if callback_requested or transfer_status == "failed" else "transferred"

        # Log call to dashboard and wait briefly so the process cannot exit before
        # the POST is sent. Dashboard returns 201 on create and 200 on duplicate update.
        try:
            await asyncio.wait_for(
                log_call_to_dashboard(
                    config=saved_config,
                    twilio_call_sid=call_id,
                    from_number=saved_caller or "",
                    to_number=saved_twilio_number,
                    duration=duration_s,
                    outcome=outcome,
                    summary=summary or "",
                    caller_name=caller_name,
                    appointment_date=appointment_date,
                    sms_consent=sms_consented,
                    transfer_reason=transfer_state.get("transfer_reason"),
                    transfer_status=transfer_status,
                    callback_requested=callback_requested,
                    callback_due_at=transfer_state.get("callback_due_at"),
                ),
                timeout=CALL_LOG_WAIT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(f"Timed out logging call {call_id} to dashboard")

        clear_pending_transfer_if_terminal(call_id)

        # Clean up per-call context after all state-dependent work is done.
        clear_call_context(call_id)

        # Schedule automated follow-up SMS only if consent was explicitly given
        if not booked and saved_caller and sms_consented is True:
            asyncio.create_task(
                _delayed_followup_sms(saved_caller, saved_config, delay_seconds=300)
            )

        await task.cancel()

    # ── Pre-Booking Silence Watcher ────────────────────

    async def _pre_booking_silence_watcher(task, context, cid: str):
        """Catch callers who never speak after the greeting (butt-dial, technical
        issues, hesitant callers). At 30s of silence, prompt once. At 45s, say a
        graceful goodbye and end. Stops as soon as the caller speaks OR a booking
        starts (post-booking watcher takes over)."""
        INITIAL_PROMPT_AT_SECONDS = 30
        GOODBYE_AT_SECONDS = 45

        try:
            # Capture the timestamp at watcher start. If it changes, caller spoke.
            initial_timestamp = last_caller_speech_time
            nudge_sent = False

            while True:
                await asyncio.sleep(3)

                # If a booking has completed, the post-booking watcher handles it
                if is_booking_complete(cid):
                    return

                # If the caller has spoken at any point, our job is done
                if last_caller_speech_time != initial_timestamp:
                    return

                now = datetime.now(ZoneInfo(timezone))
                elapsed = (now - initial_timestamp).total_seconds()

                if elapsed >= INITIAL_PROMPT_AT_SECONDS and not nudge_sent:
                    logger.info(f"Pre-booking silence at {INITIAL_PROMPT_AT_SECONDS}s for {cid} — sending nudge")
                    await _queue_context_instruction(
                        task,
                        context,
                        "The caller hasn't said anything yet. Gently check in: 'Hello? Are you still there?'",
                    )
                    nudge_sent = True

                if elapsed >= GOODBYE_AT_SECONDS:
                    logger.info(f"Pre-booking silence at {GOODBYE_AT_SECONDS}s — ending call {cid}")
                    await _queue_context_instruction(
                        task,
                        context,
                        "The caller still hasn't responded. Say briefly: 'Sounds like you might have called by mistake. Feel free to call us back anytime — have a great day!' Then stop talking.",
                    )
                    await asyncio.sleep(8)
                    await task.cancel()
                    return

        except asyncio.CancelledError:
            pass  # Call ended before silence watcher fired

    # ── Post-Booking Silence Watcher ───────────────────

    async def _post_booking_watcher(task, context, cid: str):
        """After a booking completes, wait for the LLM to finish its natural
        response, then start a silence timer. Nudge at 15s, hang up at 25s.
        If the caller speaks, the timer resets."""
        POST_BOOKING_SILENCE_SECONDS = 25
        NUDGE_AFTER_SECONDS = 15

        try:
            # Phase 1: Wait for booking to complete
            while not is_booking_complete(cid):
                await asyncio.sleep(2)

            logger.info(f"Booking complete for {cid} — starting post-booking silence watcher")

            # Give the LLM time to finish its natural response
            # (the system prompt already instructs it to ask "anything else?")
            await asyncio.sleep(10)

            # Reset speech timer so the countdown starts fresh
            nonlocal last_caller_speech_time
            last_caller_speech_time = datetime.now(ZoneInfo(timezone))
            nudge_sent = False

            # Phase 2: Silence timer with nudge and speech reset
            while True:
                now = datetime.now(ZoneInfo(timezone))
                elapsed = (now - last_caller_speech_time).total_seconds()

                # At 15s: gentle nudge
                if elapsed >= NUDGE_AFTER_SECONDS and not nudge_sent:
                    await _queue_context_instruction(
                        task,
                        context,
                        "The caller has been quiet. Gently ask: 'Was there anything else you needed help with?'",
                    )
                    nudge_sent = True

                # At 25s: say goodbye and hang up
                if elapsed >= POST_BOOKING_SILENCE_SECONDS:
                    logger.info(f"Post-booking silence ({POST_BOOKING_SILENCE_SECONDS}s) — ending call {cid}")

                    await _queue_context_instruction(
                        task,
                        context,
                        "The caller has been silent for a while. Say a brief, warm goodbye like: 'Alright, sounds like we're all set! We'll see you on your scheduled day. Have a great one!' Then stop talking.",
                    )

                    await asyncio.sleep(8)
                    await task.cancel()
                    return

                await asyncio.sleep(3)

        except asyncio.CancelledError:
            pass  # Call ended before silence timer fired

    # ── Call Timer ──────────────────────────────────────

    async def _call_timeout_watcher(task, context):
        """Enforce max call duration."""
        try:
            await asyncio.sleep(MAX_CALL_DURATION_SECONDS - 60)

            # Warn at 9 minutes
            await _queue_context_instruction(
                task,
                context,
                "The call is approaching 10 minutes. Naturally wrap up the conversation — say something like 'I want to make sure I'm not keeping you too long. Is there anything else I can quickly help with?'",
            )

            await asyncio.sleep(60)

            # Force graceful end
            await _queue_context_instruction(
                task,
                context,
                "The call has reached 10 minutes. Say goodbye warmly and end the call.",
            )
            await asyncio.sleep(10)
            await task.cancel()
        except asyncio.CancelledError:
            pass  # Call ended before timeout

    # ── Run ─────────────────────────────────────────────
    runner = PipelineRunner()
    await runner.run(task)
