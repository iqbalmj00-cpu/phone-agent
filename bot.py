"""ScaleYourJunk Multi-Tenant AI Phone Agent — Pipecat Pipeline.

Builds and runs the full voice agent pipeline per call:
  Twilio audio in → Deepgram Flux STT → Claude Sonnet tools/text → Cartesia TTS → Twilio audio out

Each call gets its own pipeline with the client's config (agent name, voice,
company name, hours, etc.) loaded dynamically from the dashboard.

Handles:
  - Pre-generated greeting for zero-latency first utterance
  - Tool registration with cancel_on_interruption settings
  - Call duration timer (10 min)
  - Post-call summary
  - Conversation context through Pipecat context aggregators
"""

import asyncio
import time
from agent.silence import watch_silence
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from anthropic import AsyncAnthropic
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService, GenerationConfig
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from agent.prosody import CartesiaContinuationTTS, ProsodyProcessor

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    ANTHROPIC_UTILITY_MODEL,
    CARTESIA_API_KEY,
    DEFAULT_CARTESIA_VOICE_ID,
    CARTESIA_MODEL,
    CARTESIA_SPEED,
    DEEPGRAM_API_KEY,
    DEEPGRAM_MODEL,
    MAX_CALL_DURATION_SECONDS,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
)
from agent.prompt import build_greeting, build_system_prompt, client_supports_dumpsters, client_supports_junk
from agent.handlers import (
    handle_create_booking,
    handle_check_container_availability,
    handle_check_available_slots,
    handle_lookup_appointment,
    handle_verify_caller_identity,
    handle_reschedule_appointment,
    handle_cancel_appointment,
    handle_schedule_callback,
    handle_transfer_to_human,
    handle_validate_promo_code,
    handle_record_caller_name,
    handle_record_sms_consent,
    handle_verify_address,
    set_call_context,
    set_pipeline_task,
    clear_call_context,
    is_booking_complete,
    was_transfer_complete,
    get_transfer_state,
    get_booking_log_state,
    get_booking_outcome,
    create_lead_from_call,
    get_caller_name,
    has_sms_consent,
    usable_inquiry,
    log_call_to_dashboard,
    get_inflight_tasks,
    is_dashboard_owned_transfer,
    _current_call_sid,
)


# ── Automated Follow-Up SMS ─────────────────────────────

INFLIGHT_TASK_WAIT_SECONDS = 17  # Covers 15s dashboard tool calls plus small scheduling overhead
CALL_LOG_WAIT_SECONDS = 12
SUMMARY_WAIT_SECONDS = 8
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 8000
SILENCE_SENTINEL = "NO_RESPONSE_NEEDED"
CARTESIA_SPEED_PRESETS = {
    "slowest": 0.85,
    "slow": 0.95,
    "normal": 1.0,
    "fast": 1.08,
    "fastest": 1.15,
}


def should_write_final_call_log(call_id: str) -> bool:
    """Final phone-agent logs must not overwrite dashboard-owned transfer results."""
    return not is_dashboard_owned_transfer(call_id)


class CallerSpeechTracker(FrameProcessor):
    """Observe final caller transcripts for silence timers.

    Pipecat's context aggregators own conversation history in the cascaded
    pipeline. This processor only records that the caller spoke.
    """

    def __init__(self, *, on_user_transcript, **kwargs):
        super().__init__(**kwargs)
        self._on_user_transcript = on_user_transcript
        self.user_speaking = False
        self.bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
            self.user_speaking = isinstance(frame, UserStartedSpeakingFrame)
            self._on_user_transcript("")
        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            self.bot_speaking = isinstance(frame, BotStartedSpeakingFrame)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            self._on_user_transcript(frame.text)

        await self.push_frame(frame, direction)


class NoResponseNeededFilter(FrameProcessor):
    """Drop the private LLM silence sentinel before it reaches TTS."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._checking_response = True
        self._dropping_sentinel = False
        self._candidate_frames: list[TextFrame] = []

    @staticmethod
    def _normalize_candidate(text: str) -> str:
        return "".join(text.strip().split()).rstrip(".!").upper()

    async def _flush_candidate_frames(self, direction: FrameDirection):
        for buffered_frame in self._candidate_frames:
            await self.push_frame(buffered_frame, direction)
        self._candidate_frames = []

    def _reset_response_state(self):
        self._checking_response = True
        self._dropping_sentinel = False
        self._candidate_frames = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMFullResponseStartFrame):
            self._reset_response_state()

        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._dropping_sentinel:
                return

            if self._checking_response:
                candidate_text = "".join(buffered.text or "" for buffered in self._candidate_frames)
                candidate_text += frame.text or ""
                normalized = self._normalize_candidate(candidate_text)

                if not normalized or SILENCE_SENTINEL.startswith(normalized):
                    self._candidate_frames.append(frame)
                    if normalized == SILENCE_SENTINEL:
                        self._candidate_frames = []
                        self._dropping_sentinel = True
                        logger.debug("Dropped LLM silence sentinel before TTS")
                    return

                self._checking_response = False
                await self._flush_candidate_frames(direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMFullResponseEndFrame):
            if self._dropping_sentinel:
                self._reset_response_state()
                await self.push_frame(frame, direction)
                return

            if self._candidate_frames:
                candidate_text = "".join(buffered.text or "" for buffered in self._candidate_frames)
                normalized = self._normalize_candidate(candidate_text)
                if normalized == SILENCE_SENTINEL:
                    logger.debug("Dropped LLM silence sentinel before TTS")
                    self._reset_response_state()
                    await self.push_frame(frame, direction)
                    return

                await self._flush_candidate_frames(direction)

            self._reset_response_state()
            await self.push_frame(frame, direction)
            return

        if self._dropping_sentinel and direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TextFrame):
                return

        await self.push_frame(frame, direction)


async def _queue_context_instruction(task: PipelineTask, context: LLMContext, content: str) -> None:
    """Append a task-specific developer instruction and trigger one response."""
    message = {"role": "developer", "content": content}
    await task.queue_frames([LLMMessagesAppendFrame([message], run_llm=True)])


def _message_content_to_text(content: Any) -> str:
    """Return readable text from Pipecat message content."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(content or "").strip()


def _format_transcript_lines(messages: list[dict]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        text = _message_content_to_text(message.get("content"))
        if not text:
            continue
        label = "Caller" if role == "user" else "Agent"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _extract_anthropic_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
        elif isinstance(block, dict) and block.get("text"):
            parts.append(str(block["text"]))
    return " ".join(parts).strip()


def _cartesia_speed_multiplier(value: str) -> float:
    """Map env-friendly speed values to Cartesia Sonic 3 generation_config.speed."""
    normalized = (value or "normal").strip().lower()
    if normalized in CARTESIA_SPEED_PRESETS:
        return CARTESIA_SPEED_PRESETS[normalized]

    try:
        numeric = float(normalized)
    except ValueError:
        logger.warning(f"Invalid CARTESIA_SPEED={value!r}; using normal speed")
        return CARTESIA_SPEED_PRESETS["normal"]

    if 0.6 <= numeric <= 1.5:
        return numeric

    logger.warning(f"CARTESIA_SPEED={value!r} is outside 0.6-1.5; using normal speed")
    return CARTESIA_SPEED_PRESETS["normal"]


async def _summarize_call_with_anthropic(transcript_messages: list[dict]) -> str:
    # The caller says why they rang in the FIRST turn or two. A trailing
    # ten-message window discards exactly that on any call past a few
    # exchanges, and this summary is what an operator reads to decide who to
    # call back. Nothing truncates `context.messages` upstream, and a call is
    # capped at ten minutes, so the whole conversation is a few thousand
    # tokens — cheap to send in full.
    transcript = _format_transcript_lines(transcript_messages)
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model=ANTHROPIC_UTILITY_MODEL,
        max_tokens=200,  # the instruction caps the length; this is headroom for the second sentence
        temperature=0,
        system=(
            "Summarize this phone call in one or two sentences using only the "
            "caller and agent transcript provided. Lead with what the caller "
            "actually wanted — the job, the item, the problem, or the question "
            "they rang about — because that is what the operator needs in order "
            "to call them back. Do not infer that a booking, cancellation, "
            "reschedule, transfer, or callback happened unless it is explicitly "
            "present in the transcript."
        ),
        messages=[{"role": "user", "content": f"Transcript:\n{transcript}"}],
    )
    return _extract_anthropic_text(response) or "Summary unavailable"


# ── Post-Call Finalization ──────────────────────────────


class _OnceGuard:
    """Claim-once flag guarding work that has two possible entry points.

    `claim()` is synchronous and flips the flag before returning, so two
    coroutines racing into the same guarded block cannot both win — asyncio
    only switches tasks at an await, and there isn't one here.
    """

    def __init__(self) -> None:
        self._claimed = False

    def claim(self) -> bool:
        if self._claimed:
            return False
        self._claimed = True
        return True


async def run_post_call(
    *,
    call_id: str,
    caller_number: str,
    client_config: dict,
    context: LLMContext,
    call_start_time: datetime,
    timezone: str,
) -> None:
    """Summarize the call, log it to the dashboard, and release per-call state.

    Module-level rather than nested inside `run_bot` for two reasons: it closes
    over nothing, and a nested version could only be reached through a live
    websocket transport, which put it beyond the reach of any test.

    Runs exactly once per call — see `_OnceGuard` at the call site.
    """
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
            summary = await asyncio.wait_for(
                _summarize_call_with_anthropic(transcript_messages),
                timeout=SUMMARY_WAIT_SECONDS,
            )
    except asyncio.TimeoutError:
        logger.error(f"Summary generation timed out after {SUMMARY_WAIT_SECONDS}s")
        try:
            transcript_parts = []
            for m in transcript_messages[-10:]:
                role = m.get("role", "")
                content = _message_content_to_text(m.get("content"))[:80]
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
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        # Fallback: build a minimal transcript from the last 10 real turns so
        # the operator has SOME context instead of "Summary unavailable".
        try:
            transcript_parts = []
            for m in transcript_messages[-10:]:
                role = m.get("role", "")
                content = _message_content_to_text(m.get("content"))[:80]
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
    dashboard_owned_transfer = not should_write_final_call_log(call_id)
    transfer_status = transfer_state.get("transfer_status")
    callback_requested = bool(transfer_state.get("callback_requested"))
    # Determine call outcome
    operation = get_booking_outcome(call_id)
    if operation in ("rescheduled", "cancelled"):
        outcome = operation
    elif operation in ("pending", "uncertain"):
        outcome = "booking_" + operation
    elif booked:
        outcome = "booked"
    elif callback_requested or transfer_status in ("failed", "unavailable_no_forwarding_phone"):
        outcome = "callback_requested"
    elif transferred or transfer_status in ("requested", "dialing", "completed"):
        outcome = "transferred"
    elif duration_s < 10:
        outcome = "voicemail"
    else:
        outcome = "info_only"

    caller_name = get_caller_name(call_id) or str(booking_log_state.get("caller_name") or "")
    appointment_date = "" if operation in ("uncertain", "pending") else str(booking_log_state.get("appointment_date") or "")

    # Dashboard owns routing; its receiver permits safe contact/summary enrichment.
    # Log call to dashboard and wait briefly so the process cannot exit before
    # the POST is sent. Dashboard returns 201 on create and 200 on duplicate update.
    #
    # Retry on the RESULT, not on a timeout. `log_call_to_dashboard` sets its
    # own 10s client timeout and swallows every failure into `False`, so the
    # 12s wait_for below almost never raises — an earlier version retried on
    # `asyncio.TimeoutError` and could therefore never fire at all. Retrying
    # is safe because the route looks the call up by its Twilio SID and
    # updates rather than inserting; lead capture below has no such key and
    # is never retried.
    log_args = dict(
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
        callback_phone=transfer_state.get("callback_phone"),
        callback_source=transfer_state.get("callback_source"),
    )
    logged = False
    for attempt in (1, 2):
        try:
            logged = await asyncio.wait_for(
                log_call_to_dashboard(**log_args),
                timeout=CALL_LOG_WAIT_SECONDS,
            )
        except asyncio.TimeoutError:
            logged = False
        if logged:
            break
        if attempt == 1:
            logger.warning(f"Call log for {call_id} did not land; retrying once")
    if not logged:
        logger.error(f"Call log for {call_id} failed twice; giving up")

    # A caller who gave their name but did not book is still worth recording —
    # that is the whole reason the opening line asks who is calling (D12).
    #
    # Only `info_only` qualifies. Every other outcome is already accounted for
    # somewhere: `booked` has a job, `voicemail` has nobody on the line, and
    # every route to `callback_requested` — the after-hours handoff refusal, an
    # explicitly scheduled callback, and a failed transfer — has already put the
    # caller in the dashboard's Callback Queue. Making those a lead as well
    # would cold-text someone the operator is already meant to be ringing back
    # (D27).
    #
    # This runs BEFORE clear_call_context below, which wipes the context that
    # `caller_name` was read from.
    if outcome == "info_only" and not dashboard_owned_transfer and caller_name and saved_caller:
        await create_lead_from_call(
            config=saved_config,
            name=caller_name,
            phone=saved_caller,
            description=usable_inquiry(summary or ""),
            sms_consent=sms_consented is True,
        )

    # Clean up per-call context after all state-dependent work is done. This also
    # drops the retained PipelineTask, which holds a per-call VAD session.
    clear_call_context(call_id)



# ── Tool Definitions ────────────────────────────────────

def _create_booking_schema(dumpster_enabled: bool, junk_enabled: bool = True) -> FunctionSchema:
    properties = {
        "name": {"type": "string", "description": "Customer full name"},
        "phone": {"type": "string", "description": "Customer phone number"},
        "address": {"type": "string", "description": "Service address"},
        "date": {
            "type": "string",
            "description": "Internal YYYY-MM-DD date value resolved from caller's natural wording. Never ask the caller to say this format.",
        },
        "time": {
            "type": "string",
            "description": "Time slot as an explicit 24-hour range, such as 08:00-11:00. Use a range returned by check_available_slots. Do not send a label like morning or afternoon — the dashboard has no 'afternoon' and silently records it as a midday window, so the caller is promised one arrival time and booked for another.",
        },
        "description": {"type": "string", "description": "Items for removal"},
        "type": {
            "type": "string",
            "enum": ["pickup"],
            "description": "Use pickup for every junk removal job, including estimates and on-site quote requests.",
        },
        "booking_reference": {"type": "string", "description": "Keep primary for the first booking and all its retries. Use a distinct stable reference only when the caller explicitly requests another separate booking; keep that reference on retries."},
        "job_id": {"type": "string", "description": "Verified lookup job ID anchoring the selected rental; required for swap/final pickup."},
        "rental_address_confirmed": {"type": "boolean", "description": "True only after caller confirms the selected rental stored address and pickup versus swap."},
        "promo_code": {"type": "string", "description": "Validated promo code to apply discount (optional)"},
    }
    description = (
        "Book a junk removal pickup. ONLY call this after reading back all details "
        "and receiving verbal confirmation from the caller."
    )

    if dumpster_enabled:
        properties["description"] = {
            "type": "string",
            "description": "Items for removal, project description for dumpster, or swap reason",
        }
        properties["type"] = {
            "type": "string",
            "enum": (["pickup"] if junk_enabled else []) + ["dumpster_rental", "dumpster_swap", "dumpster_pickup"],
            "description": (
                "Use pickup for all junk removal jobs, including estimate requests. "
                "Use dumpster_rental for new dumpster deliveries and dumpster_swap "
                "for swapping a full container for an empty one. Use dumpster_pickup for final collection with no replacement."
            ),
        }
        properties["container_size"] = {
            "type": "string",
            "enum": ["10", "15", "20", "30", "40"],
            "description": "Container size in cubic yards for dumpster_rental or dumpster_swap",
        }
        properties["rental_duration_days"] = {
            "type": "integer",
            "description": "Rental duration in days, default 7, only for dumpster_rental",
        }
        description = (
            "Book a junk removal pickup, dumpster rental, or dumpster swap. ONLY call "
            "this after reading back all details and receiving verbal confirmation "
            "from the caller."
        )

    if dumpster_enabled and not junk_enabled:
        description = "Book a dumpster rental, swap, or final pickup only after reading all details back and receiving explicit caller confirmation."
        properties["type"]["description"] = "dumpster_rental is a new delivery; dumpster_swap replaces an on-site box; dumpster_pickup collects the box with no replacement."

    required = ["name", "phone", "address", "date", "description", "type"]
    if not dumpster_enabled:
        required.insert(4, "time")

    return FunctionSchema(
        name="create_booking",
        description=description,
        properties=properties,
        required=required,
    )


def _check_container_availability_schema() -> FunctionSchema:
    return FunctionSchema(
        name="check_container_availability",
        description=(
            "Check if a dumpster container of the requested size is available for a "
            "specific delivery date and get live pricing. Call this AFTER collecting "
            "the customer's preferred delivery date and rental duration."
        ),
        properties={
            "size": {
                "type": "string",
                "enum": ["10", "15", "20", "30", "40"],
                "description": "Container size in cubic yards",
            },
            "date": {
                "type": "string",
                "description": "Internal YYYY-MM-DD delivery date resolved from caller wording. Never ask the caller to say this format.",
            },
            "days": {"type": "integer", "description": "Rental duration in days, default 7"},
        },
        required=["size"],
    )


def build_tools(dumpster_enabled: bool, junk_enabled: bool = True) -> ToolsSchema:
    """Build the LLM tool surface for the client's actual capabilities."""
    standard_tools = [
        _create_booking_schema(dumpster_enabled, junk_enabled),
        FunctionSchema(
            name="lookup_appointment",
            description=(
                "Find an existing appointment by phone number. If the caller is not "
                "ringing from the number on the booking, this returns no details and "
                "asks you to verify them first — that is expected, not an error."
            ),
            properties={
                "phone": {"type": "string", "description": "Phone number to search"},
            },
            required=["phone"],
        ),
        FunctionSchema(
            name="verify_caller_identity",
            description=(
                "Confirm a caller owns a booking when they are calling from a different "
                "number. Ask them for the service address, then pass exactly what they "
                "said. Only call this after lookup_appointment asked you to. Never tell "
                "the caller the address or any other booking detail first — they must "
                "produce it themselves."
            ),
            properties={
                "address": {
                    "type": "string",
                    "description": "The service address exactly as the caller stated it",
                },
            },
            required=["address"],
        ),
        FunctionSchema(
            name="reschedule_appointment",
            description="Reschedule an existing appointment. Call check_available_slots first to find open times, then read back new details and confirm.",
            properties={
                "phone": {"type": "string", "description": "Customer phone to find booking"},
                "new_date": {
                    "type": "string",
                    "description": "Internal YYYY-MM-DD date value resolved from caller's natural wording. Never ask the caller to say this format.",
                },
                "new_time": {"type": "string", "description": "Time slot as an explicit 24-hour range, such as 08:00-11:00. Use a range returned by check_available_slots. Do not send a label like morning or afternoon — the dashboard has no 'afternoon' and silently records it as a midday window."},
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
                "requested_time": {
                    "type": "string",
                    "description": "Internal YYYY-MM-DDTHH:MM:SS callback timestamp resolved from caller wording in the client's timezone. Never ask the caller to say this format.",
                },
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
                "service_type": {"type": "string", "enum": ["junk", "dumpster"], "description": "Service receiving this discount"},
            },
            required=["code", "service_type"],
        ),
        FunctionSchema(
            name="check_available_slots",
            description="Check which business-hour booking windows are available for a specific date. Call this BEFORE offering times to the caller. This is not a junk-removal job capacity check.",
            properties={
                "date": {
                    "type": "string",
                    "description": "Internal YYYY-MM-DD date value resolved from caller's natural wording. Never ask the caller to say this format.",
                },
            },
            required=["date"],
        ),
        FunctionSchema(
            name="record_caller_name",
            description="Record the caller's name. Call this as soon as they tell you who they are — usually in answer to the opening line. Call it once. If they never give a name, do not call this and do not keep asking.",
            properties={
                "name": {"type": "string", "description": "The caller's name exactly as they said it. Do not guess, and do not pass a placeholder like 'unknown' or 'customer'."},
            },
            required=["name"],
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
            description="Verify and normalize a service address using Google Maps. Call this after the caller gives their address; if they only gave a street number and street name, the tool will use the client's city/state context. Read back spoken_address and use formatted_address for booking.",
            properties={
                "address": {"type": "string", "description": "The address as spoken by the caller, including street, city, and state if provided"},
            },
            required=["address"],
        ),
    ]

    if dumpster_enabled:
        standard_tools.insert(1, _check_container_availability_schema())

    return ToolsSchema(standard_tools=standard_tools)

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
    dumpster_enabled = client_supports_dumpsters(client_config)
    cartesia_speed = _cartesia_speed_multiplier(CARTESIA_SPEED)

    # ── Tracking ────────────────────────────────────────
    call_start_time = datetime.now(ZoneInfo(timezone))
    last_caller_speech_time = datetime.now(ZoneInfo(timezone))

    # ── Greeting ────────────────────────────────────────
    greeting = build_greeting(company_name, agent_name)

    # ── Transport ───────────────────────────────────────
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_in_passthrough=True,
            audio_out_enabled=True,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
            serializer=TwilioFrameSerializer(
                stream_sid=stream_id,
                call_sid=call_id,
                account_sid=TWILIO_ACCOUNT_SID,
                auth_token=TWILIO_AUTH_TOKEN,
                params=TwilioFrameSerializer.InputParams(sample_rate=INPUT_SAMPLE_RATE),
            ),
        ),
    )

    # ── STT + LLM context ─────────────────────────────────
    stt = DeepgramFluxSTTService(
        api_key=DEEPGRAM_API_KEY,
        sample_rate=INPUT_SAMPLE_RATE,
        should_interrupt=True,
        settings=DeepgramFluxSTTService.Settings(
            model=DEEPGRAM_MODEL,
            language="en",
            eager_eot_threshold=0.5,
            eot_threshold=0.8,
        ),
    )

    system_prompt = build_system_prompt(client_config)
    initial_messages = []
    if caller_number:
        initial_messages.append({
            "role": "developer",
            "content": (
                "The caller's phone number from Twilio is "
                f"{caller_number}. When the booking flow asks for phone, "
                "confirm this number first instead of asking for it from scratch."
            ),
        })

    call_tools = build_tools(dumpster_enabled, client_supports_junk(client_config))
    logger.info(
        f"Call config: company={company_name!r}, dumpster_enabled={dumpster_enabled}, "
        f"cartesia_model={CARTESIA_MODEL}, cartesia_speed={cartesia_speed:.2f}, "
        f"tools={len(getattr(call_tools, 'standard_tools', []))}"
    )
    context = LLMContext(messages=initial_messages, tools=call_tools, tool_choice="auto")
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(sample_rate=INPUT_SAMPLE_RATE),
        ),
        assistant_params=LLMAssistantAggregatorParams(
            enable_auto_context_summarization=False,
        ),
    )

    # ── Claude LLM (reasoning + tools) ─────────────────────
    llm = AnthropicLLMService(
        api_key=ANTHROPIC_API_KEY,
        settings=AnthropicLLMService.Settings(
            model=ANTHROPIC_MODEL,
            system_instruction=system_prompt,
            max_tokens=512,
            temperature=0.3,
        ),
    )

    # ── TTS ─────────────────────────────────────────────
    tts = CartesiaContinuationTTS(
        api_key=CARTESIA_API_KEY,
        voice_id=voice_id,
        model=CARTESIA_MODEL,
        sample_rate=OUTPUT_SAMPLE_RATE,
        max_buffer_delay_ms=150,
        aggregate_sentences=False,
        params=CartesiaTTSService.InputParams(
            generation_config=GenerationConfig(speed=cartesia_speed),
        ),
    )

    def record_user_transcript(text: str):
        nonlocal last_caller_speech_time
        last_caller_speech_time = datetime.now(ZoneInfo(timezone))

    caller_speech_tracker = CallerSpeechTracker(
        name="CallerSpeechTracker",
        on_user_transcript=record_user_transcript,
    )

    # Set call context for handlers (passes client_config through)
    set_call_context(call_id, caller_number, client_config)

    # ── Register Tool Handlers ──────────────────────────

    llm.register_function(
        "create_booking", handle_create_booking, cancel_on_interruption=False
    )
    if dumpster_enabled:
        llm.register_function(
            "check_container_availability", handle_check_container_availability
        )
    llm.register_function("lookup_appointment", handle_lookup_appointment)
    llm.register_function("verify_caller_identity", handle_verify_caller_identity)
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
    llm.register_function("record_caller_name", handle_record_caller_name)
    llm.register_function("record_sms_consent", handle_record_sms_consent)
    llm.register_function("verify_address", handle_verify_address)

    # ── Pipeline ────────────────────────────────────────
    no_response_filter = NoResponseNeededFilter()
    sentence_aggregator = SentenceAggregator()
    prosody = ProsodyProcessor()

    pipeline = Pipeline([
        transport.input(),
        stt,
        caller_speech_tracker,
        context_aggregator.user(),
        llm,
        no_response_filter,
        sentence_aggregator,
        prosody,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
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
        _booking_watcher_task = None
        _pre_silence_watcher_task = asyncio.create_task(_pre_booking_silence_watcher(task, context, call_id))

    _finalize_guard = _OnceGuard()

    async def _finalize_call(reason: str):
        """Summarize, log and release the call, from whichever end fires first.

        Two paths lead here. A caller hangup arrives through
        `on_client_disconnected`. But when the agent ends the call itself —
        duration cap, either silence watcher, or a transfer — it does so via
        `task.cancel()`, and the CancelledError that follows is a BaseException,
        so it slips past the `except Exception` inside Pipecat's websocket
        transport and the disconnect event never fires. Those calls previously
        skipped this work entirely: no summary, no call log, and the per-call
        context (which holds the PipelineTask) was never released.
        """
        if not _finalize_guard.claim():
            return

        logger.info(f"Finalizing call {call_id} ({reason})")

        # Cancel background watcher tasks
        if _timeout_task and not _timeout_task.done():
            _timeout_task.cancel()
        if _booking_watcher_task and not _booking_watcher_task.done():
            _booking_watcher_task.cancel()
        if _pre_silence_watcher_task and not _pre_silence_watcher_task.done():
            _pre_silence_watcher_task.cancel()

        await run_post_call(
            call_id=call_id,
            caller_number=caller_number,
            client_config=client_config,
            context=context,
            call_start_time=call_start_time,
            timezone=timezone,
        )

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info(f"Call disconnected: {call_id}")
        await _finalize_call("client_disconnected")
        await task.cancel()

    # ── Pre-Booking Silence Watcher ────────────────────

    async def _pre_booking_silence_watcher(task, context, cid: str):
        try:
            await watch_silence(
                activity=lambda: last_caller_speech_time,
                busy=lambda: bool(get_inflight_tasks(cid)) or caller_speech_tracker.user_speaking or caller_speech_tracker.bot_speaking,
                booked=lambda: is_booking_complete(cid),
                speak=lambda text: _queue_context_instruction(task, context, text),
                cancel=task.cancel, now=time.monotonic,
            )
        except asyncio.CancelledError:
            pass

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
    try:
        await runner.run(task)
    finally:
        # The disconnect handler covers a caller hangup. This covers every call
        # the agent ends itself, where that handler never fires. Whichever runs
        # first claims the guard; the other returns immediately.
        await _finalize_call("pipeline_ended")
