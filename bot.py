"""ScaleYourJunk Multi-Tenant AI Phone Agent — Pipecat Pipeline.

Builds and runs the full voice agent pipeline per call:
  Twilio audio in → Deepgram STT → GPT-4.1 Mini (tools) → Cartesia TTS → Twilio audio out

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
from deepgram import LiveOptions
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.sentence import SentenceAggregator

from agent.prosody import CartesiaContinuationTTS, ProsodyProcessor

from config import (
    DEEPGRAM_API_KEY,
    OPENAI_API_KEY,
    CARTESIA_API_KEY,
    DEFAULT_CARTESIA_VOICE_ID,
    CARTESIA_MODEL,
    LLM_MODEL,
    MAX_CALL_DURATION_SECONDS,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
)
from agent.prompt import build_system_prompt
from agent.handlers import (
    handle_create_booking,
    handle_lookup_appointment,
    handle_reschedule_appointment,
    handle_cancel_appointment,
    set_call_context,
    clear_call_context,
    is_booking_complete,
)
from agent.context import should_compress, compress_context


# ── Tool Definitions ────────────────────────────────────

tools = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="create_booking",
        description="Book a junk removal pickup. ONLY call this after reading back all details and receiving verbal confirmation from the caller.",
        properties={
            "name": {"type": "string", "description": "Customer full name"},
            "phone": {"type": "string", "description": "Customer phone number"},
            "address": {"type": "string", "description": "Service address"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "time": {"type": "string", "description": "HH:MM (24-hour)"},
            "description": {"type": "string", "description": "Items for removal"},
        },
        required=["name", "phone", "address", "date", "time", "description"],
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
        description="Reschedule an existing appointment. Read back new details and confirm first.",
        properties={
            "phone": {"type": "string", "description": "Customer phone to find booking"},
            "new_date": {"type": "string", "description": "YYYY-MM-DD"},
            "new_time": {"type": "string", "description": "HH:MM"},
        },
        required=["phone", "new_date", "new_time"],
    ),
    FunctionSchema(
        name="cancel_appointment",
        description="Cancel an existing appointment",
        properties={
            "phone": {"type": "string", "description": "Customer phone to find booking"},
            "reason": {"type": "string", "description": "Reason for cancellation"},
        },
        required=["phone"],
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
            audio_in_sample_rate=8000,
            audio_out_enabled=True,
            audio_out_sample_rate=8000,
            vad_enabled=True,
            vad_audio_passthrough=True,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=TwilioFrameSerializer(
                stream_sid=stream_id,
                call_sid=call_id,
                account_sid=TWILIO_ACCOUNT_SID,
                auth_token=TWILIO_AUTH_TOKEN,
            ),
        ),
    )

    # ── STT ─────────────────────────────────────────────
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        live_options=LiveOptions(
            model="nova-3-general",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            sample_rate=8000,
            channels=1,
        ),
    )

    # ── LLM ─────────────────────────────────────────────
    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        params=OpenAILLMService.InputParams(
            temperature=0.7,
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

    context = OpenAILLMContext(messages=messages, tools=tools)
    context_aggregator = llm.create_context_aggregator(context)

    # Set call context for handlers (passes client_config through)
    set_call_context(call_id, caller_number, client_config)

    # ── Register Tool Handlers ──────────────────────────

    llm.register_function(
        "create_booking", handle_create_booking, cancel_on_interruption=False
    )
    llm.register_function("lookup_appointment", handle_lookup_appointment)
    llm.register_function(
        "reschedule_appointment", handle_reschedule_appointment, cancel_on_interruption=False
    )
    llm.register_function(
        "cancel_appointment", handle_cancel_appointment, cancel_on_interruption=False
    )

    # ── Pipeline ────────────────────────────────────────
    sentence_aggregator = SentenceAggregator()
    prosody = ProsodyProcessor()

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        sentence_aggregator,
        prosody,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ── Event Handlers ──────────────────────────────────

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info(f"Call connected: {call_id} from {caller_number} → {company_name}")

        # Play greeting immediately via TTS frame
        await task.queue_frames([TTSSpeakFrame(text=greeting)])

        # Start call duration timer and post-booking silence watcher
        asyncio.create_task(_call_timeout_watcher(task, context))
        asyncio.create_task(_post_booking_watcher(task, context, call_id))

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info(f"Call disconnected: {call_id}")

        # Clean up per-call context
        clear_call_context(call_id)

        # Generate post-call summary
        call_end_time = datetime.now(ZoneInfo(timezone))
        duration_s = int((call_end_time - call_start_time).total_seconds())

        try:
            summary_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
            summary_resp = await summary_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize this phone call in one sentence. Include the outcome (inquiry, booked, rescheduled, cancelled, or escalation).",
                    },
                    *context.messages[-10:],
                ],
                max_tokens=100,
            )
            summary = summary_resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            summary = "Summary unavailable"

        logger.info(f"Call summary [{call_id}] ({duration_s}s): {summary}")
        await task.cancel()

    # ── Track caller speech for post-booking silence timer ──

    @stt.event_handler("on_transcript")
    async def on_transcript(stt_service, transcript):
        nonlocal last_caller_speech_time
        last_caller_speech_time = datetime.now(ZoneInfo(timezone))

    # ── Post-Booking Silence Watcher ───────────────────

    async def _post_booking_watcher(task, context, cid: str):
        """After a booking completes, hang up if caller is silent for 60 seconds."""
        POST_BOOKING_SILENCE_SECONDS = 60

        try:
            # Phase 1: Wait for booking to complete
            while not is_booking_complete(cid):
                await asyncio.sleep(2)

            logger.info(f"Booking complete for {cid} — starting silence watchdog")

            # Phase 2: Monitor silence — reset timer on any caller speech
            while True:
                now = datetime.now(ZoneInfo(timezone))
                elapsed = (now - last_caller_speech_time).total_seconds()

                if elapsed >= POST_BOOKING_SILENCE_SECONDS:
                    logger.info(f"Post-booking silence ({POST_BOOKING_SILENCE_SECONDS}s) — ending call {cid}")

                    # Inject goodbye message for the LLM to speak
                    context.messages.append({
                        "role": "system",
                        "content": "The caller has been silent for a while after the booking was confirmed. Say a brief, warm goodbye like: 'Alright, it sounds like we're all set! We'll see you on your scheduled day. Have a great one!' Then stop talking.",
                    })
                    await task.queue_frames([LLMMessagesFrame(context.messages)])

                    # Wait for TTS to finish the goodbye
                    await asyncio.sleep(8)
                    await task.cancel()
                    return

                # Check every 5 seconds
                await asyncio.sleep(5)

        except asyncio.CancelledError:
            pass  # Call ended before silence timer fired

    # ── Call Timer ──────────────────────────────────────

    async def _call_timeout_watcher(task, context):
        """Enforce max call duration."""
        try:
            await asyncio.sleep(MAX_CALL_DURATION_SECONDS - 60)

            # Warn at 9 minutes
            context.messages.append({
                "role": "system",
                "content": "The call is approaching 10 minutes. Naturally wrap up the conversation — say something like 'I want to make sure I'm not keeping you too long. Is there anything else I can quickly help with?'",
            })
            await task.queue_frames([LLMMessagesFrame(context.messages)])

            await asyncio.sleep(60)

            # Force graceful end
            context.messages.append({
                "role": "system",
                "content": "The call has reached 10 minutes. Say goodbye warmly and end the call.",
            })
            await task.queue_frames([LLMMessagesFrame(context.messages)])
            await asyncio.sleep(10)
            await task.cancel()
        except asyncio.CancelledError:
            pass  # Call ended before timeout

    # ── Run ─────────────────────────────────────────────
    runner = PipelineRunner()
    await runner.run(task)
