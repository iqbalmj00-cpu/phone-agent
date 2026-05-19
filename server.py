"""Multi-tenant FastAPI server for the ScaleYourJunk AI phone agent.

Receives inbound Twilio Media Stream connections and spins up a Pipecat
pipeline per call, loading per-client config dynamically.

Endpoints:
  GET  /health               — Health check
  POST /twiml/{client_id}    — Twilio webhook: returns TwiML connecting to WebSocket
  WS   /ws/{client_id}       — WebSocket: Pipecat media stream for the call
"""

import asyncio
import json
import os
from xml.sax.saxutils import quoteattr

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from loguru import logger

from config import (
    HOST,
    PORT,
    MAX_CONCURRENT_CALLS,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    ANTHROPIC_API_KEY,
    CARTESIA_API_KEY,
    DEEPGRAM_API_KEY,
    DASHBOARD_URL,
    INGEST_API_KEY,
    PLATFORM_API_KEY,
    GOOGLE_MAPS_API_KEY,
)
from client_config import get_client_config
from bot import run_bot
from agent.phone_coverage import (
    DASHBOARD_ORIGIN_CAPACITY,
    DASHBOARD_ORIGIN_COVERAGE_OFF,
    build_dashboard_handoff_redirect_twiml,
    build_no_handoff_twiml,
    resolve_phone_coverage_decision,
)
from agent.handlers import process_transfer_status_callback, sanitize_transfer_reason_token

app = FastAPI(title="ScaleYourJunk Phone Agent")

# ── Concurrent call tracking ────────────────────────────
_active_calls: dict[str, asyncio.Task] = {}
_calls_lock = asyncio.Lock()
_twilio_client = None


def _get_twilio_client():
    """Get or create Twilio REST client for live call redirects."""
    global _twilio_client
    if _twilio_client is None:
        from twilio.rest import Client as TwilioClient
        _twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    return _twilio_client


@app.on_event("startup")
async def startup():
    logger.info("═══════════════════════════════════════════════")
    logger.info("  ScaleYourJunk Phone Agent — Multi-Tenant")
    logger.info(f"  Max concurrent calls: {MAX_CONCURRENT_CALLS}")
    logger.info("═══════════════════════════════════════════════")


@app.get("/health")
async def health():
    """Health check for Fly.io."""
    return JSONResponse({"status": "ok", "active_calls": len(_active_calls)})


@app.get("/ready")
async def ready():
    """Readiness check: verifies required env var names are configured."""
    required = {
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
        "CARTESIA_API_KEY": CARTESIA_API_KEY,
        "TWILIO_ACCOUNT_SID": TWILIO_ACCOUNT_SID,
        "TWILIO_AUTH_TOKEN": TWILIO_AUTH_TOKEN,
        "DASHBOARD_URL": DASHBOARD_URL,
        "INGEST_API_KEY": INGEST_API_KEY,
        "PLATFORM_API_KEY": PLATFORM_API_KEY,
    }
    missing = [name for name, value in required.items() if not value]
    degraded = []
    if not GOOGLE_MAPS_API_KEY:
        degraded.append("GOOGLE_MAPS_API_KEY")

    status_code = 503 if missing else 200
    return JSONResponse(
        {
            "status": "not_ready" if missing else "ready",
            "active_calls": len(_active_calls),
            "missing": missing,
            "degraded": degraded,
        },
        status_code=status_code,
    )


def _public_base_url(request: Request) -> str:
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "localhost")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    scheme = "https" if proto == "https" or "fly.dev" in host else "http"
    return f"{scheme}://{host}"


async def _redirect_live_call_to_dashboard_handoff(
    *,
    call_sid: str,
    client_id: str,
    company_name: str,
    reason: str,
    origin: str,
) -> bool:
    """Ask Twilio to move an already-connected live call to dashboard handoff."""
    if not call_sid:
        return False

    twiml = build_dashboard_handoff_redirect_twiml(
        company_name=company_name,
        client_id=client_id,
        dashboard_url=DASHBOARD_URL,
        reason=reason,
        origin=origin,
    ) or build_no_handoff_twiml(company_name)

    try:
        client = _get_twilio_client()
        await asyncio.to_thread(client.calls(call_sid).update, twiml=twiml)
        return True
    except Exception as exc:
        logger.error(
            f"Failed to redirect live call {call_sid} for client={client_id} "
            f"reason={reason}: {exc}"
        )
        return False


@app.post("/twiml/{client_id}")
async def twiml_webhook(client_id: str, request: Request):
    """Twilio calls this when a number rings. Returns TwiML that
    opens a WebSocket media stream back to /ws/{client_id}."""

    # Extract call metadata from Twilio's POST form data before deciding whether
    # the call should enter the AI stream or be handed off immediately.
    form_data = await request.form()
    from_number = str(form_data.get("From", "") or "")
    to_number = str(form_data.get("To", "") or "")
    call_sid = str(form_data.get("CallSid", "") or "")

    # Validate client exists
    try:
        config = await get_client_config(client_id, force_refresh=True)
    except Exception as e:
        logger.error(f"Client config error for {client_id}: {e}")
        raise HTTPException(status_code=404, detail="Client not found")

    # ── Client-configurable AI phone coverage ───────────────────────────────
    coverage = resolve_phone_coverage_decision(config)
    if coverage.error:
        logger.warning(
            f"Phone coverage decision warning for client={client_id}: "
            f"mode={coverage.mode} reason={coverage.reason} error={coverage.error}"
        )
    if coverage.should_handoff:
        company = config.get("companyName", "us")
        handoff_twiml = build_dashboard_handoff_redirect_twiml(
            company_name=company,
            client_id=client_id,
            dashboard_url=DASHBOARD_URL,
            reason=coverage.reason,
            origin=DASHBOARD_ORIGIN_COVERAGE_OFF,
        )
        if handoff_twiml:
            logger.info(
                f"Phone coverage dashboard handoff for {company} ({client_id}): "
                f"mode={coverage.mode} reason={coverage.reason} call_sid={call_sid or 'missing'}"
            )
            return PlainTextResponse(content=handoff_twiml, media_type="application/xml")

        logger.warning(
            f"Phone coverage required dashboard handoff for {company} ({client_id}) but DASHBOARD_URL "
            f"is missing; returning safe unavailable TwiML"
        )
        return PlainTextResponse(content=build_no_handoff_twiml(company), media_type="application/xml")

    # ── Concurrent call limit — voice message instead of silent drop ──
    async with _calls_lock:
        if len(_active_calls) >= MAX_CONCURRENT_CALLS:
            company = config.get("companyName", "us")
            logger.warning(
                f"Max concurrent calls ({MAX_CONCURRENT_CALLS}) reached at TwiML for {company}; "
                "redirecting to dashboard softphone handoff"
            )
            capacity_twiml = build_dashboard_handoff_redirect_twiml(
                company_name=company,
                client_id=client_id,
                dashboard_url=DASHBOARD_URL,
                reason="phone_agent_at_capacity",
                origin=DASHBOARD_ORIGIN_CAPACITY,
            )
            if capacity_twiml:
                return PlainTextResponse(content=capacity_twiml, media_type="application/xml")
            return PlainTextResponse(content=build_no_handoff_twiml(company), media_type="application/xml")

    # ── Coverage is ON: connect to AI agent ──
    host = request.headers.get("host", "localhost")
    scheme = "wss" if request.url.scheme == "https" or "fly.dev" in host else "ws"
    ws_url = f"{scheme}://{host}/ws/{client_id}"
    phone_agent_base_url = _public_base_url(request)

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url={quoteattr(ws_url)}>
            <Parameter name="client_id" value={quoteattr(client_id)} />
            <Parameter name="From" value={quoteattr(str(from_number))} />
            <Parameter name="phoneAgentBaseUrl" value={quoteattr(phone_agent_base_url)} />
        </Stream>
    </Connect>
</Response>"""

    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.post("/transfer-status/{client_id}/{call_sid}")
async def transfer_status_webhook(client_id: str, call_sid: str, request: Request):
    """Twilio <Dial action> callback used to record real handoff result."""
    form_data = await request.form()
    dial_status = str(form_data.get("DialCallStatus", "") or "")
    dial_call_sid = str(form_data.get("DialCallSid", "") or "")
    dial_duration = form_data.get("DialCallDuration", 0)
    raw_transfer_reason = str(request.query_params.get("reason", "") or "")
    transfer_reason = sanitize_transfer_reason_token(raw_transfer_reason)
    if raw_transfer_reason and not transfer_reason:
        logger.warning(
            f"Ignoring unknown transfer reason token for client={client_id} call={call_sid}: "
            f"{raw_transfer_reason[:80]!r}"
        )
    logger.info(
        f"Transfer status callback: client={client_id} call={call_sid} "
        f"dial_status={dial_status or 'missing'}"
    )
    twiml = await process_transfer_status_callback(
        client_id=client_id,
        call_sid=call_sid,
        dial_status=dial_status,
        dial_call_sid=dial_call_sid,
        dial_duration=dial_duration,
        callback_data=dict(form_data),
        transfer_reason=transfer_reason,
    )
    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handle inbound Twilio Media Stream WebSocket connections."""
    await websocket.accept()

    stream_id = None
    call_id = None
    caller_number = ""
    phone_agent_base_url = ""

    try:
        # ── Parse initial Twilio messages to get stream metadata ──
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("event") == "connected":
                logger.info(f"Twilio WebSocket connected for client: {client_id}")
                continue

            if msg.get("event") == "start":
                start_data = msg.get("start", {})
                stream_id = start_data.get("streamSid", "")
                call_id = start_data.get("callSid", "")
                custom_params = start_data.get("customParameters", {})
                caller_number = custom_params.get(
                    "callerNumber", ""
                )
                if not caller_number:
                    caller_number = custom_params.get(
                        "From", ""
                    )
                phone_agent_base_url = custom_params.get("phoneAgentBaseUrl", "")
                logger.info(
                    f"Stream started: stream={stream_id} call={call_id} "
                    f"from={caller_number} client={client_id}"
                )
                break
            continue

        if not stream_id or not call_id:
            logger.error("Missing stream_id or call_id from Twilio start event")
            await websocket.close()
            return

        # ── Check concurrent call limit ──
        over_capacity = False
        async with _calls_lock:
            if len(_active_calls) >= MAX_CONCURRENT_CALLS:
                over_capacity = True
            else:
                _active_calls[call_id] = None

        if over_capacity:
            logger.warning(
                f"Max concurrent calls ({MAX_CONCURRENT_CALLS}) reached. "
                f"Redirecting live call {call_id} to dashboard softphone handoff"
            )
            redirected = await _redirect_live_call_to_dashboard_handoff(
                call_sid=call_id,
                client_id=client_id,
                company_name="our team",
                reason="phone_agent_at_capacity",
                origin=DASHBOARD_ORIGIN_CAPACITY,
            )
            if not redirected:
                logger.warning(
                    f"Could not redirect at-capacity live call {call_id}; closing websocket"
                )
            await websocket.close()
            return

        # ── Fetch client config ──
        try:
            client_config = await get_client_config(client_id)
            client_config = dict(client_config)
            client_config["_clientId"] = client_id
            client_config["_phoneAgentBaseUrl"] = phone_agent_base_url
        except Exception as e:
            logger.error(f"Failed to load config for {client_id}: {e}")
            async with _calls_lock:
                _active_calls.pop(call_id, None)
            await websocket.close()
            return

        logger.info(f"Loaded config for {client_config.get('companyName', 'unknown')} "
                     f"(agent: {client_config.get('agentName', 'unknown')})")

        # ── Run the bot pipeline ──
        try:
            await run_bot(
                websocket_client=websocket,
                stream_id=stream_id,
                call_id=call_id,
                caller_number=caller_number,
                client_config=client_config,
            )
        finally:
            async with _calls_lock:
                _active_calls.pop(call_id, None)
            logger.info(f"Call ended: {call_id}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {call_id or 'unknown'}")
        async with _calls_lock:
            if call_id:
                _active_calls.pop(call_id, None)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        async with _calls_lock:
            if call_id:
                _active_calls.pop(call_id, None)


# ── Init packages ──────────────────────────────────────
def _ensure_packages():
    for pkg in ["agent"]:
        init_path = os.path.join(os.path.dirname(__file__), pkg, "__init__.py")
        os.makedirs(os.path.dirname(init_path), exist_ok=True)
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write("")

_ensure_packages()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
