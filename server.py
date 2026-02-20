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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from loguru import logger

from config import HOST, PORT, MAX_CONCURRENT_CALLS, TWILIO_ACCOUNT_SID
from client_config import get_client_config
from bot import run_bot

app = FastAPI(title="ScaleYourJunk Phone Agent")

# ── Concurrent call tracking ────────────────────────────
_active_calls: dict[str, asyncio.Task] = {}
_calls_lock = asyncio.Lock()


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


@app.post("/twiml/{client_id}")
async def twiml_webhook(client_id: str, request: Request):
    """Twilio calls this when a number rings. Returns TwiML that
    opens a WebSocket media stream back to /ws/{client_id}."""

    # Validate client exists
    try:
        config = await get_client_config(client_id)
    except Exception as e:
        logger.error(f"Client config error for {client_id}: {e}")
        raise HTTPException(status_code=404, detail="Client not found")

    # Build the WebSocket URL
    host = request.headers.get("host", "localhost")
    scheme = "wss" if request.url.scheme == "https" or "fly.dev" in host else "ws"
    ws_url = f"{scheme}://{host}/ws/{client_id}"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="client_id" value="{client_id}" />
        </Stream>
    </Connect>
</Response>"""

    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handle inbound Twilio Media Stream WebSocket connections."""
    await websocket.accept()

    stream_id = None
    call_id = None
    caller_number = ""

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
                caller_number = start_data.get("customParameters", {}).get(
                    "callerNumber", ""
                )
                if not caller_number:
                    caller_number = start_data.get("customParameters", {}).get(
                        "From", ""
                    )
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
        async with _calls_lock:
            if len(_active_calls) >= MAX_CONCURRENT_CALLS:
                logger.warning(
                    f"Max concurrent calls ({MAX_CONCURRENT_CALLS}) reached. "
                    f"Rejecting call {call_id}"
                )
                await websocket.close()
                return
            _active_calls[call_id] = None

        # ── Fetch client config ──
        try:
            client_config = await get_client_config(client_id)
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
