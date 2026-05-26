"""
ScaleYourJunk — Multi-Tenant Phone Agent Configuration

All settings loaded from environment variables.
Per-client config (agent name, voice, hours, etc.) is fetched
dynamically from the dashboard API at call time.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Twilio ──────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")

# ── Anthropic / Deepgram ────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_UTILITY_MODEL = os.getenv("ANTHROPIC_UTILITY_MODEL", ANTHROPIC_MODEL)
ANTHROPIC_STABLE_UTILITY_MODEL = os.getenv(
    "ANTHROPIC_STABLE_UTILITY_MODEL",
    "claude-haiku-4-5-20251001",
)
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "flux-general-en")

# ── Cartesia ────────────────────────────────────────────
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
DEFAULT_CARTESIA_VOICE_ID = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-3-latest")
CARTESIA_SPEED = os.getenv("CARTESIA_SPEED", "normal").strip().lower()

# ── Dashboard Connection ────────────────────────────────
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "")
DASHBOARD_LIVE_CALL_REDIRECT_PATH = os.getenv(
    "DASHBOARD_LIVE_CALL_REDIRECT_PATH",
    "/api/agent/voice/redirect-live-call",
)
INGEST_API_KEY = os.getenv("INGEST_API_KEY", "")
PLATFORM_API_KEY = os.getenv("PLATFORM_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# ── Rate Limits ─────────────────────────────────────────
MAX_CALL_DURATION_SECONDS = 600  # 10 minutes
MAX_CONCURRENT_CALLS = int(os.getenv("MAX_CONCURRENT_CALLS", "10"))

# ── Server ──────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
