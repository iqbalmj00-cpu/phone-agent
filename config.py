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

# ── Deepgram ────────────────────────────────────────────
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# ── OpenAI ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4.1-mini"

# ── Cartesia ────────────────────────────────────────────
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
DEFAULT_CARTESIA_VOICE_ID = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
CARTESIA_MODEL = "sonic-3"
CARTESIA_SAMPLE_RATE = 24000  # Generate at 24kHz, Pipecat resamples to 8kHz

# ── Dashboard Connection ────────────────────────────────
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "")
INGEST_API_KEY = os.getenv("INGEST_API_KEY", "")
PLATFORM_API_KEY = os.getenv("PLATFORM_API_KEY", "")

# ── Rate Limits ─────────────────────────────────────────
MAX_CALL_DURATION_SECONDS = 600  # 10 minutes
MAX_CONCURRENT_CALLS = int(os.getenv("MAX_CONCURRENT_CALLS", "10"))

# ── Server ──────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
