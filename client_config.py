"""
Client config fetcher — loads per-client settings from the dashboard API.

On each incoming call, we fetch the client's config from:
  GET {DASHBOARD_URL}/api/agent/config/{client_id}

The response includes: companyName, agentName, voiceId, timezone,
businessStart, businessEnd, businessDays, smsEnabled, siteToken, agentSecret, etc.

Configs are cached in-memory for 5 minutes to reduce API calls.
"""
import time
from typing import Any

import httpx
from loguru import logger

from config import DASHBOARD_URL, PLATFORM_API_KEY

# Simple in-memory cache: { client_id: (config_dict, fetched_at) }
_cache: dict[str, tuple[dict[str, Any], float]] = {}
CACHE_TTL = 300  # 5 minutes

# Clients already warned about incomplete config. The TwiML webhook fetches with
# force_refresh=True on every inbound call, so without this the warning repeats
# once per call for the life of the process.
_warned_incomplete: set[str] = set()


async def get_client_config(client_id: str, force_refresh: bool = False) -> dict[str, Any]:
    """Fetch client config from dashboard, with caching."""

    # Check cache
    if not force_refresh and client_id in _cache:
        config, fetched_at = _cache[client_id]
        if time.time() - fetched_at < CACHE_TTL:
            return config

    # Fetch from dashboard
    url = f"{DASHBOARD_URL}/api/agent/config/{client_id}"
    headers = {"x-api-key": PLATFORM_API_KEY}

    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)

    if resp.status_code != 200:
        logger.error(f"Failed to fetch config for {client_id}: {resp.status_code} {resp.text}")
        raise ValueError(f"Client config not found: {client_id}")

    config = resp.json()
    _cache[client_id] = (config, time.time())
    action = "Refreshed" if force_refresh else "Cached"
    logger.info(f"{action} config for {client_id} ({config.get('companyName', 'unknown')})")

    # Sanity-check handoff fallback fields and warn if missing. Human transfer
    # now goes to dashboard softphones first; forwardingPhone is the final
    # fallback if the dashboard does not answer.
    if not config.get("forwardingPhone"):
        logger.warning(
            f"forwardingPhone MISSING for client {client_id} "
            f"({config.get('companyName', 'unknown')}) — dashboard softphone can still ring, "
            f"but phone fallback will create a callback if no dashboard user answers. "
            f"Operator must set this in dashboard settings."
        )

    # Fields the prompt renders into spoken sentences. The dashboard sends every
    # key with an empty-string fallback, so a blank one is indistinguishable from
    # a field the operator never filled in. The prompt omits the whole clause
    # rather than speaking a fragment, which is invisible from the caller's side
    # — so say it here, once per config fetch.
    missing = [
        label
        for label, value in (
            ("city", config.get("city")),
            ("state", config.get("state")),
            ("serviceArea", config.get("serviceArea")),
            ("services", config.get("services")),
        )
        if not value
    ]
    if config.get("companyName") in (None, "", "Junk Removal"):
        missing.append("businessName")
    if missing and client_id not in _warned_incomplete:
        _warned_incomplete.add(client_id)
        logger.warning(
            f"Config fields empty for client {client_id} "
            f"({config.get('companyName', 'unknown')}): {', '.join(missing)}. "
            f"The agent leaves these out of what it says, or falls back to a "
            f"generic phrase, rather than speaking a partial sentence. "
            f"Operator must set them in dashboard settings."
        )

    return config


def clear_cache(client_id: str | None = None):
    """Clear cache for a specific client or all clients."""
    if client_id:
        _cache.pop(client_id, None)
        _warned_incomplete.discard(client_id)
    else:
        _cache.clear()
        _warned_incomplete.clear()
