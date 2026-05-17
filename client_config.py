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

    # Sanity-check critical fields and warn if missing.
    # forwardingPhone is required for transfer_to_human to work. Missing it means
    # callers asking for a human will fall back to a callback request instead of
    # being transferred live.
    if not config.get("forwardingPhone"):
        logger.critical(
            f"forwardingPhone MISSING for client {client_id} "
            f"({config.get('companyName', 'unknown')}) — human handoff will fail. "
            f"Operator must set this in dashboard settings."
        )

    return config


def clear_cache(client_id: str | None = None):
    """Clear cache for a specific client or all clients."""
    if client_id:
        _cache.pop(client_id, None)
    else:
        _cache.clear()
