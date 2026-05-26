"""Dashboard-mediated live call redirects.

Live calls for launched clients can belong to a client's Twilio subaccount.
The phone agent must not update those calls with parent Twilio credentials.
Instead it asks the dashboard to perform the redirect because the dashboard
owns tenant/subaccount credential lookup.
"""

from dataclasses import dataclass
from typing import Any

import aiohttp
from loguru import logger

from config import DASHBOARD_LIVE_CALL_REDIRECT_PATH, DASHBOARD_URL, PLATFORM_API_KEY


@dataclass
class DashboardLiveRedirectResult:
    ok: bool
    status: int | None = None
    error: str = ""
    response: dict[str, Any] | None = None


def _join_dashboard_url(base_url: str, path: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    suffix = str(path or "").strip()
    if not base or not suffix:
        return ""
    if not suffix.startswith("/"):
        suffix = f"/{suffix}"
    return f"{base}{suffix}"


async def redirect_live_call_via_dashboard(
    *,
    client_config: dict[str, Any],
    client_id: str,
    call_sid: str,
    reason: str,
    origin: str,
    dashboard_url: str = DASHBOARD_URL,
    endpoint_path: str = DASHBOARD_LIVE_CALL_REDIRECT_PATH,
    platform_api_key: str = PLATFORM_API_KEY,
    timeout_seconds: float = 8,
) -> DashboardLiveRedirectResult:
    """Ask dashboard to update a live Twilio call using tenant credentials."""
    url = _join_dashboard_url(dashboard_url, endpoint_path)
    if not url:
        return DashboardLiveRedirectResult(ok=False, error="dashboard_live_redirect_url_missing")

    platform_api_key = str(platform_api_key or "").strip()
    if not platform_api_key:
        return DashboardLiveRedirectResult(ok=False, error="platform_api_key_missing")

    agent_secret = str((client_config or {}).get("agentSecret") or "").strip()

    client_id = str(client_id or "").strip()
    call_sid = str(call_sid or "").strip()
    if not client_id or not call_sid:
        return DashboardLiveRedirectResult(ok=False, error="client_id_or_call_sid_missing")

    payload = {
        "clientId": client_id,
        "callSid": call_sid,
        "reason": str(reason or ""),
        "origin": str(origin or ""),
    }
    headers = {
        "x-api-key": platform_api_key,
        "Content-Type": "application/json",
    }
    if agent_secret:
        headers["X-AGENT-SECRET"] = agent_secret

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession() as http:
            response = await http.post(url, json=payload, headers=headers, timeout=timeout)
            body_text = await response.text()
            body_json: dict[str, Any] | None = None
            try:
                parsed = await response.json(content_type=None)
                if isinstance(parsed, dict):
                    body_json = parsed
            except Exception:
                body_json = None

        if 200 <= response.status < 300:
            return DashboardLiveRedirectResult(ok=True, status=response.status, response=body_json)

        logger.warning(
            f"Dashboard live-call redirect failed for call={call_sid} client={client_id} "
            f"status={response.status} body={body_text[:300]}"
        )
        return DashboardLiveRedirectResult(
            ok=False,
            status=response.status,
            error=f"dashboard_redirect_failed:{response.status}",
            response=body_json,
        )
    except Exception as exc:
        logger.error(
            f"Dashboard live-call redirect request failed for call={call_sid} "
            f"client={client_id}: {exc}"
        )
        return DashboardLiveRedirectResult(ok=False, error=str(exc))
