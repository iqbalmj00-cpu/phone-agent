"""Tool handler functions for the multi-tenant phone agent.

Each handler receives a FunctionCallParams object and uses the client_config
(set per-call via set_call_context) to route API calls to the correct
dashboard tenant.

ENDPOINT MAPPING (dashboard ↔ phone agent):
  - create_booking        → POST /api/agent/book             (X-AGENT-SECRET auth)
  - lookup_appointment    → GET  /api/agent/lookup            (X-AGENT-SECRET auth)
  - reschedule            → POST /api/agent/reschedule        (X-AGENT-SECRET auth)
  - cancel                → POST /api/agent/cancel            (X-AGENT-SECRET auth)
  - schedule_callback     → POST /api/agent/schedule-callback (X-AGENT-SECRET auth)
  - transfer_to_human     → Twilio REST API (call redirect)
  - validate_promo_code   → GET  /api/promo/validate          (x-api-key + x-site-token)
"""

import asyncio
import contextvars
import json
import re
import aiohttp
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote
from xml.sax.saxutils import escape, quoteattr
from zoneinfo import ZoneInfo

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from config import DASHBOARD_URL, INGEST_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, GOOGLE_MAPS_API_KEY

# ── Per-call context — keyed by call_sid for concurrency safety ──
_call_contexts: dict[str, dict[str, Any]] = {}
_current_call_sid: contextvars.ContextVar[str] = contextvars.ContextVar("current_call_sid", default="")

# Twilio <Dial action> callbacks arrive after the media stream has ended, so
# transfer state must outlive the normal per-call context.
_pending_transfers: dict[str, dict[str, Any]] = {}
_TRANSFER_SUCCESS_STATUSES = {"completed", "answered"}
_TRANSFER_FAILURE_STATUSES = {"busy", "no-answer", "failed", "canceled"}


def set_call_context(call_sid: str, caller_number: str, client_config: dict[str, Any]):
    """Set context for the current call (called from bot.py on connect)."""
    _call_contexts[call_sid] = {
        "call_sid": call_sid,
        "caller_number": caller_number,
        "client_config": client_config,
    }


def clear_call_context(call_sid: str):
    """Clean up context when call ends."""
    _call_contexts.pop(call_sid, None)


def mark_booking_complete(call_sid: str, details: dict[str, Any] | None = None):
    """Flag that a booking was successfully created for this call."""
    if call_sid in _call_contexts:
        _call_contexts[call_sid]["booking_complete"] = True
        if details:
            current = _call_contexts[call_sid].setdefault("booking_details", {})
            current.update({k: v for k, v in details.items() if v is not None})


def set_pipeline_task(call_sid: str, task):
    """Store the PipelineTask so handlers can cancel it (e.g. after transfer)."""
    if call_sid in _call_contexts:
        _call_contexts[call_sid]["pipeline_task"] = task


def is_booking_complete(call_sid: str) -> bool:
    """Check if a booking has been completed during this call."""
    ctx = _call_contexts.get(call_sid, {})
    return ctx.get("booking_complete", False)


def get_booking_log_state(call_sid: str) -> dict[str, Any]:
    """Return structured booking fields supported by the dashboard call log."""
    ctx = _call_contexts.get(call_sid, {})
    return dict(ctx.get("booking_details", {}))


def mark_sms_sent(call_sid: str):
    """Flag that an SMS was sent during this call (to avoid duplicate follow-ups)."""
    if call_sid in _call_contexts:
        _call_contexts[call_sid]["sms_sent"] = True


def was_sms_sent(call_sid: str) -> bool:
    """Check if an SMS was already sent during this call."""
    ctx = _call_contexts.get(call_sid, {})
    return ctx.get("sms_sent", False)


def mark_transfer_complete(call_sid: str):
    """Flag that the call was transferred to a human."""
    if call_sid in _call_contexts:
        _call_contexts[call_sid]["transfer_complete"] = True
        _call_contexts[call_sid]["transfer_status"] = "completed"


def was_transfer_complete(call_sid: str) -> bool:
    """Check if a transfer was completed during this call."""
    ctx = _call_contexts.get(call_sid, {})
    return ctx.get("transfer_complete", False)


def mark_transfer_state(
    call_sid: str,
    status: str,
    reason: str | None = None,
    callback_requested: bool = False,
):
    """Record structured human-handoff state for the final call log."""
    if call_sid not in _call_contexts:
        return
    _call_contexts[call_sid]["transfer_status"] = status
    if reason is not None:
        _call_contexts[call_sid]["transfer_reason"] = reason
    if callback_requested:
        _call_contexts[call_sid]["callback_requested"] = True


def mark_callback_requested(
    call_sid: str,
    due_at: str | None = None,
    reason: str | None = None,
):
    """Record scheduled callback state for the final call log."""
    if call_sid not in _call_contexts:
        return
    _call_contexts[call_sid]["callback_requested"] = True
    if due_at:
        _call_contexts[call_sid]["callback_due_at"] = due_at
    if reason:
        _call_contexts[call_sid]["callback_reason"] = reason


def get_transfer_state(call_sid: str) -> dict[str, Any]:
    """Return structured human-handoff and callback state captured during this call."""
    ctx = _call_contexts.get(call_sid, {})
    return {
        "transfer_status": ctx.get("transfer_status"),
        "transfer_reason": ctx.get("transfer_reason"),
        "callback_requested": ctx.get("callback_requested", False),
        "callback_due_at": ctx.get("callback_due_at"),
        "callback_reason": ctx.get("callback_reason"),
    }


def _remember_pending_transfer(
    call_sid: str,
    config: dict[str, Any],
    from_number: str,
    reason: str,
):
    """Keep enough context to update the dashboard when Twilio reports Dial status."""
    _pending_transfers[call_sid] = {
        "config": dict(config),
        "from_number": from_number,
        "to_number": config.get("twilioNumber", ""),
        "reason": reason,
        "transfer_status": "requested",
        "duration": 0,
        "summary": "",
        "sms_consent": None,
        "callback_requested": False,
        "callback_due_at": None,
        "caller_name": None,
        "appointment_date": None,
    }


def _update_pending_transfer_status(
    call_sid: str,
    status: str,
    *,
    callback_requested: bool = False,
):
    pending = _pending_transfers.get(call_sid)
    if not pending:
        return
    pending["transfer_status"] = status
    if callback_requested:
        pending["callback_requested"] = True


def update_pending_transfer_snapshot(
    call_sid: str,
    *,
    duration: int,
    summary: str,
    from_number: str,
    to_number: str,
    config: dict[str, Any],
    sms_consent: bool | None,
    callback_requested: bool,
    callback_due_at: str | None = None,
    caller_name: str | None = None,
    appointment_date: str | None = None,
):
    """Refresh pending transfer metadata before final call-log write."""
    pending = _pending_transfers.get(call_sid)
    if not pending:
        return
    pending.update({
        "duration": max(0, int(duration)),
        "summary": summary,
        "from_number": from_number,
        "to_number": to_number,
        "config": dict(config),
        "sms_consent": sms_consent,
        "callback_requested": bool(callback_requested),
        "callback_due_at": callback_due_at,
        "caller_name": caller_name,
        "appointment_date": appointment_date,
    })


def get_pending_transfer_state(call_sid: str) -> dict[str, Any]:
    pending = _pending_transfers.get(call_sid)
    return dict(pending) if pending else {}


def clear_pending_transfer_if_terminal(call_sid: str):
    pending = _pending_transfers.get(call_sid)
    if pending and pending.get("terminal_logged"):
        _pending_transfers.pop(call_sid, None)


def mark_sms_consent(call_sid: str, consented: bool):
    """Record the caller's SMS consent decision."""
    if call_sid in _call_contexts:
        _call_contexts[call_sid]["sms_consent"] = consented


def has_sms_consent(call_sid: str) -> bool | None:
    """Check SMS consent status. Returns None if not yet asked."""
    ctx = _call_contexts.get(call_sid, {})
    return ctx.get("sms_consent", None)


def _phone_lookup_key(phone: str) -> str:
    """Use the last 10 digits as a stable per-call lookup cache key."""
    digits = "".join(c for c in phone if c.isdigit())
    return digits[-10:] if len(digits) >= 10 else digits


def remember_lookup_results(call_sid: str, phone: str, jobs: list[dict[str, Any]]):
    """Cache lookup results so mutations can avoid ambiguous phone-only updates."""
    if call_sid not in _call_contexts:
        return
    key = _phone_lookup_key(phone)
    if not key:
        return
    lookups = _call_contexts[call_sid].setdefault("appointment_lookups", {})
    lookups[key] = jobs


def get_cached_lookup_jobs(call_sid: str, phone: str) -> list[dict[str, Any]] | None:
    """Return cached jobs for a phone, or None if lookup was not performed."""
    ctx = _call_contexts.get(call_sid, {})
    key = _phone_lookup_key(phone)
    if not key:
        return None
    return ctx.get("appointment_lookups", {}).get(key)


def resolve_job_id_from_lookup(
    call_sid: str,
    phone: str,
    provided_job_id: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Require lookup-backed job selection before mutating existing bookings."""
    if provided_job_id:
        return provided_job_id, None

    cached_jobs = get_cached_lookup_jobs(call_sid, phone)
    if cached_jobs is None:
        return None, {
            "error": "lookup_required",
            "message": (
                "Look up the caller's appointments first, then confirm which "
                "appointment they want to change before trying again."
            ),
        }

    if len(cached_jobs) == 1:
        job_id = cached_jobs[0].get("jobId") or cached_jobs[0].get("id")
        if job_id:
            return str(job_id), None

    if len(cached_jobs) > 1:
        return None, {
            "error": "multiple_bookings_require_job_id",
            "message": (
                "I found multiple active bookings for this caller. Read back "
                "the date and address for each one, ask which appointment they "
                "mean, then call this tool again with the selected job_id."
            ),
            "jobs": cached_jobs,
        }

    return None, {
        "error": "no_cached_booking",
        "message": "No active appointment was found from the lookup. Do not change or cancel anything.",
    }


def _normalize_address_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def remember_address_verification(
    call_sid: str,
    raw_address: str,
    formatted_address: str,
    service_area_status: str,
    reason: str,
):
    """Cache address verification so create_booking can block clear out-of-area jobs."""
    if call_sid not in _call_contexts:
        return
    verifications = _call_contexts[call_sid].setdefault("address_verifications", {})
    record = {
        "raw_address": raw_address,
        "formatted_address": formatted_address,
        "service_area_status": service_area_status,
        "reason": reason,
    }
    for key_source in (raw_address, formatted_address):
        key = _normalize_address_key(key_source)
        if key:
            verifications[key] = record


def get_address_verification(call_sid: str, address: str) -> dict[str, Any] | None:
    """Find the verification result that matches the booking address."""
    ctx = _call_contexts.get(call_sid, {})
    verifications = ctx.get("address_verifications", {})
    if not verifications:
        return None
    key = _normalize_address_key(address)
    if key in verifications:
        return verifications[key]

    matches = []
    for stored_key, record in verifications.items():
        if key and (key in stored_key or stored_key in key):
            matches.append(record)
    if len(matches) == 1:
        return matches[0]
    return None


def _component_long_short(result: dict[str, Any], component_type: str) -> tuple[str, str]:
    for component in result.get("address_components", []):
        if component_type in component.get("types", []):
            return component.get("long_name", ""), component.get("short_name", "")
    return "", ""


def _extract_geocode_components(result: dict[str, Any]) -> dict[str, str]:
    city = (
        _component_long_short(result, "locality")[0]
        or _component_long_short(result, "postal_town")[0]
        or _component_long_short(result, "sublocality")[0]
        or _component_long_short(result, "administrative_area_level_3")[0]
    )
    state_long, state_short = _component_long_short(result, "administrative_area_level_1")
    county = _component_long_short(result, "administrative_area_level_2")[0]
    postal_code = _component_long_short(result, "postal_code")[0]
    return {
        "city": city,
        "state": state_short or state_long,
        "state_long": state_long,
        "county": county,
        "postal_code": postal_code,
    }


def _normalize_place(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


_US_STATE_TO_ABBR = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy",
}


def _state_variants(value: str) -> set[str]:
    norm = _normalize_place(value)
    if not norm:
        return set()
    variants = {norm}
    if norm in _US_STATE_TO_ABBR:
        variants.add(_US_STATE_TO_ABBR[norm])
    for long_name, abbr in _US_STATE_TO_ABBR.items():
        if norm == abbr:
            variants.add(long_name)
    return variants


def _service_area_tokens(config: dict[str, Any]) -> list[str]:
    raw = str(config.get("serviceArea") or "")
    parts = re.split(r"[,;\n|/]+", raw)
    tokens = [_normalize_place(p) for p in parts]
    city = _normalize_place(str(config.get("city") or ""))
    if city:
        tokens.append(city)
    return [t for t in tokens if len(t) >= 3 and t not in {"area", "service area", "near me"}]


def _assess_service_area(components: dict[str, str], config: dict[str, Any]) -> tuple[str, str]:
    """Conservative area check: only block addresses that are clearly outside."""
    address_city = _normalize_place(components.get("city", ""))
    address_county = _normalize_place(components.get("county", ""))
    address_states = _state_variants(components.get("state", "")) | _state_variants(components.get("state_long", ""))
    configured_states = _state_variants(str(config.get("state") or ""))
    service_area_text = _normalize_place(str(config.get("serviceArea") or ""))
    tokens = _service_area_tokens(config)

    if configured_states and address_states and configured_states.isdisjoint(address_states):
        return "out_of_area", f"address state {components.get('state')} does not match configured state {config.get('state')}"

    if address_city and address_city in tokens:
        return "in_area", "address city matches configured service area"
    if address_county and address_county in tokens:
        return "in_area", "address county matches configured service area"

    broad_terms = {"metro", "area", "surrounding", "nearby", "county", "counties", "greater", "region"}
    has_broad_term = any(term in service_area_text.split() for term in broad_terms)
    has_explicit_list = bool(re.search(r"[,;\n|/]", str(config.get("serviceArea") or "")))
    city_like_tokens = [t for t in tokens if t and t not in configured_states]

    if address_city and has_explicit_list and len(city_like_tokens) >= 2 and not has_broad_term:
        return "out_of_area", "address city is not in the configured explicit service-area list"

    if address_city or address_states:
        return "uncertain", "address verified, but service-area text is not strict enough to prove coverage"
    return "unknown", "address components did not include enough location detail for service-area checking"


# ── Tool Failure Counter (per call, per tool) ──────────
# Tracks how many times a given tool has hit a SYSTEM error in this call.
# Validation errors (bad date format, slot_full, etc.) do NOT count — those
# are LLM-correctable. Only network/API/exception failures count.
# After 2 failures of the same tool, the handler returns {"fallback": true}
# which the prompt's HUMAN HANDOFF rule converts into an automatic transfer.

TOOL_FAILURE_THRESHOLD = 2


def increment_tool_failure(call_sid: str, tool_name: str) -> int:
    """Increment the system-failure count for a tool in this call. Returns new count."""
    if call_sid not in _call_contexts:
        return 0
    failures = _call_contexts[call_sid].setdefault("tool_failures", {})
    failures[tool_name] = failures.get(tool_name, 0) + 1
    return failures[tool_name]


def reset_tool_failure(call_sid: str, tool_name: str):
    """Reset failure count for a tool (called on success)."""
    if call_sid in _call_contexts:
        failures = _call_contexts[call_sid].get("tool_failures", {})
        failures.pop(tool_name, None)


def should_escalate_after_failure(call_sid: str, tool_name: str) -> bool:
    """Increment failure count and return True if threshold reached."""
    return increment_tool_failure(call_sid, tool_name) >= TOOL_FAILURE_THRESHOLD


def _build_error_with_tracking(call_sid: str, tool_name: str, fallback_msg: str) -> dict:
    """Track this as a system failure. If the threshold is reached, return
    fallback: True so the prompt's HUMAN HANDOFF rule auto-transfers the call.
    Otherwise return the normal error message so the LLM can recover/retry."""
    if call_sid and should_escalate_after_failure(call_sid, tool_name):
        return {
            "fallback": True,
            "error": "repeated_failure",
            "message": (
                "I'm having repeated trouble with this. Let me get someone "
                "from our team to help you right away."
            ),
        }
    return {"error": fallback_msg}


def _build_immediate_fallback(error: str, message: str) -> dict:
    """Return a tool result that prompts an immediate human handoff."""
    return {
        "fallback": True,
        "error": error,
        "message": message,
    }


# ── In-flight Tool Task Tracking (disconnect resilience) ──
# When the caller hangs up mid-tool-call (e.g. during create_booking's API
# request), we need to wait for the request to complete before snapshotting
# call state. Otherwise we get "ghost bookings" — the dashboard creates the
# booking, but our call log shows info_only because mark_booking_complete
# never fired before disconnect cancelled the handler.

def add_inflight_task(call_sid: str, task: asyncio.Task):
    """Register an in-flight tool task so the disconnect handler can await it."""
    if call_sid not in _call_contexts:
        return
    inflight = _call_contexts[call_sid].setdefault("inflight_tasks", set())
    inflight.add(task)


def remove_inflight_task(call_sid: str, task: asyncio.Task):
    """Remove a completed task from the in-flight set (called via done_callback)."""
    if call_sid not in _call_contexts:
        return
    inflight = _call_contexts[call_sid].get("inflight_tasks", set())
    inflight.discard(task)


def get_inflight_tasks(call_sid: str) -> list:
    """Return a snapshot of in-flight tool tasks for this call."""
    if call_sid not in _call_contexts:
        return []
    return list(_call_contexts[call_sid].get("inflight_tasks", set()))


async def handle_record_sms_consent(params: FunctionCallParams):
    """Record the caller's verbal SMS consent decision.

    Called by the LLM after explicitly asking the caller if they agree
    to receive text messages and receiving a clear yes/no answer.
    """
    ctx = _get_context()
    consented = params.arguments.get("consented", False)
    call_sid = ctx.get("call_sid", "")

    if call_sid:
        mark_sms_consent(call_sid, consented)
        logger.info(f"SMS consent recorded for call {call_sid}: {'opted-in' if consented else 'declined'}")

    if consented:
        await params.result_callback({
            "recorded": True,
            "message": "SMS consent recorded. You may now offer to send texts.",
        })
    else:
        await params.result_callback({
            "recorded": True,
            "message": "Customer declined SMS. Do not mention texting for the rest of this call.",
        })


def _get_context() -> dict[str, Any]:
    """Get the call context for the current pipeline.

    Uses contextvars to identify the correct call_sid for this asyncio task,
    ensuring concurrent calls always use their own config.
    """
    call_sid = _current_call_sid.get()
    if call_sid and call_sid in _call_contexts:
        return _call_contexts[call_sid]
    # No matching context — log and return empty
    if _call_contexts:
        logger.critical(f"No call context for call_sid={call_sid}. Active: {list(_call_contexts.keys())}")
    return {}


def _get_config() -> dict[str, Any]:
    return _get_context().get("client_config", {})


def _agent_headers(config: dict[str, Any]) -> dict[str, str]:
    """Build auth headers using X-AGENT-SECRET (agentSecret) — matches dashboard's agent-auth.ts."""
    return {
        "X-AGENT-SECRET": config.get("agentSecret", ""),
        "Content-Type": "application/json",
    }


def _ingest_headers(config: dict[str, Any]) -> dict[str, str]:
    """Build auth headers for ingest-style endpoints (container-availability).
    Uses x-api-key + x-site-token — matches dashboard's validate-ingest.ts."""
    return {
        "x-api-key": INGEST_API_KEY,
        "x-site-token": config.get("siteToken", ""),
    }



async def _safe_json(resp: aiohttp.ClientResponse, label: str) -> dict:
    """Parse JSON safely, logging raw text on failure."""
    text = await resp.text()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.error(f"[{label}] Non-JSON response (status {resp.status}): {text[:500]}")
        return {"error": f"Dashboard returned non-JSON (status {resp.status})"}


# ── Time Slots (must match website wizardData.ts) ───────

TIME_SLOTS = {
    "morning":   {"label": "Morning",   "start": "08:00", "end": "11:00", "start_hour": 8,  "period": "Morning"},
    "midday":    {"label": "Midday",    "start": "11:00", "end": "13:00", "start_hour": 11, "period": "Midday"},
    "afternoon": {"label": "Afternoon", "start": "13:00", "end": "16:00", "start_hour": 13, "period": "Afternoon"},
}


# ── Date / Time Validation ──────────────────────────────

def _validate_date(date_str: str) -> datetime | None:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def _validate_time_slot(slot_id: str) -> dict | None:
    """Validate a time slot ID and return the slot definition, or None.

    Accepts both old-style IDs (morning, midday, etc.) and new dynamic
    time-range format from check_available_slots (e.g. '08:00-10:00').
    """
    if not slot_id:
        return None
    # Old named slots
    old = TIME_SLOTS.get(slot_id.lower().strip())
    if old:
        return old
    # New dynamic format: "HH:MM-HH:MM"
    stripped = slot_id.strip()
    if "-" in stripped:
        parts = stripped.split("-")
        if len(parts) == 2:
            try:
                sh, sm = parts[0].split(":")
                eh, em = parts[1].split(":")
                start_hour = int(sh)
                return {
                    "label": f"{parts[0]} - {parts[1]}",
                    "start": parts[0],
                    "end": parts[1],
                    "start_hour": start_hour,
                    "period": stripped,  # Use the range as the period
                }
            except (ValueError, IndexError):
                pass
    return None


def _is_business_hours(date: datetime, hour: int, config: dict) -> bool:
    """Check if date/time falls within this client's business hours."""
    business_days = config.get("businessDays", [0, 1, 2, 3, 4, 5])
    business_start = int(config.get("businessStart", 7))
    business_end = int(config.get("businessEnd", 19))

    # Convert Python weekday (0=Mon) to JS convention (0=Sun) used by dashboard
    js_day = (date.weekday() + 1) % 7
    if js_day not in business_days:
        return False
    if hour < business_start or hour >= business_end:
        return False
    return True


# ── Tool Handlers ───────────────────────────────────────

async def handle_check_container_availability(params: FunctionCallParams):
    """Check real-time container availability via dashboard's
    /api/booking/container-availability endpoint.

    Auth: x-api-key + x-site-token (ingest-style auth).

    Available response:  { available: true, count, baseRate, includedDays, extendedDailyRate, ... }
    Unavailable response: { available: false, alternativeSizes: [10, 30, 40] }
    """
    config = _get_config()
    size = params.arguments["size"]
    date = params.arguments.get("date")
    days = params.arguments.get("days")

    try:
        async with aiohttp.ClientSession() as http:
            query: dict[str, str] = {"size": size}
            if date:
                query["date"] = date
            if days:
                query["days"] = str(days)
            resp = await http.get(
                f"{DASHBOARD_URL}/api/booking/container-availability",
                params=query,
                headers=_ingest_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )

            if resp.status == 200:
                data = await _safe_json(resp, "container-availability")
                if data.get("available"):
                    base_rate = data.get("baseRate")
                    included_days = data.get("includedDays", 7)
                    extended_rate = data.get("extendedDailyRate")

                    # Round to nearest $5 for clean customer-facing prices
                    def _round5(n: float) -> int:
                        return round(n / 5) * 5

                    price_msg = ""
                    if base_rate:
                        price_msg = f"${_round5(base_rate)} for {included_days} days"
                        if extended_rate:
                            price_msg += f", then ${_round5(extended_rate)}/day after"

                    await params.result_callback({
                        "available": True,
                        "size": size,
                        "baseRate": base_rate,
                        "includedDays": included_days,
                        "extendedDailyRate": extended_rate,
                        "message": f"{size}-yard container is available. {price_msg}" if price_msg else f"{size}-yard container is available.",
                    })
                else:
                    alternatives = data.get("alternativeSizes", [])
                    next_date = data.get("nextAvailableDate")
                    if next_date:
                        # Parse ISO date to readable format
                        try:
                            from datetime import datetime as _dt
                            nd = _dt.fromisoformat(next_date.replace("Z", "+00:00"))
                            next_date_str = nd.strftime("%A, %B %d")
                        except Exception:
                            next_date_str = next_date
                    else:
                        next_date_str = None

                    if alternatives:
                        alt_str = ", ".join(f"{s}-yard" for s in alternatives)
                        msg = f"No {size}-yard containers available"
                        if next_date_str:
                            msg += f" for that date. Next available: {next_date_str}."
                        else:
                            msg += " right now."
                        msg += f" Alternative sizes available: {alt_str}."
                        await params.result_callback({
                            "available": False,
                            "alternativeSizes": alternatives,
                            "nextAvailableDate": next_date,
                            "message": msg,
                        })
                    else:
                        msg = f"No {size}-yard containers available"
                        if next_date_str:
                            msg += f" for that date. Next available: {next_date_str}."
                        else:
                            msg += ", and no other sizes in stock right now."
                        msg += " Do not create a booking or request unless the caller chooses an available size or date. Offer to transfer to the team if they need help."
                        await params.result_callback({
                            "available": False,
                            "alternativeSizes": [],
                            "nextAvailableDate": next_date,
                            "message": msg,
                        })
            else:
                # API error — do NOT proceed with booking
                logger.warning(f"Container availability check returned {resp.status}")
                await params.result_callback({
                    "available": False,
                    "message": f"I wasn't able to check live inventory for the {size}-yard right now. Let me transfer you to our team to confirm availability.",
                    "fallback": True,
                })
    except Exception as e:
        logger.error(f"Container availability check failed: {e}")
        await params.result_callback({
            "available": False,
            "message": "I'm having trouble checking availability right now. Let me transfer you to our team to confirm.",
            "fallback": True,
        })


async def handle_validate_promo_code(params: FunctionCallParams):
    """Validate a promo/referral code via dashboard's /api/promo/validate.

    Dashboard returns: { valid: bool, discountType?: str, discountValue?: float }
    Auth: x-api-key + x-site-token (ingest-style)
    """
    config = _get_config()
    code = params.arguments.get("code", "").strip()

    if not code:
        await params.result_callback({
            "valid": False,
            "message": "No code provided.",
        })
        return

    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.get(
                f"{DASHBOARD_URL}/api/promo/validate",
                params={"code": code},
                headers=_ingest_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )
            data = await _safe_json(resp, "validate-promo")

            if data.get("valid"):
                discount_type = data.get("discountType", "percentage")
                discount_value = data.get("discountValue", 0)
                if discount_type == "percentage":
                    msg = f"Code {code.upper()} is valid — {discount_value}% discount. Include this code when booking."
                else:
                    msg = f"Code {code.upper()} is valid — ${discount_value:.0f} off. Include this code when booking."
                await params.result_callback({
                    "valid": True,
                    "code": code.upper(),
                    "discountType": discount_type,
                    "discountValue": discount_value,
                    "message": msg,
                })
            else:
                reason = data.get("reason", "invalid")
                reason_msg = {
                    "not_found": "That code doesn't exist.",
                    "expired": "That code has expired.",
                    "inactive": "That code is no longer active.",
                    "max_uses_reached": "That code has been fully redeemed.",
                }.get(reason, "That code isn't valid.")
                await params.result_callback({
                    "valid": False,
                    "message": reason_msg,
                })
    except Exception as e:
        logger.error(f"Promo code validation failed: {e}")
        await params.result_callback({
            "valid": False,
            "message": "I'm having trouble checking that code right now. Let's go ahead and book without it — you can always apply it later.",
        })




async def handle_check_available_slots(params: FunctionCallParams):
    """Check available time slots for a specific date.

    Calls GET /api/public/available-slots?date=YYYY-MM-DD
    Returns human-readable business-hour booking windows. This is not a
    junk-removal job capacity check.
    """
    config = _get_config()
    date = params.arguments["date"]

    parsed_date = _validate_date(date)
    if not parsed_date:
        await params.result_callback({
            "error": "I need the date in YYYY-MM-DD format. Please resolve relative dates first."
        })
        return

    # Business day check
    business_days = config.get("businessDays", [0, 1, 2, 3, 4, 5])
    js_day = (parsed_date.weekday() + 1) % 7
    if js_day not in business_days:
        await params.result_callback({
            "available": False,
            "slots": [],
            "message": "We're closed that day. Ask which weekday works best."
        })
        return

    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.get(
                f"{DASHBOARD_URL}/api/public/available-slots",
                params={"date": date},
                headers=_ingest_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )
            if resp.status == 200:
                data = await _safe_json(resp, "available-slots")
                slots = data.get("slots", [])
                available = [s for s in slots if s.get("available")]
                full = [s for s in slots if not s.get("available")]

                # Format human-readable times
                def fmt_time(t: str) -> str:
                    """Convert '08:00' to '8 AM', '14:00' to '2 PM'."""
                    try:
                        h = int(t.split(":")[0])
                        if h == 0: return "12 AM"
                        if h == 12: return "12 PM"
                        return f"{h} AM" if h < 12 else f"{h - 12} PM"
                    except (ValueError, IndexError):
                        return t

                if available:
                    time_strs = [f"{fmt_time(s['start'])} to {fmt_time(s['end'])}" for s in available]
                    msg = f"Available times on {date}: {', '.join(time_strs)}."
                    if full:
                        full_strs = [f"{fmt_time(s['start'])} to {fmt_time(s['end'])}" for s in full]
                        msg += f" Not available: {', '.join(full_strs)}."
                else:
                    msg = f"No booking windows are available on {date}. Suggest a different date."

                await params.result_callback({
                    "available": len(available) > 0,
                    "slots": [{"start": s["start"], "end": s["end"], "label": s.get("label", ""), "available": s["available"]} for s in slots],
                    "message": msg,
                })
            else:
                logger.error(f"Available slots API error: {resp.status} {await resp.text()}")
                await params.result_callback({
                    "available": False,
                    "fallback": True,
                    "error": "system_unavailable",
                    "slots": [],
                    "message": "I can't verify the schedule right now. Let me get someone from our team to help."
                })
    except Exception as e:
        logger.error(f"Available slots check failed: {e}")
        await params.result_callback({
            "available": False,
            "fallback": True,
            "error": "system_unavailable",
            "slots": [],
            "message": "I can't verify the schedule right now. Let me get someone from our team to help."
        })


async def handle_create_booking(params: FunctionCallParams):
    """Create a booking via dashboard's /api/agent/book.

    Dashboard expects: { customerName, customerPhone, address, volume, date, timeSlot, notes }
    Dashboard returns: { success, jobId, customerId, message } (status 201)
    Auth: X-AGENT-SECRET header (agentSecret)
    """
    config = _get_config()
    ctx = _get_context()
    name = params.arguments["name"]
    raw_phone = params.arguments.get("phone", "")
    # Use the real Twilio caller number if the LLM passed garbage (e.g. "the number you're calling from")
    import re
    if re.search(r"\d{7,}", re.sub(r"[\s\-\(\)\+]", "", raw_phone)):
        phone = raw_phone  # LLM passed something that looks like a real phone number
    else:
        phone = ctx.get("caller_number", raw_phone)  # Fall back to real Twilio caller number
        if phone != raw_phone:
            logger.info(f"Replaced LLM phone '{raw_phone}' with caller_number '{phone}'")
    address = params.arguments["address"]
    date = params.arguments["date"]
    time_slot_id = params.arguments.get("time")
    description = params.arguments["description"]
    booking_type = params.arguments.get("type", "pickup")
    container_size = params.arguments.get("container_size")
    rental_duration_days = params.arguments.get("rental_duration_days", 7)
    call_sid = ctx.get("call_sid", "")

    is_dumpster = booking_type in ("dumpster_rental", "dumpster_swap")
    is_swap = booking_type == "dumpster_swap"
    service_type = booking_type if is_dumpster else "junk_removal"

    # ── Validate date ──
    parsed_date = _validate_date(date)
    if not parsed_date:
        await params.result_callback({"error": "Date must be YYYY-MM-DD format."})
        return

    # ── Validate time slot ──
    if not time_slot_id and is_dumpster and not is_swap:
        # Dumpster rentals are all-day deliveries — no time slot needed
        slot = {"label": "All Day", "start": "08:00", "end": "17:00", "start_hour": 8, "period": "All Day"}
    else:
        slot = _validate_time_slot(time_slot_id)
        if not slot:
            await params.result_callback({
                "error": "I need a valid time. Check available slots first with check_available_slots, or use: morning, midday, or afternoon."
            })
            return

    # ── Business hours check ──
    if not _is_business_hours(parsed_date, slot["start_hour"], config):
        business_start = int(config.get("businessStart", 7))
        business_end = int(config.get("businessEnd", 19))
        await params.result_callback({
            "error": f"That's outside our hours. We're available from {_format_hour(business_start)} to {_format_hour(business_end)}."
        })
        return

    address_verification = get_address_verification(call_sid, address) if call_sid else None
    if address_verification and address_verification.get("service_area_status") == "out_of_area":
        await params.result_callback({
            "fallback": True,
            "error": "outside_service_area",
            "message": (
                "That address appears to be outside the configured service area. "
                "Do not create the booking. Let me connect the caller with the team to confirm coverage."
            ),
        })
        return

    # ── Build notes ──
    if is_swap:
        size_str = f"{container_size}-yard" if container_size else "current size"
        notes = f"Dumpster Swap: pick up full {size_str} container, drop off empty. {description}"
    elif is_dumpster:
        size_str = f"{container_size}-yard" if container_size else "TBD size"
        notes = f"Dumpster Rental Delivery: {size_str} container, {rental_duration_days} day rental. Project: {description}"
    else:
        notes = f"Junk Removal Pickup: {description}"

    # ── Create booking via /api/agent/book ──
    local_datetime = f"{date}T{slot['start']}:00"

    payload = {
        "customerName": name,
        "customerPhone": phone,
        "address": address,
        "date": local_datetime,
        "timeSlot": slot["period"],  # "08:00-10:00" or "Morning"
        "notes": notes,
        "type": booking_type,
        "serviceType": service_type,
        "twilioCallSid": ctx.get("call_sid", ""),
    }
    if is_dumpster:
        payload["containerSize"] = container_size
        payload["rentalDays"] = rental_duration_days

    promo_code = params.arguments.get("promo_code")
    if promo_code:
        payload["promoCode"] = promo_code

    # ── Include SMS consent in booking payload ──
    # Critical: dashboard sends confirmation SMS immediately on booking creation.
    # Consent must arrive WITH the booking, not after-the-fact in the call-log.
    consent_value = has_sms_consent(call_sid) if call_sid else None
    payload["smsConsent"] = {
        "optedIn": consent_value if consent_value is not None else False,
        "source": "phone_agent",
        "timestamp": datetime.now(ZoneInfo("UTC")).isoformat(),
        "consentTextVersion": "v1",
    }

    # ── API call wrapped in an independent task ──
    # If the caller hangs up mid-request, this task keeps running so:
    # 1. The dashboard's response is still processed
    # 2. mark_booking_complete fires (preventing "ghost bookings")
    # 3. The call log records outcome="booked" instead of "info_only"
    # The disconnect handler in bot.py awaits any in-flight tasks before
    # snapshotting state.

    async def _do_booking_request():
        try:
            async with aiohttp.ClientSession() as http:
                resp = await http.post(
                    f"{DASHBOARD_URL}/api/agent/book",
                    json=payload,
                    headers=_agent_headers(config),
                    timeout=aiohttp.ClientTimeout(total=15),
                )

                if resp.status == 201:
                    # Mark IMMEDIATELY — before any further awaits — so the
                    # disconnect handler always sees this booking, even if
                    # subsequent JSON parsing or callback delivery is interrupted.
                    if call_sid:
                        mark_booking_complete(call_sid, {
                            "caller_name": name,
                            "appointment_date": local_datetime,
                        })
                        reset_tool_failure(call_sid, "create_booking")

                    data = await _safe_json(resp, "create-booking")
                    job_id = data.get("jobId", "confirmed")
                    label = "Dumpster swap" if is_swap else "Dumpster rental" if is_dumpster else "Booking"
                    logger.info(f"{label} created: jobId={job_id} for {name} on {date} ({slot['period']} window)")

                    auto_booked = data.get("autoBooked", False)

                    if is_swap:
                        result = {
                            "success": True,
                            "booking_id": str(job_id),
                            "message": f"Dumpster swap scheduled for {date}, {slot['period']} window. We'll pick up the full container and drop off an empty one.",
                        }
                    elif is_dumpster and auto_booked:
                        size_label = f"{container_size}-yard" if container_size else ""
                        result = {
                            "success": True,
                            "autoBooked": True,
                            "booking_id": str(job_id),
                            "message": f"Dumpster rental CONFIRMED for {date}. {size_label} container, {rental_duration_days} day rental. Delivery is scheduled. If SMS consent was recorded earlier in the call, you may tell the caller they'll receive a confirmation text shortly. Do not promise an email unless the dashboard explicitly confirms one.",
                        }
                    elif is_dumpster:
                        size_label = f"{container_size}-yard" if container_size else ""
                        result = {
                            "success": True,
                            "autoBooked": False,
                            "booking_id": str(job_id),
                            "message": f"Dumpster rental request submitted for {date}. {size_label} container, {rental_duration_days} day rental. Our team will follow up to confirm availability and pricing.",
                        }
                    else:
                        result = {
                            "success": True,
                            "booking_id": str(job_id),
                            "message": f"Booking confirmed for {date}, {slot['period']} window ({slot['start']} - {slot['end']}).",
                        }
                elif resp.status == 409:
                    # slot_full — the time slot filled between checking and booking
                    data = await _safe_json(resp, "create-booking-conflict")
                    available_slots = data.get("availableSlots", [])
                    if available_slots:
                        def fmt_t(t: str) -> str:
                            try:
                                h = int(t.split(":")[0])
                                if h == 0: return "12 AM"
                                if h == 12: return "12 PM"
                                return f"{h} AM" if h < 12 else f"{h - 12} PM"
                            except (ValueError, IndexError):
                                return t
                        avail = [s for s in available_slots if s.get("available")]
                        if avail:
                            alt_strs = [f"{fmt_t(s['start'])} to {fmt_t(s['end'])}" for s in avail[:4]]
                            result = {
                                "error": "slot_full",
                                "message": f"That time slot just filled up. Available alternatives: {', '.join(alt_strs)}. Ask the caller which works.",
                                "availableSlots": avail,
                            }
                        else:
                            result = {
                                "error": "slot_full",
                                "message": f"That time slot and all others on {date} are now full. Suggest a different date.",
                                "availableSlots": [],
                            }
                    else:
                        result = {
                            "error": "slot_full",
                            "message": "That time slot just filled up. Ask the caller for a different time.",
                        }
                else:
                    text = await resp.text()
                    logger.error(f"Booking API error: {resp.status} {text}")
                    result = _build_error_with_tracking(
                        call_sid,
                        "create_booking",
                        "I'm having trouble with our scheduling system. Let me take your info and have someone call you back.",
                    )

                # Try to deliver result back to LLM — may no-op if the call ended
                try:
                    await params.result_callback(result)
                except Exception as cb_err:
                    logger.debug(f"Could not deliver booking result to LLM (call may have ended): {cb_err}")

        except asyncio.CancelledError:
            logger.info(f"Booking task cancelled for {call_sid}")
            raise
        except Exception as e:
            logger.error(f"Booking creation error: {e}")
            try:
                await params.result_callback(_build_error_with_tracking(
                    call_sid,
                    "create_booking",
                    "I'm having trouble creating the booking right now. Let me take your info and have someone call you back.",
                ))
            except Exception:
                pass

    # Spawn the booking request as an independent task and track it
    api_task = asyncio.create_task(_do_booking_request())
    if call_sid:
        add_inflight_task(call_sid, api_task)
        api_task.add_done_callback(lambda t, sid=call_sid: remove_inflight_task(sid, t))

    # Wait for it (shielded so cancellation of THIS handler doesn't kill api_task)
    try:
        await asyncio.shield(api_task)
    except asyncio.CancelledError:
        logger.info(f"create_booking handler cancelled for {call_sid}; API task continues independently")
        raise


async def handle_lookup_appointment(params: FunctionCallParams):
    """Look up existing appointments by phone number.

    Dashboard: GET /api/agent/lookup?phone=XXX with X-AGENT-SECRET auth
    Response: { found: bool, customer?: { id, name, phone }, jobs?: [{ jobId, title, address, scheduledDate, timeSlot, status }] }
    """
    config = _get_config()
    ctx = _get_context()
    raw_phone = params.arguments.get("phone", "")
    import re
    if re.search(r"\d{7,}", re.sub(r"[\s\-\(\)\+]", "", raw_phone)):
        phone = raw_phone
    else:
        phone = ctx.get("caller_number", raw_phone)

    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.get(
                f"{DASHBOARD_URL}/api/agent/lookup",
                params={"phone": phone},
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )

            if resp.status == 200:
                reset_tool_failure(ctx.get("call_sid", ""), "lookup_appointment")
                data = await _safe_json(resp, "lookup-appointment")
                if data.get("found"):
                    jobs = data.get("jobs", [])
                    remember_lookup_results(ctx.get("call_sid", ""), phone, jobs)
                    await params.result_callback({
                        "found": True,
                        "customer": data.get("customer", {}),
                        "jobs": jobs,
                    })
                else:
                    remember_lookup_results(ctx.get("call_sid", ""), phone, [])
                    await params.result_callback({
                        "found": False,
                        "message": "No appointments found for that number."
                    })
            else:
                await params.result_callback(_build_error_with_tracking(
                    ctx.get("call_sid", ""),
                    "lookup_appointment",
                    "I'm having trouble looking that up. Can you give me your name instead?",
                ))
    except Exception as e:
        logger.error(f"Appointment lookup failed: {e}")
        await params.result_callback(_build_error_with_tracking(
            ctx.get("call_sid", ""),
            "lookup_appointment",
            "I'm having trouble looking that up right now.",
        ))


async def handle_reschedule_appointment(params: FunctionCallParams):
    """Reschedule an appointment.

    Dashboard: POST /api/agent/reschedule { phone, newDate, newTime } with X-AGENT-SECRET auth
    """
    config = _get_config()
    ctx = _get_context()
    raw_phone = params.arguments.get("phone", "")
    import re
    if re.search(r"\d{7,}", re.sub(r"[\s\-\(\)\+]", "", raw_phone)):
        phone = raw_phone
    else:
        phone = ctx.get("caller_number", raw_phone)
    new_date = params.arguments["new_date"]
    new_time_slot_id = params.arguments["new_time"]

    parsed_date = _validate_date(new_date)
    if not parsed_date:
        await params.result_callback({"error": "Date must be YYYY-MM-DD format."})
        return

    slot = _validate_time_slot(new_time_slot_id)
    if not slot:
        await params.result_callback({
            "error": "Please use a valid time window: morning, midday, or afternoon."
        })
        return

    if not _is_business_hours(parsed_date, slot["start_hour"], config):
        await params.result_callback({
            "error": "That's outside our business hours."
        })
        return

    call_sid = ctx.get("call_sid", "")
    job_id, selection_error = resolve_job_id_from_lookup(
        call_sid,
        phone,
        params.arguments.get("job_id"),
    )
    if selection_error:
        await params.result_callback(selection_error)
        return

    try:
        async with aiohttp.ClientSession() as http:
            payload = {
                    "phone": phone,
                    "newDate": new_date,
                    "newTime": slot["start"],
                    "timeSlot": slot["period"],
                }
            payload["jobId"] = job_id
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/reschedule",
                json=payload,
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )

            if resp.status == 200:
                reset_tool_failure(ctx.get("call_sid", ""), "reschedule_appointment")
                await params.result_callback({
                    "success": True,
                    "message": f"Appointment rescheduled to {new_date}, {slot['period']} window ({slot['start']} - {slot['end']}).",
                })
                logger.info(f"Rescheduled for {phone} to {new_date} ({slot['period']} window)")
            else:
                text = await resp.text()
                logger.error(f"Reschedule API error: {resp.status} {text}")
                await params.result_callback(_build_error_with_tracking(
                    ctx.get("call_sid", ""),
                    "reschedule_appointment",
                    "Couldn't reschedule that appointment.",
                ))
    except Exception as e:
        logger.error(f"Reschedule error: {e}")
        await params.result_callback(_build_error_with_tracking(
            ctx.get("call_sid", ""),
            "reschedule_appointment",
            "I'm having trouble rescheduling right now.",
        ))


async def handle_cancel_appointment(params: FunctionCallParams):
    """Cancel an appointment.

    Dashboard: POST /api/agent/cancel { phone, reason } with X-AGENT-SECRET auth
    """
    config = _get_config()
    ctx = _get_context()
    raw_phone = params.arguments.get("phone", "")
    import re
    if re.search(r"\d{7,}", re.sub(r"[\s\-\(\)\+]", "", raw_phone)):
        phone = raw_phone
    else:
        phone = ctx.get("caller_number", raw_phone)
    reason = params.arguments.get("reason", "Customer requested cancellation")

    call_sid = ctx.get("call_sid", "")
    job_id, selection_error = resolve_job_id_from_lookup(
        call_sid,
        phone,
        params.arguments.get("job_id"),
    )
    if selection_error:
        await params.result_callback(selection_error)
        return

    try:
        async with aiohttp.ClientSession() as http:
            payload = {
                    "phone": phone,
                    "reason": reason,
                }
            payload["jobId"] = job_id
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/cancel",
                json=payload,
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )

            if resp.status == 200:
                reset_tool_failure(ctx.get("call_sid", ""), "cancel_appointment")
                await params.result_callback({
                    "success": True,
                    "message": "Appointment cancelled.",
                })
                logger.info(f"Cancelled appointment for {phone}, reason: {reason}")
            else:
                await params.result_callback(_build_error_with_tracking(
                    ctx.get("call_sid", ""),
                    "cancel_appointment",
                    "Couldn't cancel that appointment.",
                ))
    except Exception as e:
        logger.error(f"Cancel error: {e}")
        await params.result_callback(_build_error_with_tracking(
            ctx.get("call_sid", ""),
            "cancel_appointment",
            "I'm having trouble processing the cancellation.",
        ))


# ── Address Verification ──────────────────────────────

async def handle_verify_address(params: FunctionCallParams):
    """Verify and normalize a caller's address using Google Geocoding API.

    Returns the top formatted address result so the agent can read it back
    and confirm with the caller before booking.
    """
    config = _get_config()
    ctx = _get_context()
    call_sid = ctx.get("call_sid", "")
    address = params.arguments.get("address", "")
    if not address:
        await params.result_callback({
            "verified": False,
            "message": "No address provided. Ask the caller for their full address."
        })
        return

    if not GOOGLE_MAPS_API_KEY:
        await params.result_callback({
            "verified": False,
            "message": "Address verification unavailable. Read back the address as you heard it and ask the caller to confirm."
        })
        return

    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": address, "key": GOOGLE_MAPS_API_KEY},
                timeout=aiohttp.ClientTimeout(total=5),
            )
            data = await resp.json()

        if data.get("status") == "OK" and data.get("results"):
            result = data["results"][0]
            formatted = result.get("formatted_address", "")
            if formatted:
                # Strip ", USA" or ", US" suffix for cleaner readback
                formatted = formatted.replace(", USA", "").replace(", US", "")
                components = _extract_geocode_components(result)
                area_status, area_reason = _assess_service_area(components, config)
                if call_sid:
                    remember_address_verification(
                        call_sid,
                        address,
                        formatted,
                        area_status,
                        area_reason,
                    )

                if area_status == "out_of_area":
                    message = (
                        f"Verified address: {formatted}. This appears to be outside the configured service area. "
                        "Do not book it automatically; offer to transfer the caller to the team to confirm coverage."
                    )
                elif area_status == "uncertain":
                    message = (
                        f"Verified address: {formatted}. Read this back to the caller and ask if it's correct. "
                        "Service-area coverage is not fully proven from the configured area text, so transfer if the caller asks whether that location is covered."
                    )
                else:
                    message = f"Verified address: {formatted}. Read this back to the caller and ask if it's correct."

                await params.result_callback({
                    "verified": True,
                    "formatted_address": formatted,
                    "city": components.get("city"),
                    "state": components.get("state"),
                    "postal_code": components.get("postal_code"),
                    "service_area_status": area_status,
                    "service_area_reason": area_reason,
                    "message": message,
                })
                return

        # No results or bad status
        await params.result_callback({
            "verified": False,
            "message": "I couldn't find a match for that address. Ask the caller to repeat the full address including street number, street name, and city."
        })

    except Exception as e:
        logger.error(f"Address verification failed: {e}")
        await params.result_callback({
            "verified": False,
            "message": "Address verification unavailable right now. Read back the address as you heard it and ask the caller to confirm."
        })


# ── Helpers ─────────────────────────────────────────────

def _format_hour(hour: int) -> str:
    if hour == 0:
        return "12:00 AM"
    elif hour < 12:
        return f"{hour}:00 AM"
    elif hour == 12:
        return "12:00 PM"
    else:
        return f"{hour - 12}:00 PM"


# ── Twilio Client (lazy singleton for call transfers) ──

_twilio_client = None


def _get_twilio_client():
    """Get or create Twilio REST client for call operations."""
    global _twilio_client
    if _twilio_client is None:
        from twilio.rest import Client as TwilioClient
        _twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    return _twilio_client


# ── Scheduled Callbacks ────────────────────────────────

async def handle_schedule_callback(params: FunctionCallParams):
    """Schedule a human callback through the dashboard.

    Dashboard: POST /api/agent/schedule-callback with X-AGENT-SECRET auth.
    """
    config = _get_config()
    ctx = _get_context()
    call_sid = ctx.get("call_sid", "")

    requested_time = (
        params.arguments.get("requested_time")
        or params.arguments.get("requestedTime")
        or params.arguments.get("callback_due_at")
        or params.arguments.get("callbackDueAt")
        or ""
    )
    requested_time = str(requested_time).strip()
    reason = str(params.arguments.get("reason") or "Caller requested a scheduled callback").strip()
    caller_phone = (
        params.arguments.get("caller_phone")
        or params.arguments.get("callerPhone")
        or ctx.get("caller_number", "")
        or ""
    )
    caller_phone = str(caller_phone).strip()
    caller_name = (
        params.arguments.get("caller_name")
        or params.arguments.get("callerName")
        or ""
    )
    caller_name = str(caller_name).strip()

    async def _deliver_result(result: dict[str, Any]):
        try:
            await params.result_callback(result)
        except Exception as cb_err:
            logger.debug(f"Could not deliver schedule_callback result to LLM (call may have ended): {cb_err}")

    if not requested_time:
        await _deliver_result({
            "error": "requested_time_required",
            "message": "Ask the caller for the exact date and time they want someone to call them back.",
        })
        return

    if "T" not in requested_time:
        await _deliver_result({
            "error": "requested_time_must_include_time",
            "message": "The callback time needs both a date and time. Ask the caller for an exact callback time, then use YYYY-MM-DDTHH:MM:SS.",
        })
        return

    if not caller_phone:
        await _deliver_result({
            "error": "caller_phone_required",
            "message": "Confirm the best phone number for the callback, then try again.",
        })
        return

    payload: dict[str, Any] = {
        "callerPhone": caller_phone,
        "requestedTime": requested_time,
        "reason": reason,
    }
    if call_sid:
        payload["twilioCallSid"] = call_sid
    if caller_name:
        payload["callerName"] = caller_name

    async def _do_schedule_callback_request():
        try:
            async with aiohttp.ClientSession() as http:
                resp = await http.post(
                    f"{DASHBOARD_URL}/api/agent/schedule-callback",
                    json=payload,
                    headers=_agent_headers(config),
                    timeout=aiohttp.ClientTimeout(total=15),
                )

                if resp.status in (200, 201):
                    if call_sid:
                        mark_callback_requested(call_sid, requested_time, reason)
                        reset_tool_failure(call_sid, "schedule_callback")

                    data = await _safe_json(resp, "schedule-callback")
                    callback_due_at = data.get("callbackDueAt") or requested_time
                    if call_sid:
                        mark_callback_requested(call_sid, str(callback_due_at), reason)

                    await _deliver_result({
                        "success": True,
                        "callbackDueAt": callback_due_at,
                        "phoneCallId": data.get("phoneCallId"),
                        "callbackTaskId": data.get("callbackTaskId"),
                        "message": "Callback scheduled. You may now tell the caller someone from the team will call them back at the confirmed time.",
                    })
                    return

                data = await _safe_json(resp, "schedule-callback-error")
                message = data.get("error") or data.get("message") or "That callback time is not available."
                if resp.status in (400, 422):
                    await _deliver_result({
                        "error": "invalid_callback_time",
                        "message": f"{message} Ask the caller for another time during business hours.",
                    })
                    return

                logger.error(f"Schedule callback API error: {resp.status} {data}")
                if call_sid:
                    increment_tool_failure(call_sid, "schedule_callback")
                await _deliver_result(_build_immediate_fallback(
                    "schedule_callback_unavailable",
                    "I'm having trouble scheduling that callback right now. Let me get someone from our team to help.",
                ))

        except asyncio.CancelledError:
            logger.info(f"Schedule callback task cancelled for {call_sid}")
            raise
        except Exception as e:
            logger.error(f"Schedule callback error: {e}")
            if call_sid:
                increment_tool_failure(call_sid, "schedule_callback")
            await _deliver_result(_build_immediate_fallback(
                "schedule_callback_unavailable",
                "I'm having trouble scheduling that callback right now. Let me get someone from our team to help.",
            ))

    api_task = asyncio.create_task(_do_schedule_callback_request())
    if call_sid:
        add_inflight_task(call_sid, api_task)
        api_task.add_done_callback(lambda t, sid=call_sid: remove_inflight_task(sid, t))

    try:
        await asyncio.shield(api_task)
    except asyncio.CancelledError:
        logger.info(f"schedule_callback handler cancelled for {call_sid}; API task continues independently")
        raise


# ── Human Handoff ──────────────────────────────────────

async def handle_transfer_to_human(params: FunctionCallParams):
    """Transfer the live call to the client's forwarding phone number.

    Uses Twilio's REST API to update the active call with <Dial> TwiML,
    which redirects the caller to the client's number. The AI pipeline
    is then cancelled since the call audio stream will end.

    Requires 'forwardingPhone' in the client config (set via dashboard).
    """
    config = _get_config()
    ctx = _get_context()
    reason = params.arguments.get("reason", "Caller requested human agent")
    call_sid = ctx.get("call_sid")
    forwarding_phone = config.get("forwardingPhone", "")
    if call_sid:
        mark_transfer_state(call_sid, "requested", reason)

    # No forwarding number configured — take a message instead
    if not forwarding_phone:
        logger.warning(f"No forwardingPhone configured — cannot transfer call {call_sid}")
        if call_sid:
            mark_transfer_state(
                call_sid,
                "unavailable_no_forwarding_phone",
                reason,
                callback_requested=True,
            )
        await params.result_callback({
            "transferred": False,
            "message": (
                "I'm not able to transfer you right now, but I've noted your request. "
                "Someone from our team will call you back shortly. "
                "Is there anything else I can help you with in the meantime?"
            ),
        })
        return

    if not call_sid:
        logger.error("No call_sid in context — cannot transfer")
        await params.result_callback({
            "transferred": False,
            "message": "I'm having trouble with the transfer. Let me take your info and have someone call you back.",
        })
        return

    # Build TwiML to redirect the call
    company_name = config.get("companyName", "our team")
    base_url = str(config.get("_phoneAgentBaseUrl") or "").rstrip("/")
    client_id = str(config.get("_clientId") or "unknown")
    action_url = ""
    if base_url:
        action_url = (
            f"{base_url}/transfer-status/"
            f"{quote(client_id, safe='')}/{quote(call_sid, safe='')}"
        )
    else:
        logger.warning(f"No phone-agent public base URL available for transfer status callback on call {call_sid}")

    dial_attrs = ['timeout="25"']
    twilio_number = config.get("twilioNumber", "")
    if twilio_number:
        dial_attrs.append(f"callerId={quoteattr(str(twilio_number))}")
    if action_url:
        dial_attrs.append(f"action={quoteattr(action_url)}")
        dial_attrs.append('method="POST"')
    dial_attr_str = " ".join(dial_attrs)
    if call_sid:
        _remember_pending_transfer(
            call_sid,
            config,
            ctx.get("caller_number", ""),
            reason,
        )

    post_dial_fallback = (
        ""
        if action_url
        else '<Say voice="Polly.Joanna">We were unable to reach anyone at this time. We have marked this for a callback. Goodbye.</Say>'
    )
    transfer_twiml = (
        f'<Response>'
        f'<Say voice="Polly.Joanna">Please hold while we connect you with {escape(str(company_name))}.</Say>'
        f'<Dial {dial_attr_str}>{escape(str(forwarding_phone))}</Dial>'
        f'{post_dial_fallback}'
        f'</Response>'
    )

    try:
        # Twilio SDK is synchronous — run in thread executor
        client = _get_twilio_client()
        await asyncio.to_thread(
            client.calls(call_sid).update,
            twiml=transfer_twiml,
        )

        logger.info(f"Call {call_sid} transferred to {forwarding_phone} (reason: {reason})")

        # Twilio accepting the redirect only means dialing started. The final
        # completed/failed state comes later from the <Dial action> callback.
        if ctx.get("call_sid"):
            mark_transfer_state(ctx["call_sid"], "dialing", reason)
            _update_pending_transfer_status(ctx["call_sid"], "dialing")

        await params.result_callback({
            "transferred": True,
            "transferStatus": "dialing",
            "message": "Transferring the call now.",
        })

        # Cancel the AI pipeline — the call audio stream will end
        pipeline_task = ctx.get("pipeline_task")
        if pipeline_task:
            await asyncio.sleep(2)  # Brief delay so the TTS can finish speaking
            await pipeline_task.cancel()

    except Exception as e:
        logger.error(f"Transfer failed for call {call_sid}: {e}")
        if call_sid:
            mark_transfer_state(call_sid, "failed", reason, callback_requested=True)
            _update_pending_transfer_status(call_sid, "failed", callback_requested=True)
        await params.result_callback({
            "transferred": False,
            "message": "I'm having trouble with the transfer right now. Let me take your info and have someone call you back within the hour.",
        })


# ── SMS Messaging ──────────────────────────────────────

# Fixed SMS templates — LLM picks a template name and supplies variables.
_SMS_TEMPLATES: dict[str, str] = {
    "follow_up": (
        "{company_name}: Thanks for calling! Whenever you're ready, you can "
        "get a free estimate and book online in under 2 minutes — no phone "
        "call needed 👇\n\n"
        "{website_url}?utm_source=phone_agent&utm_medium=sms&utm_campaign=followup\n\n"
        "Or call us back anytime at {company_phone} — we're here "
        "{days_str}, {hours_str}."
    ),
}


def _build_sms_body(template_name: str, config: dict[str, Any]) -> str | None:
    """Render an SMS template with config values. Returns None for unknown templates."""
    template = _SMS_TEMPLATES.get(template_name)
    if not template:
        return None

    company_name = config.get("companyName", "Junk Removal")
    website_url = config.get("websiteUrl", "")
    # Use forwardingPhone as company phone; fall back to twilioNumber
    company_phone = config.get("forwardingPhone") or config.get("twilioNumber", "")

    # Build business hours string
    business_start = int(config.get("businessStart", 7))
    business_end = int(config.get("businessEnd", 19))
    business_days = config.get("businessDays", [0, 1, 2, 3, 4, 5])
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    active_days = [day_names[d] for d in business_days if 0 <= d <= 6]
    days_str = (
        f"{active_days[0]} through {active_days[-1]}"
        if len(active_days) > 1
        else active_days[0] if active_days else "Monday through Saturday"
    )

    def fmt_hour(h: int) -> str:
        if h == 0:
            return "12 AM"
        elif h < 12:
            return f"{h} AM"
        elif h == 12:
            return "12 PM"
        else:
            return f"{h - 12} PM"

    hours_str = f"{fmt_hour(business_start)} to {fmt_hour(business_end)}"

    return template.format(
        company_name=company_name,
        website_url=website_url,
        company_phone=company_phone,
        days_str=days_str,
        hours_str=hours_str,
    )


async def handle_send_sms(params: FunctionCallParams):
    """Send a fixed-template SMS to the caller via Twilio.

    Triple-gated: smsEnabled must be true, twilioNumber must exist,
    and caller_number (or explicit phone arg) must be available.

    Uses the platform's Twilio credentials but sends FROM the client's
    provisioned phone number so the SMS appears to come from them.
    """
    config = _get_config()
    ctx = _get_context()
    template_name = params.arguments.get("template", "")
    explicit_phone = params.arguments.get("phone", "")

    # ── Gate 0: SMS consent ──
    call_sid = ctx.get("call_sid", "")
    consent = has_sms_consent(call_sid) if call_sid else None
    if consent is not True:
        logger.info(f"SMS blocked: no consent for call {call_sid} (consent={consent})")
        await params.result_callback({
            "sent": False,
            "message": "I can give you the details verbally instead.",
        })
        return

    # ── Gate 1: smsEnabled ──
    if not config.get("smsEnabled", False):
        logger.info("SMS disabled for this client — skipping send_sms")
        await params.result_callback({
            "sent": False,
            "message": "I'm not able to send texts right now, but I can give you the details verbally.",
        })
        return

    # ── Gate 2: twilioNumber (from= number) ──
    twilio_number = config.get("twilioNumber")
    if not twilio_number:
        logger.warning("No twilioNumber configured — cannot send SMS")
        await params.result_callback({
            "sent": False,
            "message": "I'm not able to send texts right now, but let me give you the information.",
        })
        return

    # ── Gate 3: recipient number ──
    to_number = explicit_phone or ctx.get("caller_number", "")
    if not to_number:
        logger.warning("No recipient phone number — cannot send SMS")
        await params.result_callback({
            "sent": False,
            "message": "I don't have a number to text. What's the best number to reach you at?",
        })
        return

    # ── Gate 4: websiteUrl required for both templates ──
    website_url = config.get("websiteUrl")
    if not website_url:
        logger.info("No websiteUrl configured — cannot send website link SMS")
        await params.result_callback({
            "sent": False,
            "message": "I'm not able to send a link right now, but you can search for us online to find our website.",
        })
        return

    # ── Build message from template ──
    body = _build_sms_body(template_name, config)
    if not body:
        logger.error(f"Unknown SMS template: {template_name}")
        await params.result_callback({
            "sent": False,
            "message": "I had trouble sending that text. Let me give you the information verbally instead.",
        })
        return

    # ── Send via dashboard SMS endpoint (enforces A2P, consent, suppression gates) ──
    try:
        normalized_to = to_number if to_number.startswith("+") else f"+1{to_number.replace('-', '').replace(' ', '')}"
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/sms/send",
                json={
                    "clientId": config.get("clientId", ""),
                    "to": normalized_to,
                    "message": body,
                    "consentGranted": True,
                    "consentSource": "phone_agent",
                },
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=15),
            )
            data = await _safe_json(resp, "agent-sms-send")

        if data.get("success"):
            logger.info(f"SMS sent via dashboard: template={template_name} to={to_number} sid={data.get('sid', 'n/a')}")

            # Track that SMS was sent during this call
            ctx = _get_context()
            if ctx.get("call_sid"):
                mark_sms_sent(ctx["call_sid"])

            await params.result_callback({
                "sent": True,
                "message": "Text message sent successfully.",
            })
        else:
            error = data.get("error", "unknown")
            logger.info(f"SMS blocked by dashboard: {error} (template={template_name} to={to_number})")
            await params.result_callback({
                "sent": False,
                "message": "I'm not able to send texts right now, but I can give you the details verbally.",
            })

    except Exception as e:
        logger.error(f"SMS send via dashboard failed: {e}")
        await params.result_callback({
            "sent": False,
            "message": "I had trouble sending that text, but no worries — let me give you the details.",
        })


# ── Automated Follow-Up SMS (called from bot.py after call ends) ──

# 24-hour cooldown — keyed by normalized phone number
_sms_cooldown: dict[str, datetime] = {}
SMS_COOLDOWN_HOURS = 24


def _normalize_phone(phone: str) -> str:
    """Strip a phone to digits only for consistent cooldown keying."""
    return "".join(c for c in phone if c.isdigit())


async def send_automated_followup(caller_number: str, config: dict[str, Any], sms_consented: bool = False) -> bool:
    """Send an automated follow-up SMS after a no-booking call.

    Called from bot.py on disconnect, NOT from an LLM tool handler.
    Returns True if sent, False otherwise.
    """
    # Gate: explicit consent required
    if not sms_consented:
        logger.info(f"Automated follow-up skipped: no SMS consent for {caller_number}")
        return False

    # Gate checks
    if not config.get("smsEnabled", False):
        logger.debug("Automated follow-up skipped: SMS disabled")
        return False
    twilio_number = config.get("twilioNumber")
    if not twilio_number:
        logger.debug("Automated follow-up skipped: no twilioNumber")
        return False
    if not config.get("websiteUrl"):
        logger.debug("Automated follow-up skipped: no websiteUrl")
        return False
    if not caller_number:
        logger.debug("Automated follow-up skipped: no caller number")
        return False

    # ── Prune expired cooldown entries ──
    cutoff = datetime.now() - timedelta(hours=SMS_COOLDOWN_HOURS)
    expired_keys = [k for k, v in _sms_cooldown.items() if v < cutoff]
    for k in expired_keys:
        del _sms_cooldown[k]

    # ── 24-hour cooldown check ──
    key = _normalize_phone(caller_number)
    last_sent = _sms_cooldown.get(key)
    if last_sent and (datetime.now() - last_sent) < timedelta(hours=SMS_COOLDOWN_HOURS):
        logger.info(f"Automated follow-up skipped: SMS already sent to {caller_number} within {SMS_COOLDOWN_HOURS}h")
        return False

    body = _build_sms_body("follow_up", config)
    if not body:
        return False

    try:
        normalized_to = caller_number if caller_number.startswith("+") else f"+1{caller_number.replace('-', '').replace(' ', '')}"
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/sms/send",
                json={
                    "clientId": config.get("clientId", ""),
                    "to": normalized_to,
                    "message": body,
                    "consentGranted": True,
                    "consentSource": "phone_agent",
                },
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=15),
            )
            data = await _safe_json(resp, "agent-sms-followup")

        if data.get("success"):
            logger.info(f"Automated follow-up SMS sent via dashboard: to={caller_number} sid={data.get('sid', 'n/a')}")
            _sms_cooldown[key] = datetime.now()
            return True
        else:
            error = data.get("error", "unknown")
            logger.info(f"Automated follow-up SMS blocked by dashboard: {error} (to={caller_number})")
            return False
    except Exception as e:
        logger.error(f"Automated follow-up SMS failed: {e}")
        return False


# ── Dashboard Logging Helpers ──────────────────────────────────

async def _log_sms_to_dashboard(
    config: dict[str, Any],
    to_number: str,
    from_number: str,
    template_name: str,
    sms_type: str,
    twilio_call_sid: str = "",
    twilio_sms_sid: str = "",
) -> None:
    """Fire-and-forget POST to dashboard SMS log endpoint."""
    try:
        payload = {
            "toNumber": to_number,
            "fromNumber": from_number,
            "templateName": template_name,
            "type": sms_type,
        }
        if twilio_call_sid:
            payload["twilioCallSid"] = twilio_call_sid
        if twilio_sms_sid:
            payload["twilioSmsSid"] = twilio_sms_sid

        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/sms-log",
                json=payload,
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )
            if resp.status == 201:
                logger.debug(f"SMS log recorded for {to_number}")
            else:
                body = await resp.text()
                logger.warning(f"SMS log failed ({resp.status}): {body[:200]}")
    except Exception as e:
        logger.error(f"SMS log POST error: {e}")


async def log_call_to_dashboard(
    config: dict[str, Any],
    twilio_call_sid: str,
    from_number: str,
    to_number: str,
    duration: int,
    outcome: str,
    summary: str = "",
    caller_name: str = "",
    appointment_date: str = "",
    sms_consent: bool | None = None,
    transfer_reason: str | None = None,
    transfer_status: str | None = None,
    callback_requested: bool | None = None,
    callback_due_at: str | None = None,
) -> bool:
    """POST to dashboard call log endpoint. Returns True when accepted."""
    try:
        payload: dict[str, Any] = {
            "twilioCallSid": twilio_call_sid,
            "fromNumber": from_number,
            "duration": duration,
            "outcome": outcome,
        }
        if to_number:
            payload["toNumber"] = to_number
        if caller_name:
            payload["callerName"] = caller_name
        if summary:
            payload["summary"] = summary
        if appointment_date:
            payload["appointmentDate"] = appointment_date
        # Include SMS consent decision for audit trail
        if sms_consent is not None:
            payload["smsConsent"] = sms_consent
        if transfer_reason:
            payload["transferReason"] = transfer_reason
        if transfer_status:
            payload["transferStatus"] = transfer_status
        if callback_requested is not None:
            payload["callbackRequested"] = callback_requested
        if callback_due_at:
            payload["callbackDueAt"] = callback_due_at

        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/call-log",
                json=payload,
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )
            if resp.status in (200, 201):
                logger.info(f"Call log recorded: {twilio_call_sid} outcome={outcome}")
                return True
            else:
                body = await resp.text()
                logger.warning(f"Call log failed ({resp.status}): {body[:200]}")
                return False
    except Exception as e:
        logger.error(f"Call log POST error: {e}")
        return False


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def process_transfer_status_callback(
    call_sid: str,
    dial_status: str,
    dial_call_sid: str = "",
    dial_duration: str | int | None = None,
) -> str:
    """Handle Twilio <Dial action> status and update the dashboard call log."""
    status = (dial_status or "").strip().lower()
    pending = _pending_transfers.get(call_sid)

    if status in _TRANSFER_SUCCESS_STATUSES:
        transfer_status = "completed"
        outcome = "transferred"
        callback_requested = False
    elif status in _TRANSFER_FAILURE_STATUSES:
        transfer_status = "failed"
        outcome = "callback_requested"
        callback_requested = True
    else:
        transfer_status = "failed"
        outcome = "callback_requested"
        callback_requested = True
        logger.warning(f"Unknown DialCallStatus for {call_sid}: {dial_status!r}")

    if call_sid in _call_contexts:
        mark_transfer_state(
            call_sid,
            transfer_status,
            pending.get("reason") if pending else None,
            callback_requested=callback_requested,
        )
        if transfer_status == "completed":
            mark_transfer_complete(call_sid)

    _update_pending_transfer_status(
        call_sid,
        transfer_status,
        callback_requested=callback_requested,
    )

    if not pending:
        logger.warning(f"Transfer status callback received without pending context for {call_sid}")
        if callback_requested:
            return (
                '<Response><Say voice="Polly.Joanna">'
                "We were not able to reach anyone right now. Please try again later. Goodbye."
                "</Say><Hangup/></Response>"
            )
        return "<Response><Hangup/></Response>"

    base_duration = _parse_int(pending.get("duration"), 0)
    dial_seconds = _parse_int(dial_duration, 0)
    duration = max(base_duration, base_duration + max(dial_seconds, 0))
    reason = pending.get("reason") or "Caller requested human handoff"
    summary_base = (pending.get("summary") or "").strip()
    if transfer_status == "completed":
        summary = (
            f"{summary_base} Transfer completed to human."
            if summary_base
            else f"Caller requested human handoff ({reason}); transfer completed."
        )
    else:
        summary = (
            f"{summary_base} Transfer failed or was not answered; callback needed."
            if summary_base
            else f"Caller requested human handoff ({reason}); transfer failed or was not answered, so callback is needed."
        )

    await log_call_to_dashboard(
        config=pending.get("config", {}),
        twilio_call_sid=call_sid,
        from_number=pending.get("from_number", ""),
        to_number=pending.get("to_number", ""),
        duration=duration,
        outcome=outcome,
        summary=summary,
        caller_name=pending.get("caller_name") or "",
        appointment_date=pending.get("appointment_date") or "",
        sms_consent=pending.get("sms_consent"),
        transfer_reason=reason,
        transfer_status=transfer_status,
        callback_requested=callback_requested,
        callback_due_at=pending.get("callback_due_at"),
    )

    logger.info(
        f"Transfer status recorded for {call_sid}: "
        f"status={transfer_status} dial_status={status or 'missing'} dial_call_sid={dial_call_sid or 'missing'}"
    )
    pending["terminal_logged"] = True
    pending["transfer_status"] = transfer_status
    pending["callback_requested"] = callback_requested
    pending["duration"] = duration
    pending["summary"] = summary
    if call_sid not in _call_contexts:
        _pending_transfers.pop(call_sid, None)

    if callback_requested:
        return (
            '<Response><Say voice="Polly.Joanna">'
            "We were not able to reach anyone right now. We have marked this for a callback. Goodbye."
            "</Say><Hangup/></Response>"
        )
    return "<Response><Hangup/></Response>"
