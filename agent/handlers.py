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
  - transfer_to_human     → POST /api/agent/voice/redirect-live-call (x-api-key auth)
  - validate_promo_code   → GET  /api/promo/validate          (x-api-key + x-site-token)
"""

import asyncio
import contextvars
import json
import re
import time
import aiohttp
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from agent.business_hours import (
    evaluate_current_business_hours,
    format_business_hours_for_day as _shared_format_business_hours_for_day,
    is_business_hours as _shared_is_business_hours,
    is_open_day as _shared_is_open_day,
)
from agent.dashboard_redirect import redirect_live_call_via_dashboard
from agent.phone_coverage import DASHBOARD_ORIGIN_AI_TRANSFER
from agent.prosody import format_address_for_speech
from agent.prompt import client_supports_dumpsters
from client_config import get_client_config
from config import (
    DASHBOARD_URL,
    GOOGLE_MAPS_API_KEY,
    INGEST_API_KEY,
    PLATFORM_API_KEY,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
)

# ── Per-call context — keyed by call_sid for concurrency safety ──
_call_contexts: dict[str, dict[str, Any]] = {}
_current_call_sid: contextvars.ContextVar[str] = contextvars.ContextVar("current_call_sid", default="")

# Twilio <Dial action> callbacks arrive after the media stream has ended, so
# transfer state must outlive the normal per-call context.
_terminal_transfer_states: dict[str, dict[str, Any]] = {}
_TRANSFER_SUCCESS_STATUSES = {"completed", "answered"}
_TRANSFER_FAILURE_STATUSES = {"busy", "no-answer", "failed", "canceled"}
_TRANSFER_TERMINAL_STATUSES = {"completed", "failed", "unavailable_no_forwarding_phone"}


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
        # A caller refused a handoff after hours who then books does not need
        # calling back — they got what they rang for. Leaving the flag set puts
        # a completed booking in the Callback Queue, which is the one place that
        # has to keep meaning "still needs attention". An explicitly scheduled
        # callback is a different status and is left alone.
        if _call_contexts[call_sid].get("transfer_status") == "suppressed_after_hours":
            _call_contexts[call_sid]["callback_requested"] = False
        if details:
            current = _call_contexts[call_sid].setdefault("booking_details", {})
            current.update({k: v for k, v in details.items() if v is not None})


def set_pipeline_task(call_sid: str, task):
    """Store the PipelineTask so handlers can cancel it (e.g. after transfer)."""
    if call_sid in _call_contexts:
        _call_contexts[call_sid]["pipeline_task"] = task


def mark_dashboard_transfer_ownership(call_sid: str, origin: str, reason: str | None = None):
    """Record that dashboard Twilio routes own the final transfer result."""
    if call_sid not in _call_contexts:
        return
    _call_contexts[call_sid]["dashboard_owns_transfer_result"] = True
    _call_contexts[call_sid]["dashboard_transfer_origin"] = origin
    _call_contexts[call_sid]["transfer_status"] = "dashboard_redirected"
    if reason is not None:
        _call_contexts[call_sid]["transfer_reason"] = reason


def is_dashboard_owned_transfer(call_sid: str) -> bool:
    """Return True once a live call has been redirected to dashboard handoff."""
    return bool(_call_contexts.get(call_sid, {}).get("dashboard_owns_transfer_result"))


def is_booking_complete(call_sid: str) -> bool:
    """Check if a booking has been completed during this call."""
    ctx = _call_contexts.get(call_sid, {})
    return ctx.get("booking_complete", False)


def get_booking_log_state(call_sid: str) -> dict[str, Any]:
    """Return structured booking fields supported by the dashboard call log."""
    ctx = _call_contexts.get(call_sid, {})
    return dict(ctx.get("booking_details", {}))


def was_transfer_complete(call_sid: str) -> bool:
    """Check if a transfer was completed during this call.

    Nothing sets this any more. Its only writer was the Twilio transfer-status
    callback, whose route was never reachable in production (the TwiML that
    pointed at it was built by a function no live path called) and which has
    since been removed. So this returns False for every call, and the outcome
    ladder in `bot.py` is driven entirely by `transfer_status`. Kept because that
    ladder still reads it; it is a no-op, not a signal.
    """
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
    if status in _TRANSFER_TERMINAL_STATUSES:
        remember_terminal_transfer_state(
            call_sid,
            status,
            reason=reason,
            callback_requested=callback_requested,
        )


def remember_terminal_transfer_state(
    call_sid: str,
    status: str,
    *,
    reason: str | None = None,
    callback_requested: bool = False,
    summary: str | None = None,
):
    """Remember terminal transfer state so later final logs cannot downgrade it."""
    if not call_sid or status not in _TRANSFER_TERMINAL_STATUSES:
        return
    _terminal_transfer_states[call_sid] = {
        "transfer_status": status,
        "transfer_reason": reason,
        "callback_requested": callback_requested,
        "summary": summary,
        "terminal_logged": True,
    }


def get_terminal_transfer_state(call_sid: str) -> dict[str, Any]:
    terminal = _terminal_transfer_states.get(call_sid)
    return dict(terminal) if terminal else {}


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
    state = {
        "transfer_status": ctx.get("transfer_status"),
        "transfer_reason": ctx.get("transfer_reason"),
        "callback_requested": ctx.get("callback_requested", False),
        "callback_due_at": ctx.get("callback_due_at"),
        "callback_reason": ctx.get("callback_reason"),
    }
    terminal = get_terminal_transfer_state(call_sid)
    if terminal:
        state.update({
            "transfer_status": terminal.get("transfer_status") or state.get("transfer_status"),
            "transfer_reason": terminal.get("transfer_reason") or state.get("transfer_reason"),
            "callback_requested": bool(terminal.get("callback_requested")),
        })
    return state


def mark_sms_consent(call_sid: str, consented: bool):
    """Record the caller's SMS consent decision."""
    if call_sid in _call_contexts:
        _call_contexts[call_sid]["sms_consent"] = consented


def has_sms_consent(call_sid: str) -> bool | None:
    """Check SMS consent status. Returns None if not yet asked."""
    ctx = _call_contexts.get(call_sid, {})
    return ctx.get("sms_consent", None)


_CALLER_NAME_PLACEHOLDERS = {
    "unknown", "n/a", "na", "none", "caller", "customer", "client",
    "anonymous", "no name", "not given", "test",
}


def _normalize_caller_name(name: Any) -> str:
    """Return a usable caller name, or "" when there is not one.

    D12 says no name, no lead — so this is the gate, and every caller of it
    tests plain truthiness. The model will happily pass "unknown" or the
    caller's phone number when it has not actually been told a name, and a lead
    row named "Customer" is worse than no lead at all.
    """
    text = " ".join(str(name or "").split())
    if not text or len(text) > 80:
        return ""
    if text.lower().strip(".") in _CALLER_NAME_PLACEHOLDERS:
        return ""
    if not any(char.isalpha() for char in text):
        return ""
    return text


def mark_caller_name(call_sid: str, name: Any):
    """Record the caller's name for the call log and lead capture."""
    if call_sid in _call_contexts:
        normalized = _normalize_caller_name(name)
        if normalized:
            _call_contexts[call_sid]["caller_name"] = normalized


def get_caller_name(call_sid: str) -> str:
    """Return the caller's name if one was given during this call."""
    ctx = _call_contexts.get(call_sid, {})
    return str(ctx.get("caller_name") or "")


def _normalize_container_size(size: Any) -> str:
    text = str(size or "").lower()
    match = re.search(r"\d+", text)
    return match.group(0) if match else text.strip()


def _normalize_rental_days(days: Any) -> int:
    try:
        normalized = int(days)
    except (TypeError, ValueError):
        return 7
    return normalized if normalized > 0 else 7


def _dashboard_safe_duration_notes(text: Any) -> str:
    """Avoid triggering dashboard's legacy week-based rental duration regex."""
    return re.sub(r"week", "wk", str(text or ""), flags=re.IGNORECASE)


def _availability_key(size: Any, date: Any, days: Any) -> str:
    return "|".join((
        _normalize_container_size(size),
        str(date or "").strip(),
        str(_normalize_rental_days(days)),
    ))


def remember_container_availability(
    call_sid: str,
    size: Any,
    date: Any,
    days: Any,
    data: dict[str, Any],
):
    """Cache date-specific positive container availability for this call."""
    if call_sid not in _call_contexts or not date or not data.get("available"):
        return
    key = _availability_key(size, date, days)
    if not key.startswith("|"):
        availability = _call_contexts[call_sid].setdefault("container_availability", {})
        availability[key] = {
            "size": _normalize_container_size(size),
            "date": str(date).strip(),
            "days": _normalize_rental_days(days),
            "checked_at": time.time(),
        }


def has_container_availability(call_sid: str, size: Any, date: Any, days: Any) -> bool:
    """Return true only when this call already checked matching live inventory."""
    if not call_sid or not size or not date:
        return False
    ctx = _call_contexts.get(call_sid, {})
    availability = ctx.get("container_availability", {})
    return _availability_key(size, date, days) in availability


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


# ── Caller identity ──────────────────────────────────────────────


def caller_id_matches(call_sid: str, phone: str) -> bool:
    """True when the number being asked about is the number actually calling.

    Twilio supplies the calling number on the webhook and it is stored as
    `caller_number`. A number the caller merely *says* is not evidence of
    anything — before this check existed, stating any phone number returned that
    customer's name, service address, appointment date and price, and then
    allowed rescheduling or cancelling their job.
    """
    caller_number = _call_contexts.get(call_sid, {}).get("caller_number", "")
    a = _phone_lookup_key(caller_number)
    b = _phone_lookup_key(phone)
    return bool(a) and bool(b) and a == b


def mark_identity_verified(call_sid: str, phone: str, job_ids: list[str] | None = None):
    """Record that this caller proved the booking is theirs, for this call only.

    `job_ids` scopes what they proved. Passing None means unrestricted, which is
    only correct when the caller ID matches the number on the booking — they
    already hold the phone, so every booking on it is theirs.

    Passing a list restricts them to exactly those jobs. Proving you know the
    address of ONE booking must not hand you the rest of the account.
    """
    if call_sid not in _call_contexts:
        return
    key = _phone_lookup_key(phone)
    if not key:
        return
    _call_contexts[call_sid].setdefault("verified_identities", set()).add(key)
    scopes = _call_contexts[call_sid].setdefault("verified_job_scope", {})
    if job_ids is None:
        scopes.pop(key, None)
    else:
        scopes[key] = {str(j) for j in job_ids if j}


def is_identity_verified(call_sid: str, phone: str) -> bool:
    """True if the caller owns this number or has passed the address challenge."""
    if caller_id_matches(call_sid, phone):
        return True
    key = _phone_lookup_key(phone)
    if not key:
        return False
    return key in _call_contexts.get(call_sid, {}).get("verified_identities", set())


def verified_job_scope(call_sid: str, phone: str) -> set[str] | None:
    """Job ids this caller is limited to, or None for unrestricted access.

    Unrestricted is the caller-ID-matches case. Anyone who got in by answering
    the address challenge is pinned to the job they answered for.
    """
    if caller_id_matches(call_sid, phone):
        return None
    key = _phone_lookup_key(phone)
    if not key:
        return None
    return _call_contexts.get(call_sid, {}).get("verified_job_scope", {}).get(key)


def limit_jobs_to_scope(call_sid: str, phone: str, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop any job this caller has not proved is theirs.

    Applied on every lookup, not just the first. Without it a second
    `lookup_appointment` — which the prompt tells the model to run before any
    reschedule or cancel — refilled the cache with the whole account, and the
    "which booking did you mean?" reply then read out the addresses and dates of
    bookings the caller had proved nothing about.
    """
    scope = verified_job_scope(call_sid, phone)
    if scope is None:
        return jobs
    return [j for j in jobs if str(j.get("jobId") or j.get("id") or "") in scope]


# Words that identify a street type rather than the street itself. Matching on
# these would let "500 Avenue" stand in for "500 Pine Avenue".
_ADDRESS_GENERIC_WORDS = {
    "street", "str", "avenue", "ave", "road", "rd", "drive", "dr", "lane", "ln",
    "boulevard", "blvd", "court", "ct", "terrace", "ter", "place", "pl",
    "circle", "cir", "way", "parkway", "pkwy", "highway", "hwy", "route",
    "trail", "path", "plaza", "square", "sq", "loop", "crossing", "extension",
    "north", "south", "east", "west", "the", "apt", "apartment", "unit",
    "suite", "ste", "floor", "building", "bldg",
}


def _address_tokens(address: str) -> tuple[str, set[str]]:
    """Split an address into its house number and the words that name the street.

    Only the segment before the first comma is considered. The rest is city,
    state and ZIP — constants for a single-metro operator, and public. Including
    them meant "12 Houston" passed the challenge for "12 Oak St, Houston, TX",
    reducing the secret to the house number alone.
    """
    street_line = (address or "").split(",")[0]
    cleaned = re.sub(r"[^a-z0-9\s]", " ", street_line.lower())
    parts = cleaned.split()
    house = next((p for p in parts if p.isdigit()), "")
    words = {
        p for p in parts
        if len(p) >= 3 and not p.isdigit() and p not in _ADDRESS_GENERIC_WORDS
    }
    return house, words


def address_matches(stated: str, on_file: str) -> bool:
    """Forgiving comparison between a spoken address and the stored one.

    A caller says "twelve Oak Street"; the record holds
    "12 Oak St, Houston, TX 77008". Requiring an exact match would reject real
    customers, so this requires the house number to match exactly and at least
    one distinctive street word to appear in both. Deliberately lenient on
    everything else — this checks that the caller knows where the job is, it is
    not an address parser.

    Fails closed: if either side yields no distinctive word, there is nothing to
    prove knowledge of and the answer is no.
    """
    stated_house, stated_words = _address_tokens(stated)
    file_house, file_words = _address_tokens(on_file)
    if not stated_house or not file_house or stated_house != file_house:
        return False
    return bool(stated_words & file_words)


def record_failed_identity_challenge(call_sid: str, phone: str):
    """Lock this number out of the address challenge for the rest of the call."""
    if call_sid not in _call_contexts:
        return
    key = _phone_lookup_key(phone)
    if key:
        _call_contexts[call_sid].setdefault("identity_challenge_failed", set()).add(key)


def identity_challenge_locked(call_sid: str, phone: str) -> bool:
    """True once this number's challenge has been failed on this call."""
    key = _phone_lookup_key(phone)
    if not key:
        return False
    return key in _call_contexts.get(call_sid, {}).get("identity_challenge_failed", set())


# Held server-side between the lookup and the challenge. Deliberately NOT
# returned to the model: it cannot disclose what it was never given.
def stash_pending_verification(call_sid: str, phone: str, payload: dict[str, Any]):
    """Arm one address challenge.

    Refuses to re-arm after a failure. Dropping the stash on a failed guess is
    not enough on its own: `lookup_appointment` can simply be called again, and
    the prompt actively encourages that, so each new lookup handed the caller a
    fresh guess. Unlimited guesses against a house number and street name is not
    a challenge.
    """
    if call_sid not in _call_contexts:
        return
    if identity_challenge_locked(call_sid, phone):
        return
    _call_contexts[call_sid]["pending_verification"] = {"phone": phone, **payload}


def get_pending_verification(call_sid: str) -> dict[str, Any] | None:
    return _call_contexts.get(call_sid, {}).get("pending_verification")


def resolve_job_id_from_lookup(
    call_sid: str,
    phone: str,
    provided_job_id: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Require lookup-backed job selection before mutating existing bookings."""
    cached_jobs = get_cached_lookup_jobs(call_sid, phone)
    # Belt and braces: the cache is already scoped at lookup time, but this is
    # the only door to reschedule and cancel, so re-apply it here too.
    if cached_jobs is not None:
        cached_jobs = limit_jobs_to_scope(call_sid, phone, cached_jobs)

    # Identity first. Without this, a caller who learned someone else's job id
    # could reschedule or cancel it — the dashboard ignores `phone` entirely once
    # a job id is supplied, so this is the only place it can be stopped.
    if not is_identity_verified(call_sid, phone):
        return None, {
            "error": "identity_not_verified",
            "fallback": True,
            "message": (
                "This caller has not confirmed the booking is theirs. Do not "
                "change or cancel it. Transfer them to a human."
            ),
        }

    if provided_job_id:
        # Only accept a job id that came from this call's own lookup. Previously
        # any job id was taken on trust.
        known = {
            str(job.get("jobId") or job.get("id") or "")
            for job in (cached_jobs or [])
        }
        if str(provided_job_id) not in known:
            return None, {
                "error": "job_id_not_from_lookup",
                "message": (
                    "That appointment was not one of the ones found for this "
                    "caller. Look up their appointments again and confirm which "
                    "one they mean."
                ),
            }
        return provided_job_id, None

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
    spoken_address: str,
    service_area_status: str,
    reason: str,
    unit: str = "",
):
    """Cache address verification so create_booking can block clear out-of-area jobs."""
    if call_sid not in _call_contexts:
        return
    verifications = _call_contexts[call_sid].setdefault("address_verifications", {})
    record = {
        "raw_address": raw_address,
        "formatted_address": formatted_address,
        "spoken_address": spoken_address,
        "service_area_status": service_area_status,
        "reason": reason,
        # Apartment/suite, from Google's subpremise or parsed out of what the
        # caller actually said. The geocoder's formatted_address drops it.
        "unit": unit,
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


_UNIT_PREFIXES = (
    "apt", "apartment", "unit", "ste", "suite", "#",
    "fl", "floor", "rm", "room", "bldg", "building", "lot", "trlr",
)

# The prefixes safe to EXTRACT from a full address. Deliberately not the same
# tuple as above: "fl" is the Florida abbreviation, so "Miami, FL 33101" was
# being read as floor 33101 and folded back into the street line. "floor" still
# works. A caller who says "fl" loses the unit, which is the safe direction —
# a missing unit is a phone call, a corrupted street line is a wasted truck.
_UNIT_PREFIX_WORDS = "apt|apartment|unit|ste|suite|floor|rm|room|bldg|building|lot|trlr"

# Matches "apartment 4B", "apt 4b", "unit 12", "suite 200", "#3".
#
# Two subtleties, both of which cost a real unit number when got wrong:
#   - The captured value must contain a digit or be a single letter, so a street
#     named "Building Road" isn't read as unit "Road".
#   - A separator after a word prefix is REQUIRED. Without it "88 Unity Ave Apt 3"
#     matches "unit" + "y" first, and because re.search takes the leftmost match
#     the genuine "Apt 3" is never seen.
_UNIT_RE = re.compile(
    rf"(?:\b(?:{_UNIT_PREFIX_WORDS})[.\s#]+|#\s*)"
    r"((?=[a-z0-9\-]*\d)[a-z0-9\-]{1,8}|[a-z])\b",
    re.IGNORECASE,
)


def extract_address_unit(address: str) -> str:
    """Pull an apartment / suite / unit out of an address as the caller said it.

    Google's geocoder returns a `formatted_address` that almost never carries a
    unit number, and `resolve_booking_address` substitutes it for the caller's
    own words — so before this existed an apartment caller silently lost their
    unit and the crew was dispatched to the building with no way to find them.
    """
    match = _UNIT_RE.search(address or "")
    if not match:
        return ""
    return match.group(0).strip()


def _bare_unit_value(unit: str) -> str:
    """The unit with any prefix word or leading punctuation stripped: "Apt 4" → "4"."""
    stripped = re.sub(rf"^\s*(?:{_UNIT_PREFIX_WORDS})\b\.?\s*", "", unit.strip(), flags=re.IGNORECASE)
    return stripped.lstrip("#").strip()


def _street_already_has_unit(street: str, unit: str) -> bool:
    """Is this unit already in the street line, however it happens to be written?

    A literal substring test is not enough. Google renders a subpremise as "#4",
    so composing unit "4" produced the suffix "Unit 4", failed to find it in
    "123 Main St #4", and appended a second copy — the caller then heard
    "123 Main St #4 Unit 4" read back.
    """
    value = _bare_unit_value(unit)
    if not value:
        return False
    pattern = rf"(?:#|\b(?:{_UNIT_PREFIX_WORDS})\b\.?)\s*{re.escape(value)}\b"
    return re.search(pattern, street, re.IGNORECASE) is not None


def compose_address(address: str, unit: str | None) -> str:
    """Fold a unit back into a street address, before the first comma.

    Mirrors `composeAddress` in website-template/lib/wizardData.ts and
    booking-widget/src/lib/wizardData.ts. Keep the three in step.
    """
    street = (address or "").strip()
    clean = (unit or "").strip()
    if not clean:
        return street

    if _street_already_has_unit(street, clean):
        return street

    has_prefix = any(clean.lower().startswith(p) for p in _UNIT_PREFIXES)
    suffix = clean if has_prefix else f"Unit {clean}"

    comma = street.find(",")
    if comma == -1:
        return f"{street} {suffix}".strip()
    return f"{street[:comma]} {suffix}{street[comma:]}"


def resolve_booking_address(call_sid: str, address: str) -> tuple[str, dict[str, Any] | None]:
    """Return the canonical verified address for booking when available."""
    verification = get_address_verification(call_sid, address) if call_sid else None
    if verification and verification.get("formatted_address"):
        # Re-attach the unit. The geocoder drops it, and the address stored on
        # the job is what the driver app and the confirmation SMS both show.
        unit = str(verification.get("unit") or "") or extract_address_unit(address)
        return compose_address(str(verification["formatted_address"]), unit), verification
    return address, verification


def _component_long_short(result: dict[str, Any], component_type: str) -> tuple[str, str]:
    for component in result.get("address_components", []):
        if component_type in component.get("types", []):
            return component.get("long_name", ""), component.get("short_name", "")
    return "", ""


def _extract_geocode_components(result: dict[str, Any]) -> dict[str, str]:
    street_number = _component_long_short(result, "street_number")[0]
    route = _component_long_short(result, "route")[0]
    city = (
        _component_long_short(result, "locality")[0]
        or _component_long_short(result, "postal_town")[0]
        or _component_long_short(result, "sublocality")[0]
        or _component_long_short(result, "administrative_area_level_3")[0]
    )
    state_long, state_short = _component_long_short(result, "administrative_area_level_1")
    county = _component_long_short(result, "administrative_area_level_2")[0]
    postal_code = _component_long_short(result, "postal_code")[0]
    # Google returns the apartment/suite as `subpremise` when it has one. It
    # usually doesn't, which is why the caller's own words are the fallback.
    subpremise = _component_long_short(result, "subpremise")[0]
    return {
        "street_number": street_number,
        "route": route,
        "city": city,
        "state": state_short or state_long,
        "state_long": state_long,
        "county": county,
        "postal_code": postal_code,
        "subpremise": subpremise,
    }


def _normalize_place(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_HOUSE_NUMBER_RE = re.compile(r"\b\d{1,6}\b")


def _contains_house_number(address: str) -> bool:
    return bool(_HOUSE_NUMBER_RE.search(address or ""))


def _address_has_location_context(address: str, config: dict[str, Any]) -> bool:
    if _ZIP_RE.search(address or ""):
        return True

    normalized = f" {_normalize_place(address)} "
    city = _normalize_place(str(config.get("city") or ""))
    if city and f" {city} " in normalized:
        return True

    state_variants = _state_variants(str(config.get("state") or ""))
    if any(f" {variant} " in normalized for variant in state_variants):
        return True

    for token in _service_area_tokens(config):
        if f" {token} " in normalized:
            return True

    return False


def _primary_address_lookup_place(config: dict[str, Any]) -> str:
    city = str(config.get("city") or "").strip()
    state = str(config.get("state") or "").strip()
    if city and state:
        return f"{city}, {state}"
    if city:
        return city
    if state:
        return state

    for token in _service_area_tokens(config):
        if not any(term in token.split() for term in {"area", "nearby", "surrounding", "greater"}):
            return token.title()
    return ""


def build_geocode_query(address: str, config: dict[str, Any]) -> tuple[str, bool]:
    """Append tenant city/state to partial street addresses before geocoding."""
    cleaned = re.sub(r"\s+", " ", (address or "").strip())
    if not cleaned:
        return "", False

    if _contains_house_number(cleaned) and not _address_has_location_context(cleaned, config):
        place = _primary_address_lookup_place(config)
        if place:
            return f"{cleaned}, {place}", True

    return cleaned, False


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
# Validation errors (a bad date format, a full calendar, etc.) do NOT count —
# those are LLM-correctable. Only network/API/exception failures count.
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


async def handle_record_caller_name(params: FunctionCallParams):
    """Record the caller's name once they give it.

    Pure state, no network. The opening line asks who is calling precisely so a
    caller who never books can still be recorded as a lead, and this is where
    that answer is kept.
    """
    ctx = _get_context()
    call_sid = ctx.get("call_sid", "")
    name = _normalize_caller_name(params.arguments.get("name"))

    if not name:
        await params.result_callback({
            "recorded": False,
            "message": "That was not a usable name. Do not ask again unless it comes up naturally, and carry on helping.",
        })
        return

    if call_sid:
        mark_caller_name(call_sid, name)
        logger.info(f"Caller name recorded for call {call_sid}")

    await params.result_callback({
        "recorded": True,
        "message": f"Name recorded as {name}. Use it naturally, and do not ask for it again.",
    })


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



# The three strings `bot.py` falls back to when the summariser fails or times
# out. They are fine on a call log, but the lead's description renders verbatim
# under "Customer's Message" in the operator's pipeline, so a lead ships with no
# description rather than machine wording plus a raw transcript.
_SUMMARY_FALLBACK_PREFIXES = (
    "Summary unavailable",
    "No caller conversation captured",
    "AI summary unavailable",
)


def usable_inquiry(summary: str) -> str:
    """Return the call summary if it is a real one, else ""."""
    text = str(summary or "").strip()
    if not text or text.startswith(_SUMMARY_FALLBACK_PREFIXES):
        return ""
    return text


async def create_lead_from_call(
    *,
    config: dict[str, Any],
    name: str,
    phone: str,
    description: str,
    sms_consent: bool,
) -> bool:
    """Record a caller who did not book as a lead on the dashboard.

    D12: no name, no lead — the caller has to have told us who they are.

    Deliberately NOT sent: `serviceType` (a value of "dumpster_rental" makes the
    dashboard auto-approve a container reservation, so the agent would silently
    reserve a dumpster), and `notes` (the pipeline regex-splits that field into
    chips, so a sentence arrives as fragments). The inquiry belongs in
    `description`, which renders verbatim.

    Never retried. `/api/ingest/lead` has no dedup on create, so a retry is a
    second lead row, a second operator alert, and possibly a second text.
    """
    if not name or not phone:
        return False

    payload: dict[str, Any] = {
        "name": name,
        "phone": phone,
        "source": "phone_ai",
        "status": "new",
    }
    if description:
        payload["description"] = description
    if sms_consent:
        payload["smsOptIn"] = True
        payload["consentText"] = "Caller agreed on a recorded phone call to receive text messages."

    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{DASHBOARD_URL}/api/ingest/lead",
                json=payload,
                headers=_ingest_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )
            if resp.status in (200, 201):
                logger.info(f"Lead created from phone call for {config.get('companyName', 'unknown')}")
                return True
            if resp.status == 429:
                # Per-IP hourly budget on the dashboard side, and the phone
                # agent shares egress IPs across every tenant. Nothing retries,
                # so this lead is gone — say so loudly enough to be noticed.
                logger.error(
                    "Lead POST rate-limited (429). This lead is LOST — the agent "
                    "does not retry, because the route has no dedup on create."
                )
                return False
            body = await _safe_json(resp, "ingest-lead-error")
            logger.error(f"Lead POST failed: {resp.status} {body}")
            return False
    except Exception as e:
        logger.error(f"Lead POST error: {e}")
        return False


async def _safe_json(resp: aiohttp.ClientResponse, label: str) -> dict:
    """Parse JSON safely, logging raw text on failure."""
    text = await resp.text()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.error(f"[{label}] Non-JSON response (status {resp.status}): {text[:500]}")
        return {"error": f"Dashboard returned non-JSON (status {resp.status})"}


_DAY_WORDS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth", 21: "twenty first",
    22: "twenty second", 23: "twenty third", 24: "twenty fourth",
    25: "twenty fifth", 26: "twenty sixth", 27: "twenty seventh",
    28: "twenty eighth", 29: "twenty ninth", 30: "thirtieth",
    31: "thirty first",
}
_DAY_KEYS = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]


def _parse_hhmm(value: Any) -> int | None:
    try:
        hour_text, minute_text = str(value).strip().split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text[:2])
    except (TypeError, ValueError):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _format_minutes_spoken(minutes: int | None) -> str:
    if minutes is None:
        return "that time"
    hour = (minutes // 60) % 24
    minute = minutes % 60
    suffix = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12 or 12
    if minute == 0:
        return f"{hour_12} {suffix}"
    return f"{hour_12}:{minute:02d} {suffix}"


def _format_spoken_time(value: Any) -> str:
    return _format_minutes_spoken(_parse_hhmm(value))


def _format_spoken_date(value: Any) -> str:
    if not value:
        return "that day"
    text = str(value).strip()
    try:
        if "T" in text:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d")
    except (TypeError, ValueError):
        return "that day"
    day = _DAY_WORDS.get(parsed.day, str(parsed.day))
    return f"{parsed.strftime('%A')}, {parsed.strftime('%B')} {day}"


def _format_spoken_slot(slot: dict[str, Any]) -> str:
    if str(slot.get("period", "")).lower() == "all day":
        return "all day"
    start = _format_spoken_time(slot.get("start"))
    end = _format_spoken_time(slot.get("end"))
    if start == "that time" or end == "that time":
        return str(slot.get("period") or "that time")
    return f"{start} to {end}"


def _slot_window_minutes(slot_or_hour: Any) -> tuple[int | None, int | None]:
    if isinstance(slot_or_hour, dict):
        start = slot_or_hour.get("start_minute")
        end = slot_or_hour.get("end_minute")
        if start is None:
            start = _parse_hhmm(slot_or_hour.get("start"))
        if end is None:
            end = _parse_hhmm(slot_or_hour.get("end"))
        return start, end
    try:
        hour = int(slot_or_hour)
    except (TypeError, ValueError):
        return None, None
    return hour * 60, None


def _coerce_business_window(day_value: Any) -> tuple[bool, tuple[int, int] | None]:
    if day_value is None or day_value is False:
        return True, None

    if isinstance(day_value, str):
        value = day_value.strip().lower()
        if value in {"closed", "off", "false", "none"}:
            return True, None
        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start = _parse_hhmm(start_text)
            end = _parse_hhmm(end_text)
            if start is not None and end is not None:
                return True, (start, end)
        return False, None

    if isinstance(day_value, list):
        if not day_value:
            return True, None
        day_value = day_value[0]

    if isinstance(day_value, dict):
        closed_value = day_value.get("closed", day_value.get("isClosed"))
        if closed_value is True or str(closed_value).strip().lower() in {"1", "true", "yes", "closed"}:
            return True, None
        enabled_value = day_value.get("enabled", day_value.get("isOpen"))
        if enabled_value is False or str(enabled_value).strip().lower() in {"0", "false", "no"}:
            return True, None
        start_text = (
            day_value.get("open")
            or day_value.get("start")
            or day_value.get("from")
            or day_value.get("opens")
        )
        end_text = (
            day_value.get("close")
            or day_value.get("end")
            or day_value.get("to")
            or day_value.get("closes")
        )
        start = _parse_hhmm(start_text)
        end = _parse_hhmm(end_text)
        if start is not None and end is not None:
            return True, (start, end)
        return False, None

    return False, None


def _get_exact_business_window(config: dict[str, Any], js_day: int) -> tuple[bool, tuple[int, int] | None]:
    hours = config.get("businessHours")
    if not isinstance(hours, dict):
        return False, None

    candidate_keys = (_DAY_KEYS[js_day], str(js_day), js_day)
    for key in candidate_keys:
        if key in hours:
            return _coerce_business_window(hours.get(key))
    return False, None


def _is_open_day(date: datetime, config: dict[str, Any]) -> bool:
    js_day = (date.weekday() + 1) % 7
    exact_found, exact_window = _get_exact_business_window(config, js_day)
    if exact_found:
        return exact_window is not None
    business_days = config.get("businessDays", [0, 1, 2, 3, 4, 5])
    return js_day in business_days


def _format_business_hours_for_day(config: dict[str, Any], date: datetime) -> str:
    js_day = (date.weekday() + 1) % 7
    exact_found, exact_window = _get_exact_business_window(config, js_day)
    if exact_found and exact_window:
        return (
            f"That's outside our hours. We're available that day from "
            f"{_format_minutes_spoken(exact_window[0])} to {_format_minutes_spoken(exact_window[1])}."
        )
    if exact_found and exact_window is None:
        return "We're closed that day. Ask which other day works best."
    business_days = config.get("businessDays", [0, 1, 2, 3, 4, 5])
    if js_day not in business_days:
        return "We're closed that day. Ask which other day works best."
    business_start = int(config.get("businessStart", 7))
    business_end = int(config.get("businessEnd", 19))
    return (
        f"That's outside our hours. We're available from "
        f"{_format_minutes_spoken(business_start * 60)} to {_format_minutes_spoken(business_end * 60)}."
    )


def _normalize_e164_phone(value: Any) -> str | None:
    text = str(value or "").strip()
    digits = re.sub(r"\D+", "", text)
    if not digits:
        return None
    if text.startswith("+") and 10 <= len(digits) <= 15:
        return f"+{digits}"
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return None


# ── Time Slots (must match website wizardData.ts) ───────

TIME_SLOTS = {
    "morning":   {"label": "Morning",   "start": "08:00", "end": "11:00", "start_hour": 8,  "start_minute": 480, "end_minute": 660, "period": "Morning"},
    "midday":    {"label": "Midday",    "start": "11:00", "end": "13:00", "start_hour": 11, "start_minute": 660, "end_minute": 780, "period": "Midday"},
    "afternoon": {"label": "Afternoon", "start": "13:00", "end": "16:00", "start_hour": 13, "start_minute": 780, "end_minute": 960, "period": "Afternoon"},
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
            start_minute = _parse_hhmm(parts[0])
            end_minute = _parse_hhmm(parts[1])
            if start_minute is not None and end_minute is not None:
                return {
                    "label": f"{parts[0]} - {parts[1]}",
                    "start": parts[0],
                    "end": parts[1],
                    "start_hour": start_minute // 60,
                    "start_minute": start_minute,
                    "end_minute": end_minute,
                    "period": stripped,  # Use the range as the period
                }
    return None


def _slot_wire_value(slot: dict[str, Any]) -> str:
    """The value to put in `timeSlot` when talking to the dashboard.

    Always an explicit "HH:MM-HH:MM" range, never a label.

    The agent's vocabulary and the dashboard's do not match: the dashboard has
    MORNING / MIDDAY / EVENING and no "afternoon" at all, so it mapped our
    "Afternoon" onto MIDDAY. A caller told "1 PM to 4 PM" ended up with a job
    recorded as 11:00 AM – 2:00 PM — which is what the customer portal, the
    driver app, the reminder email and this agent's own later lookup all read.
    Sending the range removes the translation step entirely.

    "All Day" is the dumpster-delivery sentinel, not a time window. It must stay
    a label: expanded to 08:00-17:00 it would be read back as a MIDDAY slot.
    """
    period = str(slot.get("period") or "")
    if period.lower() == "all day":
        return "All Day"
    start = str(slot.get("start") or "")
    end = str(slot.get("end") or "")
    if start and end:
        return f"{start}-{end}"
    return period


def _is_business_hours(date: datetime, slot_or_hour: Any, config: dict) -> bool:
    """Check if date/time falls within this client's business hours."""
    # Convert Python weekday (0=Mon) to JS convention (0=Sun) used by dashboard
    js_day = (date.weekday() + 1) % 7
    start_minute, end_minute = _slot_window_minutes(slot_or_hour)
    if start_minute is None:
        return False

    exact_found, exact_window = _get_exact_business_window(config, js_day)
    if exact_found:
        if exact_window is None:
            return False
        open_minute, close_minute = exact_window
        if start_minute < open_minute:
            return False
        if end_minute is not None and end_minute > close_minute:
            return False
        return start_minute < close_minute

    business_days = config.get("businessDays", [0, 1, 2, 3, 4, 5])
    business_start = int(config.get("businessStart", 7)) * 60
    business_end = int(config.get("businessEnd", 19)) * 60

    if js_day not in business_days:
        return False
    if start_minute < business_start or start_minute >= business_end:
        return False
    if end_minute is not None and end_minute > business_end:
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
    rental_days = _normalize_rental_days(params.arguments.get("days"))
    call_sid = _current_call_sid.get()
    started_at = time.perf_counter()

    if not client_supports_dumpsters(config):
        await params.result_callback({
            "available": False,
            "error": "unsupported_service",
            "message": (
                "This client does not offer dumpster rentals. Do not quote dumpster "
                "prices or discuss container sizes. Redirect to junk removal pickup "
                "or offer a human transfer."
            ),
        })
        return

    try:
        async with aiohttp.ClientSession() as http:
            query: dict[str, str] = {"size": size}
            if date:
                query["date"] = date
            query["days"] = str(rental_days)
            logger.info(
                f"container_availability start call={call_sid} "
                f"size={size} date={date or ''} days={rental_days}"
            )
            resp = await http.get(
                f"{DASHBOARD_URL}/api/booking/container-availability",
                params=query,
                headers=_ingest_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                f"container_availability response call={call_sid} "
                f"status={resp.status} elapsed_ms={elapsed_ms}"
            )

            if resp.status == 200:
                data = await _safe_json(resp, "container-availability")
                if data.get("available"):
                    base_rate = data.get("baseRate")
                    included_days = data.get("includedDays", 7)
                    extended_rate = data.get("extendedDailyRate")
                    remember_container_availability(
                        call_sid,
                        size,
                        date,
                        rental_days,
                        data,
                    )

                    # Speak the configured price. Rounding it to the nearest $5
                    # read cleanly but quoted a number the customer is not
                    # billed — the dashboard charges the configured amount.
                    price_msg = ""
                    if base_rate:
                        price_msg = f"${base_rate:.0f} for {included_days} days"
                        if extended_rate:
                            price_msg += f", then ${extended_rate:.0f}/day after"

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
                        next_date_str = _format_spoken_date(next_date)
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
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.error(f"Container availability check failed after {elapsed_ms}ms: {e}")
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
            "error": "invalid_date",
            "message": "Ask the caller which day they mean in natural language, then resolve the tool date internally before trying again. Do not ask the caller for a date format.",
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

                if available:
                    time_strs = [f"{_format_spoken_time(s['start'])} to {_format_spoken_time(s['end'])}" for s in available]
                    msg = f"Available times on {_format_spoken_date(date)}: {', '.join(time_strs)}."
                    if full:
                        full_strs = [f"{_format_spoken_time(s['start'])} to {_format_spoken_time(s['end'])}" for s in full]
                        msg += f" Not available: {', '.join(full_strs)}."
                else:
                    msg = f"No booking windows are available on {_format_spoken_date(date)}. Suggest a different date."

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


# Reason codes no amount of date-shopping can fix. `no_trucks` means the
# operator has no vehicles set up at all, and `material_not_accepted` means the
# tip will not take it — both come back with an empty `alternatives` list, so
# offering another day would be a lie dressed up as helpfulness.
_UNFIXABLE_BY_DATE = {"no_trucks", "material_not_accepted"}

# Two capacity refusals per call, then stop. Every POST to /api/agent/book
# creates another Lead row dashboard-side, so an unbounded date-shopping loop
# quietly fills the operator's pipeline with duplicates of one caller.
_CAPACITY_REFUSAL_LIMIT = 2


def _build_capacity_refusal(call_sid: str, data: dict[str, Any], spoken_date: str) -> dict[str, Any]:
    """Turn the dashboard's deliberate "not that day" into something sayable.

    Deliberately does NOT share the `create_booking` failure counter. That one
    escalates with "I'm having repeated trouble with this", which is false here
    — nothing is broken, the day is just full — and it would let one real
    outage plus one full day trip a handoff.
    """
    feasibility = data.get("feasibility") or {}
    reason = str(feasibility.get("reasonCode") or "")
    alternatives = [a for a in (data.get("alternatives") or []) if a.get("date")]
    attempts = increment_tool_failure(call_sid, "create_booking_capacity") if call_sid else 1

    if reason == "no_trucks":
        return {
            "error": "no_crew_configured",
            "message": "This operator has no crew set up for bookings, so no date will work. Do not offer another day. Tell the caller you cannot get it booked from here and that someone from the team will call them straight back, then follow HUMAN HANDOFF.",
        }

    if reason == "material_not_accepted":
        return {
            "error": "material_not_accepted",
            "message": "The tip will not take what this caller described, so changing the date fixes nothing. Do not offer another day and do not guess at what is acceptable. Say you do not want to promise a day you cannot keep, and that someone from the team will sort out where it can go. Then follow HUMAN HANDOFF.",
        }

    if attempts >= _CAPACITY_REFUSAL_LIMIT or not alternatives:
        return {
            "error": "no_dates_available",
            "message": f"Nothing is open near {spoken_date}. Do not offer any more dates and do not ask the caller for another one — you have already tried. Tell them you cannot find anything in the next week or so and that someone from the team will call to sort it out, then follow HUMAN HANDOFF.",
        }

    spoken = [_format_spoken_date(a["date"]) for a in alternatives[:2]]
    # These are other DAYS, never other times on the same day — the dashboard
    # probes forward from tomorrow and keeps the caller's original time window.
    offer = spoken[0] if len(spoken) == 1 else f"{spoken[0]}, or {spoken[1]}"
    return {
        "error": "date_unavailable",
        "alternatives": [a["date"] for a in alternatives[:2]],
        "message": f"{spoken_date} is full. Offer these instead, as days rather than times: {offer}. Say it briefly — the caller already told you what day they wanted, so do not ask 'what day works for you' again. If they pick one, call create_booking with it.",
    }


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
    rental_duration_days = _normalize_rental_days(params.arguments.get("rental_duration_days", 7))
    call_sid = ctx.get("call_sid", "")

    is_dumpster = booking_type in ("dumpster_rental", "dumpster_swap")
    is_swap = booking_type == "dumpster_swap"
    service_type = booking_type if is_dumpster else "junk_removal"

    if is_dumpster and not client_supports_dumpsters(config):
        await params.result_callback({
            "fallback": True,
            "error": "unsupported_service",
            "message": (
                "This client does not offer dumpster rentals. Do not create a "
                "dumpster booking. Redirect to junk removal pickup or offer a "
                "human transfer."
            ),
        })
        return

    # ── Validate date ──
    parsed_date = _validate_date(date)
    if not parsed_date:
        await params.result_callback({
            "error": "invalid_date",
            "message": "Ask the caller which day they mean in natural language, then resolve the tool date internally before trying again. Do not ask the caller for a date format.",
        })
        return

    if is_dumpster and not is_swap:
        if not container_size:
            await params.result_callback({
                "error": "container_size_required",
                "message": "Ask what dumpster size they want, or recommend a size based on their project, before creating the rental.",
            })
            return
        if not has_container_availability(call_sid, container_size, date, rental_duration_days):
            await params.result_callback({
                "error": "availability_check_required",
                "message": (
                    "Check live container availability for that size, delivery day, "
                    "and rental duration before creating this dumpster rental. Do not "
                    "create the booking yet."
                ),
            })
            return

    # ── Validate time slot ──
    if not time_slot_id and is_dumpster and not is_swap:
        # Dumpster rentals are all-day deliveries — no time slot needed
        slot = {
            "label": "All Day",
            "start": "08:00",
            "end": "17:00",
            "start_hour": 8,
            "start_minute": 480,
            "end_minute": 1020,
            "period": "All Day",
        }
    else:
        slot = _validate_time_slot(time_slot_id)
        if not slot:
            await params.result_callback({
                "error": "I need a valid time as a 24-hour range like 08:00-11:00. Call check_available_slots first and pass back one of the ranges it returns."
            })
            return

    # ── Business hours check ──
    if str(slot.get("period", "")).lower() == "all day":
        is_valid_window = _shared_is_open_day(parsed_date, config)
    else:
        is_valid_window = _shared_is_business_hours(parsed_date, slot, config)
    if not is_valid_window:
        await params.result_callback({
            "error": _shared_format_business_hours_for_day(config, parsed_date)
        })
        return

    address, address_verification = resolve_booking_address(call_sid, address)
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
        # A final pickup is booked as a swap because the dashboard has no
        # pickup-only type, and the prompt says so in the description. The note
        # the crew reads must not then promise a replacement container the
        # caller explicitly turned down.
        lowered = str(description or "").lower()
        pickup_only = any(
            phrase in lowered
            for phrase in ("pickup-only", "pickup only", "pick up only", "no replacement", "final pickup")
        )
        if pickup_only:
            notes = f"FINAL PICKUP — collect the {size_str} container, do NOT drop off a replacement. {description}"
        else:
            notes = f"Dumpster Swap: pick up full {size_str} container, drop off empty. {description}"
    elif is_dumpster:
        size_str = f"{container_size}-yard" if container_size else "TBD size"
        safe_description = _dashboard_safe_duration_notes(description)
        notes = (
            f"Dumpster Rental Delivery: {size_str} container. "
            f"rental_duration: {rental_duration_days}. "
            f"Rental length: {rental_duration_days} days. Project: {safe_description}"
        )
    else:
        notes = f"Junk Removal Pickup: {description}"

    # ── Create booking via /api/agent/book ──
    local_datetime = f"{date}T{slot['start']}:00"

    payload = {
        "customerName": name,
        "customerPhone": phone,
        "address": address,
        "date": local_datetime,
        "timeSlot": _slot_wire_value(slot),  # "13:00-16:00", or "All Day" for dumpsters
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
    spoken_date = _format_spoken_date(date)
    spoken_slot = _format_spoken_slot(slot)

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
                        reset_tool_failure(call_sid, "create_booking_capacity")

                    data = await _safe_json(resp, "create-booking")
                    job_id = data.get("jobId", "confirmed")
                    label = "Dumpster swap" if is_swap else "Dumpster rental" if is_dumpster else "Booking"
                    logger.info(f"{label} created: jobId={job_id} for {name} on {date} ({slot['period']} window)")

                    auto_booked = data.get("autoBooked", False)
                    # The dashboard branches its own SMS copy on `scheduled`, so
                    # branching on the same field keeps the voice and the text
                    # saying the same thing by construction. They move together
                    # for swaps; `autoBooked` is the fallback for older shapes.
                    scheduled = data.get("scheduled", auto_booked)

                    if is_swap and scheduled:
                        # No text is sent for a scheduled swap — by design, the
                        # voice is the only confirmation the customer gets.
                        result = {
                            "success": True,
                            "scheduled": True,
                            "booking_id": str(job_id),
                            "message": f"Swap confirmed for {spoken_date}, {spoken_slot}. Tell the caller we'll be out then. No confirmation text goes out for a booked swap, so what you say now is the only confirmation they get.",
                        }
                    elif is_swap:
                        # A container has not been matched yet. The dashboard
                        # texts "your swap request is in", so the voice must not
                        # claim it is scheduled.
                        result = {
                            "success": True,
                            "scheduled": False,
                            "booking_id": str(job_id),
                            "message": f"Swap request logged for {spoken_date}, but no container has been assigned yet. Do NOT tell the caller it is scheduled or give them a time. Say the request is in and the team will confirm the exact time with them. They will also get a text confirming we have it.",
                        }
                    elif is_dumpster and auto_booked:
                        size_label = f"{container_size}-yard" if container_size else ""
                        result = {
                            "success": True,
                            "autoBooked": True,
                            "booking_id": str(job_id),
                            "message": f"Dumpster rental CONFIRMED for {spoken_date}. {size_label} container, {rental_duration_days} day rental. Delivery is scheduled. If SMS consent was recorded earlier in the call, you may tell the caller they'll receive a confirmation text shortly. Do not promise an email unless the dashboard explicitly confirms one.",
                        }
                    elif is_dumpster:
                        size_label = f"{container_size}-yard" if container_size else ""
                        result = {
                            "success": True,
                            "autoBooked": False,
                            "booking_id": str(job_id),
                            "message": f"Dumpster rental request submitted for {spoken_date}. {size_label} container, {rental_duration_days} day rental. Our team will follow up to confirm availability and pricing.",
                        }
                    else:
                        result = {
                            "success": True,
                            "booking_id": str(job_id),
                            "message": f"Booking confirmed for {spoken_date}, {spoken_slot}.",
                        }
                elif resp.status == 200:
                    # 201 books; a 200 is the dashboard declining the date on
                    # purpose. It is NOT an error — the old code sent it down the
                    # generic path, so the caller heard "trouble with our
                    # scheduling system" and the alternative dates were binned.
                    #
                    # Gate on the `success` field, never the status: a proxy or
                    # auth wall also answers 200, and `_safe_json` turns that
                    # body into {"error": ...}, so `success` is None and it
                    # correctly falls through to the error path below.
                    data = await _safe_json(resp, "create-booking-refused")
                    if data.get("success") is False:
                        result = _build_capacity_refusal(call_sid, data, spoken_date)
                    else:
                        logger.error(f"Booking API returned 200 without success:false: {data}")
                        result = _build_error_with_tracking(
                            call_sid,
                            "create_booking",
                            "I'm having trouble with our scheduling system. Let me take your info and have someone call you back.",
                        )
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
    # A spoken number is fine as a *search key* — it is not evidence of identity.
    # Whether the details may be disclosed is decided after the lookup, below.
    if re.search(r"\d{7,}", re.sub(r"[\s\-\(\)\+]", "", raw_phone)):
        phone = raw_phone
    else:
        phone = ctx.get("caller_number", raw_phone)
    call_sid = ctx.get("call_sid", "")

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
                    customer = data.get("customer", {})
                    # A caller who passed the address challenge earlier in this
                    # call is pinned to the job they proved. Re-running the
                    # lookup must not hand back the rest of the account.
                    jobs = limit_jobs_to_scope(call_sid, phone, jobs)
                    remember_lookup_results(call_sid, phone, jobs)

                    if is_identity_verified(call_sid, phone):
                        if caller_id_matches(call_sid, phone):
                            # Calling from the number on the booking — nothing
                            # to prove, and every job on it is theirs.
                            mark_identity_verified(call_sid, phone)
                        # Otherwise they already passed the challenge earlier in
                        # this call. Do not re-challenge, and do not re-mark:
                        # marking again here would widen their scope back to the
                        # whole account. `jobs` is already limited to it.
                        await params.result_callback({
                            "found": True,
                            "customer": customer,
                            "jobs": jobs,
                        })
                        return

                    # Already failed the challenge on this call. Do not offer
                    # another one, and still disclose nothing.
                    if identity_challenge_locked(call_sid, phone):
                        await params.result_callback({
                            "found": True,
                            "verification_required": True,
                            "fallback": True,
                            "error": "identity_verification_failed",
                            "message": (
                                "This caller already failed to confirm this booking on "
                                "this call. Do not ask again and do not reveal any detail "
                                "of it — not the name, address, date, price, or whether "
                                "it exists. Say you cannot pull it up from this number "
                                "and that you will get someone onto it, then call "
                                "transfer_to_human now. If it records a callback instead "
                                "of connecting, that is fine: say someone will ring them "
                                "back on this number, and still reveal nothing."
                            ),
                        })
                        return

                    # Different number. Hold everything server-side and make the
                    # caller produce the service address first. The details are
                    # deliberately NOT returned here — the model cannot read out
                    # what it was never given.
                    stash_pending_verification(call_sid, phone, {"customer": customer, "jobs": jobs})
                    await params.result_callback({
                        "found": True,
                        "verification_required": True,
                        "message": (
                            "A booking exists, but this caller is not ringing from the "
                            "number on it, so I cannot show you any of its details yet. "
                            "Do not guess or imply anything about it. Ask the caller for "
                            "the service address for the job, then call "
                            "verify_caller_identity with what they say. If they already "
                            "told you the address, confirm the phone number on the "
                            "account with them and still pass the address through."
                        ),
                    })
                else:
                    remember_lookup_results(call_sid, phone, [])
                    await params.result_callback({
                        "found": False,
                        "message": "No appointments found for that number."
                    })
            else:
                await params.result_callback(_build_error_with_tracking(
                    ctx.get("call_sid", ""),
                    "lookup_appointment",
                    "Lookup failed. Ask for a name only if the caller has not already given one this call, then try again.",
                ))
    except Exception as e:
        logger.error(f"Appointment lookup failed: {e}")
        await params.result_callback(_build_error_with_tracking(
            ctx.get("call_sid", ""),
            "lookup_appointment",
            "I'm having trouble looking that up right now.",
        ))


async def handle_verify_caller_identity(params: FunctionCallParams):
    """Release a booking's details once the caller proves the job is theirs.

    Only reached when someone rings about a booking from a number other than the
    one on it. The caller must state the service address; it is compared against
    the record held server-side by `handle_lookup_appointment`.

    On failure the caller is handed to a human. After hours that handoff records
    a callback instead (D26), which is safe here: the callback number is the
    Twilio `From` for the call (`server.py` reads it from the stream's
    customParameters), never a number the caller supplied — so an unverified
    caller cannot steer where the business rings back to. What must not leak is
    any detail of the booking, and the returned instruction says so either way.
    """
    ctx = _get_context()
    call_sid = ctx.get("call_sid", "")
    stated_address = str(params.arguments.get("address") or "").strip()

    pending = get_pending_verification(call_sid)
    if not pending:
        await params.result_callback({
            "verified": False,
            "error": "no_pending_verification",
            "message": "Look up the caller's appointment first.",
        })
        return

    if not stated_address:
        await params.result_callback({
            "verified": False,
            "message": "Ask the caller for the service address for the job, then try again.",
        })
        return

    jobs = pending.get("jobs", [])
    matched = [job for job in jobs if address_matches(stated_address, str(job.get("address") or ""))]

    if not matched:
        # Drop the stashed payload AND lock the number out, so a fresh
        # lookup_appointment cannot re-arm the challenge for another guess.
        # The model is not a security control; the prompt asking it not to retry
        # was the only thing standing between a caller and unlimited guesses.
        _call_contexts.get(call_sid, {}).pop("pending_verification", None)
        record_failed_identity_challenge(call_sid, str(pending.get("phone") or ""))
        # Do not say what was expected, do not offer another guess, do not hint
        # at how close they were. Hand off.
        await params.result_callback({
            "verified": False,
            "fallback": True,
            "error": "identity_verification_failed",
            "message": (
                "That does not match the address on the booking. Do not reveal any "
                "detail of it and do not offer another attempt. Say you cannot pull it "
                "up from this number and that you will get someone onto it, then call "
                "transfer_to_human now. If it records a callback instead of connecting, "
                "that is fine: say someone will ring them back on this number, and "
                "still reveal nothing."
            ),
        })
        return

    phone = str(pending.get("phone") or "")
    # Pin them to the job they proved. The scope is what makes the narrowing
    # below durable — a second lookup re-applies it instead of undoing it.
    mark_identity_verified(
        call_sid,
        phone,
        job_ids=[str(j.get("jobId") or j.get("id") or "") for j in matched],
    )
    remember_lookup_results(call_sid, phone, matched)
    _call_contexts.get(call_sid, {}).pop("pending_verification", None)

    await params.result_callback({
        "verified": True,
        "customer": pending.get("customer", {}),
        "jobs": matched,
        "message": "Address confirmed. You can now discuss and change this booking.",
    })


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
        await params.result_callback({
            "error": "invalid_date",
            "message": "Ask the caller which day they mean in natural language, then resolve the tool date internally before trying again. Do not ask the caller for a date format.",
        })
        return

    slot = _validate_time_slot(new_time_slot_id)
    if not slot:
        await params.result_callback({
            "error": "Please use a 24-hour range like 08:00-11:00. Call check_available_slots first and pass back one of the ranges it returns."
        })
        return

    if not _shared_is_business_hours(parsed_date, slot, config):
        await params.result_callback({
            "error": _shared_format_business_hours_for_day(config, parsed_date)
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
                    "timeSlot": _slot_wire_value(slot),
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
                    "message": f"Appointment rescheduled to {_format_spoken_date(new_date)}, {_format_spoken_slot(slot)}.",
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

    if not _contains_house_number(address):
        await params.result_callback({
            "verified": False,
            "error": "street_number_required",
            "message": "Ask the caller for the street number, then verify the full street address again."
        })
        return

    if not GOOGLE_MAPS_API_KEY:
        await params.result_callback({
            "verified": False,
            "message": "Address verification unavailable. Read back the address as you heard it and ask the caller to confirm."
        })
        return

    try:
        lookup_query, added_location_context = build_geocode_query(address, config)
        geocode_params = {
            "address": lookup_query,
            "key": GOOGLE_MAPS_API_KEY,
            "region": "us",
        }
        state = str(config.get("state") or "").strip()
        if state:
            geocode_params["components"] = f"country:US|administrative_area:{state}"

        async with aiohttp.ClientSession() as http:
            resp = await http.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params=geocode_params,
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
                if not components.get("street_number") or not components.get("route"):
                    await params.result_callback({
                        "verified": False,
                        "error": "street_number_required",
                        "message": "I found the street, but not a specific service address. Ask the caller for the full address including the street number."
                    })
                    return
                if not components.get("postal_code"):
                    await params.result_callback({
                        "verified": False,
                        "error": "postal_code_missing",
                        "message": "The address resolved but with no ZIP. Read back what you have and ask only for the ZIP — do not ask for the whole address again."
                    })
                    return

                # Prefer Google's subpremise; fall back to whatever unit the
                # caller actually said, since the geocoder rarely returns one.
                unit = components.get("subpremise") or extract_address_unit(address)
                # Read the unit back too — otherwise the agent confirms an
                # address the caller will not recognise as their own.
                formatted = compose_address(formatted, unit)

                spoken_address = format_address_for_speech(formatted)
                area_status, area_reason = _assess_service_area(components, config)
                if call_sid:
                    remember_address_verification(
                        call_sid,
                        address,
                        formatted,
                        spoken_address,
                        area_status,
                        area_reason,
                        unit=unit,
                    )

                if area_status == "out_of_area":
                    message = (
                        f"Verified address: {formatted}. Spoken readback: {spoken_address}. "
                        "This appears to be outside the configured service area. "
                        "Do not book it automatically; offer to transfer the caller to the team to confirm coverage."
                    )
                elif area_status == "uncertain":
                    message = (
                        f"Verified address: {formatted}. Spoken readback: {spoken_address}. "
                        "Read the spoken readback to the caller and ask if it's correct. "
                        "Service-area coverage is not fully proven from the configured area text, so transfer if the caller asks whether that location is covered."
                    )
                else:
                    message = (
                        f"Verified address: {formatted}. Spoken readback: {spoken_address}. "
                        "Read the spoken readback to the caller and ask if it's correct."
                    )

                await params.result_callback({
                    "verified": True,
                    "formatted_address": formatted,
                    "spoken_address": spoken_address,
                    "lookup_query": lookup_query,
                    "added_location_context": added_location_context,
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
    # No production caller. Kept deliberately: test_dashboard_transfer patches
    # this to assert the AI never updates Twilio directly, which is the
    # regression guard for the subaccount bug that made dashboard-owned
    # transfers necessary in the first place. Deleting it would delete the guard.
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
    # Same guard create_booking uses: the model can pass a phrase like "the
    # number you're calling from" instead of digits. The dashboard rejects that
    # with a 400 that used to be reported to the caller as a bad TIME.
    if not re.search(r"\d{7,}", re.sub(r"[\s\-\(\)\+]", "", caller_phone)):
        fallback_phone = ctx.get("caller_number", "")
        if fallback_phone:
            logger.info(
                f"Replaced callback phone '{caller_phone}' with caller_number "
                f"'{fallback_phone}'"
            )
            caller_phone = fallback_phone
    caller_name = (
        params.arguments.get("caller_name")
        or params.arguments.get("callerName")
        or ""
    )
    caller_name = str(caller_name).strip()
    # This is the one non-booking path that is already handed the caller's name,
    # and it used to keep it only for the outgoing payload. Recording it here
    # means a caller who asks for a callback and nothing else is still named on
    # the call log.
    if call_sid and caller_name:
        mark_caller_name(call_sid, caller_name)

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
            "message": "Ask the caller for the exact callback day and time in natural language, then resolve the tool timestamp internally before trying again. Do not ask the caller for a technical timestamp.",
        })
        return

    if not caller_phone:
        await _deliver_result({
            "error": "caller_phone_required",
            "message": "No usable callback number. If the caller already confirmed a number earlier in this call, use that one — only ask again if none was confirmed.",
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
                        reset_tool_failure(call_sid, "callback_time")

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
                error_text = str(data.get("error") or "")

                # 400 is about the PHONE, 422 is about the TIME. Collapsing them
                # told a caller with an unusable number that their time was
                # unavailable, and they would offer time after time.
                if resp.status == 400:
                    await _deliver_result({
                        "error": "invalid_callback_phone",
                        "message": "That phone number was not usable. Confirm the best number to reach them on, read it back, and try again. Do not ask for a different time.",
                    })
                    return

                if resp.status == 403:
                    # Tenant mismatch. Not retryable, and it must not return a
                    # fallback: after hours the mandated transfer is refused.
                    logger.error(f"Schedule callback rejected as cross-tenant: {data}")
                    await _deliver_result({
                        "error": "callback_not_available",
                        "message": "A callback cannot be booked on this call. Apologise briefly, do not try again, and carry on helping with anything else.",
                    })
                    return

                if resp.status == 422:
                    # Four shapes, and one of them cannot be recovered by
                    # offering a different time: hours the operator has
                    # misconfigured. "Outside business hours" USUALLY just means
                    # the caller picked a bad slot — but it is also what an
                    # all-closed schedule returns for every candidate date, and
                    # the two are indistinguishable from the response. So the
                    # first are treated as retryable and a repeat is not, which
                    # stops the caller being asked for time after time.
                    unusable_hours = "not configured correctly" in error_text
                    attempts = increment_tool_failure(call_sid, "callback_time") if call_sid else 1
                    if unusable_hours or attempts >= 2:
                        logger.warning(
                            f"Callback hours unusable for this client after "
                            f"{attempts} attempt(s): {error_text}"
                        )
                        await _deliver_result({
                            "error": "callback_hours_unavailable",
                            "message": "A callback cannot be scheduled for this business right now. Tell the caller you have noted their request and someone will get back to them, then move on. Do not ask for another time.",
                        })
                        return
                    await _deliver_result({
                        "error": "invalid_callback_time",
                        "message": "That callback time is not available. Ask the caller for another time during business hours.",
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
    """Transfer the live call to the dashboard human handoff route.

    Asks the dashboard to update the active call because launched clients may
    use Twilio subaccounts that the phone-agent parent credentials do not own.
    The dashboard rings registered softphone users first and falls back to the
    handoff phone if the dashboard does not answer.
    """
    config = _get_config()
    ctx = _get_context()
    reason = params.arguments.get("reason", "Caller requested human agent")
    call_sid = ctx.get("call_sid")

    # D21: outside business hours there is nobody to transfer to, so no handoff
    # is attempted at all. The call is noted in the dashboard's Callback Queue
    # and continues normally — no dashboard redirect, no pipeline cancel.
    #
    # This sits ABOVE the "requested" write below. If the caller hangs up
    # between the two, that write would stand on its own and the call would be
    # logged as a transfer that never happened.
    #
    # The result deliberately carries no "fallback": true. Several prompt sites
    # mandate a transfer on any fallback, so returning one here would send the
    # model straight back into the tool it was just refused by.
    if not evaluate_current_business_hours(config):
        if call_sid:
            mark_transfer_state(
                call_sid, "suppressed_after_hours", reason, callback_requested=True
            )
        logger.info(
            f"Transfer suppressed outside business hours for call {call_sid} "
            f"(reason: {reason}); caller added to the callback queue"
        )
        await params.result_callback({
            "transferred": False,
            "after_hours": True,
            "say": "The office is closed right now, but I've left a note for the team and someone will give you a call back on this number.",
            "note": (
                "AFTER HOURS. No transfer was attempted and none is possible; the "
                "caller is already on the callback queue. Say the line above in your "
                "own words. It is normal, not a fault, so do not apologise for a "
                "problem and do not try to transfer again on this call. Then carry on "
                "— you can still book, look up, reschedule and cancel."
            ),
        })
        return

    if call_sid:
        mark_transfer_state(call_sid, "requested", reason)

    if not call_sid:
        logger.error("No call_sid in context — cannot transfer")
        await params.result_callback({
            "transferred": False,
            "message": "I'm having trouble with the transfer. Let me take your info and have someone call you back.",
        })
        return

    company_name = config.get("companyName", "our team")
    client_id = str(config.get("_clientId") or "unknown")

    try:
        redirect_result = await redirect_live_call_via_dashboard(
            client_config=config,
            client_id=client_id,
            call_sid=call_sid,
            reason=str(reason),
            origin=DASHBOARD_ORIGIN_AI_TRANSFER,
            dashboard_url=DASHBOARD_URL,
            platform_api_key=PLATFORM_API_KEY,
        )
        if not redirect_result.ok:
            raise RuntimeError(
                "dashboard live-call redirect failed "
                f"status={redirect_result.status} error={redirect_result.error}"
            )

        logger.info(
            f"Call {call_sid} redirected to dashboard human handoff via dashboard "
            f"endpoint for {company_name} (reason: {reason})"
        )

        # Dashboard Twilio routes own the final softphone/handoff result from
        # this point forward. Do not write legacy phone-agent transfer updates
        # that can downgrade dashboard_answered/handoff_answered later.
        if ctx.get("call_sid"):
            mark_dashboard_transfer_ownership(
                ctx["call_sid"],
                DASHBOARD_ORIGIN_AI_TRANSFER,
                reason=str(reason),
            )

        await params.result_callback({
            "transferred": True,
            "transferStatus": "dashboard_redirected",
            "note": "The transfer is going through. Say nothing further — the line hands over immediately.",
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
        await params.result_callback({
            "transferred": False,
            "say": "I couldn't get anyone on the line just now, but I've put a note in and someone from the team will get back to you.",
            "note": "The transfer did not connect. The caller is on the callback queue. Do not promise a time and do not try again.",
        })


# ── SMS Messaging ──────────────────────────────────────

# handle_send_sms is intentionally not registered as a live LLM tool in bot.py.
# Current in-call behavior is consent capture; actual booking and follow-up SMS
# sends are handled by dashboard/post-call paths.

# Fixed SMS templates — LLM picks a template name and supplies variables.


# ── Automated Follow-Up SMS (called from bot.py after call ends) ──

# 24-hour cooldown — keyed by normalized phone number


# ── Dashboard Logging Helpers ──────────────────────────────────


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
