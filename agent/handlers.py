"""Tool handler functions for the multi-tenant phone agent.

Each handler receives a FunctionCallParams object and uses the client_config
(set per-call via set_call_context) to route API calls to the correct
dashboard tenant.

ENDPOINT MAPPING (dashboard ↔ phone agent):
  - create_booking        → POST /api/agent/book             (X-AGENT-SECRET auth)
  - lookup_appointment    → GET  /api/agent/lookup            (X-AGENT-SECRET auth)
  - reschedule            → POST /api/agent/reschedule        (X-AGENT-SECRET auth)
  - cancel                → POST /api/agent/cancel            (X-AGENT-SECRET auth)
  - transfer_to_human     → Twilio REST API (call redirect)
  - validate_promo_code   → GET  /api/promo/validate          (x-api-key + x-site-token)
"""

import asyncio
import contextvars
import json
import aiohttp
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from config import DASHBOARD_URL, INGEST_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN

# ── Per-call context — keyed by call_sid for concurrency safety ──
_call_contexts: dict[str, dict[str, Any]] = {}
_current_call_sid: contextvars.ContextVar[str] = contextvars.ContextVar("current_call_sid", default="")


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


def mark_booking_complete(call_sid: str):
    """Flag that a booking was successfully created for this call."""
    if call_sid in _call_contexts:
        _call_contexts[call_sid]["booking_complete"] = True


def set_pipeline_task(call_sid: str, task):
    """Store the PipelineTask so handlers can cancel it (e.g. after transfer)."""
    if call_sid in _call_contexts:
        _call_contexts[call_sid]["pipeline_task"] = task


def is_booking_complete(call_sid: str) -> bool:
    """Check if a booking has been completed during this call."""
    ctx = _call_contexts.get(call_sid, {})
    return ctx.get("booking_complete", False)


def mark_sms_sent(call_sid: str):
    """Flag that an SMS was sent during this call (to avoid duplicate follow-ups)."""
    if call_sid in _call_contexts:
        _call_contexts[call_sid]["sms_sent"] = True


def was_sms_sent(call_sid: str) -> bool:
    """Check if an SMS was already sent during this call."""
    ctx = _call_contexts.get(call_sid, {})
    return ctx.get("sms_sent", False)


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
    """Build auth headers using X-AGENT-SECRET (siteToken) — matches dashboard's agent-auth.ts."""
    return {
        "X-AGENT-SECRET": config.get("siteToken", ""),
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
    business_start = config.get("businessStart", 7)
    business_end = config.get("businessEnd", 19)

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
                        msg += " Offer to submit a request."
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
    Returns human-readable list of available/full time windows.
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
                        msg += f" Fully booked: {', '.join(full_strs)}."
                else:
                    msg = f"All time slots are fully booked on {date}. Suggest a different date."

                await params.result_callback({
                    "available": len(available) > 0,
                    "slots": [{"start": s["start"], "end": s["end"], "label": s.get("label", ""), "available": s["available"]} for s in slots],
                    "message": msg,
                })
            else:
                # API error — fall back to offering any time
                await params.result_callback({
                    "available": True,
                    "slots": [],
                    "message": "I couldn't check slot availability. Go ahead and offer their preferred time."
                })
    except Exception as e:
        logger.error(f"Available slots check failed: {e}")
        await params.result_callback({
            "available": True,
            "slots": [],
            "message": "I couldn't check the schedule right now. Go ahead and offer their preferred time."
        })


async def handle_create_booking(params: FunctionCallParams):
    """Create a booking via dashboard's /api/agent/book.

    Dashboard expects: { customerName, customerPhone, address, volume, date, timeSlot, notes }
    Dashboard returns: { success, jobId, customerId, message } (status 201)
    Auth: X-AGENT-SECRET header (siteToken)
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

    is_dumpster = booking_type in ("dumpster_rental", "dumpster_swap")
    is_swap = booking_type == "dumpster_swap"

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
        business_start = config.get("businessStart", 7)
        business_end = config.get("businessEnd", 19)
        await params.result_callback({
            "error": f"That's outside our hours. We're available from {_format_hour(business_start)} to {_format_hour(business_end)}."
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
    company_name = config.get("companyName", "the company")
    tz = config.get("timezone", "America/Chicago")
    tz_offset = datetime.now(ZoneInfo(tz)).strftime("%z")
    # Format offset as -06:00 (insert colon)
    tz_offset_formatted = f"{tz_offset[:3]}:{tz_offset[3:]}" if len(tz_offset) == 5 else tz_offset

    payload = {
        "customerName": name,
        "customerPhone": phone,
        "address": address,
        "date": f"{date}T{slot['start']}:00{tz_offset_formatted}",
        "timeSlot": slot["period"],  # "08:00-10:00" or "Morning"
        "notes": notes,
        "type": booking_type,
        "serviceType": "dumpster_rental" if is_dumpster else "junk_removal",
    }
    if is_dumpster:
        payload["containerSize"] = container_size
        payload["rentalDays"] = rental_duration_days

    promo_code = params.arguments.get("promo_code")
    if promo_code:
        payload["promoCode"] = promo_code

    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/book",
                json=payload,
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=15),
            )

            if resp.status == 201:
                data = await _safe_json(resp, "create-booking")
                job_id = data.get("jobId", "confirmed")
                label = "Dumpster rental" if is_dumpster else "Booking"
                logger.info(f"{label} created: jobId={job_id} for {name} on {date} ({slot['period']} window)")

                # Signal bot.py to start post-booking silence timer
                ctx = _get_context()
                if ctx.get("call_sid"):
                    mark_booking_complete(ctx["call_sid"])

                auto_booked = data.get("autoBooked", False)

                if is_swap:
                    await params.result_callback({
                        "success": True,
                        "booking_id": str(job_id),
                        "message": f"Dumpster swap scheduled for {date}, {slot['period']} window. We'll pick up the full container and drop off an empty one.",
                    })
                elif is_dumpster and auto_booked:
                    size_label = f"{container_size}-yard" if container_size else ""
                    await params.result_callback({
                        "success": True,
                        "autoBooked": True,
                        "booking_id": str(job_id),
                        "message": f"Dumpster rental CONFIRMED for {date}. {size_label} container, {rental_duration_days} day rental. Delivery is scheduled. Customer will receive a confirmation text and email with a link to their customer portal to add a card on file before delivery.",
                    })
                elif is_dumpster:
                    size_label = f"{container_size}-yard" if container_size else ""
                    await params.result_callback({
                        "success": True,
                        "autoBooked": False,
                        "booking_id": str(job_id),
                        "message": f"Dumpster rental request submitted for {date}. {size_label} container, {rental_duration_days} day rental. Our team will follow up to confirm availability and pricing.",
                    })
                else:
                    await params.result_callback({
                        "success": True,
                        "booking_id": str(job_id),
                        "message": f"Booking confirmed for {date}, {slot['period']} window ({slot['start']} - {slot['end']}).",
                    })
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
                        await params.result_callback({
                            "error": "slot_full",
                            "message": f"That time slot just filled up. Available alternatives: {', '.join(alt_strs)}. Ask the caller which works.",
                            "availableSlots": avail,
                        })
                    else:
                        await params.result_callback({
                            "error": "slot_full",
                            "message": f"That time slot and all others on {date} are now full. Suggest a different date.",
                            "availableSlots": [],
                        })
                else:
                    await params.result_callback({
                        "error": "slot_full",
                        "message": "That time slot just filled up. Ask the caller for a different time.",
                    })
            else:
                text = await resp.text()
                logger.error(f"Booking API error: {resp.status} {text}")
                await params.result_callback({
                    "error": "I'm having trouble with our scheduling system. Let me take your info and have someone call you back."
                })

    except Exception as e:
        logger.error(f"Booking creation error: {e}")
        await params.result_callback({
            "error": "I'm having trouble creating the booking right now. Let me take your info and have someone call you back."
        })


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
                data = await _safe_json(resp, "lookup-appointment")
                if data.get("found"):
                    await params.result_callback({
                        "found": True,
                        "customer": data.get("customer", {}),
                        "jobs": data.get("jobs", []),
                    })
                else:
                    await params.result_callback({
                        "found": False,
                        "message": "No appointments found for that number."
                    })
            else:
                await params.result_callback({
                    "error": "I'm having trouble looking that up. Can you give me your name instead?"
                })
    except Exception as e:
        logger.error(f"Appointment lookup failed: {e}")
        await params.result_callback({
            "error": "I'm having trouble looking that up right now."
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

    try:
        async with aiohttp.ClientSession() as http:
            job_id = params.arguments.get("job_id")
            payload = {
                    "phone": phone,
                    "newDate": new_date,
                    "newTime": slot["start"],
                    "timeSlot": slot["period"],
                }
            if job_id:
                payload["jobId"] = job_id
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/reschedule",
                json=payload,
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )

            if resp.status == 200:
                await params.result_callback({
                    "success": True,
                    "message": f"Appointment rescheduled to {new_date}, {slot['period']} window ({slot['start']} - {slot['end']}).",
                })
                logger.info(f"Rescheduled for {phone} to {new_date} ({slot['period']} window)")
            else:
                text = await resp.text()
                logger.error(f"Reschedule API error: {resp.status} {text}")
                await params.result_callback({"error": "Couldn't reschedule that appointment."})
    except Exception as e:
        logger.error(f"Reschedule error: {e}")
        await params.result_callback({"error": "I'm having trouble rescheduling right now."})


async def handle_cancel_appointment(params: FunctionCallParams):
    """Cancel an appointment.

    Dashboard: POST /api/agent/cancel { phone, reason } with X-AGENT-SECRET auth
    """
    config = _get_config()
    ctx = _get_context()
    phone = params.arguments.get("phone", ctx.get("caller_number", ""))
    reason = params.arguments.get("reason", "Customer requested cancellation")

    try:
        async with aiohttp.ClientSession() as http:
            job_id = params.arguments.get("job_id")
            payload = {
                    "phone": phone,
                    "reason": reason,
                }
            if job_id:
                payload["jobId"] = job_id
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/cancel",
                json=payload,
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )

            if resp.status == 200:
                await params.result_callback({
                    "success": True,
                    "message": "Appointment cancelled.",
                })
                logger.info(f"Cancelled appointment for {phone}, reason: {reason}")
            else:
                await params.result_callback({"error": "Couldn't cancel that appointment."})
    except Exception as e:
        logger.error(f"Cancel error: {e}")
        await params.result_callback({"error": "I'm having trouble processing the cancellation."})


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

    # No forwarding number configured — take a message instead
    if not forwarding_phone:
        logger.warning(f"No forwardingPhone configured — cannot transfer call {call_sid}")
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
    transfer_twiml = (
        f'<Response>'
        f'<Say voice="Polly.Joanna">Please hold while we connect you with {company_name}.</Say>'
        f'<Dial callerId="{config.get("twilioNumber", "")}">{forwarding_phone}</Dial>'
        f'<Say voice="Polly.Joanna">We were unable to reach anyone at this time. Please try again later. Goodbye.</Say>'
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

        await params.result_callback({
            "transferred": True,
            "message": "Transferring the call now.",
        })

        # Cancel the AI pipeline — the call audio stream will end
        pipeline_task = ctx.get("pipeline_task")
        if pipeline_task:
            await asyncio.sleep(2)  # Brief delay so the TTS can finish speaking
            await pipeline_task.cancel()

    except Exception as e:
        logger.error(f"Transfer failed for call {call_sid}: {e}")
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
    business_start = config.get("businessStart", 7)
    business_end = config.get("businessEnd", 19)
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

    # ── Send via Twilio ──
    try:
        client = _get_twilio_client()
        result = await asyncio.to_thread(
            client.messages.create,
            to=to_number if to_number.startswith("+") else f"+1{to_number.replace('-', '').replace(' ', '')}",
            from_=twilio_number,
            body=body,
        )

        logger.info(f"SMS sent (SID: {result.sid}) template={template_name} to={to_number} from={twilio_number}")

        # Track that SMS was sent during this call
        ctx = _get_context()
        if ctx.get("call_sid"):
            mark_sms_sent(ctx["call_sid"])

        await params.result_callback({
            "sent": True,
            "message": "Text message sent successfully.",
        })

    except Exception as e:
        logger.error(f"SMS send failed: {e}")
        await params.result_callback({
            "sent": False,
            "message": "I had trouble sending that text, but no worries — let me give you the details.",
        })


# ── Automated Follow-Up SMS (called from bot.py after call ends) ──

async def send_automated_followup(caller_number: str, config: dict[str, Any]) -> bool:
    """Send an automated follow-up SMS after a no-booking call.

    Called from bot.py on disconnect, NOT from an LLM tool handler.
    Returns True if sent, False otherwise.
    """
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

    body = _build_sms_body("follow_up", config)
    if not body:
        return False

    try:
        client = _get_twilio_client()
        to = caller_number if caller_number.startswith("+") else f"+1{caller_number.replace('-', '').replace(' ', '')}"
        result = await asyncio.to_thread(
            client.messages.create,
            to=to,
            from_=twilio_number,
            body=body,
        )
        logger.info(f"Automated follow-up SMS sent (SID: {result.sid}) to={caller_number} from={twilio_number}")
        return True
    except Exception as e:
        logger.error(f"Automated follow-up SMS failed: {e}")
        return False
