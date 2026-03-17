"""Tool handler functions for the multi-tenant phone agent.

Each handler receives a FunctionCallParams object and uses the client_config
(set per-call via set_call_context) to route API calls to the correct
dashboard tenant.

ENDPOINT MAPPING (dashboard ↔ phone agent):
  - check_availability    → GET  /api/agent/capacity-check  (X-AGENT-SECRET auth)
  - create_booking        → POST /api/agent/book             (X-AGENT-SECRET auth)
  - lookup_appointment    → GET  /api/agent/lookup            (X-AGENT-SECRET auth)
  - reschedule            → POST /api/agent/reschedule        (X-AGENT-SECRET auth)
  - cancel                → POST /api/agent/cancel            (X-AGENT-SECRET auth)
  - transfer_to_human     → Twilio REST API (call redirect)
  - validate_promo_code   → GET  /api/promo/validate          (x-api-key + x-site-token)
"""

import asyncio
import contextvars
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


def _get_context() -> dict[str, Any]:
    """Get the call context for the current pipeline.

    Uses contextvars to identify the correct call_sid for this asyncio task,
    ensuring concurrent calls always use their own config.
    """
    call_sid = _current_call_sid.get()
    if call_sid and call_sid in _call_contexts:
        return _call_contexts[call_sid]
    # Fallback for single-call scenarios
    if not _call_contexts:
        return {}
    return list(_call_contexts.values())[-1]


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


# ── Time Slots (must match website wizardData.ts) ───────

TIME_SLOTS = {
    "morning":   {"label": "Morning",        "start": "08:00", "end": "10:00", "start_hour": 8,  "period": "Morning"},
    "midday":    {"label": "Midday",          "start": "10:00", "end": "12:00", "start_hour": 10, "period": "Midday"},
    "afternoon": {"label": "Afternoon",       "start": "12:00", "end": "14:00", "start_hour": 12, "period": "Afternoon"},
    "late":      {"label": "Late Afternoon",  "start": "14:00", "end": "16:00", "start_hour": 14, "period": "Late Afternoon"},
}


# ── Date / Time Validation ──────────────────────────────

def _validate_date(date_str: str) -> datetime | None:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def _validate_time_slot(slot_id: str) -> dict | None:
    """Validate a time slot ID and return the slot definition, or None."""
    return TIME_SLOTS.get(slot_id.lower().strip()) if slot_id else None


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
                data = await resp.json()
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
                # API error — fall back to static pricing from config
                logger.warning(f"Container availability check returned {resp.status}")
                await params.result_callback({
                    "available": True,
                    "message": f"I wasn't able to check live inventory, but let's go ahead and get you booked for a {size}-yard. Our team will confirm availability.",
                    "fallback": True,
                })
    except Exception as e:
        logger.error(f"Container availability check failed: {e}")
        await params.result_callback({
            "available": True,
            "message": f"I'm having trouble checking availability right now, but let's go ahead and get you scheduled. Our team will follow up to confirm.",
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
            data = await resp.json()

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


async def handle_check_availability(params: FunctionCallParams):
    """Check day-level capacity via dashboard's /api/agent/capacity-check.

    Dashboard returns: { hasCapacity: bool, date: str, availableTrucks?: int, reason?: str }
    """
    config = _get_config()
    date = params.arguments["date"]
    time_preference = params.arguments.get("time_preference", "any")

    parsed_date = _validate_date(date)
    if not parsed_date:
        await params.result_callback({
            "error": "I need the date in YYYY-MM-DD format. Please resolve relative dates first."
        })
        return

    business_days = config.get("businessDays", [0, 1, 2, 3, 4, 5])
    js_day = (parsed_date.weekday() + 1) % 7
    if js_day not in business_days:
        await params.result_callback({
            "available": False,
            "message": "That day we're closed. Let me find a day that works.",
            "suggestion": "Ask the caller which weekday works best."
        })
        return

    # Call dashboard's capacity-check endpoint (uses X-AGENT-SECRET auth)
    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.get(
                f"{DASHBOARD_URL}/api/agent/capacity-check",
                params={"date": date, "volumeEstimate": "4"},
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )
            if resp.status == 200:
                data = await resp.json()
                if data.get("hasCapacity"):
                    await params.result_callback({
                        "available": True,
                        "message": f"We have capacity on {date}. {data.get('availableTrucks', 'Multiple')} trucks available."
                    })
                else:
                    await params.result_callback({
                        "available": False,
                        "message": f"No capacity on {date}. {data.get('reason', 'All trucks are full.')}",
                    })
            else:
                # API error — default to "available" so we don't block bookings
                await params.result_callback({
                    "available": True,
                    "message": "Schedule looks open, go ahead and book."
                })
    except Exception as e:
        logger.error(f"Capacity check failed: {e}")
        await params.result_callback({
            "available": True,
            "message": "I can't check the schedule right now, but let's go ahead and get you booked."
        })


async def handle_create_booking(params: FunctionCallParams):
    """Create a booking via dashboard's /api/agent/book.

    Dashboard expects: { customerName, customerPhone, address, volume, date, timeSlot, notes }
    Dashboard returns: { success, jobId, customerId, message } (status 201)
    Auth: X-AGENT-SECRET header (siteToken)
    """
    config = _get_config()
    name = params.arguments["name"]
    phone = params.arguments["phone"]
    address = params.arguments["address"]
    date = params.arguments["date"]
    time_slot_id = params.arguments["time"]
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
    slot = _validate_time_slot(time_slot_id)
    if not slot:
        await params.result_callback({
            "error": "Please use a valid time window: morning, midday, afternoon, or late."
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
        "timeSlot": slot["period"],
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
                data = await resp.json()
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
    phone = params.arguments["phone"]

    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.get(
                f"{DASHBOARD_URL}/api/agent/lookup",
                params={"phone": phone},
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )

            if resp.status == 200:
                data = await resp.json()
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
    phone = params.arguments["phone"]
    new_date = params.arguments["new_date"]
    new_time_slot_id = params.arguments["new_time"]

    parsed_date = _validate_date(new_date)
    if not parsed_date:
        await params.result_callback({"error": "Date must be YYYY-MM-DD format."})
        return

    slot = _validate_time_slot(new_time_slot_id)
    if not slot:
        await params.result_callback({
            "error": "Please use a valid time window: morning, midday, afternoon, or late."
        })
        return

    if not _is_business_hours(parsed_date, slot["start_hour"], config):
        await params.result_callback({
            "error": "That's outside our business hours."
        })
        return

    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/reschedule",
                json={
                    "phone": phone,
                    "newDate": new_date,
                    "newTime": slot["start"],
                },
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
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/cancel",
                json={
                    "phone": phone,
                    "reason": reason,
                },
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
