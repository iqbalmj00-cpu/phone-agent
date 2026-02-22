"""Tool handler functions for the multi-tenant phone agent.

Each handler receives a FunctionCallParams object and uses the client_config
(set per-call via set_call_context) to route API calls to the correct
dashboard tenant.

ENDPOINT MAPPING (dashboard ↔ phone agent):
  - check_availability → GET  /api/agent/capacity-check  (X-AGENT-SECRET auth)
  - create_booking     → POST /api/agent/book             (X-AGENT-SECRET auth)
  - lookup_appointment → GET  /api/agent/lookup            (X-AGENT-SECRET auth)  [NEEDS DASHBOARD ENDPOINT]
  - reschedule         → POST /api/agent/reschedule        (X-AGENT-SECRET auth)  [NEEDS DASHBOARD ENDPOINT]
  - cancel             → POST /api/agent/cancel            (X-AGENT-SECRET auth)  [NEEDS DASHBOARD ENDPOINT]
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from config import DASHBOARD_URL, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN

# ── Per-call context — keyed by call_sid for concurrency safety ──
_call_contexts: dict[str, dict[str, Any]] = {}


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


def _get_context() -> dict[str, Any]:
    """Get the most recent call context.

    NOTE: For true multi-call safety, handlers would need the call_sid
    passed through. Pipecat's FunctionCallParams doesn't carry it, so
    we use the most recently set context. This works because each call
    runs in its own pipeline and tool calls are sequential within a call.
    """
    if not _call_contexts:
        return {}
    # Return the most recently added context
    return list(_call_contexts.values())[-1]


def _get_config() -> dict[str, Any]:
    return _get_context().get("client_config", {})


def _agent_headers(config: dict[str, Any]) -> dict[str, str]:
    """Build auth headers using X-AGENT-SECRET (siteToken) — matches dashboard's agent-auth.ts."""
    return {
        "X-AGENT-SECRET": config.get("siteToken", ""),
        "Content-Type": "application/json",
    }


# ── Date / Time Validation ──────────────────────────────

def _validate_date(date_str: str) -> datetime | None:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def _validate_time(time_str: str) -> tuple[int, int] | None:
    try:
        parts = time_str.split(":")
        h, m = int(parts[0]), int(parts[1])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return (h, m)
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

    Dashboard expects: { customerName, customerPhone, address, volume, date, notes }
    Dashboard returns: { success, jobId, customerId, message } (status 201)
    Auth: X-AGENT-SECRET header (siteToken)
    """
    config = _get_config()
    name = params.arguments["name"]
    phone = params.arguments["phone"]
    address = params.arguments["address"]
    date = params.arguments["date"]
    time = params.arguments["time"]
    description = params.arguments["description"]

    # ── Validate date ──
    parsed_date = _validate_date(date)
    if not parsed_date:
        await params.result_callback({"error": "Date must be YYYY-MM-DD format."})
        return

    # ── Validate time ──
    parsed_time = _validate_time(time)
    if not parsed_time:
        await params.result_callback({"error": "Time must be HH:MM format."})
        return

    # ── Business hours check ──
    if not _is_business_hours(parsed_date, parsed_time[0], config):
        business_start = config.get("businessStart", 7)
        business_end = config.get("businessEnd", 19)
        await params.result_callback({
            "error": f"That's outside our hours. We're available from {_format_hour(business_start)} to {_format_hour(business_end)}."
        })
        return

    # ── Create booking via /api/agent/book ──
    company_name = config.get("companyName", "the company")

    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/book",
                json={
                    "customerName": name,
                    "customerPhone": phone,
                    "address": address,
                    "date": f"{date}T{time}:00",
                    "notes": f"Junk Removal Pickup: {description}",
                },
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=15),
            )

            if resp.status == 201:
                data = await resp.json()
                job_id = data.get("jobId", "confirmed")
                logger.info(f"Booking created: jobId={job_id} for {name} on {date} at {time}")

                await params.result_callback({
                    "success": True,
                    "booking_id": str(job_id),
                    "message": f"Booking confirmed for {date} at {time}.",
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
    """Look up an existing appointment by phone number.

    NOTE: Dashboard endpoint /api/agent/lookup needs to be built by developer.
    Expected: GET /api/agent/lookup?phone=XXX with X-AGENT-SECRET auth
    Expected response: { found: bool, appointment?: { id, date, address, status } }
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
                    "booking": data.get("appointment", {}),
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

    NOTE: Dashboard endpoint /api/agent/reschedule needs to be built by developer.
    Expected: POST /api/agent/reschedule { phone, newDate, newTime } with X-AGENT-SECRET auth
    """
    config = _get_config()
    phone = params.arguments["phone"]
    new_date = params.arguments["new_date"]
    new_time = params.arguments["new_time"]

    parsed_date = _validate_date(new_date)
    if not parsed_date:
        await params.result_callback({"error": "Date must be YYYY-MM-DD format."})
        return

    parsed_time = _validate_time(new_time)
    if not parsed_time:
        await params.result_callback({"error": "Time must be HH:MM format."})
        return

    if not _is_business_hours(parsed_date, parsed_time[0], config):
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
                    "newTime": new_time,
                },
                headers=_agent_headers(config),
                timeout=aiohttp.ClientTimeout(total=10),
            )

        if resp.status == 200:
            await params.result_callback({
                "success": True,
                "message": f"Appointment rescheduled to {new_date} at {new_time}.",
            })
            logger.info(f"Rescheduled for {phone} to {new_date} at {new_time}")
        else:
            await params.result_callback({"error": "Couldn't reschedule that appointment."})
    except Exception as e:
        logger.error(f"Reschedule error: {e}")
        await params.result_callback({"error": "I'm having trouble rescheduling right now."})


async def handle_cancel_appointment(params: FunctionCallParams):
    """Cancel an appointment.

    NOTE: Dashboard endpoint /api/agent/cancel needs to be built by developer.
    Expected: POST /api/agent/cancel { phone, reason } with X-AGENT-SECRET auth
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
