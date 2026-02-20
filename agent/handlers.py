"""Tool handler functions for the multi-tenant phone agent.

Each handler receives a FunctionCallParams object and uses the client_config
(set per-call via set_call_context) to route API calls to the correct
dashboard tenant.

All booking operations go through the ScaleYourJunk dashboard API
instead of Google Calendar/Sheets.
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from config import DASHBOARD_URL, PLATFORM_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN

# ── Per-call context (set in bot.py before pipeline starts) ──
_call_context: dict[str, Any] = {}


def set_call_context(call_sid: str, caller_number: str, client_config: dict[str, Any]):
    """Set context for the current call (called from bot.py on connect)."""
    _call_context["call_sid"] = call_sid
    _call_context["caller_number"] = caller_number
    _call_context["client_config"] = client_config


def _get_config() -> dict[str, Any]:
    return _call_context.get("client_config", {})


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

    if date.weekday() not in business_days:
        return False
    if hour < business_start or hour >= business_end:
        return False
    return True


# ── Tool Handlers ───────────────────────────────────────

async def handle_check_availability(params: FunctionCallParams):
    """Check available appointment slots via dashboard API."""
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
    if parsed_date.weekday() not in business_days:
        await params.result_callback({
            "available": False,
            "message": "That day we're closed. Let me find a day that works.",
            "suggestion": "Ask the caller which weekday works best."
        })
        return

    # Check availability via dashboard API
    try:
        client_id = config.get("clientId", "")
        async with aiohttp.ClientSession() as http:
            resp = await http.get(
                f"{DASHBOARD_URL}/api/agent/availability",
                params={"date": date, "preference": time_preference, "clientId": client_id},
                headers={"x-api-key": PLATFORM_API_KEY},
                timeout=aiohttp.ClientTimeout(total=10),
            )
            if resp.status == 200:
                data = await resp.json()
                if data.get("slots"):
                    await params.result_callback({
                        "available": True,
                        "open_slots": data["slots"],
                        "message": f"Found {len(data['slots'])} available slots on {date}."
                    })
                else:
                    await params.result_callback({
                        "available": False,
                        "message": f"No openings on {date}.",
                        "next_available": data.get("nextAvailable"),
                    })
            else:
                await params.result_callback({
                    "available": True,
                    "message": "Schedule looks open, go ahead and book."
                })
    except Exception as e:
        logger.error(f"Availability check failed: {e}")
        await params.result_callback({
            "available": True,
            "message": "I can't check the schedule right now, but let's go ahead and get you booked."
        })


async def handle_create_booking(params: FunctionCallParams):
    """Create a booking via the dashboard API."""
    config = _get_config()
    name = params.arguments["name"]
    phone = params.arguments["phone"]
    address = params.arguments["address"]
    date = params.arguments["date"]
    time = params.arguments["time"]
    description = params.arguments["description"]
    appt_type = params.arguments.get("type", "pickup")

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

    # ── Create booking via dashboard ──
    company_name = config.get("companyName", "the company")
    site_token = config.get("siteToken", "")

    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{DASHBOARD_URL}/api/ingest/website",
                json={
                    "name": name,
                    "phone": phone,
                    "address": address,
                    "requestedDate": f"{date}T{time}:00",
                    "description": description,
                    "source": "PHONE_AGENT",
                    "metadata": {
                        "type": appt_type,
                        "bookedBy": config.get("agentName", "AI Agent"),
                        "callSid": _call_context.get("call_sid", ""),
                    },
                },
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": PLATFORM_API_KEY,
                    "x-site-token": site_token,
                },
                timeout=aiohttp.ClientTimeout(total=15),
            )

            if resp.status in (200, 201):
                data = await resp.json()
                booking_id = data.get("leadId", data.get("jobId", "confirmed"))
                logger.info(f"Booking created: {booking_id} for {name} on {date} at {time}")

                # Send SMS confirmation (non-blocking)
                if config.get("smsEnabled") and phone:
                    asyncio.create_task(_send_sms(
                        phone, company_name, date, time, address, config
                    ))

                await params.result_callback({
                    "success": True,
                    "booking_id": str(booking_id),
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
    """Look up an existing appointment by phone number via dashboard."""
    config = _get_config()
    phone = params.arguments["phone"]
    client_id = config.get("clientId", "")

    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.get(
                f"{DASHBOARD_URL}/api/agent/lookup",
                params={"phone": phone, "clientId": client_id},
                headers={"x-api-key": PLATFORM_API_KEY},
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
    """Reschedule an appointment via dashboard API."""
    config = _get_config()
    phone = params.arguments["phone"]
    new_date = params.arguments["new_date"]
    new_time = params.arguments["new_time"]
    client_id = config.get("clientId", "")

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
                    "clientId": client_id,
                    "newDate": new_date,
                    "newTime": new_time,
                },
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": PLATFORM_API_KEY,
                },
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
    """Cancel an appointment via dashboard API."""
    config = _get_config()
    phone = params.arguments.get("phone", _call_context.get("caller_number", ""))
    reason = params.arguments.get("reason", "Customer requested cancellation")
    client_id = config.get("clientId", "")

    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{DASHBOARD_URL}/api/agent/cancel",
                json={
                    "phone": phone,
                    "clientId": client_id,
                    "reason": reason,
                },
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": PLATFORM_API_KEY,
                },
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


async def _send_sms(phone: str, company: str, date: str, time: str, address: str, config: dict):
    """Send SMS confirmation via Twilio (non-blocking fire-and-forget)."""
    from_number = config.get("twilioNumber", "")
    if not from_number or not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.warning("SMS skipped — missing Twilio credentials or number")
        return

    body = (
        f"Thanks for booking with {company}! 🚛\n\n"
        f"Your junk removal pickup is confirmed for {date} at {time} at {address}.\n"
        f"We'll reach out before arrival. Questions? Just reply to this text."
    )

    try:
        async with aiohttp.ClientSession() as http:
            await http.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json",
                data={"To": phone, "From": from_number, "Body": body},
                auth=aiohttp.BasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            )
        logger.info(f"SMS confirmation sent to {phone}")
    except Exception as e:
        logger.error(f"SMS send error: {e}")
