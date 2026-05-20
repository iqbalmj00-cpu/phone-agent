"""AI phone coverage routing helpers.

This module is intentionally separate from business-hours booking logic. The
dashboard resolves ``business_hours`` as after-hours AI coverage: AI answers
outside configured business hours and hands off during business hours.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import quote, urlencode
from xml.sax.saxutils import escape, quoteattr
from zoneinfo import ZoneInfo

from agent.business_hours import is_currently_within_business_hours, parse_hhmm

SHORT_DAY_KEYS = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
LONG_DAY_KEYS = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
VALID_COVERAGE_MODES = {"business_hours", "custom_hours", "always_on", "always_handoff"}
DASHBOARD_ORIGIN_AI_TRANSFER = "phone_agent_ai_transfer"
DASHBOARD_ORIGIN_CAPACITY = "phone_agent_capacity"
DASHBOARD_ORIGIN_COVERAGE_OFF = "phone_agent_coverage_off"


@dataclass(frozen=True)
class CoverageDecision:
    mode: str
    should_answer_ai: bool
    should_handoff: bool
    reason: str
    error: str | None = None


@dataclass(frozen=True)
class WeeklyHoursEvaluation:
    is_within: bool
    is_valid: bool
    error: str | None = None


def normalize_e164_phone(value: Any) -> str | None:
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


def _handoff(mode: str, reason: str, error: str | None = None) -> CoverageDecision:
    return CoverageDecision(
        mode=mode,
        should_answer_ai=False,
        should_handoff=True,
        reason=reason,
        error=error,
    )


def _answer_ai(mode: str, reason: str) -> CoverageDecision:
    return CoverageDecision(
        mode=mode,
        should_answer_ai=True,
        should_handoff=False,
        reason=reason,
    )


def _localized_now(timezone: Any, now: datetime | None = None) -> tuple[datetime | None, str | None]:
    tz_name = str(timezone or "America/Chicago").strip() or "America/Chicago"
    try:
        tz = ZoneInfo(tz_name)
    except Exception as exc:
        return None, f"Invalid timezone {tz_name!r}: {exc}"

    if now is None:
        return datetime.now(tz), None
    if now.tzinfo is None:
        return now.replace(tzinfo=tz), None
    return now.astimezone(tz), None


def _day_value(hours: dict[Any, Any], js_day: int) -> Any:
    for key in (SHORT_DAY_KEYS[js_day], LONG_DAY_KEYS[js_day], str(js_day), js_day):
        if key in hours:
            return hours[key]
    return None


def _is_closed_value(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "closed", "off"}
    return False


def _coverage_window(day_value: Any, day_key: str) -> tuple[bool, tuple[int, int] | None, str | None]:
    if not isinstance(day_value, dict) or isinstance(day_value, list):
        return False, None, f"{day_key} coverage hours must be an object."

    if _is_closed_value(day_value.get("closed")):
        return True, None, None

    open_text = day_value.get("open", day_value.get("start"))
    close_text = day_value.get("close", day_value.get("end"))
    open_minute = parse_hhmm(open_text)
    close_minute = parse_hhmm(close_text)
    if open_minute is None or close_minute is None:
        return False, None, f"{day_key} coverage hours must include valid open and close times."
    if close_minute <= open_minute:
        return False, None, f"{day_key} coverage close time must be after open time."

    return True, (open_minute, close_minute), None


def evaluate_weekly_hours(
    hours: Any,
    timezone: Any,
    now: datetime | None = None,
) -> WeeklyHoursEvaluation:
    """Evaluate explicit weekly AI coverage hours without business-hours fallback."""
    if not isinstance(hours, dict) or isinstance(hours, list):
        return WeeklyHoursEvaluation(False, False, "phoneCoverageHours must be a weekly hours object.")

    current, timezone_error = _localized_now(timezone, now)
    if timezone_error or current is None:
        return WeeklyHoursEvaluation(False, False, timezone_error)

    windows: dict[int, tuple[int, int] | None] = {}
    for js_day, key in enumerate(SHORT_DAY_KEYS):
        value = _day_value(hours, js_day)
        if value is None:
            return WeeklyHoursEvaluation(False, False, f"{key} coverage hours are missing.")
        valid, window, error = _coverage_window(value, key)
        if not valid:
            return WeeklyHoursEvaluation(False, False, error)
        windows[js_day] = window

    js_day = (current.weekday() + 1) % 7
    window = windows.get(js_day)
    if window is None:
        return WeeklyHoursEvaluation(False, True)

    current_minutes = current.hour * 60 + current.minute
    return WeeklyHoursEvaluation(window[0] <= current_minutes < window[1], True)


def resolve_phone_coverage_decision(config: dict[str, Any], now: datetime | None = None) -> CoverageDecision:
    raw_mode = config.get("phoneCoverageMode")
    mode = str(raw_mode or "").strip().lower()

    if not mode:
        return _handoff(
            "missing",
            "phone_coverage_missing_mode",
            "Dashboard config did not include phoneCoverageMode.",
        )
    if mode == "plan_default":
        return _handoff(
            mode,
            "phone_coverage_unresolved_plan_default",
            "Dashboard config returned unresolved plan_default.",
        )
    if mode not in VALID_COVERAGE_MODES:
        return _handoff(
            mode,
            "phone_coverage_invalid_mode",
            f"Dashboard config returned invalid phoneCoverageMode {mode!r}.",
        )

    if mode == "always_on":
        return _answer_ai(mode, "phone_coverage_always_on")
    if mode == "always_handoff":
        return _handoff(mode, "phone_coverage_always_handoff")

    current, timezone_error = _localized_now(config.get("timezone"), now)
    if timezone_error or current is None:
        return _handoff(mode, "phone_coverage_invalid_timezone", timezone_error)

    if mode == "business_hours":
        try:
            within_hours = is_currently_within_business_hours(config, current)
        except Exception as exc:
            return _handoff(
                mode,
                "phone_coverage_business_hours_invalid",
                f"Could not evaluate business hours: {exc}",
            )
        return (
            _handoff(mode, "phone_coverage_off")
            if within_hours
            else _answer_ai(mode, "phone_coverage_after_hours_on")
        )

    evaluation = evaluate_weekly_hours(config.get("phoneCoverageHours"), config.get("timezone"), current)
    if not evaluation.is_valid:
        return _handoff(mode, "phone_coverage_custom_hours_invalid", evaluation.error)
    return (
        _answer_ai(mode, "phone_coverage_custom_hours_on")
        if evaluation.is_within
        else _handoff(mode, "phone_coverage_off")
    )


def build_transfer_status_action_url(
    base_url: str,
    client_id: str,
    call_sid: str,
    reason: str,
) -> str:
    if not base_url or not call_sid:
        return ""
    path = (
        f"{base_url.rstrip('/')}/transfer-status/"
        f"{quote(str(client_id), safe='')}/{quote(str(call_sid), safe='')}"
    )
    query = urlencode({"reason": reason}) if reason else ""
    return f"{path}?{query}" if query else path


def build_dashboard_handoff_redirect_twiml(
    *,
    company_name: Any,
    dashboard_url: str,
    client_id: str,
    reason: str = "phone_coverage_off",
    origin: str = "",
) -> str | None:
    if not dashboard_url or not client_id:
        return None

    path = (
        f"{dashboard_url.rstrip('/')}/api/voice/twilio/agent-transfer/"
        f"{quote(str(client_id), safe='')}"
    )
    query_params = {}
    if reason:
        query_params["reason"] = reason
    if origin:
        query_params["origin"] = origin
    query = urlencode(query_params) if query_params else ""
    redirect_url = f"{path}?{query}" if query else path

    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Say voice="Polly.Joanna">Please hold while we connect you with {escape(str(company_name or "our team"))}.</Say>'
        f'<Redirect method="POST">{escape(redirect_url)}</Redirect>'
        "</Response>"
    )


def build_handoff_twiml(
    *,
    company_name: Any,
    forwarding_phone: Any,
    caller_id: Any = "",
    base_url: str = "",
    client_id: str = "",
    call_sid: str = "",
    reason: str = "phone_coverage_off",
) -> str | None:
    normalized_forwarding_phone = normalize_e164_phone(forwarding_phone)
    if not normalized_forwarding_phone:
        return None

    action_url = build_transfer_status_action_url(base_url, client_id, call_sid, reason)
    dial_attrs = ['timeout="25"']
    normalized_caller_id = normalize_e164_phone(caller_id)
    if normalized_caller_id:
        dial_attrs.append(f"callerId={quoteattr(normalized_caller_id)}")
    if action_url:
        dial_attrs.append(f"action={quoteattr(action_url)}")
        dial_attrs.append('method="POST"')

    post_dial_fallback = (
        ""
        if action_url
        else '<Say voice="Polly.Joanna">We were unable to reach anyone at this time. Please try again later. Goodbye.</Say>'
    )
    dial_attr_str = " ".join(dial_attrs)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Say voice="Polly.Joanna">Please hold while we connect you with {escape(str(company_name or "our team"))}.</Say>'
        f"<Dial {dial_attr_str}>{escape(normalized_forwarding_phone)}</Dial>"
        f"{post_dial_fallback}"
        "</Response>"
    )


def build_no_handoff_twiml(company_name: Any) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Say voice="Polly.Joanna">Thanks for calling {escape(str(company_name or "us"))}. '
        "We're not able to answer or transfer this call right now. "
        "Please try again later, or visit our website to book online. Goodbye.</Say>"
        "<Hangup/>"
        "</Response>"
    )
