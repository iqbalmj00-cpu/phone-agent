"""Shared business-hours helpers for phone-agent runtime checks."""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

SHORT_DAY_KEYS = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
LONG_DAY_KEYS = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def parse_hhmm(value: Any) -> int | None:
    """Return minutes since midnight for HH:MM-ish values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        hour = int(value)
        return hour * 60 if 0 <= hour <= 23 else None

    text = str(value or "").strip()
    if not text:
        return None
    try:
        if ":" in text:
            hour_text, minute_text = text.split(":", 1)
            hour = int(hour_text)
            minute = int(minute_text[:2])
        else:
            hour = int(text)
            minute = 0
    except (TypeError, ValueError):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def format_minutes_spoken(minutes: int | None) -> str:
    if minutes is None:
        return "that time"
    hour = (minutes // 60) % 24
    minute = minutes % 60
    suffix = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12 or 12
    if minute == 0:
        return f"{hour_12} {suffix}"
    return f"{hour_12}:{minute:02d} {suffix}"


def _coerce_business_window(day_value: Any) -> tuple[bool, tuple[int, int] | None]:
    """Return (recognized, window). window=None means recognized closed."""
    if day_value is None or day_value is False:
        return True, None

    if isinstance(day_value, str):
        value = day_value.strip().lower()
        if value in {"closed", "off", "false", "none"}:
            return True, None
        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start = parse_hhmm(start_text)
            end = parse_hhmm(end_text)
            if start is not None and end is not None and end > start:
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
        start = parse_hhmm(start_text)
        end = parse_hhmm(end_text)
        if start is not None and end is not None and end > start:
            return True, (start, end)
        return False, None

    return False, None


def get_exact_business_window(config: dict[str, Any], js_day: int) -> tuple[bool, tuple[int, int] | None]:
    hours = config.get("businessHours")
    if not isinstance(hours, dict):
        return False, None

    candidate_keys = (
        SHORT_DAY_KEYS[js_day],
        LONG_DAY_KEYS[js_day],
        str(js_day),
        js_day,
    )
    for key in candidate_keys:
        if key in hours:
            return _coerce_business_window(hours.get(key))
    return False, None


def slot_window_minutes(slot_or_hour: Any) -> tuple[int | None, int | None]:
    if isinstance(slot_or_hour, dict):
        start = slot_or_hour.get("start_minute")
        end = slot_or_hour.get("end_minute")
        if start is None:
            start = parse_hhmm(slot_or_hour.get("start"))
        if end is None:
            end = parse_hhmm(slot_or_hour.get("end"))
        return start, end
    try:
        hour = int(slot_or_hour)
    except (TypeError, ValueError):
        return None, None
    return hour * 60, None


def is_open_day(date: datetime, config: dict[str, Any]) -> bool:
    js_day = (date.weekday() + 1) % 7
    exact_found, exact_window = get_exact_business_window(config, js_day)
    if exact_found:
        return exact_window is not None
    business_days = config.get("businessDays", [0, 1, 2, 3, 4, 5])
    return js_day in business_days


def is_business_hours(date: datetime, slot_or_hour: Any, config: dict[str, Any]) -> bool:
    js_day = (date.weekday() + 1) % 7
    start_minute, end_minute = slot_window_minutes(slot_or_hour)
    if start_minute is None:
        return False

    exact_found, exact_window = get_exact_business_window(config, js_day)
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


def is_currently_within_business_hours(config: dict[str, Any], now: datetime | None = None) -> bool:
    tz_str = config.get("timezone", "America/Chicago")
    current = now or datetime.now(ZoneInfo(tz_str))
    if current.tzinfo is None:
        current = current.replace(tzinfo=ZoneInfo(tz_str))
    current_minutes = current.hour * 60 + current.minute
    return is_business_hours(current, current_minutes // 60 if current.minute == 0 else {
        "start_minute": current_minutes,
        "end_minute": current_minutes + 1,
    }, config)


def _names_an_open_day(config: dict[str, Any]) -> bool:
    """True when the per-day weekly map marks at least one day open."""
    if not isinstance(config.get("businessHours"), dict):
        return False
    return any(
        found and window is not None
        for found, window in (get_exact_business_window(config, day) for day in range(7))
    )


def evaluate_current_business_hours(config: dict[str, Any], now: datetime | None = None) -> bool:
    """Is the business open right now? Decides whether a handoff can connect.

    Delegates to `is_currently_within_business_hours`, which `phone_coverage`
    also uses, with one correction: a weekly map whose every day is closed means
    the operator never configured hours, not that they are never open. The
    dashboard produces exactly that from an empty `businessDays`, and taken
    literally it would refuse every handoff forever while the agent went on
    telling callers the business was open — `format_days_label` falls back to
    the aggregate fields for the same reason, and the gate has to agree with
    what the caller is being told.
    """
    if isinstance(config.get("businessHours"), dict) and not _names_an_open_day(config):
        config = {key: value for key, value in config.items() if key != "businessHours"}
    return is_currently_within_business_hours(config, now)


def format_business_hours_for_day(config: dict[str, Any], date: datetime) -> str:
    js_day = (date.weekday() + 1) % 7
    exact_found, exact_window = get_exact_business_window(config, js_day)
    if exact_found and exact_window:
        return (
            "That's outside our hours. We're available that day from "
            f"{format_minutes_spoken(exact_window[0])} to {format_minutes_spoken(exact_window[1])}."
        )
    if exact_found and exact_window is None:
        return "We're closed that day. Ask which other day works best."
    business_days = config.get("businessDays", [0, 1, 2, 3, 4, 5])
    if js_day not in business_days:
        return "We're closed that day. Ask which other day works best."
    business_start = int(config.get("businessStart", 7))
    business_end = int(config.get("businessEnd", 19))
    return (
        "That's outside our hours. We're available from "
        f"{format_minutes_spoken(business_start * 60)} to {format_minutes_spoken(business_end * 60)}."
    )


def format_hours_label(config: dict[str, Any], now: datetime | None = None) -> str:
    tz_str = config.get("timezone", "America/Chicago")
    current = now or datetime.now(ZoneInfo(tz_str))
    js_day = (current.weekday() + 1) % 7
    exact_found, exact_window = get_exact_business_window(config, js_day)
    if exact_found and exact_window:
        return f"{format_minutes_spoken(exact_window[0])} to {format_minutes_spoken(exact_window[1])}"
    start = int(config.get("businessStart", 8))
    end = int(config.get("businessEnd", 18))
    return f"{format_minutes_spoken(start * 60)} to {format_minutes_spoken(end * 60)}"


def format_days_label(config: dict[str, Any]) -> str:
    hours = config.get("businessHours")
    if isinstance(hours, dict):
        open_days: list[int] = []
        for day in range(7):
            found, window = get_exact_business_window(config, day)
            if found and window is not None:
                open_days.append(day)
        if open_days:
            active = [DAY_NAMES[d] for d in open_days]
            if _is_contiguous(open_days) and len(active) > 1:
                return f"{active[0]} through {active[-1]}"
            return _spoken_day_list(active)

    biz_days = config.get("businessDays", [0, 1, 2, 3, 4, 5])
    # The dashboard stores this array unfiltered, so it can arrive with repeats
    # or out of order. A repeat makes _is_contiguous see a zero delta and the
    # label names the same day twice.
    sorted_days = sorted({d for d in biz_days if 0 <= d <= 6})
    active = [DAY_NAMES[d] for d in sorted_days]
    if not active:
        return "Monday through Saturday"
    if _is_contiguous(sorted_days) and len(active) > 1:
        return f"{active[0]} through {active[-1]}"
    return _spoken_day_list(active)


def _spoken_day_list(active: list[str]) -> str:
    """Join day names the way a person says them out loud.

    "Monday, Wednesday, Friday" read aloud is a flat comma list that runs
    straight into the opening time. A person says "Mondays, Wednesdays and
    Fridays". A single open day becomes "Saturdays only", because bare
    "Saturday" sounds like an offer of one particular date.
    """
    plural = [f"{day}s" for day in active]
    if len(plural) == 1:
        return f"{plural[0]} only"
    return f"{', '.join(plural[:-1])} and {plural[-1]}"


def _is_contiguous(days: list[int]) -> bool:
    return all(days[i + 1] - days[i] == 1 for i in range(len(days) - 1))
