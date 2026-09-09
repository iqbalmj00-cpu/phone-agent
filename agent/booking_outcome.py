"""Semantic booking responses. HTTP acceptance alone is never a commitment."""
from typing import Any


def booking_outcome(status: int, data: dict[str, Any]) -> str:
    if isinstance(data, dict) and data.get("success") is False and data.get("outcome") == "refused":
        return "refused"
    if status not in (200, 201, 202):
        return "uncertain" if status >= 500 else "failed"
    if not isinstance(data, dict):
        return "uncertain"
    if data.get("outcome") == "uncertain" or status == 202:
        return "uncertain"
    if data.get("success") is False:
        return "refused" if data.get("outcome") == "refused" or isinstance(data.get("feasibility"), dict) else "uncertain"
    if data.get("success") is not True:
        return "uncertain"
    if data.get("outcome") in ("closed", "active"):
        return data["outcome"]
    if data.get("scheduled") is True and isinstance(data.get("jobId"), str) and data["jobId"]:
        return "scheduled"
    if data.get("scheduled") is False and (data.get("leadId") or data.get("jobId")):
        return "pending"
    return "uncertain"


def sms_guidance(data: dict[str, Any]) -> str:
    sms = data.get("sms") if isinstance(data.get("sms"), dict) else {}
    status = sms.get("status")
    label = "card-on-file link text" if sms.get("kind") == "card_on_file" else "request receipt text" if sms.get("kind") == "request_receipt" else "text"
    if status == "accepted":
        return f"A {label} was accepted for sending; delivery is not confirmed."
    if status == "queued":
        return f"A {label} is queued for the permitted sending window; do not promise it shortly."
    return "Do not promise a text; no accepted send is confirmed."
