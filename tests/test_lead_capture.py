"""A caller who gives their name and does not book is now recorded.

Until now the agent recorded a caller in the dashboard only if they completed a
booking. Everyone else — the person who rang, described a garage full of
furniture, and said they would think about it — left no trace. That is the
reason the opening line asks who is calling (D12), and the rule is strict: no
name, no lead.

The gate is narrow on purpose. Only an `info_only` call becomes a lead. Every
other outcome is already accounted for somewhere else, and the one that matters
is `callback_requested`: the after-hours handoff refusal, an explicitly
scheduled callback, and a failed transfer all put the caller in the dashboard's
Callback Queue, and D27 says those callers must not also be captured as leads
and cold-texted by the follow-up cron.

The lead POST is never retried. `/api/ingest/lead` has no dedup on create, so a
retry means a second lead row, a second alert email to the operator, and
possibly a second automated text to the customer.
"""
import unittest
from unittest.mock import AsyncMock, patch

from agent import handlers
from agent.handlers import _normalize_caller_name, usable_inquiry


class CallerNameTests(unittest.TestCase):
    """D12's gate. Everything downstream is a truthiness test on this."""

    def test_a_real_name_survives(self):
        self.assertEqual(_normalize_caller_name("Jane"), "Jane")
        self.assertEqual(_normalize_caller_name("Jean-Luc O'Brien"), "Jean-Luc O'Brien")

    def test_whitespace_is_collapsed(self):
        self.assertEqual(_normalize_caller_name("  jane   doe "), "jane doe")

    def test_placeholders_are_rejected(self):
        """The model offers these when it has not actually been told a name."""
        for value in ("unknown", "N/A", "none", "Customer", "caller", "anonymous", "none."):
            with self.subTest(value=value):
                self.assertEqual(_normalize_caller_name(value), "")

    def test_a_phone_number_is_not_a_name(self):
        self.assertEqual(_normalize_caller_name("5125551234"), "")

    def test_empty_and_oversized_are_rejected(self):
        for value in ("", None, "   ", "x" * 81):
            with self.subTest(value=repr(value)[:20]):
                self.assertEqual(_normalize_caller_name(value), "")

    def test_marking_a_name_is_readable_back(self):
        handlers._call_contexts.clear()
        handlers.set_call_context("CA-N", "+15550000000", {})
        handlers.mark_caller_name("CA-N", "  Jane  ")
        self.assertEqual(handlers.get_caller_name("CA-N"), "Jane")
        handlers._call_contexts.clear()

    def test_a_placeholder_never_overwrites_a_real_name(self):
        handlers._call_contexts.clear()
        handlers.set_call_context("CA-N", "+15550000000", {})
        handlers.mark_caller_name("CA-N", "Jane")
        handlers.mark_caller_name("CA-N", "unknown")
        self.assertEqual(handlers.get_caller_name("CA-N"), "Jane")
        handlers._call_contexts.clear()

    def test_an_unknown_call_is_a_no_op(self):
        """Matches mark_sms_consent, which early-returns the same way."""
        handlers._call_contexts.clear()
        handlers.mark_caller_name("CA-MISSING", "Jane")
        self.assertEqual(handlers.get_caller_name("CA-MISSING"), "")


class InquiryTests(unittest.TestCase):
    """The description renders verbatim under "Customer's Message"."""

    def test_a_real_summary_is_kept(self):
        self.assertEqual(
            usable_inquiry("Caller wants a garage cleared next week."),
            "Caller wants a garage cleared next week.",
        )

    def test_every_fallback_string_is_suppressed(self):
        """One interpolates the transcript, so this must be a prefix test."""
        for value in (
            "Summary unavailable",
            "No caller conversation captured before the call ended.",
            "AI summary unavailable. Recent transcript: Caller: hi | Agent: hello",
        ):
            with self.subTest(value=value[:32]):
                self.assertEqual(usable_inquiry(value), "")

    def test_blank_is_suppressed(self):
        self.assertEqual(usable_inquiry(""), "")
        self.assertEqual(usable_inquiry(None), "")


class LeadPayloadTests(unittest.IsolatedAsyncioTestCase):
    """What actually goes on the wire."""

    async def _post(self, status=201, **kwargs):
        args = {
            "config": {"companyName": "Test Co", "siteToken": "tok"},
            "name": "Jane",
            "phone": "+15125551234",
            "description": "Caller wants a garage cleared.",
            "sms_consent": False,
        }
        args.update(kwargs)
        self.sent = None

        class _Session:
            async def __aenter__(_s):
                return _s

            async def __aexit__(_s, *e):
                return False

            async def post(_s, url, json=None, headers=None, timeout=None):
                self.sent = json
                self.url = url

                class _R:
                    def __init__(self):
                        self.status = status

                    async def text(self):
                        return "{}"

                return _R()

        with patch.object(handlers.aiohttp, "ClientSession", _Session):
            return await handlers.create_lead_from_call(**args)

    async def test_the_payload_carries_what_the_operator_needs(self):
        self.assertTrue(await self._post())
        self.assertEqual(self.sent["name"], "Jane")
        self.assertEqual(self.sent["phone"], "+15125551234")
        self.assertEqual(self.sent["source"], "phone_ai")
        self.assertEqual(self.sent["status"], "new")
        self.assertEqual(self.sent["description"], "Caller wants a garage cleared.")

    async def test_it_never_sends_service_type(self):
        """"dumpster_rental" makes the dashboard auto-approve a container."""
        await self._post()
        self.assertNotIn("serviceType", self.sent)

    async def test_it_never_sends_notes(self):
        """The pipeline regex-splits notes into chips, breaking a sentence up."""
        await self._post()
        self.assertNotIn("notes", self.sent)

    async def test_it_sends_no_lead_id_value_or_requested_date(self):
        await self._post()
        for field in ("leadId", "value", "requestedDate"):
            with self.subTest(field=field):
                self.assertNotIn(field, self.sent)

    async def test_consent_travels_only_when_it_was_given(self):
        await self._post(sms_consent=False)
        self.assertNotIn("smsOptIn", self.sent)
        await self._post(sms_consent=True)
        self.assertTrue(self.sent["smsOptIn"])
        self.assertIn("phone call", self.sent["consentText"])

    async def test_a_missing_description_is_simply_absent(self):
        await self._post(description="")
        self.assertNotIn("description", self.sent)

    async def test_no_name_means_no_request_at_all(self):
        self.assertFalse(await self._post(name=""))
        self.assertIsNone(self.sent)

    async def test_no_phone_means_no_request_at_all(self):
        self.assertFalse(await self._post(phone=""))
        self.assertIsNone(self.sent)

    async def test_a_rate_limit_is_reported_as_a_lost_lead(self):
        self.assertFalse(await self._post(status=429))

    async def test_a_failure_is_never_retried(self):
        """No dedup on create: a retry is a second row and a second text."""
        calls = []

        class _Session:
            async def __aenter__(_s):
                return _s

            async def __aexit__(_s, *e):
                return False

            async def post(_s, url, json=None, headers=None, timeout=None):
                calls.append(json)

                class _R:
                    status = 500

                    async def text(self):
                        return "{}"

                return _R()

        with patch.object(handlers.aiohttp, "ClientSession", _Session):
            result = await handlers.create_lead_from_call(
                config={"siteToken": "tok"},
                name="Jane",
                phone="+15125551234",
                description="x",
                sms_consent=False,
            )
        self.assertFalse(result)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
