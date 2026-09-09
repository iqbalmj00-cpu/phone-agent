"""A caller with a bad phone number was told their TIME was unavailable.

`schedule_callback` collapsed HTTP 400 and 422 into one message. The dashboard
returns 422 for problems with the requested time and 400 for a missing or
unusable phone, so a caller whose number never made it through was asked, over
and over, for a different time — a question that could not fix anything.

Two more things could loop. A 403 (the call belongs to another tenant) fell into
the generic branch and returned `fallback: true`, and every prompt site mandates
a transfer on a fallback — which, after the D21 gate, is refused. And an operator
whose stored hours reject every possible date returns the same "outside business
hours" 422 as an ordinary bad slot, so asking again never terminates.

The phone itself is now substituted the way `create_booking` already does it:
the model can pass "the number you're calling from" instead of digits, and the
real Twilio number is used instead.
"""
import json
import unittest
from unittest.mock import patch

from agent import handlers


class FakeResponse:
    """The handler parses `text()` with `json.loads`, not `json()`."""

    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload

    async def text(self):
        return json.dumps(self._payload)


class FakeParams:
    def __init__(self, arguments):
        self.arguments = arguments
        self.results = []

    async def result_callback(self, payload):
        self.results.append(payload)


class ScheduleCallbackTests(unittest.IsolatedAsyncioTestCase):
    CALL_SID = "CA-CALLBACK"
    CALLER = "+15559990000"

    def setUp(self):
        handlers._call_contexts.clear()
        self.token = handlers._current_call_sid.set(self.CALL_SID)
        handlers.set_call_context(
            self.CALL_SID,
            self.CALLER,
            {
                "_clientId": "client-1",
                "companyName": "Callback Test Co",
                "agentSecret": "secret",
            },
        )

    def tearDown(self):
        handlers._current_call_sid.reset(self.token)
        handlers._call_contexts.clear()

    async def _schedule(self, status, payload, **arguments):
        """Run the handler against one dashboard response."""
        args = {
            "requested_time": "2026-09-10T14:00:00-05:00",
            "caller_phone": self.CALLER,
            "reason": "Caller asked for a callback",
        }
        args.update(arguments)
        params = FakeParams(args)
        self.sent = {}

        class _Session:
            async def __aenter__(_self):
                return _self

            async def __aexit__(_self, *exc):
                return False

            async def post(_self, url, json=None, headers=None, timeout=None):
                self.sent.update(json or {})
                return FakeResponse(status, payload)

        with patch.object(handlers.aiohttp, "ClientSession", _Session):
            await handlers.handle_schedule_callback(params)
        return params.results[-1] if params.results else {}


class PhoneVersusTimeTests(ScheduleCallbackTests):
    async def test_a_bad_phone_is_reported_as_a_phone_problem(self):
        result = await self._schedule(400, {"error": "callerPhone is required"})
        self.assertEqual(result["error"], "invalid_callback_phone")
        self.assertIn("number", result["message"].lower())
        self.assertNotIn("another time", result["message"].lower())

    async def test_a_bad_time_is_still_reported_as_a_time_problem(self):
        result = await self._schedule(
            422, {"error": "requestedTime cannot be in the past"}
        )
        self.assertEqual(result["error"], "invalid_callback_time")
        self.assertIn("another time", result["message"].lower())

    async def test_a_spoken_phrase_is_replaced_with_the_real_number(self):
        """The model passes prose here as often as digits."""
        await self._schedule(
            201,
            {"success": True, "callbackTaskId": "task-fixture", "callbackDueAt": "2026-09-10T14:00:00-05:00"},
            caller_phone="the number you're calling from",
        )
        self.assertEqual(self.sent["callerPhone"], self.CALLER)

    async def test_a_real_number_is_left_alone(self):
        await self._schedule(
            201, {"callbackDueAt": "x"}, caller_phone="(512) 555-1234"
        )
        self.assertEqual(self.sent["callerPhone"], "(512) 555-1234")


class UnrecoverableTests(ScheduleCallbackTests):
    async def test_misconfigured_hours_stop_the_asking_immediately(self):
        result = await self._schedule(
            422, {"error": "Business hours are not configured correctly"}
        )
        self.assertEqual(result["error"], "callback_hours_unavailable")
        self.assertIn("do not ask for another time", result["message"].lower())

    async def test_repeated_time_rejections_stop_the_asking(self):
        """An all-closed schedule rejects every date with the ordinary message."""
        first = await self._schedule(422, {"error": "requestedTime is outside business hours"})
        self.assertEqual(first["error"], "invalid_callback_time")
        second = await self._schedule(422, {"error": "requestedTime is outside business hours"})
        self.assertEqual(second["error"], "callback_hours_unavailable")

    async def test_a_success_clears_the_count(self):
        await self._schedule(422, {"error": "requestedTime is outside business hours"})
        await self._schedule(201, {"success": True, "callbackTaskId": "task-fixture", "callbackDueAt": "2026-09-10T14:00:00-05:00"})
        again = await self._schedule(422, {"error": "requestedTime is outside business hours"})
        self.assertEqual(again["error"], "invalid_callback_time")

    async def test_a_cross_tenant_rejection_does_not_mandate_a_transfer(self):
        """A fallback here would demand a transfer the gate then refuses."""
        result = await self._schedule(
            403, {"error": "twilioCallSid does not belong to this tenant"}
        )
        self.assertNotIn("fallback", result)
        self.assertEqual(result["error"], "callback_not_available")


if __name__ == "__main__":
    unittest.main()
