"""After hours there is nobody to transfer to, so the agent takes a callback.

D21: outside business hours the agent attempts no handoff at all. It notes the
call in the dashboard's Callback Queue instead, which is driven purely by
`PhoneCall.callbackRequested`, and carries on with the call normally.

Two config shapes decide whether that gate is correct or catastrophic, so they
are tested first.

The first is the `business_hours` coverage mode, which inverts intuition: the AI
answers only when the office is CLOSED, because that is the whole point of it
answering. So for those clients the gate is in force on 100% of their traffic.

The second is a weekly map with every day closed. The dashboard produces exactly
that from an empty `businessDays`, and read literally it would refuse every
handoff forever while the agent went on telling callers the business was open
Monday to Friday. An all-closed map means hours were never configured, so the
gate falls back to the same aggregate fields the agent quotes.
"""
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

from agent import handlers
from agent.business_hours import evaluate_current_business_hours
from agent.dashboard_redirect import DashboardLiveRedirectResult


TZ = ZoneInfo("America/Chicago")
WED_NOON = datetime(2026, 9, 2, 12, 0, tzinfo=TZ)
WED_9PM = datetime(2026, 9, 2, 21, 0, tzinfo=TZ)
WED_7AM = datetime(2026, 9, 2, 7, 0, tzinfo=TZ)
SUN_NOON = datetime(2026, 9, 6, 12, 0, tzinfo=TZ)

DAYS = ("sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday")


def _weekly(open_days, opens="08:00", closes="17:00"):
    return {
        day: (
            {"open": opens, "close": closes, "closed": False}
            if day in open_days
            else {"open": None, "close": None, "closed": True}
        )
        for day in DAYS
    }


BASE = {
    "timezone": "America/Chicago",
    "businessDays": [1, 2, 3, 4, 5],
    "businessStart": 8,
    "businessEnd": 17,
}
MON_FRI = {**BASE, "businessHours": _weekly({"monday", "tuesday", "wednesday", "thursday", "friday"})}
ALL_CLOSED = {**BASE, "businessHours": _weekly(set())}


class CurrentHoursTests(unittest.TestCase):
    """The ordinary shapes, so the corrections below are visibly corrections."""

    def test_open_during_configured_hours(self):
        self.assertTrue(evaluate_current_business_hours(MON_FRI, WED_NOON))

    def test_closed_in_the_evening(self):
        self.assertFalse(evaluate_current_business_hours(MON_FRI, WED_9PM))

    def test_closed_before_opening(self):
        self.assertFalse(evaluate_current_business_hours(MON_FRI, WED_7AM))

    def test_closed_on_a_closed_day(self):
        self.assertFalse(evaluate_current_business_hours(MON_FRI, SUN_NOON))

    def test_aggregate_fields_are_used_when_no_weekly_map_exists(self):
        self.assertTrue(evaluate_current_business_hours(BASE, WED_NOON))
        self.assertFalse(evaluate_current_business_hours(BASE, WED_9PM))


class AllDaysClosedTests(unittest.TestCase):
    """An all-closed map means unconfigured, not never-open.

    Reachable today: PUT /api/dashboard/agent-config accepts `businessDays: []`
    with no length check, and that serializes to a non-null all-closed map.
    """

    def test_the_gate_agrees_with_the_hours_the_agent_quotes(self):
        self.assertTrue(evaluate_current_business_hours(ALL_CLOSED, WED_NOON))

    def test_it_still_closes_outside_the_aggregate_hours(self):
        """Falling back must not mean always open."""
        self.assertFalse(evaluate_current_business_hours(ALL_CLOSED, WED_9PM))
        self.assertFalse(evaluate_current_business_hours(ALL_CLOSED, SUN_NOON))

    def test_one_open_day_is_enough_to_trust_the_weekly_map(self):
        """A genuinely Sunday-only operator is configured, not broken."""
        sunday_only = {**BASE, "businessHours": _weekly({"sunday"})}
        self.assertTrue(evaluate_current_business_hours(sunday_only, SUN_NOON))
        self.assertFalse(evaluate_current_business_hours(sunday_only, WED_NOON))

    def test_the_weekly_map_still_wins_when_it_names_open_days(self):
        """Mon/Wed/Fri per-day hours beat a Mon-Fri aggregate."""
        mwf = {**BASE, "businessHours": _weekly({"monday", "wednesday", "friday"})}
        tue_noon = datetime(2026, 9, 1, 12, 0, tzinfo=TZ)
        self.assertTrue(evaluate_current_business_hours(mwf, WED_NOON))
        self.assertFalse(evaluate_current_business_hours(mwf, tue_noon))


ALWAYS_OPEN = {
    "timezone": "America/Chicago",
    "businessDays": [0, 1, 2, 3, 4, 5, 6],
    "businessStart": 0,
    "businessEnd": 24,
}
ALWAYS_SHUT = {**ALWAYS_OPEN, "businessDays": []}


class FakeParams:
    def __init__(self, arguments):
        self.arguments = arguments
        self.results = []

    async def result_callback(self, payload):
        self.results.append(payload)


class HandoffGateTests(unittest.IsolatedAsyncioTestCase):
    """What the caller gets when they ask for a person after hours."""

    CALL_SID = "CA-AFTER-HOURS"

    def setUp(self):
        handlers._call_contexts.clear()
        handlers._terminal_transfer_states.clear()
        self.token = handlers._current_call_sid.set(self.CALL_SID)

    def tearDown(self):
        handlers._current_call_sid.reset(self.token)
        handlers._call_contexts.clear()
        handlers._terminal_transfer_states.clear()

    def _context(self, hours):
        handlers.set_call_context(
            self.CALL_SID,
            "+15559990000",
            {
                "_clientId": "client-1",
                "companyName": "After Hours Co",
                "twilioNumber": "+15550000000",
                "agentSecret": "secret",
                **hours,
            },
        )

    async def _transfer(self, hours, reason="Caller asked for a person"):
        self._context(hours)
        params = FakeParams({"reason": reason})
        with patch.object(handlers, "_get_twilio_client") as twilio:
            await handlers.handle_transfer_to_human(params)
            self.twilio_used = twilio.called
        return params.results[-1]

    async def test_after_hours_no_transfer_is_attempted(self):
        result = await self._transfer(ALWAYS_SHUT)
        self.assertFalse(result["transferred"])
        self.assertTrue(result["after_hours"])
        self.assertFalse(self.twilio_used)

    async def test_after_hours_lands_the_caller_in_the_callback_queue(self):
        """`callbackRequested` is the only thing that tab reads."""
        await self._transfer(ALWAYS_SHUT)
        state = handlers.get_transfer_state(self.CALL_SID)
        self.assertTrue(state["callback_requested"])
        self.assertEqual(state["transfer_status"], "suppressed_after_hours")

    async def test_the_refusal_is_not_logged_as_a_transfer(self):
        """A gate below the "requested" write would log a phantom transfer."""
        await self._transfer(ALWAYS_SHUT)
        self.assertNotEqual(
            handlers.get_transfer_state(self.CALL_SID)["transfer_status"], "requested"
        )

    async def test_the_refusal_does_not_reuse_the_failed_status(self):
        """"failed" fires the dashboard's "Phone transfer failed" alert."""
        await self._transfer(ALWAYS_SHUT)
        self.assertNotEqual(
            handlers.get_transfer_state(self.CALL_SID)["transfer_status"], "failed"
        )

    async def test_the_refusal_does_not_mandate_another_transfer(self):
        """Prompt sites mandate a transfer on any fallback, so this must carry none."""
        result = await self._transfer(ALWAYS_SHUT)
        self.assertNotIn("fallback", result)

    async def test_the_call_continues(self):
        """No pipeline cancel: the caller can still book after being refused."""
        result = await self._transfer(ALWAYS_SHUT)
        self.assertIn("carry on", result["note"].lower())

    async def test_the_caller_is_addressed_directly(self):
        """The spoken half must never talk ABOUT the caller in front of them."""
        result = await self._transfer(ALWAYS_SHUT)
        said = result["say"].lower()
        self.assertIn("you", said)
        for third_person in ("this caller", "the caller", "them ", "they "):
            with self.subTest(phrase=third_person):
                self.assertNotIn(third_person, said)

    async def test_the_spoken_line_carries_no_stage_directions(self):
        """`say` is read aloud; instructions belong in `note`."""
        said = (await self._transfer(ALWAYS_SHUT))["say"].lower()
        for leak in ("do not", "tell the caller", "say that", "warmly"):
            with self.subTest(leak=leak):
                self.assertNotIn(leak, said)

    async def test_in_hours_the_transfer_still_proceeds(self):
        self._context(ALWAYS_OPEN)
        params = FakeParams({"reason": "Caller asked for dispatch"})
        redirect = AsyncMock(return_value=DashboardLiveRedirectResult(ok=True, status=200))
        with patch.object(handlers, "redirect_live_call_via_dashboard", redirect):
            await handlers.handle_transfer_to_human(params)
        self.assertTrue(redirect.called)
        self.assertNotEqual(
            handlers.get_transfer_state(self.CALL_SID)["transfer_status"],
            "suppressed_after_hours",
        )

    async def test_booking_after_a_refusal_clears_the_queue_flag(self):
        """They got what they rang for, so nobody needs to call them back."""
        await self._transfer(ALWAYS_SHUT)
        self.assertTrue(handlers.get_transfer_state(self.CALL_SID)["callback_requested"])
        handlers.mark_booking_complete(self.CALL_SID)
        self.assertFalse(handlers.get_transfer_state(self.CALL_SID)["callback_requested"])


if __name__ == "__main__":
    unittest.main()
