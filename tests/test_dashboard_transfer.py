import unittest
from unittest.mock import AsyncMock, patch

from agent import handlers
from agent.dashboard_redirect import DashboardLiveRedirectResult
import bot


class FakeParams:
    def __init__(self, arguments):
        self.arguments = arguments
        self.results = []

    async def result_callback(self, payload):
        self.results.append(payload)


class DashboardTransferTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        handlers._call_contexts.clear()
        handlers._terminal_transfer_states.clear()
        self.call_sid = "CA-DASHBOARD"
        self.token = handlers._current_call_sid.set(self.call_sid)
        handlers.set_call_context(
            self.call_sid,
            "+15559990000",
            {
                "_clientId": "client-1",
                "companyName": "Transfer Test Co",
                "twilioNumber": "+15550000000",
                "agentSecret": "secret",
                # Open every hour of every day. These tests cover the in-hours
                # transfer path; without explicit hours they would inherit the
                # 7am-7pm default and start failing overnight. The after-hours
                # refusal has its own tests in test_after_hours_handoff.py.
                "timezone": "America/Chicago",
                "businessDays": [0, 1, 2, 3, 4, 5, 6],
                "businessStart": 0,
                "businessEnd": 24,
            },
        )

    def tearDown(self):
        handlers._current_call_sid.reset(self.token)
        handlers._call_contexts.clear()
        handlers._terminal_transfer_states.clear()

    async def test_the_ai_never_updates_twilio_directly(self):
        params = FakeParams({"reason": "Caller asked for dispatch"})
        redirect_mock = AsyncMock(return_value=DashboardLiveRedirectResult(ok=True, status=200))

        with patch.object(handlers, "DASHBOARD_URL", "https://dashboard.example.com"), patch.object(
            handlers, "PLATFORM_API_KEY", "platform-secret"
        ), patch.object(
            handlers,
            "redirect_live_call_via_dashboard",
            redirect_mock,
        ), patch.object(
            handlers,
            "_get_twilio_client",
            side_effect=AssertionError("AI transfer must not update Twilio directly"),
        ):
            await handlers.handle_transfer_to_human(params)

        redirect_mock.assert_awaited_once()
        redirect_kwargs = redirect_mock.await_args.kwargs
        self.assertEqual(redirect_kwargs["client_id"], "client-1")
        self.assertEqual(redirect_kwargs["call_sid"], self.call_sid)
        self.assertEqual(redirect_kwargs["reason"], "Caller asked for dispatch")
        self.assertEqual(redirect_kwargs["origin"], "phone_agent_ai_transfer")
        self.assertEqual(redirect_kwargs["dashboard_url"], "https://dashboard.example.com")
        self.assertEqual(redirect_kwargs["platform_api_key"], "platform-secret")
        self.assertTrue(handlers.is_dashboard_owned_transfer(self.call_sid))
        self.assertFalse(bot.should_write_final_call_log(self.call_sid))
        self.assertEqual(params.results[0]["transferStatus"], "dashboard_redirected")

    async def test_failed_dashboard_redirect_can_still_be_logged_by_phone_agent(self):
        params = FakeParams({"reason": "Caller asked for dispatch"})

        with patch.object(handlers, "DASHBOARD_URL", ""):
            await handlers.handle_transfer_to_human(params)

        self.assertFalse(handlers.is_dashboard_owned_transfer(self.call_sid))
        self.assertTrue(bot.should_write_final_call_log(self.call_sid))
        self.assertEqual(handlers.get_transfer_state(self.call_sid)["transfer_status"], "failed")
        self.assertEqual(params.results[0]["transferred"], False)
