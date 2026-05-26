import copy
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import server
from agent.dashboard_redirect import DashboardLiveRedirectResult


OPEN_ALL_WEEK = {
    day: {"open": "00:00", "close": "23:59", "closed": False}
    for day in ("sun", "mon", "tue", "wed", "thu", "fri", "sat")
}
CLOSED_ALL_WEEK = {
    day: {"open": "00:00", "close": "00:00", "closed": True}
    for day in ("sun", "mon", "tue", "wed", "thu", "fri", "sat")
}

BASE_CONFIG = {
    "companyName": "Coverage Test Co",
    "twilioNumber": "+15550000000",
    "forwardingPhone": "+15551234567",
    "timezone": "UTC",
    "agentSecret": "secret",
    "businessDays": [0, 1, 2, 3, 4, 5, 6],
    "businessStart": 0,
    "businessEnd": 23,
    "businessHours": OPEN_ALL_WEEK,
}


class TwimlCoverageRouteTests(unittest.TestCase):
    def setUp(self):
        server._active_calls.clear()
        self.client = TestClient(server.app, base_url="https://agent.example.com")

    def _post_twiml(self, config):
        async def fake_get_client_config(client_id, force_refresh=False):
            self.assertEqual(client_id, "client-1")
            self.assertTrue(force_refresh)
            return config

        with patch.object(server, "get_client_config", fake_get_client_config), patch.object(server, "DASHBOARD_URL", "https://dashboard.example.com"):
            return self.client.post(
                "/twiml/client-1",
                data={"From": "+15559990000", "To": "+15550000000", "CallSid": "CA123"},
                headers={"host": "agent.example.com", "x-forwarded-proto": "https"},
            )

    def test_always_on_returns_stream_twiml(self):
        response = self._post_twiml(copy.deepcopy({**BASE_CONFIG, "phoneCoverageMode": "always_on"}))
        self.assertEqual(response.status_code, 200)
        self.assertIn("<Stream", response.text)
        self.assertIn("wss://agent.example.com/ws/client-1", response.text)
        self.assertNotIn("<Dial", response.text)

    def test_always_handoff_redirects_to_dashboard_handoff(self):
        response = self._post_twiml(copy.deepcopy({**BASE_CONFIG, "phoneCoverageMode": "always_handoff"}))
        self.assertEqual(response.status_code, 200)
        self.assertIn("<Redirect", response.text)
        self.assertIn("https://dashboard.example.com/api/voice/twilio/agent-transfer/client-1", response.text)
        self.assertIn("reason=phone_coverage_always_handoff", response.text)
        self.assertIn("origin=phone_agent_coverage_off", response.text)
        self.assertNotIn("<Stream", response.text)

    def test_business_hours_mode_during_hours_redirects_to_dashboard_handoff(self):
        config = copy.deepcopy(BASE_CONFIG)
        config["phoneCoverageMode"] = "business_hours"
        config["businessHours"] = OPEN_ALL_WEEK
        response = self._post_twiml(config)
        self.assertEqual(response.status_code, 200)
        self.assertIn("<Redirect", response.text)
        self.assertIn("https://dashboard.example.com/api/voice/twilio/agent-transfer/client-1", response.text)
        self.assertIn("reason=phone_coverage_off", response.text)
        self.assertIn("origin=phone_agent_coverage_off", response.text)

    def test_business_hours_mode_after_hours_returns_stream_twiml(self):
        config = copy.deepcopy(BASE_CONFIG)
        config["phoneCoverageMode"] = "business_hours"
        config["businessHours"] = CLOSED_ALL_WEEK
        response = self._post_twiml(config)
        self.assertEqual(response.status_code, 200)
        self.assertIn("<Stream", response.text)
        self.assertNotIn("<Dial", response.text)

    def test_custom_hours_on_returns_stream_twiml(self):
        config = copy.deepcopy(BASE_CONFIG)
        config["phoneCoverageMode"] = "custom_hours"
        config["phoneCoverageHours"] = OPEN_ALL_WEEK
        response = self._post_twiml(config)
        self.assertEqual(response.status_code, 200)
        self.assertIn("<Stream", response.text)
        self.assertNotIn("<Dial", response.text)

    def test_agent_at_capacity_redirects_to_dashboard_handoff(self):
        for index in range(server.MAX_CONCURRENT_CALLS):
            server._active_calls[f"existing-call-{index}"] = None
        response = self._post_twiml(copy.deepcopy({**BASE_CONFIG, "phoneCoverageMode": "always_on"}))
        self.assertEqual(response.status_code, 200)
        self.assertIn("<Redirect", response.text)
        self.assertIn("https://dashboard.example.com/api/voice/twilio/agent-transfer/client-1", response.text)
        self.assertIn("reason=phone_agent_at_capacity", response.text)
        self.assertIn("origin=phone_agent_capacity", response.text)
        self.assertNotIn("<Stream", response.text)

    def test_websocket_capacity_redirects_live_call_to_dashboard_handoff(self):
        for index in range(server.MAX_CONCURRENT_CALLS):
            server._active_calls[f"existing-call-{index}"] = None

        redirect_mock = AsyncMock(return_value=DashboardLiveRedirectResult(ok=True, status=200))

        with patch.object(server, "DASHBOARD_URL", "https://dashboard.example.com"), patch.object(
            server, "PLATFORM_API_KEY", "platform-secret"
        ), patch.object(server, "redirect_live_call_via_dashboard", redirect_mock):
            with self.client.websocket_connect("/ws/client-1") as websocket:
                websocket.send_json({"event": "connected"})
                websocket.send_json({
                    "event": "start",
                    "start": {
                        "streamSid": "MZ123",
                        "callSid": "CA-CAPACITY",
                        "customParameters": {"From": "+15559990000"},
                    },
                })
                with self.assertRaises(WebSocketDisconnect):
                    websocket.receive_text()

        redirect_mock.assert_awaited_once()
        redirect_kwargs = redirect_mock.await_args.kwargs
        self.assertEqual(redirect_kwargs["client_id"], "client-1")
        self.assertEqual(redirect_kwargs["call_sid"], "CA-CAPACITY")
        self.assertEqual(redirect_kwargs["reason"], "phone_agent_at_capacity")
        self.assertEqual(redirect_kwargs["origin"], "phone_agent_capacity")
        self.assertEqual(redirect_kwargs["dashboard_url"], "https://dashboard.example.com")
        self.assertEqual(redirect_kwargs["platform_api_key"], "platform-secret")


if __name__ == "__main__":
    unittest.main()
