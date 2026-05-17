import copy
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import server


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

        with patch.object(server, "get_client_config", fake_get_client_config):
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

    def test_always_handoff_returns_dial_twiml(self):
        response = self._post_twiml(copy.deepcopy({**BASE_CONFIG, "phoneCoverageMode": "always_handoff"}))
        self.assertEqual(response.status_code, 200)
        self.assertIn("<Dial", response.text)
        self.assertIn("+15551234567", response.text)
        self.assertIn("reason=phone_coverage_always_handoff", response.text)
        self.assertNotIn("<Stream", response.text)

    def test_business_hours_off_returns_dial_twiml(self):
        config = copy.deepcopy(BASE_CONFIG)
        config["phoneCoverageMode"] = "business_hours"
        config["businessHours"] = CLOSED_ALL_WEEK
        response = self._post_twiml(config)
        self.assertEqual(response.status_code, 200)
        self.assertIn("<Dial", response.text)
        self.assertIn("reason=phone_coverage_off", response.text)

    def test_custom_hours_on_returns_stream_twiml(self):
        config = copy.deepcopy(BASE_CONFIG)
        config["phoneCoverageMode"] = "custom_hours"
        config["phoneCoverageHours"] = OPEN_ALL_WEEK
        response = self._post_twiml(config)
        self.assertEqual(response.status_code, 200)
        self.assertIn("<Stream", response.text)
        self.assertNotIn("<Dial", response.text)


if __name__ == "__main__":
    unittest.main()
