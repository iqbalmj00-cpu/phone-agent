import unittest
from unittest.mock import patch

from agent import dashboard_redirect


class FakeResponse:
    def __init__(self, status=200, text='{"ok":true}', json_body=None):
        self.status = status
        self._text = text
        self._json_body = {"ok": True} if json_body is None else json_body

    async def text(self):
        return self._text

    async def json(self, content_type=None):
        return self._json_body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, json, headers, timeout):
        self.requests.append({
            "url": url,
            "json": json,
            "headers": headers,
            "timeout": timeout,
        })
        return self.response


class DashboardRedirectTests(unittest.IsolatedAsyncioTestCase):
    async def test_redirect_live_call_posts_secure_dashboard_payload(self):
        session = FakeSession(FakeResponse(status=202, json_body={"accepted": True}))

        with patch.object(dashboard_redirect.aiohttp, "ClientSession", return_value=session):
            result = await dashboard_redirect.redirect_live_call_via_dashboard(
                client_config={"agentSecret": "secret"},
                client_id="client-1",
                call_sid="CA123",
                reason="Caller asked for dispatch",
                origin="phone_agent_ai_transfer",
                dashboard_url="https://dashboard.example.com/",
                endpoint_path="api/agent/voice/redirect-live-call",
                platform_api_key="platform-secret",
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, 202)
        self.assertEqual(len(session.requests), 1)
        request = session.requests[0]
        self.assertEqual(
            request["url"],
            "https://dashboard.example.com/api/agent/voice/redirect-live-call",
        )
        self.assertEqual(request["headers"]["x-api-key"], "platform-secret")
        self.assertEqual(request["headers"]["X-AGENT-SECRET"], "secret")
        self.assertEqual(request["json"]["clientId"], "client-1")
        self.assertEqual(request["json"]["callSid"], "CA123")
        self.assertEqual(request["json"]["reason"], "Caller asked for dispatch")
        self.assertEqual(request["json"]["origin"], "phone_agent_ai_transfer")

    async def test_redirect_live_call_requires_platform_api_key(self):
        session = FakeSession(FakeResponse(status=202))

        with patch.object(dashboard_redirect.aiohttp, "ClientSession", return_value=session):
            result = await dashboard_redirect.redirect_live_call_via_dashboard(
                client_config={},
                client_id="client-1",
                call_sid="CA123",
                reason="Caller asked for dispatch",
                origin="phone_agent_ai_transfer",
                dashboard_url="https://dashboard.example.com",
                platform_api_key="",
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.error, "platform_api_key_missing")
        self.assertEqual(session.requests, [])


if __name__ == "__main__":
    unittest.main()
