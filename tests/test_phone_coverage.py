import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from agent.phone_coverage import (
    build_handoff_twiml,
    build_no_handoff_twiml,
    evaluate_weekly_hours,
    resolve_phone_coverage_decision,
)


WEEKLY_HOURS = {
    "sun": {"open": "00:00", "close": "00:00", "closed": True},
    "mon": {"open": "08:00", "close": "17:00", "closed": False},
    "tue": {"open": "08:00", "close": "17:00", "closed": False},
    "wed": {"open": "08:00", "close": "17:00", "closed": False},
    "thu": {"open": "08:00", "close": "17:00", "closed": False},
    "fri": {"open": "08:00", "close": "17:00", "closed": False},
    "sat": {"open": "00:00", "close": "00:00", "closed": True},
}


class PhoneCoverageTests(unittest.TestCase):
    def test_always_on_answers_with_ai(self):
        decision = resolve_phone_coverage_decision({"phoneCoverageMode": "always_on"})
        self.assertTrue(decision.should_answer_ai)
        self.assertFalse(decision.should_handoff)

    def test_always_handoff_forwards(self):
        decision = resolve_phone_coverage_decision({"phoneCoverageMode": "always_handoff"})
        self.assertFalse(decision.should_answer_ai)
        self.assertTrue(decision.should_handoff)
        self.assertEqual(decision.reason, "phone_coverage_always_handoff")

    def test_business_hours_uses_client_timezone(self):
        config = {
            "phoneCoverageMode": "business_hours",
            "timezone": "America/Los_Angeles",
            "businessHours": WEEKLY_HOURS,
        }
        inside = datetime(2026, 5, 18, 15, 30, tzinfo=ZoneInfo("UTC"))  # Monday 8:30 AM Pacific
        outside = datetime(2026, 5, 18, 23, 30, tzinfo=ZoneInfo("UTC"))  # Monday 4:30 PM Pacific
        self.assertTrue(resolve_phone_coverage_decision(config, inside).should_answer_ai)
        self.assertTrue(resolve_phone_coverage_decision(config, outside).should_answer_ai)

        after_close = datetime(2026, 5, 19, 1, 30, tzinfo=ZoneInfo("UTC"))  # Monday 6:30 PM Pacific
        self.assertTrue(resolve_phone_coverage_decision(config, after_close).should_handoff)

    def test_custom_hours_inside_and_outside(self):
        config = {
            "phoneCoverageMode": "custom_hours",
            "timezone": "America/Chicago",
            "phoneCoverageHours": WEEKLY_HOURS,
        }
        inside = datetime(2026, 5, 18, 14, 0, tzinfo=ZoneInfo("UTC"))  # Monday 9 AM Central
        outside = datetime(2026, 5, 18, 23, 0, tzinfo=ZoneInfo("UTC"))  # Monday 6 PM Central

        self.assertTrue(resolve_phone_coverage_decision(config, inside).should_answer_ai)
        self.assertTrue(resolve_phone_coverage_decision(config, outside).should_handoff)

    def test_custom_hours_missing_or_invalid_fails_safe_to_handoff(self):
        missing = {
            "phoneCoverageMode": "custom_hours",
            "timezone": "America/Chicago",
            "phoneCoverageHours": None,
        }
        invalid = {
            "phoneCoverageMode": "custom_hours",
            "timezone": "America/Chicago",
            "phoneCoverageHours": {
                **WEEKLY_HOURS,
                "mon": {"open": "17:00", "close": "08:00", "closed": False},
            },
        }

        self.assertTrue(resolve_phone_coverage_decision(missing).should_handoff)
        self.assertEqual(resolve_phone_coverage_decision(missing).reason, "phone_coverage_custom_hours_invalid")
        self.assertTrue(resolve_phone_coverage_decision(invalid).should_handoff)

    def test_plan_default_is_not_resolved_by_runtime(self):
        decision = resolve_phone_coverage_decision({"phoneCoverageMode": "plan_default"})
        self.assertTrue(decision.should_handoff)
        self.assertEqual(decision.reason, "phone_coverage_unresolved_plan_default")

    def test_evaluate_weekly_hours_requires_explicit_week(self):
        partial = {"mon": {"open": "08:00", "close": "17:00", "closed": False}}
        result = evaluate_weekly_hours(partial, "America/Chicago")
        self.assertFalse(result.is_valid)
        self.assertIn("sun", result.error or "")

    def test_handoff_twiml_includes_dial_action_and_escaped_values(self):
        twiml = build_handoff_twiml(
            company_name="A&B Junk",
            forwarding_phone="555-123-4567",
            caller_id="+15559990000",
            base_url="https://agent.example.com",
            client_id="client/id",
            call_sid="CA/123",
            reason="phone_coverage_off",
        )
        self.assertIsNotNone(twiml)
        assert twiml is not None
        self.assertIn("A&amp;B Junk", twiml)
        self.assertIn("<Dial ", twiml)
        self.assertIn("+15551234567", twiml)
        self.assertIn('callerId="+15559990000"', twiml)
        self.assertIn("/transfer-status/client%2Fid/CA%2F123?reason=phone_coverage_off", twiml)

    def test_handoff_twiml_without_call_sid_has_post_dial_fallback(self):
        twiml = build_handoff_twiml(
            company_name="Test Co",
            forwarding_phone="+15551234567",
            caller_id="",
            base_url="https://agent.example.com",
            client_id="client",
            call_sid="",
            reason="phone_coverage_off",
        )
        self.assertIsNotNone(twiml)
        assert twiml is not None
        self.assertNotIn("action=", twiml)
        self.assertIn("We were unable to reach anyone", twiml)

    def test_missing_forwarding_phone_returns_no_handoff_twiml_builder(self):
        self.assertIsNone(build_handoff_twiml(company_name="Test", forwarding_phone=""))
        fallback = build_no_handoff_twiml("A&B Junk")
        self.assertIn("A&amp;B Junk", fallback)
        self.assertIn("<Hangup/>", fallback)


if __name__ == "__main__":
    unittest.main()
