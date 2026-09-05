"""What the agent says when a booking does not simply succeed.

Two things were wrong.

A dumpster swap said "scheduled" on every 201, including the case where no
container had been matched yet. The dashboard then texted the same customer
"your swap request is in" — so the voice and the text contradicted each other,
and the voice was the one that was wrong. The dashboard picks its wording from
`scheduled`, so the agent now reads the same field.

And a caller who asked for a final pickup with no replacement was told we would
drop off an empty container, because the pickup-only flow books as a swap and
the swap message asserted a replacement unconditionally.

Separately, the dashboard can decline a date on purpose: 201 means booked, and a
200 carrying `success: false` means "not that day, here are others". The agent
had no branch for it, so a deliberate business answer was read out as "trouble
with our scheduling system" and the alternative dates were thrown away. That
policy is off by default today — nothing in the dashboard writes it — so this is
insurance rather than a live fix.
"""
import unittest

from agent.handlers import _build_capacity_refusal, _UNFIXABLE_BY_DATE
from agent import handlers


def _refusal(reason, alternatives=(), call_sid="CA-CAP"):
    return {
        "success": False,
        "feasibility": {"feasible": False, "reasonCode": reason},
        "alternatives": [{"date": d} for d in alternatives],
    }


class CapacityRefusalTests(unittest.TestCase):
    """A full calendar is a business answer, not a system fault."""

    def setUp(self):
        handlers._call_contexts.clear()
        handlers.set_call_context("CA-CAP", "+15550000000", {})

    def tearDown(self):
        handlers._call_contexts.clear()

    def test_alternative_dates_are_offered(self):
        result = _build_capacity_refusal(
            "CA-CAP", _refusal("day_at_capacity", ["2026-09-10", "2026-09-11"]), "Tuesday"
        )
        self.assertEqual(result["error"], "date_unavailable")
        self.assertIn("September tenth", result["message"])
        self.assertIn("September eleventh", result["message"])

    def test_it_does_not_re_ask_a_question_the_caller_answered(self):
        """They already said which day they wanted."""
        result = _build_capacity_refusal(
            "CA-CAP", _refusal("day_at_capacity", ["2026-09-10"]), "Tuesday"
        )
        self.assertIn("do not ask", result["message"].lower())

    def test_alternatives_are_described_as_days_not_times(self):
        """The dashboard probes forward from tomorrow, keeping the same window."""
        result = _build_capacity_refusal(
            "CA-CAP", _refusal("slot_full", ["2026-09-10"]), "Tuesday"
        )
        self.assertIn("as days rather than times", result["message"])

    def test_no_crew_configured_never_offers_a_date(self):
        """No day works, so offering one would be a lie."""
        result = _build_capacity_refusal("CA-CAP", _refusal("no_trucks"), "Tuesday")
        self.assertEqual(result["error"], "no_crew_configured")
        self.assertIn("do not offer another day", result["message"].lower())

    def test_material_not_accepted_never_offers_a_date(self):
        result = _build_capacity_refusal("CA-CAP", _refusal("material_not_accepted"), "Tuesday")
        self.assertEqual(result["error"], "material_not_accepted")
        self.assertIn("do not offer another day", result["message"].lower())

    def test_both_unfixable_reasons_are_covered(self):
        self.assertEqual(_UNFIXABLE_BY_DATE, {"no_trucks", "material_not_accepted"})

    def test_a_date_fixable_reason_with_no_dates_still_stops(self):
        """The probe can come back empty even when another day would do."""
        result = _build_capacity_refusal("CA-CAP", _refusal("day_at_capacity"), "Tuesday")
        self.assertEqual(result["error"], "no_dates_available")

    def test_date_shopping_is_bounded(self):
        """Every POST creates another lead row, so this cannot loop."""
        first = _build_capacity_refusal(
            "CA-CAP", _refusal("day_at_capacity", ["2026-09-10"]), "Tuesday"
        )
        self.assertEqual(first["error"], "date_unavailable")
        second = _build_capacity_refusal(
            "CA-CAP", _refusal("day_at_capacity", ["2026-09-11"]), "Wednesday"
        )
        self.assertEqual(second["error"], "no_dates_available")

    def test_a_refusal_never_demands_a_transfer(self):
        """A fallback would mandate a handoff the after-hours gate refuses."""
        for reason in ("day_at_capacity", "no_trucks", "material_not_accepted"):
            with self.subTest(reason=reason):
                result = _build_capacity_refusal("CA-CAP", _refusal(reason, ["2026-09-10"]), "Tuesday")
                self.assertNotIn("fallback", result)

    def test_it_does_not_claim_the_system_is_broken(self):
        """Nothing is broken — the day is full. Saying otherwise is a lie."""
        result = _build_capacity_refusal(
            "CA-CAP", _refusal("day_at_capacity", ["2026-09-10"]), "Tuesday"
        )
        self.assertNotIn("trouble", result["message"].lower())
        self.assertNotIn("repeated", result["message"].lower())

    def test_the_capacity_counter_is_separate_from_system_failures(self):
        """Sharing it would let one outage plus one full day trip a handoff."""
        _build_capacity_refusal("CA-CAP", _refusal("day_at_capacity", ["2026-09-10"]), "Tuesday")
        failures = handlers._call_contexts["CA-CAP"].get("tool_failures", {})
        self.assertIn("create_booking_capacity", failures)
        self.assertNotIn("create_booking", failures)


if __name__ == "__main__":
    unittest.main()
