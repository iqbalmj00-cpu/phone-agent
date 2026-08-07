"""The agent must never send a time-slot label to the dashboard.

The two systems have different slot vocabularies. The dashboard has
MORNING / MIDDAY / EVENING and no "afternoon" at all, so the agent's
"Afternoon" was mapped onto MIDDAY: a caller told "1 PM to 4 PM" had a job
recorded as 11:00 AM - 2:00 PM, and that recorded value is what the customer
portal, the driver app, the reminder email and the agent's own later lookup
all read back.

Sending an explicit "HH:MM-HH:MM" range removes the translation step.
"""
import unittest

from agent import handlers


class SlotWireValueTests(unittest.TestCase):
    def test_named_slots_go_out_as_explicit_ranges(self):
        expected = {
            "morning": "08:00-11:00",
            "midday": "11:00-13:00",
            "afternoon": "13:00-16:00",
        }
        for slot_id, wire in expected.items():
            with self.subTest(slot=slot_id):
                slot = handlers._validate_time_slot(slot_id)
                self.assertIsNotNone(slot)
                self.assertEqual(handlers._slot_wire_value(slot), wire)

    def test_afternoon_never_leaves_as_a_label(self):
        # The specific regression: "Afternoon" reaching the dashboard becomes MIDDAY.
        slot = handlers._validate_time_slot("afternoon")
        self.assertNotEqual(handlers._slot_wire_value(slot), "Afternoon")
        self.assertEqual(handlers._slot_wire_value(slot), "13:00-16:00")

    def test_dynamic_ranges_pass_through_unchanged(self):
        slot = handlers._validate_time_slot("09:30-12:30")
        self.assertEqual(handlers._slot_wire_value(slot), "09:30-12:30")

    def test_all_day_dumpster_sentinel_is_preserved(self):
        # Not a time window. Turned into 08:00-17:00 it would read as MIDDAY.
        slot = {
            "label": "All Day", "start": "08:00", "end": "17:00",
            "start_hour": 8, "start_minute": 480, "end_minute": 1020,
            "period": "All Day",
        }
        self.assertEqual(handlers._slot_wire_value(slot), "All Day")

    def test_slot_table_still_matches_the_website(self):
        # The comment above TIME_SLOTS says it must match the website's
        # wizardData.ts. Nothing enforces that, so at least pin it here.
        self.assertEqual(
            {k: (v["start"], v["end"]) for k, v in handlers.TIME_SLOTS.items()},
            {
                "morning": ("08:00", "11:00"),
                "midday": ("11:00", "13:00"),
                "afternoon": ("13:00", "16:00"),
            },
        )


if __name__ == "__main__":
    unittest.main()
