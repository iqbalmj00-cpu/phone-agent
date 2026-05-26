import unittest

from agent import handlers
from agent.prosody import format_address_for_speech, inject_prosody


class AddressVerificationTests(unittest.TestCase):
    def setUp(self):
        handlers._call_contexts.clear()
        self.call_sid = "CA-ADDRESS"
        handlers.set_call_context(
            self.call_sid,
            "+15559990000",
            {"city": "Houston", "state": "TX", "serviceArea": "Houston, Katy, Cypress"},
        )

    def tearDown(self):
        handlers._call_contexts.clear()

    def test_partial_street_address_uses_client_city_state_for_google_lookup(self):
        query, added_context = handlers.build_geocode_query(
            "3121 Main Street",
            {"city": "Houston", "state": "TX", "serviceArea": "Greater Houston Area"},
        )

        self.assertEqual(query, "3121 Main Street, Houston, TX")
        self.assertTrue(added_context)

    def test_address_with_city_or_zip_is_not_rewritten_to_primary_city(self):
        config = {"city": "Houston", "state": "TX", "serviceArea": "Houston, Katy, Cypress"}

        city_query, city_added = handlers.build_geocode_query("3121 Main Street, Katy", config)
        zip_query, zip_added = handlers.build_geocode_query("3121 Main Street 77494", config)

        self.assertEqual(city_query, "3121 Main Street, Katy")
        self.assertFalse(city_added)
        self.assertEqual(zip_query, "3121 Main Street 77494")
        self.assertFalse(zip_added)

    def test_street_name_without_house_number_is_not_treated_as_verifiable_address(self):
        self.assertFalse(handlers._contains_house_number("Main Street"))
        self.assertTrue(handlers._contains_house_number("3121 Main Street"))

    def test_geocode_component_extraction_includes_full_address_parts(self):
        result = {
            "address_components": [
                {"long_name": "3121", "short_name": "3121", "types": ["street_number"]},
                {"long_name": "Main Street", "short_name": "Main St", "types": ["route"]},
                {"long_name": "Houston", "short_name": "Houston", "types": ["locality"]},
                {"long_name": "Harris County", "short_name": "Harris County", "types": ["administrative_area_level_2"]},
                {"long_name": "Texas", "short_name": "TX", "types": ["administrative_area_level_1"]},
                {"long_name": "77008", "short_name": "77008", "types": ["postal_code"]},
            ]
        }

        self.assertEqual(
            handlers._extract_geocode_components(result),
            {
                "street_number": "3121",
                "route": "Main Street",
                "city": "Houston",
                "state": "TX",
                "state_long": "Texas",
                "county": "Harris County",
                "postal_code": "77008",
            },
        )

    def test_booking_uses_cached_formatted_address_instead_of_raw_partial_address(self):
        handlers.remember_address_verification(
            self.call_sid,
            "3121 Main Street",
            "3121 Main Street, Houston, TX 77008",
            "3 1 2 1 Main Street, Houston, Texas 7 7 0 0 8",
            "in_area",
            "address city matches configured service area",
        )

        address, verification = handlers.resolve_booking_address(self.call_sid, "3121 Main Street")

        self.assertEqual(address, "3121 Main Street, Houston, TX 77008")
        self.assertEqual(verification["service_area_status"], "in_area")

    def test_format_address_for_speech_reads_house_number_and_zip_as_digits(self):
        self.assertEqual(
            format_address_for_speech("3121 Main Street, Houston, TX 77008"),
            "3 1 2 1 Main Street, Houston, Texas 7 7 0 0 8",
        )

    def test_prosody_spells_address_digits_without_touching_non_address_numbers(self):
        self.assertEqual(
            inject_prosody("I've got 3121 Main Street, Houston, TX 77008."),
            "I've got <spell>3121</spell> Main Street, Houston, Texas <spell>77008</spell>.",
        )
        self.assertEqual(inject_prosody("That is a 10-yard dumpster for 7 days."), "That is a 10-yard dumpster for 7 days.")


if __name__ == "__main__":
    unittest.main()
