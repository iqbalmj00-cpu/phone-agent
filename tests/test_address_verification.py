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
                # Google supplies this only for addresses that have a unit.
                "subpremise": "",
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


class AddressUnitTests(unittest.TestCase):
    """Apartment callers used to lose their unit number.

    Google's `formatted_address` almost never carries a unit, and
    `resolve_booking_address` substitutes it for what the caller actually said,
    so the crew was dispatched to the building with no way to find them.
    """

    def setUp(self):
        handlers._call_contexts.clear()
        self.call_sid = "CA-UNIT"
        handlers.set_call_context(
            self.call_sid,
            "+15559990000",
            {"city": "Houston", "state": "TX", "serviceArea": "Houston"},
        )

    def tearDown(self):
        handlers._call_contexts.clear()

    def test_extracts_unit_from_spoken_address(self):
        cases = {
            "3121 Main Street apartment 4B": "apartment 4B",
            "3121 Main Street Apt 4b": "Apt 4b",
            "3121 Main Street, unit 12": "unit 12",
            "3121 Main Street suite 200": "suite 200",
            "3121 Main Street #3": "#3",
            "3121 Main Street unit B": "unit B",
        }
        for address, expected in cases.items():
            with self.subTest(address=address):
                self.assertEqual(handlers.extract_address_unit(address), expected)

    def test_ignores_street_names_that_look_like_unit_prefixes(self):
        for address in ("3121 Main Street", "500 Building Road", "77 Floorwood Lane"):
            with self.subTest(address=address):
                self.assertEqual(handlers.extract_address_unit(address), "")

    def test_a_state_abbreviation_is_not_a_floor(self):
        """"FL 33101" is Florida, not floor 33101.

        This one corrupted the street line rather than merely losing the unit:
        the bogus unit was folded back in, so the caller heard
        "123 Main St FL 33101, Miami, FL 33130" read back and the crew got it.
        """
        for address in (
            "123 Main St, Miami, FL 33101",
            "500 Ocean Dr, Tampa, FL 33602",
            "1 Elm St, Orlando, Florida 32801",
        ):
            with self.subTest(address=address):
                self.assertEqual(handlers.extract_address_unit(address), "")

    def test_a_street_starting_with_a_prefix_does_not_steal_the_real_unit(self):
        """re.search takes the leftmost match, so "Unity" beat the genuine "Apt 3"."""
        cases = {
            "88 Unity Ave Apt 3": "Apt 3",
            "12 Fly Rd Unit 7": "Unit 7",
            "9 Roomy Lane Suite 200": "Suite 200",
        }
        for address, expected in cases.items():
            with self.subTest(address=address):
                self.assertEqual(handlers.extract_address_unit(address), expected)

    def test_compose_does_not_duplicate_a_unit_google_already_rendered(self):
        """Google writes a subpremise as "#4"; the old literal check missed it
        and appended "Unit 4" alongside."""
        self.assertEqual(
            handlers.compose_address("123 Main St #4, Houston, TX 77002", "4"),
            "123 Main St #4, Houston, TX 77002",
        )
        self.assertEqual(
            handlers.compose_address("123 Main St Apt 4, Houston, TX", "4"),
            "123 Main St Apt 4, Houston, TX",
        )

    def test_compose_inserts_unit_before_first_comma(self):
        self.assertEqual(
            handlers.compose_address("3121 Main St, Houston, TX 77008", "Apt 4B"),
            "3121 Main St Apt 4B, Houston, TX 77008",
        )

    def test_compose_adds_unit_prefix_when_caller_gave_a_bare_value(self):
        self.assertEqual(
            handlers.compose_address("3121 Main St, Houston, TX", "4B"),
            "3121 Main St Unit 4B, Houston, TX",
        )

    def test_compose_is_a_no_op_without_a_unit_and_never_duplicates(self):
        self.assertEqual(handlers.compose_address("3121 Main St", ""), "3121 Main St")
        self.assertEqual(handlers.compose_address("3121 Main St", None), "3121 Main St")
        self.assertEqual(
            handlers.compose_address("3121 Main St Apt 4B, Houston", "Apt 4B"),
            "3121 Main St Apt 4B, Houston",
        )

    def test_geocode_extraction_returns_googles_subpremise(self):
        result = {
            "address_components": [
                {"long_name": "3121", "short_name": "3121", "types": ["street_number"]},
                {"long_name": "Main Street", "short_name": "Main St", "types": ["route"]},
                {"long_name": "4B", "short_name": "4B", "types": ["subpremise"]},
            ]
        }
        self.assertEqual(handlers._extract_geocode_components(result)["subpremise"], "4B")

    def test_booking_address_keeps_the_unit_google_dropped(self):
        handlers.remember_address_verification(
            self.call_sid,
            "3121 Main Street apartment 4B",
            "3121 Main Street, Houston, TX 77008",
            "3 1 2 1 Main Street, Houston, Texas 7 7 0 0 8",
            "in_area",
            "address city matches configured service area",
            unit="apartment 4B",
        )

        address, _ = handlers.resolve_booking_address(self.call_sid, "3121 Main Street apartment 4B")

        self.assertEqual(address, "3121 Main Street apartment 4B, Houston, TX 77008")

    def test_unit_is_recovered_from_the_raw_address_when_none_was_stored(self):
        # Covers a verification cached before the unit was captured.
        handlers.remember_address_verification(
            self.call_sid,
            "3121 Main Street apartment 4B",
            "3121 Main Street, Houston, TX 77008",
            "spoken",
            "in_area",
            "reason",
        )

        address, _ = handlers.resolve_booking_address(self.call_sid, "3121 Main Street apartment 4B")

        self.assertEqual(address, "3121 Main Street apartment 4B, Houston, TX 77008")


if __name__ == "__main__":
    unittest.main()
