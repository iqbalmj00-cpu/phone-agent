"""What the prompt says about the business has to be true, or absent.

Two faults sat in the rendered prompt, and both reached the caller as speech.

An operator who has not filled in a field gets an empty string from the
dashboard, not a missing key, so the prompt rendered the punctuation around the
gap: "a receptionist at Junk Removal in , .", "We cover the whole .", and a bare
"- Services: ". The fix omits the whole clause instead, so the agent says less
rather than saying something malformed.

Separately, an operator open Monday, Wednesday and Friday was described as
"Monday through Friday" at four places in the prompt, because the label was
built from the first and last open day with no contiguity check. The agent would
offer to book a Tuesday.
"""
import unittest

from agent.prompt import build_system_prompt
from agent.prosody import inject_prosody


def _config(**overrides):
    """The shape `api/agent/config/[clientId]` sends: every key present."""
    base = {
        "agentName": "Sarah",
        "companyName": "Junk Removal",
        "city": "",
        "state": "",
        "serviceArea": "",
        "services": [],
        "timezone": "America/Chicago",
        "businessDays": [1, 2, 3, 4, 5],
        "businessStart": 8,
        "businessEnd": 17,
    }
    base.update(overrides)
    return base


CONFIGURED = _config(
    companyName="Hauling Co",
    city="Austin",
    state="TX",
    serviceArea="Greater Austin",
    services=["furniture", "appliances"],
)


class EmptyFieldRenderingTests(unittest.TestCase):
    """An unset field costs the caller a clause, never a broken sentence."""

    def test_blank_config_speaks_no_fragments(self):
        rendered = build_system_prompt(_config())
        for fragment in (", .", "in , ", "whole .", "- Services: \n", "removal, \n"):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, rendered)

    def test_blank_city_and_state_drop_the_location_clause(self):
        rendered = build_system_prompt(_config())
        self.assertIn("receptionist at Junk Removal.", rendered)

    def test_blank_service_area_drops_both_of_its_clauses(self):
        rendered = build_system_prompt(_config())
        self.assertIn("full-service junk removal\n", rendered)
        self.assertNotIn("We cover the whole", rendered)

    def test_a_dumpster_client_with_no_services_drops_the_services_line(self):
        rendered = build_system_prompt(
            _config(dumpsterRentalsEnabled=True, dumpsterPricing=[])
        )
        self.assertNotIn("- Services:", rendered)

    def test_services_none_never_reaches_the_caller_as_a_word(self):
        rendered = build_system_prompt(_config(services=None))
        self.assertNotIn("Services: None", rendered)

    def test_city_without_state_still_reads_as_a_sentence(self):
        rendered = build_system_prompt(_config(city="Austin"))
        self.assertIn("receptionist at Junk Removal in Austin.", rendered)

    def test_whitespace_only_fields_count_as_empty(self):
        """The dashboard stores what was typed, spaces included."""
        rendered = build_system_prompt(
            _config(city="   ", state="  ", serviceArea="  ")
        )
        self.assertIn("receptionist at Junk Removal.", rendered)
        self.assertNotIn("We cover the whole", rendered)

    def test_the_company_and_agent_names_are_spoken_in_sentences(self):
        """Both are read aloud, so neither may render blank or as "None"."""
        for label, config in (
            ("companyName is None", _config(companyName=None)),
            ("companyName is blank", _config(companyName="  ")),
            ("agentName is blank", _config(agentName="")),
        ):
            with self.subTest(config=label):
                first_line = build_system_prompt(config).split("\n")[0]
                self.assertNotIn("None", first_line)
                self.assertNotIn("You are ,", first_line)
                self.assertNotIn("at .", first_line)

    def test_a_configured_client_keeps_every_clause(self):
        rendered = build_system_prompt(CONFIGURED)
        self.assertIn("receptionist at Hauling Co in Austin, TX.", rendered)
        self.assertIn("full-service junk removal, Greater Austin", rendered)
        self.assertIn("- Services: furniture, appliances", rendered)
        self.assertIn('We cover the whole Greater Austin."', rendered)


class BusinessDaysLabelTests(unittest.TestCase):
    """The days the agent names must be the days the operator actually works."""

    def _hours_line(self, config):
        for line in build_system_prompt(config).split("\n"):
            if line.startswith("- Hours: "):
                return line
        self.fail("no hours line in the rendered prompt")

    def test_non_contiguous_days_are_listed_not_bridged(self):
        line = self._hours_line(_config(businessDays=[1, 3, 5]))
        self.assertIn("Mondays, Wednesdays and Fridays", line)
        self.assertNotIn("Monday through Friday", line)

    def test_contiguous_days_still_read_as_a_range(self):
        self.assertIn("Monday through Friday", self._hours_line(_config()))

    def test_a_single_open_day_names_only_that_day(self):
        line = self._hours_line(_config(businessDays=[6]))
        self.assertIn("Saturdays only", line)
        self.assertNotIn("through", line)

    def test_per_day_hours_win_over_the_day_numbers(self):
        """`businessHours` is the richer field, and the dashboard sends it."""
        open_days = {"monday", "wednesday", "friday"}
        business_hours = {
            day: {
                "open": "08:00" if day in open_days else None,
                "close": "17:00" if day in open_days else None,
                "closed": day not in open_days,
            }
            for day in (
                "sunday", "monday", "tuesday", "wednesday",
                "thursday", "friday", "saturday",
            )
        }
        line = self._hours_line(
            _config(businessHours=business_hours, businessDays=[1, 2, 3, 4, 5])
        )
        self.assertIn("Mondays, Wednesdays and Fridays", line)

    def test_repeated_day_numbers_do_not_repeat_the_day(self):
        """`businessDays` is stored unfiltered, so it can arrive with repeats."""
        line = self._hours_line(_config(businessDays=[1, 1, 2, 3, 4, 5, 6]))
        self.assertIn("Monday through Saturday", line)
        self.assertNotIn("Monday, Monday", line)

    def test_unsorted_day_numbers_still_read_in_order(self):
        line = self._hours_line(_config(businessDays=[6, 0, 1, 2, 3, 4, 5]))
        self.assertIn("Sunday through Saturday", line)

    def test_every_site_that_names_the_days_agrees(self):
        """Five sites render the label, and none may disagree.

        COMPANY INFO, the scheduling rule, the closed-day reply, the hours
        example that used to be hardcoded, and the handoff rule that tells the
        agent when a transfer can actually connect.
        """
        rendered = build_system_prompt(_config(businessDays=[1, 3, 5]))
        self.assertEqual(rendered.count("Mondays, Wednesdays and Fridays"), 5)
        self.assertNotIn("Monday through Friday", rendered)
        self.assertNotIn("Monday through Saturday, 8 AM to 6 PM", rendered)


class WebsiteMentionTests(unittest.TestCase):
    """The agent could not say the website at all — it had no placeholder for it.

    `WEBSITE_UPSELL` carried zero placeholders, so even when it rendered it told
    the caller to "search for our company name online". It also promised "after
    we hang up, I'll text you the link", which only the agent's own follow-up
    SMS ever sent.
    """

    WITH_SITE = _config(websiteUrl="https://www.acmehauling.com/")
    FULL = _config(
        websiteUrl="https://www.acmehauling.com/",
        smsEnabled=True,
        twilioNumber="+15125551234",
    )

    def test_the_agent_can_say_the_website(self):
        rendered = build_system_prompt(self.WITH_SITE)
        self.assertIn("- Website: acmehauling.com", rendered)

    def test_answering_the_website_question_does_not_need_sms(self):
        """D1 is about speech, so it cannot depend on texting being set up."""
        rendered = build_system_prompt(self.WITH_SITE)
        self.assertIn('say "acmehauling.com"', rendered)

    def test_no_website_configured_says_nothing_about_one(self):
        rendered = build_system_prompt(_config(websiteUrl=None))
        self.assertNotIn("- Website:", rendered)
        self.assertNotIn("acmehauling", rendered)

    def test_an_unspeakable_url_is_treated_as_no_url(self):
        rendered = build_system_prompt(_config(websiteUrl="not a url"))
        self.assertNotIn("- Website:", rendered)

    def test_the_website_is_never_promised_as_a_text(self):
        rendered = build_system_prompt(self.FULL)
        self.assertNotIn("I'll text you the link", rendered)
        self.assertNotIn("texted you a link", rendered)
        self.assertNotIn("search for our company name online", rendered)

    def test_the_website_is_offered_as_a_second_path_not_a_replacement(self):
        rendered = build_system_prompt(self.WITH_SITE)
        self.assertIn("never as a replacement for the on-site quote", rendered)
        self.assertIn("crew gives you the exact price on site", rendered)

    def test_the_agent_may_never_state_a_junk_price_itself(self):
        """The invariant holds whether or not a website is configured."""
        for label, config in (
            ("with website", self.WITH_SITE),
            ("without website", _config()),
        ):
            with self.subTest(config=label):
                self.assertIn(
                    "Never state a junk removal price yourself",
                    build_system_prompt(config),
                )


class CompanyPhoneTests(unittest.TestCase):
    """The agent can say the company's number, and only the public one.

    Two numbers reach the agent from `api/agent/config/[clientId]`.
    `twilioNumber` is the line the caller just dialled — the inbound webhook
    resolves the tenant from it — so it is already public. `forwardingPhone`
    is the human-handoff fallback, resolved from the operator's own forwarding
    or profile number and deliberately never the Twilio one, so it is often a
    personal mobile. Only the first may ever be spoken.
    """

    WITH_NUMBER = _config(twilioNumber="+15125551234")
    BOTH_NUMBERS = _config(
        twilioNumber="+15125551234", forwardingPhone="+15129998888"
    )

    def test_the_agent_can_say_the_company_number(self):
        rendered = build_system_prompt(self.WITH_NUMBER)
        self.assertIn("- Main line: (512) 555-1234", rendered)

    def test_the_number_is_asked_for_and_answered_in_the_scenarios(self):
        rendered = build_system_prompt(self.WITH_NUMBER)
        self.assertIn('"what\'s your number?"', rendered)
        self.assertIn("it's (512) 555-1234", rendered)

    def test_e164_is_reshaped_into_the_form_a_person_writes(self):
        """The model repeats what it reads, and "+1" gets paraphrased."""
        rendered = build_system_prompt(self.WITH_NUMBER)
        self.assertNotIn("+15125551234", rendered)

    def test_the_rendered_number_survives_the_prosody_pass(self):
        """Rendering a form `inject_prosody` cannot match wastes the readback."""
        spoken = inject_prosody("Sure, it's (512) 555-1234.")
        self.assertIn("<spell>512</spell>", spoken)
        self.assertIn("<spell>555</spell>", spoken)
        self.assertIn("<spell>1234</spell>", spoken)

    def test_no_number_configured_says_nothing_about_one(self):
        for label, config in (
            ("key absent", _config()),
            ("twilioNumber is None", _config(twilioNumber=None)),
        ):
            with self.subTest(config=label):
                rendered = build_system_prompt(config)
                self.assertNotIn("- Main line:", rendered)
                self.assertNotIn("the same line they reached you on", rendered)

    def test_a_blank_number_drops_the_whole_clause(self):
        """A field the operator never filled arrives as "", not as missing."""
        for label, value in (("empty string", ""), ("whitespace", "   ")):
            with self.subTest(value=label):
                rendered = build_system_prompt(_config(twilioNumber=value))
                self.assertNotIn("- Main line:", rendered)
                self.assertNotIn("the only number you hand out", rendered)

    def test_an_unusable_number_is_treated_as_no_number(self):
        """Anything the pipeline cannot spell is worse than silence.

        A long value is read out raw, and a short one is regrouped into a
        different number said with complete confidence.
        """
        for label, value in (
            ("not north american", "+442071234567"),
            ("a digit short", "+1512555123"),
            ("not a number at all", "call us"),
            ("carries an extension", "+15125551234x99"),
        ):
            with self.subTest(value=label):
                rendered = build_system_prompt(_config(twilioNumber=value))
                self.assertNotIn("- Main line:", rendered)
                self.assertNotIn(value, rendered)

    def test_the_handoff_number_is_never_spoken(self):
        """The one that may be a personal mobile. This is the safety case."""
        rendered = build_system_prompt(self.BOTH_NUMBERS)
        for fragment in ("+15129998888", "(512) 999-8888", "9998888", "999-8888"):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, rendered)

    def test_the_handoff_number_is_not_a_fallback_for_a_missing_one(self):
        """With no public number, the agent stays quiet rather than substituting."""
        rendered = build_system_prompt(_config(forwardingPhone="+15129998888"))
        self.assertNotIn("- Main line:", rendered)
        self.assertNotIn("999-8888", rendered)

    def test_the_digit_rule_covers_the_company_number_too(self):
        """The readback carve-out is one rule, not two that contradict."""
        rendered = build_system_prompt(self.WITH_NUMBER)
        self.assertIn("or the company's own if COMPANY INFO lists a main line", rendered)
        self.assertIn("Do not volunteer digits at any other time.", rendered)


if __name__ == "__main__":
    unittest.main()
