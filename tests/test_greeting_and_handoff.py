"""The opening line, the handoff wording, and reading a number back.

The opening now asks for the caller's name, because a call that never reaches a
booking still needs a name to be recorded under. It asks two things at once,
which the prompt's one-question-at-a-time rule has to carve out explicitly, or
the model will drop half of it.

The greeting is spoken as a TTS frame before the model produces anything. The
prompt refers to it and never repeats the words, which is correct whether or not
the spoken line reaches the model's context: what makes a model greet twice is
being handed the words a second time. Both places that used to ask for a name
unconditionally now use the name already given.

Reading a phone number back needs no code. `inject_prosody` already spells every
number the model emits; what never happened was the model emitting one.
"""
import unittest

from agent.prompt import build_greeting, build_system_prompt
from agent.prosody import inject_prosody


CONFIG = {
    "agentName": "Sarah",
    "companyName": "Acme Hauling",
    "city": "Austin",
    "state": "TX",
    "serviceArea": "Greater Austin",
    "services": ["furniture"],
    "timezone": "America/Chicago",
    "businessDays": [1, 2, 3, 4, 5],
    "businessStart": 8,
    "businessEnd": 17,
}


class GreetingTests(unittest.TestCase):
    """The opening line, as amended.

    D11 fixed the wording and D20 said it shipped exactly as worded. It no
    longer does: the single 24-word sentence was split into three after a read
    of how it actually sounds — "this is Sarah, who do I have..." parsed for a
    beat as a description of Sarah rather than a new question, and a blank
    company name left a dangling "thanks for calling,". The owner approved the
    change; D20 is superseded on the wording, not on the intent, which was to
    capture the caller's name in the greeting. That intent is still tested below.
    """

    def test_the_greeting_is_the_agreed_wording(self):
        self.assertEqual(
            build_greeting("Acme Hauling", "Sarah"),
            "Thanks for calling Acme Hauling! This is Sarah. "
            "Who am I speaking with, and what can I help you with?",
        )

    def test_it_asks_who_the_caller_is(self):
        """The whole reason the line changed — no name, no lead."""
        self.assertIn("Who am I speaking with", build_greeting("Acme", "Sarah"))

    def test_a_blank_company_still_reads_as_a_sentence(self):
        greeting = build_greeting("", "Sarah")
        self.assertIn("Thanks for calling! This is Sarah.", greeting)
        self.assertNotIn(" ,", greeting)
        self.assertNotIn("calling,", greeting)

    def test_a_blank_agent_name_falls_back(self):
        self.assertIn("This is Sarah", build_greeting("Acme", ""))


class GreetingIsNotRepeatedTests(unittest.TestCase):
    """The model already has the greeting; the prompt must not hand it a second."""

    def test_the_prompt_does_not_contain_the_greeting_words(self):
        """The prompt may forbid the phrase; it must never supply the line."""
        rendered = build_system_prompt(CONFIG)
        self.assertNotIn(build_greeting("Acme Hauling", "Sarah"), rendered)
        self.assertNotIn("Who am I speaking with", rendered)

    def test_the_prompt_forbids_greeting_twice(self):
        rendered = build_system_prompt(CONFIG)
        self.assertIn("Never greet the caller again", rendered)

    def test_the_opening_is_carved_out_of_one_question_at_a_time(self):
        rendered = build_system_prompt(CONFIG)
        self.assertIn("ask only one question at a time", rendered)
        self.assertIn("THE OPENING IS ALREADY DONE", rendered)

    def test_answering_the_need_but_not_the_name_is_the_normal_path(self):
        self.assertIn(
            "Most callers answer the need and not the name", build_system_prompt(CONFIG)
        )


class NameIsNotAskedTwiceTests(unittest.TestCase):
    """Both name-ask sites defer to a name the caller already gave."""

    def test_booking_flow_uses_a_name_already_given(self):
        rendered = build_system_prompt(CONFIG)
        self.assertIn("if they already gave it, use it and do NOT ask again", rendered)
        self.assertNotIn('1. Start with their name: "Can I get your name?"', rendered)

    def test_the_website_block_does_not_re_ask_either(self):
        rendered = build_system_prompt(
            {
                **CONFIG,
                "smsEnabled": True,
                "twilioNumber": "+15125551234",
                "websiteUrl": "https://acme.com",
            }
        )
        self.assertNotIn("Can I start with your name?", rendered)


class ManagerHandoffTests(unittest.TestCase):
    """D5: the spoken line names a manager."""

    def test_the_handoff_line_says_manager(self):
        rendered = build_system_prompt(CONFIG)
        self.assertIn("Let me hand you off to a manager.", rendered)
        self.assertNotIn("I'd be happy to connect you with someone from our team.", rendered)

    def test_the_trigger_vocabulary_is_left_wide(self):
        """Narrowing this would make the agent miss callers who ask for "the owner"."""
        rendered = build_system_prompt(CONFIG)
        for word in ("manager", "supervisor", "owner", "boss", "the team", "the office"):
            with self.subTest(word=word):
                self.assertIn(word, rendered)


class PhoneReadbackTests(unittest.TestCase):
    """D9: read the number back when asked, and only when asked."""

    def test_the_prompt_allows_a_readback_on_request(self):
        rendered = build_system_prompt(CONFIG)
        self.assertIn("What number do you have for me?", rendered)
        self.assertIn("read the number back digit by digit", rendered)

    def test_the_prompt_does_not_invite_unprompted_digits(self):
        self.assertIn(
            "Do not volunteer digits at any other time.", build_system_prompt(CONFIG)
        )

    def test_the_pipeline_spells_whatever_the_model_emits(self):
        """Every format the dashboard or the caller can produce."""
        for number in (
            "+15125551234",
            "5125551234",
            "(512) 555-1234",
            "512-555-1234",
        ):
            with self.subTest(number=number):
                spoken = inject_prosody(f"I have {number} on file.")
                self.assertIn("<spell>512</spell>", spoken)
                self.assertIn("<spell>1234</spell>", spoken)
                self.assertNotIn("+1", spoken)


if __name__ == "__main__":
    unittest.main()
