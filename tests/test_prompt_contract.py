"""The prompt templates' placeholders and their render kwargs must stay in step.

`build_system_prompt` renders a 138-line template through `str.format`, plus
three nested templates: two for the dumpster sections and one for the website. A placeholder with no matching
kwarg raises `KeyError` at render time, and that happens before the greeting --
the caller hears nothing and the call dies. A kwarg with no placeholder is the
quiet half of the same fault: `str.format` drops it without a word, so a whole
section can stop reaching the model with no visible symptom.

Every prompt edit touches these templates. This file is the wiring guard for
that work. It checks all four format contracts in both directions, and pins the
config shapes that render today so a later edit cannot narrow them.

It deliberately does not assert what the prompt *says*. Those tests belong with
the changes that alter the wording.
"""
import contextlib
import string
import unittest
from zoneinfo import ZoneInfoNotFoundError

from agent import prompt
from agent.prompt import build_system_prompt


class _AnyValue:
    """Stands in for a config value so a probe render can run to completion."""

    def __format__(self, format_spec):
        return ""

    def __getitem__(self, key):
        return self

    def __getattr__(self, name):
        return self


class _FieldProbe(dict):
    """Records every field name a template asks `str.format` for.

    `string.Formatter().parse()` is not enough on its own. It reports the field
    of `{n:{w}}` as `n` and never mentions `w`, even though a missing `w` is
    exactly what raises at render time, and it reports `{cfg[city]}` as the
    single name `cfg[city]`. Driving a real `vformat` records what the template
    genuinely asks for, nested specs and item lookups included.
    """

    def __init__(self):
        super().__init__()
        self.asked = set()

    def __missing__(self, key):
        self.asked.add(key)
        return _AnyValue()


def _asked_fields(template):
    probe = _FieldProbe()
    string.Formatter().vformat(template, (), probe)
    return probe.asked


class _RecordingTemplate(str):
    """A template that remembers the arguments its `.format()` was called with."""

    def __new__(cls, value):
        template = super().__new__(cls, value)
        template.recorded = None
        template.positional = None
        return template

    def format(self, *args, **kwargs):
        self.recorded = set(kwargs)
        self.positional = len(args)
        return str.format(str(self), *args, **kwargs)


@contextlib.contextmanager
def _recording(attr_name):
    """Swap a module-level template for one that records how it was rendered."""
    original = getattr(prompt, attr_name)
    template = _RecordingTemplate(original)
    setattr(prompt, attr_name, template)
    try:
        yield template
    finally:
        setattr(prompt, attr_name, original)


# Shaped like what `api/agent/config/[clientId]/route.ts` actually sends.
JUNK_ONLY = {
    "agentName": "Sarah",
    "companyName": "Junk Removal",
    "city": "Austin",
    "state": "TX",
    "serviceArea": "Greater Austin",
    "services": ["furniture", "appliances"],
    "timezone": "America/Chicago",
    "businessDays": [1, 2, 3, 4, 5],
    "businessStart": 7,
    "businessEnd": 19,
}

DUMPSTER_WITH_TIERS = {
    **JUNK_ONLY,
    "dumpsterRentalsEnabled": True,
    "dumpsterPricing": [
        {"sizeCuYd": 10, "baseRate": 300, "includedDays": 7, "extendedDailyRate": 10},
        {"sizeCuYd": 20, "baseRate": 450, "baseRateMax": 650, "includedDays": 7},
    ],
    "swapOutFee": 125,
}

DUMPSTER_NO_TIERS = {
    **JUNK_ONLY,
    "dumpsterRentalsEnabled": True,
    "dumpsterPricing": [],
}

# The website block renders only when SMS, a Twilio number and a speakable URL
# are all configured.
WEBSITE_ENABLED = {
    **JUNK_ONLY,
    "smsEnabled": True,
    "twilioNumber": "+15125551234",
    "websiteUrl": "https://www.example.com/",
}


class TemplateContractTests(unittest.TestCase):
    """The four format contracts, checked in both directions.

    Each nested template renders only for the client shape that enables it, so
    each one needs the config that switches it on.
    """

    def _assert_contract(self, attr_name, config):
        with _recording(attr_name) as template:
            build_system_prompt(config)
        self.assertIsNotNone(
            template.recorded, f"{attr_name} was never rendered by this config"
        )
        self.assertEqual(
            template.positional, 0, f"{attr_name} gained a positional field"
        )
        self.assertEqual(_asked_fields(getattr(prompt, attr_name)), template.recorded)

    def test_system_prompt_template_matches_its_render_kwargs(self):
        self._assert_contract("SYSTEM_PROMPT_TEMPLATE", DUMPSTER_WITH_TIERS)

    def test_dumpster_company_info_matches_its_render_kwargs(self):
        self._assert_contract("DUMPSTER_COMPANY_INFO", DUMPSTER_WITH_TIERS)

    def test_dumpster_booking_flow_matches_its_render_kwargs(self):
        self._assert_contract("DUMPSTER_BOOKING_FLOW", DUMPSTER_WITH_TIERS)

    def test_website_upsell_matches_its_render_kwargs(self):
        self._assert_contract("WEBSITE_UPSELL", WEBSITE_ENABLED)

    def test_no_placeholder_survives_a_render(self):
        """A section injected without its own `.format()` call shows up here.

        Every config below is brace-free, so any brace in the output came from
        the templates -- `str.format` does not recurse into the values it
        substitutes, so a config value cannot put one there.
        """
        for label, config in (
            ("empty config", {}),
            ("junk only", JUNK_ONLY),
            ("dumpster with tiers", DUMPSTER_WITH_TIERS),
            ("dumpster no tiers", DUMPSTER_NO_TIERS),
            ("website enabled", WEBSITE_ENABLED),
        ):
            with self.subTest(config=label):
                rendered = build_system_prompt(config)
                self.assertNotIn("{", rendered)
                self.assertNotIn("}", rendered)


class RenderMatrixTests(unittest.TestCase):
    """Config shapes the dashboard can send today must all render."""

    def test_supported_config_shapes_render(self):
        for label, config in (
            ("empty config", {}),
            ("junk only", JUNK_ONLY),
            ("dumpster with tiers", DUMPSTER_WITH_TIERS),
            ("dumpster no tiers", DUMPSTER_NO_TIERS),
            ("no services", {**JUNK_ONLY, "services": []}),
            ("services is None", {**JUNK_ONLY, "services": None}),
            ("no business days", {**JUNK_ONLY, "businessDays": []}),
            ("non-contiguous days", {**JUNK_ONLY, "businessDays": [1, 3, 5]}),
            (
                "blank company fields",
                {**JUNK_ONLY, "city": "", "state": "", "serviceArea": ""},
            ),
            ("website url", {**JUNK_ONLY, "websiteUrl": "https://example.com"}),
            (
                "sms configured",
                {**JUNK_ONLY, "smsEnabled": True, "twilioNumber": "+15125551234"},
            ),
            ("zero swap fee", {**DUMPSTER_WITH_TIERS, "swapOutFee": 0}),
        ):
            with self.subTest(config=label):
                rendered = build_system_prompt(config)
                self.assertIsInstance(rendered, str)
                self.assertTrue(rendered.strip())


class MalformedConfigTests(unittest.TestCase):
    """Shapes that kill the render today, pinned so drift surfaces here first.

    None of these reach the agent right now. The dashboard coerces hours and
    days before sending them -- `src/lib/business-hours.ts` runs both hours
    through `clampHour` and filters days to integers 0-6 -- and both `timezone`
    columns are non-null with defaults. These assertions record that
    `build_system_prompt` leans on that coercion. If the dashboard ever stops
    doing it, these tests fail instead of every call for that client dying
    before the greeting.
    """

    def test_non_integer_hours_and_days_raise(self):
        for label, config in (
            ("businessStart is None", {"businessStart": None}),
            ("businessStart is a string", {"businessStart": "7"}),
            ("businessEnd is None", {"businessEnd": None}),
            ("businessDays is None", {"businessDays": None}),
            ("businessDays is a string", {"businessDays": "Mon"}),
            ("businessDays holds strings", {"businessDays": ["1"]}),
        ):
            with self.subTest(config=label):
                with self.assertRaises(TypeError):
                    build_system_prompt(config)

    def test_unusable_timezone_raises(self):
        for label, config, expected in (
            ("timezone is None", {"timezone": None}, TypeError),
            ("timezone is blank", {"timezone": ""}, ValueError),
            ("timezone is not an IANA key", {"timezone": "Not/AZone"}, ZoneInfoNotFoundError),
        ):
            with self.subTest(config=label):
                with self.assertRaises(expected):
                    build_system_prompt(config)


class DumpsterPricingContractTests(unittest.TestCase):
    """`_format_dumpster_pricing` feeds a five-value unpack at its call site.

    It has two return paths -- an early one for a client with no configured
    tiers, and the main one -- and the unpack sees whichever fires. Both are
    pinned so a change to one cannot ship green on the strength of the other.
    """

    def test_both_return_paths_give_five_values(self):
        for label, tiers in (
            ("no tiers", []),
            ("one tier", [{"sizeCuYd": 10, "baseRate": 300, "includedDays": 7}]),
            ("several tiers", DUMPSTER_WITH_TIERS["dumpsterPricing"]),
        ):
            with self.subTest(tiers=label):
                self.assertEqual(len(prompt._format_dumpster_pricing(tiers)), 5)


if __name__ == "__main__":
    unittest.main()
