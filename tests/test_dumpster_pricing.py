"""Every dumpster price the agent speaks has to exist in the client's config.

Three ways that was not true.

The prompt carried a hardcoded example — "a 20-yard runs $375 to $650" — that
was read aloud to clients whose real prices were nothing like it.

`included_days` and the extra-day rate were assigned inside the tier loop with
no reset, so after the loop they held values from different tiers. A client with
a 10-yard at $10/day for 3 days and a 40-yard for 14 days had the agent say
"$10 per extra day after the first 14 days" — a pair that existed in no tier.

And a tier whose base rate was left at 0 was quoted as "starting at $0", which
is the client's real configured pricing by the letter of the rule and a disaster
by its intent.

Separately, the two surfaces disagreed: the static table rendered the configured
price while the live availability tool rounded to the nearest $5, so a caller
could hear $372 from one and $370 from the other on the same call. Both now
speak the configured amount, which is also the amount the dashboard bills.
"""
import unittest

from agent.prompt import _format_dumpster_pricing


def _tier(size, rate=None, days=7, daily=None, rate_min=None, rate_max=None):
    tier = {"sizeCuYd": size, "includedDays": days}
    if rate is not None:
        tier["baseRate"] = rate
    if rate_min is not None:
        tier["baseRateMin"] = rate_min
    if rate_max is not None:
        tier["baseRateMax"] = rate_max
    if daily is not None:
        tier["extendedDailyRate"] = daily
    return tier


class CrossTierContaminationTests(unittest.TestCase):
    """A term and a rate may only be spoken together if one tier holds both."""

    MIXED = [
        _tier(10, rate=300, days=3, daily=10),
        _tier(40, rate=600, days=14),
    ]

    def test_a_rate_is_never_paired_with_another_tiers_term(self):
        *_, extension, _ = _format_dumpster_pricing(self.MIXED)
        self.assertNotIn("$10 per extra day after the first 14 days", extension)

    def test_disagreeing_tiers_send_the_agent_back_to_the_table(self):
        *_, extension, rental_period = _format_dumpster_pricing(self.MIXED)
        self.assertEqual(rental_period, "the included rental period")
        self.assertIn("THEIR size", extension)

    def test_agreeing_tiers_may_still_state_the_specific_pair(self):
        tiers = [
            _tier(10, rate=300, days=7, daily=10),
            _tier(20, rate=450, days=7, daily=10),
        ]
        *_, extension, rental_period = _format_dumpster_pricing(tiers)
        self.assertEqual(rental_period, "7 days")
        self.assertIn("$10 per extra day after the first 7 days", extension)

    def test_a_size_with_no_extra_day_rate_is_never_quoted_another_size_s(self):
        """Deduping the rates was not enough — every size must carry one.

        A 10-yard at $12/day beside a 20-yard with no configured rate left one
        distinct value in the set, so a 20-yard renter was told "$12 per extra
        day" — a price their size does not have.
        """
        tiers = [
            _tier(10, rate=300, days=7, daily=12),
            _tier(20, rate=450, days=7),
        ]
        *_, extension, _ = _format_dumpster_pricing(tiers)
        self.assertNotIn("$12 per extra day", extension)
        self.assertIn("THEIR size", extension)

    def test_the_stated_pair_belongs_to_a_real_tier(self):
        """Same term, different daily rates — no single pair is safe to state."""
        tiers = [
            _tier(10, rate=300, days=7, daily=10),
            _tier(40, rate=800, days=7, daily=25),
        ]
        *_, extension, _ = _format_dumpster_pricing(tiers)
        self.assertNotIn("$10 per extra day", extension)
        self.assertNotIn("$25 per extra day", extension)


class HalfConfiguredTierTests(unittest.TestCase):
    """A missing price is not a price of zero."""

    def test_a_zero_rate_tier_is_not_quoted(self):
        block, *_ = _format_dumpster_pricing(
            [_tier(20, rate=0), _tier(30, rate=500)]
        )
        self.assertNotIn("$0", block)
        self.assertIn("30-yard", block)
        self.assertNotIn("20-yard", block)

    def test_all_tiers_unpriced_falls_back_to_team_follow_up(self):
        block, *_ = _format_dumpster_pricing([_tier(20, rate=0)])
        self.assertIn("our team must confirm the availability and price before it is scheduled", block)

    def test_no_tiers_falls_back_to_team_follow_up(self):
        block, *_ = _format_dumpster_pricing([])
        self.assertIn("our team must confirm the availability and price before it is scheduled", block)


class ConfiguredAmountsTests(unittest.TestCase):
    """Every amount spoken is the amount configured, and the amount billed.

    Prices used to be rounded to the nearest $5 for a cleaner read. That is
    invisible on a $475 base rate and misleading on a $12 daily rate, and the
    dashboard bills the configured figure either way — so a caller could be
    quoted $10 a day and charged $12.
    """

    def test_base_rates_are_spoken_as_configured(self):
        block, *_ = _format_dumpster_pricing(
            [_tier(30, rate_min=372, rate_max=648, days=7)]
        )
        self.assertIn("$372", block)
        self.assertIn("$648", block)

    def test_extra_day_rates_are_spoken_as_configured(self):
        block, *_, extension, _ = _format_dumpster_pricing(
            [_tier(10, rate=300, daily=12)]
        )
        self.assertIn("$12/day", block)
        self.assertIn("$12 per extra day", extension)

    def test_swap_fees_are_spoken_as_configured(self):
        block, _, spoken, *_ = _format_dumpster_pricing(
            [_tier(10, rate=300)], swap_fee=123
        )
        self.assertIn("$123 flat", block)
        self.assertIn("A swap is a flat $123", spoken)

    def test_a_small_but_real_price_is_still_quoted(self):
        """Only a missing price is suppressed, never a low one."""
        block, *_ = _format_dumpster_pricing([_tier(20, rate=2, daily=2)])
        self.assertIn("$2", block)
        self.assertNotIn("$0", block)

    def test_the_live_tool_is_authoritative_for_a_specific_date(self):
        """A range tier shows two numbers; the tool returns one."""
        _, instruction, *_ = _format_dumpster_pricing(
            [_tier(20, rate_min=425, rate_max=650)]
        )
        self.assertIn("check_container_availability is authoritative", instruction)
        self.assertNotIn("never disagree", instruction)


class ConfiguredPricingOnlyTests(unittest.TestCase):
    """D18: the example must come from the client, or not exist."""

    def test_no_hardcoded_example_survives(self):
        _, instruction, *_ = _format_dumpster_pricing([_tier(10, rate=300)])
        self.assertNotIn("A 20-yard runs $375", instruction)

    def test_a_range_tier_supplies_the_example(self):
        _, instruction, *_ = _format_dumpster_pricing(
            [_tier(20, rate_min=400, rate_max=700, days=5)]
        )
        self.assertIn("A 20-yard runs $400 to $700 for the first 5 days", instruction)

    def test_a_single_starting_at_tier_gives_no_example(self):
        """`baseRateMax` is nullable; the ordinary row is not a range."""
        _, instruction, *_ = _format_dumpster_pricing([_tier(10, rate=300)])
        self.assertNotIn("For example", instruction)


if __name__ == "__main__":
    unittest.main()
