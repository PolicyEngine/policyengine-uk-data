"""Tests for stochastic variable generation in the data package.

These tests verify that:
1. Take-up rate parameters load correctly
2. Stochastic variables are generated with correct types and ranges
3. Generation is deterministic (seeded RNG)
4. Take-up rates produce plausible proportions
"""

import pytest
import numpy as np
from policyengine_uk_data.parameters import (
    load_take_up_rate,
    load_parameter,
)


class TestTakeUpRateParameters:
    """Test that take-up rate parameters load correctly."""

    def test_child_benefit_rate_loads(self):
        rate = load_take_up_rate("child_benefit", 2024)
        assert 0 < rate <= 1
        assert rate > 0.8  # Child benefit has high take-up

    def test_pension_credit_rate_loads(self):
        rate = load_take_up_rate("pension_credit", 2024)
        assert 0 < rate <= 1

    def test_universal_credit_rate_loads(self):
        rate = load_take_up_rate("universal_credit", 2024)
        assert 0 < rate <= 1

    def test_marriage_allowance_rate_loads(self):
        rate = load_take_up_rate("marriage_allowance", 2024)
        assert 0 < rate <= 1

    def test_child_benefit_opts_out_rate_loads(self):
        rate = load_take_up_rate("child_benefit_opts_out_rate", 2024)
        assert 0 <= rate <= 1

    def test_tax_free_childcare_rate_loads(self):
        rate = load_take_up_rate("tax_free_childcare", 2024)
        assert 0 < rate <= 1


class TestStochasticParameters:
    """Test that stochastic parameters load correctly."""

    def test_tv_ownership_rate_loads(self):
        rate = load_parameter("stochastic", "tv_ownership_rate", 2024)
        assert 0 < rate <= 1
        assert rate > 0.9  # Most households own TVs

    def test_tv_licence_evasion_rate_loads(self):
        rate = load_parameter("stochastic", "tv_licence_evasion_rate", 2024)
        assert 0 <= rate <= 1
        assert rate < 0.2  # Evasion rate should be low

    def test_tax_free_childcare_spend_routed_share_loads(self):
        share = load_parameter(
            "stochastic", "tax_free_childcare_spend_routed_share", 2024
        )
        assert 0 < share <= 1
        # HMRC's 2024-25 ratio: 7,715,605 monthly child-account observations
        # over 1,085,020 annual unique children, over twelve months.
        assert share == pytest.approx(0.593, abs=0.001)

    def test_tax_free_childcare_spend_routed_share_series_is_bounded(self):
        # Every published year must be a usable share; a value outside 0-1
        # would produce a negative or supra-statutory award downstream.
        for year in range(2018, 2027):
            share = load_parameter(
                "stochastic", "tax_free_childcare_spend_routed_share", year
            )
            assert 0 < share <= 1, f"{year}: {share}"

    def test_tax_free_childcare_spend_routed_share_launch_year(self):
        # Tax-Free Childcare launched part-way through 2017-18, so its first
        # year is much lower than the settled 0.58-0.59 of later years.
        assert load_parameter(
            "stochastic", "tax_free_childcare_spend_routed_share", 2017
        ) == pytest.approx(0.359, abs=0.001)

    def test_first_time_buyer_rate_loads(self):
        rate = load_parameter("stochastic", "first_time_buyer_rate", 2024)
        assert 0 <= rate <= 1


class TestSeededRandomness:
    """Test that stochastic generation is deterministic."""

    def test_same_seed_produces_same_results(self):
        """Using the same seed should produce identical results."""
        seed = 100
        n = 1000

        generator1 = np.random.default_rng(seed=seed)
        result1 = generator1.random(n)

        generator2 = np.random.default_rng(seed=seed)
        result2 = generator2.random(n)

        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different results."""
        n = 1000

        generator1 = np.random.default_rng(seed=100)
        result1 = generator1.random(n)

        generator2 = np.random.default_rng(seed=200)
        result2 = generator2.random(n)

        assert not np.array_equal(result1, result2)


class TestTakeUpProportions:
    """Test that take-up rates produce plausible proportions."""

    def test_take_up_produces_expected_proportion(self):
        """Simulated take-up should match the rate approximately."""
        rate = 0.7
        n = 10000
        generator = np.random.default_rng(seed=42)

        take_up = generator.random(n) < rate
        actual_proportion = take_up.mean()

        # Should be within 5 percentage points of the rate
        assert abs(actual_proportion - rate) < 0.05

    def test_boolean_generation(self):
        """Take-up decisions should be boolean."""
        rate = 0.5
        n = 100
        generator = np.random.default_rng(seed=42)

        take_up = generator.random(n) < rate

        assert take_up.dtype == bool
        assert set(take_up).issubset({True, False})
