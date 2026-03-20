"""Tests for post-calibration population rescaling (#217).

Verifies that the calibrated dataset's weighted population matches the
ONS target, rather than drifting ~6% high as it did before the fix.
"""

POPULATION_TARGET = 69.5  # ONS 2022-based projection for 2025, millions
TOLERANCE = 0.03  # 3% — was 7% before rescaling fix


def test_weighted_population_matches_ons_target(baseline):
    """Weighted UK population should be within 3% of the ONS target."""
    population = baseline.calculate("people", 2025).sum() / 1e6
    assert abs(population / POPULATION_TARGET - 1) < TOLERANCE, (
        f"Weighted population {population:.1f}M is >{TOLERANCE:.0%} "
        f"from ONS target {POPULATION_TARGET:.1f}M."
    )


def test_household_count_reasonable(baseline):
    """Total weighted households should be roughly 28-30M (ONS estimate)."""
    total_hh = baseline.calculate("household_weight", 2025).sum() / 1e6
    assert 25 < total_hh < 33, (
        f"Total weighted households {total_hh:.1f}M outside 25-33M range."
    )


def test_population_not_inflated(baseline):
    """Population should not exceed 72M (the pre-fix inflated level)."""
    population = baseline.calculate("people", 2025).sum() / 1e6
    assert population < 72, (
        f"Population {population:.1f}M exceeds 72M — rescaling may not be working."
    )


def test_country_populations_sum_to_uk(baseline):
    """England + Scotland + Wales + NI populations should sum to UK total."""
    people = baseline.calculate("people_in_household", 2025)
    country = baseline.calculate("country", map_to="household")

    uk_pop = people.sum()
    country_sum = sum(
        people[country == c].sum() for c in country.unique()
    )

    assert abs(country_sum / uk_pop - 1) < 0.001, (
        f"Country populations sum to {country_sum/1e6:.1f}M "
        f"but UK total is {uk_pop/1e6:.1f}M."
    )
