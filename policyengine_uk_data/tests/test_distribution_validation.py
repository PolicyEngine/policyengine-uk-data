"""Single-age and HBAI distribution checks (utils/distribution_validation.py)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.utils.distribution_validation import (
    SCHEMA,
    TOP_AGE_BAND,
    compare,
    hbai_measures,
    load_hbai_reference,
    render,
    single_ages,
    summarise_single_ages,
    weighted_gini,
    weighted_median,
)

samples = st.lists(
    st.tuples(st.floats(-1e4, 1e5, allow_nan=False), st.integers(1, 5)),
    min_size=1,
    max_size=40,
)


def _expand(pairs):
    return np.repeat([v for v, _ in pairs], [w for _, w in pairs]).astype(float)


@settings(max_examples=300, deadline=None)
@given(pairs=samples)
def test_weighted_median_equals_the_median_of_the_expanded_sample(pairs):
    """Differential: integer weights are repeated records."""
    values = np.array([v for v, _ in pairs])
    weights = np.array([w for _, w in pairs], float)
    expanded = np.sort(_expand(pairs))
    # The lower median of the expanded sample.
    lower = expanded[(len(expanded) - 1) // 2]
    upper = expanded[len(expanded) // 2]
    assert lower <= weighted_median(values, weights) <= upper
    # Scaling every weight leaves it unchanged.
    assert weighted_median(values, weights * 7.5) == weighted_median(values, weights)


@settings(max_examples=300, deadline=None)
@given(
    pairs=st.lists(
        st.tuples(st.floats(0.01, 1e5), st.integers(1, 5)), min_size=2, max_size=40
    )
)
def test_weighted_gini_equals_the_mean_difference_gini_of_the_expanded_sample(pairs):
    expanded = _expand(pairs)
    n = len(expanded)
    mean_difference = np.abs(expanded[:, None] - expanded[None, :]).sum() / (n * n)
    reference = mean_difference / (2 * expanded.mean())
    values = np.array([v for v, _ in pairs])
    weights = np.array([w for _, w in pairs], float)
    assert weighted_gini(values, weights) == pytest.approx(reference, abs=1e-9)


def test_gini_of_equal_incomes_is_zero():
    assert weighted_gini([5.0, 5.0, 5.0], [1, 2, 3]) == pytest.approx(0)


def test_hbai_measures_on_a_known_population():
    income = np.array([100.0, 200.0, 300.0, 400.0, 1000.0])
    weights = np.ones(5)
    child = np.array([1, 0, 0, 0, 0], bool)
    pensioner = np.array([0, 0, 0, 0, 1], bool)
    m = hbai_measures(income, income - 50, weights, child, pensioner)
    assert m["median_bhc"] == 300.0
    assert m["mean_bhc"] == 400.0
    # Below 60% of 300: 100 only, a child.
    assert m["relative_low_income_bhc_all"] == pytest.approx(20.0)
    assert m["relative_low_income_bhc_children"] == pytest.approx(100.0)
    assert m["relative_low_income_bhc_working_age"] == pytest.approx(0.0)
    assert m["relative_low_income_bhc_pensioners"] == pytest.approx(0.0)


def test_the_hbai_reference_is_the_published_table():
    """Spot values from DWP HBAI FYE 2025, tables 8a and 8b."""
    ref = load_hbai_reference()
    assert ref["period"] == 2024
    m = ref["measures"]
    assert m["median_bhc"]["estimate"] == pytest.approx(719.481062)
    assert m["median_ahc"]["estimate"] == pytest.approx(623.144059)
    assert m["relative_low_income_ahc_children"]["estimate"] == pytest.approx(27.41076)
    assert len(m) == 14


def test_single_ages_band_the_oldest_and_split_by_sex():
    age = np.array([0, 0, 79, 80, 85, 90])
    male = np.array([1, 0, 1, 1, 0, 1], bool)
    weights = np.array([2.0, 3.0, 1.0, 1.0, 4.0, 5.0])
    reference = {
        (s, a): 1.0 for s in ("male", "female") for a in range(TOP_AGE_BAND + 1)
    }
    rows = single_ages(age, male, weights, reference)
    assert rows["male"]["0"]["value"] == 2.0 and rows["female"]["0"]["value"] == 3.0
    assert rows["male"]["80"]["value"] == 6.0 and rows["female"]["80"]["value"] == 4.0
    summary = summarise_single_ages(rows)
    assert summary["error_80_plus"] == {"male": 5.0, "female": 3.0}


def _report(**hbai_values):
    ref = load_hbai_reference()["measures"]
    hbai = {}
    for name, r in ref.items():
        value = hbai_values.get(name, r["estimate"])
        hbai[name] = {
            "value": value,
            "reference": r["estimate"],
            "low": r["low"],
            "high": r["high"],
            "inside": r["low"] <= value <= r["high"],
        }
    single = {
        "mean_abs_error_0_79": 0.2,
        "max_abs_error_0_79": 0.9,
        "max_abs_error_15_19": 0.6,
        "error_80_plus": {"male": 0.0, "female": 0.0},
    }
    return {
        "schema": SCHEMA,
        "periods": {"hbai": 2024, "single_age": 2025},
        "hbai": hbai,
        "summary": {"single_age": single},
    }


def test_compare_flags_a_measure_that_leaves_its_interval_but_does_not_fail():
    result = compare(_report(), _report(median_bhc=760.0))
    assert result["status"] == "pass"
    assert result["flags"] == [
        "HBAI median_bhc leaves its 95% interval: 719.48 -> 760.00 (HBAI 719.48, 703.57 to 735.58)."
    ]
    text = render(result)
    assert "760.00 (outside)" in text and "do not fail yet" in text


def test_compare_with_itself_passes_with_no_flags():
    result = compare(_report(), _report())
    assert result["status"] == "pass" and result["flags"] == []
    assert all(r["further_from_reference"] == 0 for r in result["rows"])


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: r.update(schema="other"), "malformed"),
        (lambda r: r["periods"].update(hbai=2025), "different periods"),
        (lambda r: r["hbai"]["median_bhc"].update(value=float("nan")), "not finite"),
    ],
)
def test_compare_fails_on_a_broken_or_mismatched_report(change, message):
    head = _report()
    change(head)
    result = compare(_report(), head)
    assert result["status"] == "fail"
    assert any(message in f for f in result["failures"])


def test_negative_incomes_before_housing_costs_are_reset_to_zero_as_hbai_does():
    """Sol review of #600: HBAI resets negative BHC to zero and derives AHC from
    the adjusted figure, so AHC can still be negative."""
    from policyengine_uk_data.utils.distribution_validation import (
        WEEKS_PER_YEAR,
        hbai_incomes,
    )

    weekly_bhc = np.array([-100.0, 100.0, 200.0, 300.0, 400.0])
    annual_bhc = weekly_bhc * WEEKS_PER_YEAR
    annual_ahc = (weekly_bhc - 50) * WEEKS_PER_YEAR
    bhc, ahc = hbai_incomes(annual_bhc, annual_ahc, np.ones(5), np.ones(5))
    np.testing.assert_allclose(bhc, [0, 100, 200, 300, 400])
    np.testing.assert_allclose(ahc, [-50, 50, 150, 250, 350])
    m = hbai_measures(bhc, ahc, np.ones(5), np.zeros(5, bool), np.zeros(5, bool))
    assert m["mean_bhc"] == pytest.approx(200) and m["mean_ahc"] == pytest.approx(150)
    assert m["gini_bhc"] == pytest.approx(40.0)
    assert m["gini_ahc"] == pytest.approx(53.333333, rel=1e-6)


def test_an_incomplete_or_non_finite_report_fails():
    """Sol review of #600: every reference measure and age summary must be there."""
    head = _report()
    del head["hbai"]["median_bhc"]
    result = compare(_report(), head)
    assert result["status"] == "fail"
    assert result["failures"] == ["The head report has no median_bhc."]
    head = _report()
    head["summary"]["single_age"]["mean_abs_error_0_79"] = float("nan")
    result = compare(_report(), head)
    assert result["status"] == "fail" and "not finite" in result["failures"][0]


def test_the_workflow_fails_when_a_distribution_report_is_missing():
    from pathlib import Path

    text = (
        Path(__file__).parents[2] / ".github" / "workflows" / "pull_request.yaml"
    ).read_text()
    block = text[
        text.index("The income distribution and single-age reports are missing") :
    ]
    assert 'echo "status=1" >> "$GITHUB_OUTPUT"' in block.split("\n          fi", 1)[0]
