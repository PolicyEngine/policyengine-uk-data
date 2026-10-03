"""Tests for the HMRC salary sacrifice relief targets (Tables 6.1 and 6.2).

The July 2025 CSV returned 410 Gone once HMRC published the July 2026
release, and the source module logged the error and returned without the
relief targets, so builds calibrated without them. These tests pin: the
committed table parses to every relief target, a failed download uses it, a
changed table raises, and the rate-band columns split relief the way HMRC
does (across the bands a contribution straddles).
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import requests

from policyengine_uk_data.targets.compute.income import (
    compute_ss_it_relief,
    split_relief_by_band,
)
from policyengine_uk_data.targets.schema import Target, Unit
from policyengine_uk_data.targets.sources import hmrc_salary_sacrifice as hmrc_ss

# Tables 6.1 and 6.2, tax year 2024-25 (£m), from the committed CSV.
RELIEF_2024_25 = {
    "hmrc/salary_sacrifice_it_relief_total": 8_800,
    "hmrc/salary_sacrifice_it_relief_basic_rate": 1_600,
    "hmrc/salary_sacrifice_it_relief_higher_rate": 5_500,
    "hmrc/salary_sacrifice_it_relief_additional_rate": 1_800,
    "hmrc/salary_sacrifice_employee_nics_relief": 1_000,
    "hmrc/salary_sacrifice_employer_nics_relief": 3_400,
}


def _committed_table() -> pd.DataFrame:
    return pd.read_csv(hmrc_ss.FALLBACK_CSV, dtype=str, encoding="utf-8-sig")


def test_committed_table_gives_every_relief_target():
    targets = {t.name: t for t in hmrc_ss._relief_targets(_committed_table(), "x")}
    assert set(targets) == set(RELIEF_2024_25)
    for name, millions in RELIEF_2024_25.items():
        values = targets[name].values
        # Tax year 2024-25 is PolicyEngine year 2024.
        assert min(values) == 2024
        assert values[2024] == pytest.approx(millions * 1e6)
        assert values[2025] == pytest.approx(millions * 1e6 * 1.03)
    assert targets["hmrc/salary_sacrifice_employee_nics_relief"].variable == (
        "ni_employee"
    )
    assert targets["hmrc/salary_sacrifice_employer_nics_relief"].variable == (
        "ni_employer"
    )


def _gone(*args, **kwargs):
    response = requests.Response()
    response.status_code = 410
    response.url = args[0]
    return response


def _offline(*args, **kwargs):
    raise requests.ConnectionError("offline")


@pytest.mark.parametrize("get", [_gone, _offline], ids=["410", "offline"])
def test_failed_download_uses_committed_table(get):
    with patch.object(hmrc_ss.requests, "get", side_effect=get):
        names = {t.name for t in hmrc_ss.get_targets()}
    assert set(RELIEF_2024_25) <= names
    assert "hmrc/salary_sacrifice_contributions" in names


@pytest.mark.parametrize(
    "change",
    [
        lambda df: df[df["tax_rate"] != "Higher Rate"],
        lambda df: df[df["nics_relief_class"] != "Class 1 Secondary (employer)"],
        lambda df: pd.concat(
            [df.iloc[:-1], df.iloc[-1:].assign(tax_year="2025 to 2026")]
        ),
        lambda df: df.assign(tax_year="2024 to 2026"),
        lambda df: df.drop(columns="sector_scheme"),
    ],
    ids=["no-higher-rate", "no-employer-nics", "two-years", "bad-year", "no-column"],
)
def test_changed_table_raises(change):
    with pytest.raises((ValueError, KeyError)):
        hmrc_ss._relief_targets(change(_committed_table()), "x")


BANDS = ("basic", "higher", "additional")


def test_split_straddling_relief():
    # £4k sacrificed from £52k pay (2025-26 rUK): earned taxable income falls
    # from £39,430 to £35,430, so £1,730 leaves the higher band and £2,270
    # the basic band.
    split = split_relief_by_band(
        np.array([1_146.0]),
        {"basic": [7_540.0], "higher": [692.0], "additional": [0.0]},
        {"basic": [7_086.0], "higher": [0.0], "additional": [0.0]},
    )
    assert split["basic"] == pytest.approx([454.0])
    assert split["higher"] == pytest.approx([692.0])
    assert split["additional"] == pytest.approx([0.0])


def test_split_invariants_on_random_inputs():
    """Bands add up to the relief, keep its sign, and default to basic rate."""
    rng = np.random.default_rng(0)
    n = 5_000
    relief = rng.normal(500, 1_000, n) * rng.integers(0, 2, n)
    base = {b: rng.uniform(0, 20_000, n) * rng.integers(0, 2, n) for b in BANDS}
    # A smaller sacrifice never raises band tax; some people see no change.
    cf = {b: base[b] + rng.uniform(0, 3_000, n) * rng.integers(0, 2, n) for b in BANDS}
    split = split_relief_by_band(relief, cf, base)

    np.testing.assert_allclose(sum(split.values()), relief, rtol=1e-12, atol=1e-9)
    for band in BANDS:
        assert np.all(split[band] * np.sign(relief) >= -1e-9)
    no_drop = sum(cf[b] - base[b] for b in BANDS) == 0
    assert no_drop.any()
    np.testing.assert_array_equal(split["basic"][no_drop], relief[no_drop])


class _Ctx:
    """The parts of build_loss_matrix._SimContext the compute function reads."""

    time_period = 2025

    def __init__(self, base_pay, sacrifice):
        from policyengine_uk import Simulation

        def sim(pay):
            return Simulation(
                situation={
                    "people": {
                        "a": {"age": {2025: 40}, "employment_income": {2025: pay}}
                    },
                    "benunits": {"b": {"members": ["a"]}},
                    "households": {"h": {"members": ["a"]}},
                }
            )

        # The counterfactual adds the sacrifice back to pay, as
        # _SimContext.counterfactual_sim does.
        self.sim = sim(base_pay)
        self.counterfactual_sim = sim(base_pay + sacrifice)

    @staticmethod
    def household_from_person(values):
        return np.asarray(values)


def _relief(ctx, suffix):
    target = Target(
        name=f"hmrc/salary_sacrifice_it_relief_{suffix}",
        variable="income_tax",
        source="hmrc",
        unit=Unit.GBP,
        values={2025: 1.0},
    )
    return float(compute_ss_it_relief(target, ctx)[0])


def test_compute_splits_relief_for_a_basic_rate_taxpayer_near_the_threshold():
    """Pay of £48k after a £4k sacrifice: 40% relief only above £50,270."""
    ctx = _Ctx(base_pay=48_000, sacrifice=4_000)
    by_band = {
        s: _relief(ctx, s) for s in ("basic_rate", "higher_rate", "additional_rate")
    }
    assert _relief(ctx, "total") == pytest.approx(1_146)
    assert by_band == pytest.approx(
        {"basic_rate": 454, "higher_rate": 692, "additional_rate": 0}
    )
