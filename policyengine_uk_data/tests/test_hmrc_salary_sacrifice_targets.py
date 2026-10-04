"""Tests for the HMRC salary sacrifice relief targets (Tables 6.1 and 6.2).

The July 2025 CSV returned 410 Gone once HMRC published the July 2026
release, and the source module logged the error and returned without the
relief targets, so builds calibrated without them. These tests pin: the
committed table parses to every relief target, a failed download uses it, a
changed table raises, the rate-band columns measure relief the way HMRC does
(tax on pay, relieved at each rate a contribution straddles), and on a built
dataset every relief target produces a loss-matrix column.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import requests

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.targets.build_loss_matrix import (
    _compute_column,
    _resolve_value,
    _SimContext,
)
from policyengine_uk_data.targets.compute.income import (
    compute_ss_it_relief,
    tax_by_band,
)
from policyengine_uk_data.targets.schema import Target, Unit
from policyengine_uk_data.targets.sources import hmrc_salary_sacrifice as hmrc_ss

# Tables 6.1 and 6.2, tax year 2024-25 (£m), from the committed CSV.
RELIEF_2024_25 = {
    "hmrc/salary_sacrifice_it_relief_basic_rate": 1_600,
    "hmrc/salary_sacrifice_it_relief_higher_rate": 5_500,
    "hmrc/salary_sacrifice_it_relief_additional_rate": 1_800,
    "hmrc/salary_sacrifice_employee_nics_relief": 1_000,
    "hmrc/salary_sacrifice_employer_nics_relief": 3_400,
}
# Class 1 secondary (employer) rate: 13.8% in 2024-25, 15% from 6 April 2025
# (National Insurance Contributions (Secondary Class 1 Contributions) Act
# 2025). The primary rates (8% main, 2% additional) did not change.
EMPLOYER_RATE_RISE = 0.15 / 0.138


def _committed_table() -> pd.DataFrame:
    return pd.read_csv(hmrc_ss.FALLBACK_CSV, dtype=str, encoding="utf-8-sig")


def _offline(*args, **kwargs):
    raise requests.ConnectionError("offline")


def _offline_targets() -> list[Target]:
    with patch.object(hmrc_ss.requests, "get", side_effect=_offline):
        return hmrc_ss.get_targets()


def test_committed_table_gives_every_relief_target():
    targets = {t.name: t for t in hmrc_ss._relief_targets(_committed_table(), "x")}
    assert set(targets) == set(RELIEF_2024_25)
    for name, millions in RELIEF_2024_25.items():
        values = targets[name].values
        rate_rise = EMPLOYER_RATE_RISE if "employer" in name else 1
        # Tax year 2024-25 is PolicyEngine year 2024.
        assert min(values) == 2024
        assert values[2024] == pytest.approx(millions * 1e6)
        assert values[2025] == pytest.approx(millions * 1e6 * 1.03 * rate_rise)
        assert values[2026] == pytest.approx(millions * 1e6 * 1.03**2 * rate_rise)
    assert targets["hmrc/salary_sacrifice_employee_nics_relief"].variable == (
        "ni_employee"
    )
    assert targets["hmrc/salary_sacrifice_employer_nics_relief"].variable == (
        "ni_employer"
    )


def test_fallback_file_matches_the_tax_year_it_is_checked_against():
    assert list(_committed_table()["tax_year"].unique()) == [
        f"{hmrc_ss._TAX_YEAR} to {hmrc_ss._TAX_YEAR + 1}"
    ]


def _gone(*args, **kwargs):
    response = requests.Response()
    response.status_code = 410
    response.url = args[0]
    return response


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
        lambda df: df[
            ~(
                (df["nics_relief_class"] == "Class 1 Secondary (employer)")
                & (df["tax_rate"] == "Main Rate")
            )
        ],
        lambda df: pd.concat(
            [df.iloc[:-1], df.iloc[-1:].assign(tax_year="2025 to 2026")]
        ),
        lambda df: df.assign(tax_year="2025 to 2026"),
        lambda df: df.drop(columns="sector_scheme"),
        lambda df: df.assign(
            value_of_relief=df["value_of_relief"].where(df["tax_rate"] != "Basic Rate")
        ),
    ],
    ids=[
        "no-higher-rate",
        "no-employer-nics",
        "no-employer-rate-split",
        "two-years",
        "next-release",
        "no-column",
        "blank-cell",
    ],
)
def test_changed_table_raises(change):
    with pytest.raises((ValueError, KeyError)):
        hmrc_ss._relief_targets(change(_committed_table()), "x")


def _scales(year):
    from policyengine_uk import CountryTaxBenefitSystem

    rates = CountryTaxBenefitSystem().parameters(year).gov.hmrc.income_tax.rates
    return {"uk": rates.uk, "scotland": rates.scotland.rates}


@pytest.mark.parametrize("year", ["2023", "2024", "2025"])
@pytest.mark.parametrize("schedule", ["uk", "scotland"])
def test_tax_by_band_adds_up_to_the_schedule_and_rises_with_income(year, schedule):
    """Differential against PolicyEngine's own scale, plus monotonicity."""
    scale = _scales(year)[schedule]
    rng = np.random.default_rng(0)
    income = np.concatenate(
        [rng.uniform(-1_000, 200_000, 5_000), np.asarray(scale.thresholds[1:])]
    )
    bands = tax_by_band(income, scale.thresholds, scale.rates)
    np.testing.assert_allclose(sum(bands.values()), scale.calc(income), atol=1e-6)
    more = tax_by_band(
        income + rng.uniform(0, 20_000, income.size), scale.thresholds, scale.rates
    )
    for band in bands:
        assert np.all(more[band] >= bands[band] - 1e-9)


def test_tax_by_band_groups_scottish_rates_into_hmrc_categories():
    """Starter, basic and intermediate → basic; higher and advanced → higher;
    top → additional. Thresholds come from PolicyEngine's 2025 schedule
    (policyengine-uk#2130: its top-rate threshold is £112,570 rather than
    the statutory £125,140)."""
    scale = _scales("2025")["scotland"]
    (_, _, _, higher, advanced, top) = scale.thresholds
    assert list(scale.rates) == [0.19, 0.20, 0.21, 0.42, 0.45, 0.48]
    income = np.array(
        [higher - 1, (higher + advanced) / 2, (advanced + top) / 2, top + 10_000]
    )
    bands = tax_by_band(income, scale.thresholds, scale.rates)
    assert bands["higher"][0] == 0 and bands["additional"][0] == 0
    assert bands["higher"][1] == pytest.approx(0.42 * (income[1] - higher))
    # The advanced rate counts as higher rate.
    assert bands["higher"][2] == pytest.approx(
        0.42 * (advanced - higher) + 0.45 * (income[2] - advanced)
    )
    assert bands["additional"][2] == 0
    assert bands["additional"][3] == pytest.approx(0.48 * 10_000)


class _Ctx:
    """The parts of build_loss_matrix._SimContext the compute function reads."""

    time_period = 2025

    def __init__(self, base_pay, sacrifice, region="LONDON", dividends=0):
        from policyengine_uk import Simulation

        def sim(pay):
            return Simulation(
                situation={
                    "people": {
                        "a": {
                            "age": {2025: 40},
                            "employment_income": {2025: pay},
                            "dividend_income": {2025: dividends},
                        }
                    },
                    "benunits": {"b": {"members": ["a"]}},
                    "households": {"h": {"members": ["a"], "region": {2025: region}}},
                }
            )

        # The counterfactual adds the sacrifice back to pay, as
        # _SimContext.counterfactual_sim does.
        self.sim = sim(base_pay)
        self.counterfactual_sim = sim(base_pay + sacrifice)

    @staticmethod
    def household_from_person(values):
        return np.asarray(values)


def _relief(ctx, band):
    target = Target(
        name=f"hmrc/salary_sacrifice_it_relief_{band}_rate",
        variable="income_tax",
        source="hmrc",
        unit=Unit.GBP,
        values={2025: 1.0},
    )
    return float(compute_ss_it_relief(target, ctx)[0])


@pytest.mark.parametrize(
    "base_pay, sacrifice, region, expected",
    [
        # rUK, £48k after a £4k sacrifice: taxable income £35,430 → £39,430,
        # 40% only above £37,700.
        (48_000, 4_000, "LONDON", {"basic": 454, "higher": 692, "additional": 0}),
        # Scotland, £42k after £4k: taxable £29,430 → £33,430, 21% to £31,092
        # (basic category), 42% above.
        (
            42_000,
            4_000,
            "SCOTLAND",
            {"basic": 349.02, "higher": 981.96, "additional": 0},
        ),
        # rUK, £100k after £10k: £5k of personal allowance is withdrawn, so
        # relief is 60% of the sacrifice, all in the higher band.
        (100_000, 10_000, "LONDON", {"basic": 0, "higher": 6_000, "additional": 0}),
    ],
    ids=["ruk-straddle", "scotland-straddle", "allowance-taper"],
)
def test_compute_relieves_each_slice_at_its_rate(base_pay, sacrifice, region, expected):
    ctx = _Ctx(base_pay, sacrifice, region)
    by_band = {band: _relief(ctx, band) for band in expected}
    assert by_band == pytest.approx(expected, abs=0.01)
    # With pay as the only income, the bands add up to the fall in income tax.
    income_tax = [
        float(s.calculate("income_tax", 2025)[0])
        for s in (ctx.counterfactual_sim, ctx.sim)
    ]
    assert sum(by_band.values()) == pytest.approx(
        income_tax[0] - income_tax[1], abs=0.01
    )


def test_relief_targets_produce_loss_matrix_columns(enhanced_frs):
    """create_target_matrix skips a target, with only a warning, when
    _resolve_value or _compute_column raises or returns None. On the built
    dataset neither may happen for these targets."""
    from policyengine_uk import Microsimulation

    year = CURRENT_FRS_RELEASE.calibration_year
    sim = Microsimulation(dataset=enhanced_frs)
    sim.default_calculation_period = year
    ctx = _SimContext(sim, year, enhanced_frs, None)
    weights = np.asarray(sim.calculate("household_weight", year))
    targets = [t for t in _offline_targets() if t.name in RELIEF_2024_25]
    assert len(targets) == len(RELIEF_2024_25)
    for target in targets:
        assert _resolve_value(target, year) is not None, target.name
        column = np.asarray(_compute_column(target, ctx, year), dtype=float)
        assert np.isfinite(column).all(), target.name
        assert column @ weights > 0, target.name


def test_relief_is_tax_on_pay_not_on_other_income():
    """HMRC applies income tax rates to pay. Paying the sacrifice as salary
    also pushes £10k of dividends from the basic into the higher dividend
    band; that extra dividend tax is not salary sacrifice relief."""
    ctx = _Ctx(48_000, 4_000, dividends=10_000)
    by_band = {band: _relief(ctx, band) for band in ("basic", "higher", "additional")}
    assert by_band == pytest.approx(
        {"basic": 454, "higher": 692, "additional": 0}, abs=0.01
    )
    income_tax = [
        float(s.calculate("income_tax", 2025)[0])
        for s in (ctx.counterfactual_sim, ctx.sim)
    ]
    assert income_tax[0] - income_tax[1] > sum(by_band.values()) + 100
