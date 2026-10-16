"""MOCK FRS capital reaches the locked UK model's Universal Credit rules."""

from pathlib import Path

import pandas as pd
import pytest
from policyengine_uk import Simulation

from policyengine_uk_data.tests.test_legacy_benefit_proxies import (
    create_single_adult_frs,
)


def _simulation_from_mock_frs(tmp_path, monkeypatch, capital_fields):
    read_csv = pd.read_csv

    def read_mock_frs_table(path, *args, **kwargs):
        table = read_csv(path, *args, **kwargs)
        if Path(path).stem == "benunit":
            table = table.assign(**capital_fields)
        return table

    # The existing MOCK fixture stubs processing-time policy calculations.
    # Restore those stubs before exercising the real UK model below.
    with monkeypatch.context() as frs_patch:
        frs_patch.setattr(pd, "read_csv", read_mock_frs_table)
        dataset = create_single_adult_frs(tmp_path, frs_patch)

    # -1 is the UK model's fallback if create_frs omits the reported input.
    # Keeping that fallback makes the regression on main a capital/UC
    # assertion failure, rather than a missing-column or helper failure.
    reported_capital = float(
        dataset.benunit.get("benunit_reported_capital", pd.Series([-1.0])).iloc[0]
    )
    return Simulation(
        situation={
            "people": {
                "claimant": {
                    "age": {2025: 30},
                    "is_claimant_or_partner": {2025: True},
                }
            },
            "benunits": {
                "benunit": {
                    "members": ["claimant"],
                    "benunit_reported_capital": {2025: reported_capital},
                    "would_claim_uc": {2025: True},
                }
            },
            "households": {
                "household": {
                    "members": ["claimant"],
                    # Reported benefit-unit capital must override the
                    # household wealth proxy, including an explicit zero.
                    "savings": {2025: 50_000},
                }
            },
        }
    )


@pytest.mark.parametrize(
    "capital_fields, capital, tariff_steps, eligible",
    [
        ({"totcapb4": 6_000}, 6_000, 0, True),
        ({"totcapb4": 6_001}, 6_001, 1, True),
        ({"totcapb4": 6_250}, 6_250, 1, True),
        ({"totcapb4": 6_251}, 6_251, 2, True),
        ({"totcapb4": 16_000}, 16_000, 40, True),
        ({"totcapb4": 16_001}, 16_001, 0, False),
        ({"totcapb3": 6_251}, 6_251, 2, True),
        ({"totcapb4": -9, "totcapb3": 6_001}, 6_001, 1, True),
        ({"totcapb4": None, "totcapb3": 6_250}, 6_250, 1, True),
        ({"totcapb4": 0, "totcapb3": 16_001}, 0, 0, True),
    ],
    ids=[
        "tariff-threshold",
        "first-pound-over-threshold",
        "first-tariff-step-end",
        "second-tariff-step-start",
        "capital-limit",
        "over-capital-limit",
        "older-frs-total",
        "negative-new-total-falls-back",
        "missing-new-total-falls-back",
        "explicit-zero-overrides-old-total-and-wealth",
    ],
)
def test_mock_frs_capital_reaches_universal_credit(
    tmp_path, monkeypatch, capital_fields, capital, tariff_steps, eligible
):
    sim = _simulation_from_mock_frs(tmp_path, monkeypatch, capital_fields)

    def value(variable):
        return sim.calculate(variable, 2025)[0]

    for variable in (
        "benunit_reported_capital",
        "uc_reported_capital",
        "uc_assessable_capital",
    ):
        assert value(variable) == capital, variable
    assert bool(value("is_uc_eligible")) is eligible

    # UC Regs 2013 regs 18 and 72: each started £250 over £6,000
    # contributes £4.35 a month, until capital exceeds £16,000.
    annual_tariff = tariff_steps * 4.35 * 12
    assert value("uc_tariff_income") == pytest.approx(annual_tariff)
    standard_allowance = value("uc_standard_allowance")
    assert standard_allowance > 0
    expected_uc = standard_allowance - annual_tariff if eligible else 0
    assert value("universal_credit") == pytest.approx(expected_uc)
