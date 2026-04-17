"""Tests for parameterised VAT constants in `datasets/imputations/vat.py`.

Covers bug-hunt finding U7: the original code hardcoded
``CURRENT_VAT_RATE = 0.2``, ``CONSUMPTION_PCT_REDUCED_RATE = 0.03`` and
the ``etb.year == 2020`` filter inline, so any change to VAT rates,
reduced-rate share, or training vintage required a code edit across
multiple scattered lines.
"""

from __future__ import annotations

import pandas as pd
import pytest


def test_get_vat_parameters_reads_from_policyengine_uk():
    """Standard rate should come from `policyengine_uk` parameters."""
    try:
        from policyengine_uk.system import system
    except Exception:
        pytest.skip("policyengine_uk not available")

    from policyengine_uk_data.datasets.imputations.vat import (
        _get_vat_parameters,
    )

    expected_standard = float(system.parameters.gov.hmrc.vat.standard_rate("2020"))
    expected_reduced = float(system.parameters.gov.hmrc.vat.reduced_rate_share("2020"))
    standard, reduced = _get_vat_parameters(2020)
    assert standard == pytest.approx(expected_standard)
    assert reduced == pytest.approx(expected_reduced)


def test_vat_rate_by_year_fallback_matches_2020_statute():
    """Offline fallback must stay aligned with the statutory 2020-21 rates."""
    from policyengine_uk_data.datasets.imputations.vat import (
        VAT_RATE_BY_YEAR,
    )

    assert VAT_RATE_BY_YEAR[2020] == (0.2, 0.03)


def test_generate_etb_table_uses_year_param():
    """Changing the `year` arg filters ETB rows by that year.

    The original implementation hardcoded ``etb.year == 2020``. After the
    fix the year is a parameter with a sensible default.
    """
    from policyengine_uk_data.datasets.imputations.vat import (
        generate_etb_table,
    )

    etb = pd.DataFrame(
        {
            "year": [2020, 2020, 2021, 2021],
            "adults": [1, 2, 1, 2],
            "childs": [0, 1, 0, 1],
            "noretd": [0, 0, 1, 1],
            "disinc": [500.0, 800.0, 600.0, 900.0],
            "totvat": [50.0, 80.0, 60.0, 90.0],
            "expdis": [500.0, 800.0, 600.0, 900.0],
        }
    )

    out_2020 = generate_etb_table(etb, year=2020)
    out_2021 = generate_etb_table(etb, year=2021)

    # Filtering is by year column — disjoint row counts confirm the filter
    # actually moved.
    assert len(out_2020) == 2
    assert len(out_2021) == 2
    # Trained features use household_net_income = disinc * 52.
    assert set(out_2020["household_net_income"].to_numpy()) == {500 * 52, 800 * 52}
    assert set(out_2021["household_net_income"].to_numpy()) == {600 * 52, 900 * 52}


def test_generate_etb_table_uses_year_specific_vat_rate(monkeypatch):
    """The ``full_rate_vat_expenditure_rate`` column scales with VAT rate."""
    from policyengine_uk_data.datasets.imputations import vat as vat_module

    etb = pd.DataFrame(
        {
            "year": [2020, 2030],
            "adults": [1, 1],
            "childs": [0, 0],
            "noretd": [0, 0],
            "disinc": [1000.0, 1000.0],
            "totvat": [100.0, 100.0],
            "expdis": [1000.0, 1000.0],
        }
    )

    def _fake_params(year: int):
        return (0.2, 0.0) if year == 2020 else (0.25, 0.0)

    monkeypatch.setattr(vat_module, "_get_vat_parameters", _fake_params)

    out_2020 = vat_module.generate_etb_table(etb, year=2020)
    out_hypothetical = vat_module.generate_etb_table(etb, year=2030)

    # Higher standard rate → lower implied full-rate expenditure (divide
    # totvat by a bigger denominator), so the computed rate must drop.
    assert (
        out_hypothetical["full_rate_vat_expenditure_rate"].iloc[0]
        < (out_2020["full_rate_vat_expenditure_rate"].iloc[0])
    )
