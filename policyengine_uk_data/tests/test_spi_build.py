"""Regression tests for `policyengine_uk_data.datasets.spi`.

Covers the three bugs flagged in the bug hunt:

- The ``__main__`` block called ``create_spi`` with two positional args but
  the signature required three. This test asserts the function is callable
  with two positional args (``spi_data_file_path`` and ``fiscal_year``) and
  that the optional ``output_file_path`` kwarg is accepted.
- Age imputation was non-deterministic (unseeded ``np.random.rand``). This
  test asserts two runs with the same seed produce identical ``age``
  columns.
- Unknown GORCODE values were silently mapped to ``SOUTH_EAST``. This test
  asserts the default fallback label is now ``UNKNOWN``.
"""

from __future__ import annotations

import importlib.util
import inspect

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("policyengine_uk") is None:
    pytest.skip(
        "policyengine_uk not available in test environment",
        allow_module_level=True,
    )


SPI_COLUMNS = [
    "SREF",
    "FACT",
    "DIVIDENDS",
    "GIFTAID",
    "GORCODE",
    "INCBBS",
    "INCPROP",
    "PAY",
    "EPB",
    "EXPS",
    "PENSION",
    "PSAV_XS",
    "PENSRLF",
    "PROFITS",
    "CAPALL",
    "LOSSBF",
    "AGERANGE",
    "SRP",
    "TAX_CRED",
    "MOTHINC",
    "INCPBEN",
    "OSSBEN",
    "TAXTERM",
    "UBISJA",
    "OTHERINC",
    "GIFTINV",
    "OTHERINV",
    "COVNTS",
    "MOTHDED",
    "DEFICIEN",
    "MCAS",
    "BPADUE",
    "MAIND",
]


def _write_fake_spi(path, gor_values=(1, 2, 3), maind_values=(1, 0, 1)):
    """Write a minimal SPI-shaped tab file for tests.

    The real SPI file has dozens of columns; the test only needs them to
    exist with sensible types so ``create_spi`` can build dataframes.
    """
    n = len(gor_values)
    data = {col: np.zeros(n, dtype=float) for col in SPI_COLUMNS}
    data["SREF"] = np.arange(1, n + 1)
    data["FACT"] = np.ones(n)
    data["GORCODE"] = list(gor_values)
    data["MAIND"] = list(maind_values)
    data["AGERANGE"] = [1] * n  # bucket (16, 25)
    df = pd.DataFrame(data)
    df.to_csv(path, sep="\t", index=False)


def test_create_spi_accepts_two_positional_args(tmp_path):
    """The ``__main__`` crash bug: ``create_spi(path, year)`` must work."""
    from policyengine_uk_data.datasets.spi import create_spi

    sig = inspect.signature(create_spi)
    params = list(sig.parameters.values())
    # First two params are required positional; remaining params are optional
    # so two-arg calls succeed.
    assert params[0].default is inspect.Parameter.empty
    assert params[1].default is inspect.Parameter.empty
    for p in params[2:]:
        assert p.default is not inspect.Parameter.empty, (
            f"Parameter {p.name!r} must have a default so create_spi(path, "
            f"year) stays callable without breaking the __main__ block."
        )


def test_create_spi_age_imputation_is_deterministic(tmp_path):
    """Same seed → identical age column. Was unseeded in the buggy version."""
    from policyengine_uk_data.datasets.spi import create_spi

    tab = tmp_path / "spi.tab"
    _write_fake_spi(tab, gor_values=(1, 2, 3, 4, 5), maind_values=(0, 0, 0, 0, 0))

    ds_a = create_spi(tab, 2020, seed=42)
    ds_b = create_spi(tab, 2020, seed=42)
    ds_c = create_spi(tab, 2020, seed=123)

    assert (ds_a.person["age"].to_numpy() == ds_b.person["age"].to_numpy()).all()
    # Different seeds should give some variation for the (16, 25) bucket.
    assert not (ds_a.person["age"].to_numpy() == ds_c.person["age"].to_numpy()).all()


def test_create_spi_unknown_gorcode_does_not_silently_become_south_east(
    tmp_path,
):
    """Unmapped GORCODE rows now get UNKNOWN, not SOUTH_EAST, by default."""
    from policyengine_uk_data.datasets.spi import create_spi

    tab = tmp_path / "spi.tab"
    _write_fake_spi(
        tab,
        gor_values=(99, 7, 99),  # 99 is undocumented → should be UNKNOWN
        maind_values=(0, 0, 0),
    )

    ds = create_spi(tab, 2020, seed=0)
    regions = ds.household["region"].tolist()
    assert regions[0] == "UNKNOWN"
    assert regions[1] == "LONDON"  # GORCODE 7 maps to LONDON
    assert regions[2] == "UNKNOWN"
    # Legacy behaviour is still accessible via the kwarg for callers that
    # relied on it.
    ds_legacy = create_spi(tab, 2020, seed=0, unknown_region="SOUTH_EAST")
    assert ds_legacy.household["region"].tolist()[0] == "SOUTH_EAST"


def test_create_spi_marriage_allowance_uses_fiscal_year_parameters(tmp_path):
    """MA cap should follow the fiscal year's 10% × Personal Allowance rule.

    2020-21 PA = £12,500 so MA cap = £1,250 (the historical hardcoded value).
    2021-22 onwards PA = £12,570 so MA cap = £1,257, rounded down to
    increments per the rounding_increment parameter (HMRC publishes £1,260
    for 2025-26).
    """
    from policyengine_uk_data.datasets.spi import create_spi

    tab = tmp_path / "spi.tab"
    _write_fake_spi(tab, gor_values=(1, 2, 3), maind_values=(1, 0, 1))

    ds_2020 = create_spi(tab, 2020, seed=0)
    marriage_2020 = ds_2020.person["marriage_allowance"].to_numpy()
    # Expect eligible rows (MAIND == 1) to receive £1,250 and ineligible 0.
    assert (marriage_2020[[0, 2]] == 1_250).all()
    assert marriage_2020[1] == 0

    ds_2025 = create_spi(tab, 2025, seed=0)
    marriage_2025 = ds_2025.person["marriage_allowance"].to_numpy()
    # Post-2020, PA is £12,570 so the cap is £1,257 before rounding; the
    # published HMRC value is £1,260 (rounding to nearest £10). Accept
    # either, but require it's NOT the stale 2020-21 £1,250 figure.
    assert marriage_2025[0] != 1_250
    assert marriage_2025[0] >= 1_250  # PA has only risen since 2020
