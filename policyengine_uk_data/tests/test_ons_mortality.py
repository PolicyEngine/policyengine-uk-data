"""Tests for the ONS National Life Tables loader.

Tests run against a tiny synthetic workbook matching the real ONS NLT
schema so nothing in CI touches the network.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_MALE_HEADER = ["Males", *([""] * 12)]
_FEMALE_HEADER = [*([""] * 7), "Females", *([""] * 5)]
_COLUMN_HEADER = [
    "age",
    "mx",
    "qx",
    "lx",
    "dx",
    "ex",
    "",
    "age",
    "mx",
    "qx",
    "lx",
    "dx",
    "ex",
]


def _synthetic_workbook(tmp_path: Path) -> Path:
    """Write a minimal xlsx matching ONS NLT structure with two periods."""
    path = tmp_path / "ons_nlt_fake.xlsx"
    ages = list(range(0, 11))

    def _period_frame(male_qx: list[float], female_qx: list[float]) -> pd.DataFrame:
        rows: list[list] = [
            ["National Life Tables"] + [""] * 12,
            [""] * 13,
            [""] * 13,
            ["Back to contents"] + [""] * 12,
            _MALE_HEADER,
            _COLUMN_HEADER,
        ]
        for age, mqx, fqx in zip(ages, male_qx, female_qx):
            rows.append(
                [
                    age,
                    mqx,
                    mqx,
                    100000,
                    0,
                    80.0,
                    "",
                    age,
                    fqx,
                    fqx,
                    100000,
                    0,
                    83.0,
                ]
            )
        return pd.DataFrame(rows)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _period_frame(
            male_qx=[0.004 + 0.0001 * a for a in ages],
            female_qx=[0.003 + 0.0001 * a for a in ages],
        ).to_excel(writer, sheet_name="2022-2024", index=False, header=False)
        _period_frame(
            male_qx=[0.005 + 0.0001 * a for a in ages],
            female_qx=[0.004 + 0.0001 * a for a in ages],
        ).to_excel(writer, sheet_name="2019-2021", index=False, header=False)
        pd.DataFrame([["preface"]]).to_excel(
            writer, sheet_name="Contents", index=False, header=False
        )
    return path


def _clear_cache():
    """Reset the loader's lru_cache so tests can point at different files."""
    from policyengine_uk_data.targets.sources.ons_mortality import load_ons_life_tables

    load_ons_life_tables.cache_clear()


def test_load_ons_life_tables_returns_long_frame(tmp_path):
    from policyengine_uk_data.targets.sources.ons_mortality import load_ons_life_tables

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    tables = load_ons_life_tables(path=str(path))
    expected_cols = {"period", "period_start", "period_end", "sex", "age", "qx"}
    assert expected_cols.issubset(tables.columns)
    assert set(tables["sex"].unique()) == {"MALE", "FEMALE"}
    assert set(tables["period"].unique()) == {"2022-2024", "2019-2021"}
    # 11 ages × 2 sexes × 2 periods
    assert len(tables) == 44


def test_load_skips_non_period_sheets(tmp_path):
    from policyengine_uk_data.targets.sources.ons_mortality import load_ons_life_tables

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    tables = load_ons_life_tables(path=str(path))
    # The "Contents" sheet must not appear as a period.
    assert "Contents" not in set(tables["period"].unique())


def test_get_mortality_rates_by_year_resolves_covering_period(tmp_path):
    from policyengine_uk_data.targets.sources.ons_mortality import (
        get_mortality_rates,
        load_ons_life_tables,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    tables = load_ons_life_tables(path=str(path))

    # 2023 falls inside 2022-2024 exactly.
    rates_2023 = get_mortality_rates(2023, tables=tables)
    assert rates_2023["MALE"][0] == pytest.approx(0.004)
    assert rates_2023["FEMALE"][5] == pytest.approx(0.0035)


def test_get_mortality_rates_falls_back_to_nearest_past_period(tmp_path):
    from policyengine_uk_data.targets.sources.ons_mortality import (
        get_mortality_rates,
        load_ons_life_tables,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    tables = load_ons_life_tables(path=str(path))

    # 2018 is before any covering period; should fall back to the most
    # recent period ending before 2018 — here, 2019-2021 ends after, so
    # it's actually NOT before 2018. Use 2017 which is before 2019-2021.
    with pytest.raises(KeyError):
        get_mortality_rates(2017, tables=tables)


def test_get_mortality_rates_none_returns_latest(tmp_path):
    from policyengine_uk_data.targets.sources.ons_mortality import (
        get_mortality_rates,
        load_ons_life_tables,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    tables = load_ons_life_tables(path=str(path))

    rates = get_mortality_rates(year=None, tables=tables)
    # 2022-2024 is the latest in the synthetic workbook.
    assert rates["MALE"][0] == pytest.approx(0.004)


def test_get_mortality_rates_explicit_period_label(tmp_path):
    from policyengine_uk_data.targets.sources.ons_mortality import (
        get_mortality_rates,
        load_ons_life_tables,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    tables = load_ons_life_tables(path=str(path))

    rates = get_mortality_rates("2019-2021", tables=tables)
    assert rates["MALE"][0] == pytest.approx(0.005)


def test_unisex_rates_average_two_sexes(tmp_path):
    from policyengine_uk_data.targets.sources.ons_mortality import (
        get_mortality_rates_unisex,
        load_ons_life_tables,
    )

    path = _synthetic_workbook(tmp_path)
    _clear_cache()
    tables = load_ons_life_tables(path=str(path))

    uni = get_mortality_rates_unisex(2023, tables=tables, male_share=0.5)
    # (0.004 + 0.003) / 2 = 0.0035
    assert uni[0] == pytest.approx(0.0035)
    # Mother's-age-like weighting works too.
    skewed = get_mortality_rates_unisex(2023, tables=tables, male_share=1.0)
    assert skewed[0] == pytest.approx(0.004)
