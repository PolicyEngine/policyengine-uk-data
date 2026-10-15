"""Local UC targets retain their area identity through both calibration paths.

The oracle reads only committed public Stat-Xplore/DfC workbooks and geography
metadata. A four-household simulation stub exercises the target builders without
opening survey data or building/recalibrating a dataset.
"""

from functools import lru_cache
from pathlib import Path
import re
from types import SimpleNamespace
import unicodedata

from hypothesis import given, settings, strategies as st
import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.targets.sources import local_uc
from policyengine_uk_data.utils import uc_data


_READ_EXCEL = pd.read_excel
_AREA_FILES = {
    "constituency": "constituencies_2024.csv",
    "la": "local_authorities_2021.csv",
}
_CHILD_COLUMNS = [
    "uc_hh_0_children",
    "uc_hh_1_child",
    "uc_hh_2_children",
    "uc_hh_3plus_children",
]
_COUNTRIES = {
    "E": "ENGLAND",
    "W": "WALES",
    "S": "SCOTLAND",
    "N": "NORTHERN_IRELAND",
}


@lru_cache
def _workbook(filename, sheet_name=0):
    return _READ_EXCEL(STORAGE_FOLDER / filename, sheet_name=sheet_name, header=None)


def _name_key(name):
    # Stat-Xplore's Welsh LA labels have the English name before the slash.
    english = str(name).split(" / ")[0].casefold().replace("&", "and")
    unaccented = "".join(
        c
        for c in unicodedata.normalize("NFKD", english)
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]", "", unaccented)


@lru_cache
def _area_metadata(area_type):
    return pd.read_csv(STORAGE_FOLDER / _AREA_FILES[area_type])


@lru_cache
def _national_payment_total():
    table = _workbook("uc_national_payment_dist.xlsx")
    rows = table.iloc[9:]
    paying = rows[~rows[1].isin(["No payment", "Total"])]
    return sum(
        pd.to_numeric(paying[column], errors="coerce").fillna(0).sum()
        for column in range(3, 7)
    )


@lru_cache
def _workbook_oracle(area_type):
    """Independent raw-name join and the existing uniform national rescale."""
    areas = _area_metadata(area_type)
    codes_by_name = {_name_key(row["name"]): row["code"] for _, row in areas.iterrows()}
    assert len(codes_by_name) == len(areas)
    name_column = 1 if area_type == "constituency" else 2
    filename = (
        "uc_pc_households.xlsx"
        if area_type == "constituency"
        else "uc_la_households.xlsx"
    )
    gb = _workbook(filename)
    counts = {}
    for _, row in gb.iloc[8:].iterrows():
        name, count = row[name_column], row[3]
        if pd.isna(name) or pd.isna(count) or name in {"Unknown", "Total"}:
            continue
        code = codes_by_name[_name_key(name)]
        assert code not in counts
        counts[code] = int(count)

    ni = _workbook(
        "dfc-ni-uc-stats-supp-tables-may-2025.ods",
        "5b" if area_type == "constituency" else "5c",
    )
    month = ni.loc[ni[0].eq("May 2025")].iloc[0]
    stop = 19 if area_type == "constituency" else 12
    for column in range(1, stop):
        name = ni.iloc[2, column]
        if pd.isna(name) or (area_type == "la" and name == "Ards and North Down"):
            continue
        code = codes_by_name[_name_key(name)]
        assert code not in counts
        counts[code] = int(month[column])

    raw = pd.Series(counts, name="household_count").rename_axis("code").sort_index()
    assert set(raw.index) == set(areas["code"])
    scaled = (raw * (_national_payment_total() / raw.sum())).round().astype(int)
    return raw, scaled


def _get_targets(area_type):
    getter = (
        local_uc.get_constituency_uc_targets
        if area_type == "constituency"
        else local_uc.get_la_uc_targets
    )
    return getter()


@pytest.mark.parametrize("area_type", ["constituency", "la"])
def test_all_uc_rows_and_country_totals_match_public_workbooks(area_type):
    """Keep the national payment rescale and per-area rounding unchanged."""
    raw, expected = _workbook_oracle(area_type)
    actual = _get_targets(area_type).sort_index()
    pd.testing.assert_series_equal(actual, expected, check_dtype=False)
    for country in _COUNTRIES:
        codes = expected.index[expected.index.str.startswith(country)]
        assert actual.loc[codes].sum() == expected.loc[codes].sum()
    assert actual.sum() == (raw * (_national_payment_total() / raw.sum())).round().sum()
    # The national scale is deliberate; individual integer rounding can drift.
    assert abs(actual.sum() - _national_payment_total()) <= len(actual) / 2


@pytest.mark.parametrize("area_type", ["constituency", "la"])
@settings(max_examples=8, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_permuting_workbook_areas_preserves_code_indexed_targets(area_type, seed):
    _, expected = _workbook_oracle(area_type)
    filename = (
        "uc_pc_households.xlsx"
        if area_type == "constituency"
        else "uc_la_households.xlsx"
    )
    name_column = 1 if area_type == "constituency" else 2
    rng = np.random.default_rng(seed)
    shuffled = _workbook(filename).copy()
    valid = shuffled.index[
        (shuffled.index >= 8)
        & shuffled[name_column].notna()
        & shuffled[3].notna()
        & ~shuffled[name_column].isin(["Unknown", "Total"])
    ]
    shuffled.loc[valid] = shuffled.loc[rng.permutation(valid)].to_numpy()

    def read_excel(path, *args, **kwargs):
        if Path(path).name == filename:
            return shuffled.copy()
        if Path(path).suffix == ".ods":
            table = _workbook(Path(path).name, kwargs["sheet_name"]).copy()
            stop = 19 if kwargs["sheet_name"] == "5b" else 12
            columns = list(range(1, stop))
            table.iloc[:, columns] = table.iloc[:, rng.permutation(columns)].to_numpy()
            return table
        return _workbook(Path(path).name).copy()

    parser = (
        uc_data._parse_uc_pc_households
        if area_type == "constituency"
        else uc_data._parse_uc_la_households
    )
    attribute = (
        "uc_pc_households" if area_type == "constituency" else "uc_la_households"
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(uc_data.pd, "read_excel", read_excel)
        patch.setattr(uc_data, attribute, parser())
        pd.testing.assert_series_equal(
            _get_targets(area_type).sort_index(), expected, check_dtype=False
        )


def test_children_split_sums_exactly_and_uses_matched_country_shares():
    _, totals = _workbook_oracle("constituency")
    actual = local_uc.get_constituency_uc_by_children_targets().sort_index()
    assert list(actual.columns) == _CHILD_COLUMNS
    pd.testing.assert_index_equal(actual.index, totals.index)
    np.testing.assert_array_equal(actual.sum(axis=1), totals)
    gb_total = totals.loc[totals.index.str[0].isin(["E", "W", "S"])].sum()
    buckets = local_uc._scaled_uc_children_by_country(gb_total)
    for code, total in totals.items():
        shares = buckets[code[0]] / buckets[code[0]].sum()
        # Integer apportionment must stay within one household of each share.
        assert (np.abs(actual.loc[code].to_numpy() - total * shares) < 1).all(), code


@pytest.mark.parametrize("problem", ["unmatched", "duplicate", "missing"])
def test_area_matching_rejects_incomplete_or_ambiguous_names(problem):
    areas = pd.DataFrame({"code": ["E01", "W01"], "name": ["A & B", "Cardiff"]})
    names = {
        "unmatched": ["Unknown place", "Cardiff"],
        "duplicate": ["A and B", "A & B"],
        "missing": ["A and B"],
    }[problem]
    data = pd.DataFrame({"area_name": names, "household_count": [1] * len(names)})
    with pytest.raises(ValueError):
        uc_data._match_uc_area_codes(data, areas, "area_name")


@pytest.mark.parametrize("bilingual", ["CARDIFF / CAERDYDD", "CAERDYDD / CARDIFF"])
def test_area_matching_normalises_labels_and_prefers_explicit_codes(bilingual):
    areas = pd.DataFrame(
        {
            "code": ["E01", "W01", "W02"],
            "name": ["King's Lynn and West Norfolk", "Cardiff", "Ynys Môn"],
        }
    )
    names = pd.DataFrame(
        {
            "area_name": [bilingual, "kings lynn & west-norfolk", "YNYS MON"],
            "household_count": [10, 20, 30],
        }
    )
    matched = uc_data._match_uc_area_codes(names, areas, "area_name")
    assert matched["code"].tolist() == ["W01", "E01", "W02"]
    coded = names.assign(code=["E01", "W01", "W02"], area_name="Unrecognised label")
    assert uc_data._match_uc_area_codes(coded, areas, "area_name")["code"].tolist() == [
        "E01",
        "W01",
        "W02",
    ]


@pytest.mark.parametrize("codes", [["E01", "E01"], ["E01", "X01"]])
def test_area_matching_rejects_duplicate_or_unknown_explicit_codes(codes):
    areas = pd.DataFrame({"code": ["E01", "W01"], "name": ["Area A", "Area B"]})
    data = pd.DataFrame({"code": codes, "area_name": ["Area A", "Area B"]})
    with pytest.raises(ValueError):
        uc_data._match_uc_area_codes(data, areas, "area_name")


@pytest.mark.parametrize("problem", ["ambiguous_alias", "duplicate_code"])
def test_area_matching_rejects_ambiguous_reference_geography(problem):
    areas = pd.DataFrame(
        {
            "code": ["E01", "W01"] if problem == "ambiguous_alias" else ["E01", "E01"],
            "name": ["A & B", "A and B"],
        }
    )
    data = pd.DataFrame({"area_name": ["A and B"], "household_count": [1]})
    with pytest.raises(ValueError):
        uc_data._match_uc_area_codes(data, areas, "area_name")


class _FakeSim:
    """Four artificial households, one in each UK country; no dataset IO."""

    def __init__(self, *args, **kwargs):
        self.default_calculation_period = 2025

    def calculate(self, variable, *args, **kwargs):
        if variable == "country":
            values = list(_COUNTRIES.values())
        elif variable == "tenure_type":
            values = ["RENT_PRIVATELY"] * 4
        elif variable == "council_tax_band":
            values = ["A"] * 4
        elif variable == "age":
            values = [20, 30, 40, 50]
        elif variable == "is_child":
            values = [False] * 4
        else:
            values = [1.0] * 4
        return SimpleNamespace(values=np.asarray(values))

    def map_result(self, values, source_entity, target_entity):
        return np.asarray(values)


def _patch_unrelated_sources(monkeypatch, dense_loss):
    from policyengine_uk_data.targets.sources import local_income, local_la_extras

    for module in [local_income, dense_loss]:
        monkeypatch.setattr(module, "INCOME_VARIABLES", [])
        monkeypatch.setattr(
            module, "get_national_income_projections", lambda year: pd.DataFrame()
        )
    monkeypatch.setattr(dense_loss, "Microsimulation", _FakeSim)
    if hasattr(dense_loss, "add_private_rent_targets"):
        monkeypatch.setattr(
            dense_loss, "add_private_rent_targets", lambda *args, **kwargs: None
        )

    codes = _area_metadata("la")["code"].to_numpy()
    loaders = {
        "load_ons_la_income": pd.DataFrame(
            {"la_code": codes, "net_income_bhc": 1.0, "net_income_ahc": 1.0}
        ),
        "load_household_counts": pd.DataFrame({"la_code": codes, "households": 1.0}),
        "load_tenure_data": pd.DataFrame(
            {
                "la_code": codes,
                **{
                    name + "_pct": 25.0
                    for name in [
                        "owned_outright",
                        "owned_mortgage",
                        "private_rent",
                        "social_rent",
                    ]
                },
            }
        ),
        "load_private_rents": pd.DataFrame(
            {"area_code": codes, "median_annual_rent": 1.0}
        ),
    }
    for name, frame in loaders.items():
        for module in [local_la_extras, dense_loss]:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, lambda frame=frame: frame.copy())
    for module in [local_la_extras, dense_loss]:
        if hasattr(module, "get_ons_income_uprating_factors"):
            monkeypatch.setattr(
                module, "get_ons_income_uprating_factors", lambda year: (1.0, 1.0)
            )


@pytest.mark.parametrize("area_type", ["constituency", "la"])
@pytest.mark.parametrize("builder", ["dense", "oa"])
def test_each_builder_uc_row_matches_its_area_and_country(
    monkeypatch, area_type, builder
):
    from policyengine_uk_data.calibration import matrix_builder
    from policyengine_uk_data.datasets.local_areas.constituencies import (
        loss as constituency_loss,
    )
    from policyengine_uk_data.datasets.local_areas.local_authorities import (
        loss as la_loss,
    )
    from policyengine_uk_data.targets.sources import local_age

    loss = constituency_loss if area_type == "constituency" else la_loss
    _patch_unrelated_sources(monkeypatch, loss)
    dataset = SimpleNamespace(time_period=2025)
    areas = _area_metadata(area_type).copy()
    if builder == "dense":
        create = (
            loss.create_constituency_target_matrix
            if area_type == "constituency"
            else loss.create_local_authority_target_matrix
        )
        matrix, y, mask = create(dataset)
        expected_mask = np.equal.outer(
            areas["code"].str[0].map(_COUNTRIES).to_numpy(), list(_COUNTRIES.values())
        )
        np.testing.assert_array_equal(mask, expected_mask)
        assert list(matrix.columns) == list(y.columns)
    else:
        # Assignment rows can arrive in any code order, including interleaved countries.
        areas = areas.sample(frac=1, random_state=468).reset_index(drop=True)
        y = matrix_builder._load_area_targets(area_type, areas, dataset, _FakeSim())
    _, expected = _workbook_oracle(area_type)
    np.testing.assert_array_equal(y["uc_households"], expected.loc[areas["code"]])
    source_age = (
        local_age.get_constituency_age_targets()
        if area_type == "constituency"
        else local_age.get_la_age_targets()
    )
    age_columns = [column for column in source_age if column.startswith("age/")]
    adjustment = (
        0.9
        * local_age.get_uk_total_population(dataset.time_period)
        / source_age[age_columns].to_numpy().sum()
    )
    expected_country_age = (
        source_age[age_columns].groupby(source_age["code"].str[0]).sum() * adjustment
    )
    actual_country_age = y[age_columns].groupby(areas["code"].str[0].to_numpy()).sum()
    pd.testing.assert_frame_equal(
        actual_country_age.sort_index(),
        expected_country_age.sort_index(),
        check_names=False,
        rtol=1e-12,
        atol=1e-7,
    )
    if area_type == "la":
        np.testing.assert_allclose(
            y[age_columns].to_numpy(),
            source_age.set_index("code").loc[areas["code"], age_columns].to_numpy()
            * adjustment,
        )
    if area_type == "constituency":
        children = local_uc.get_constituency_uc_by_children_targets().loc[areas["code"]]
        np.testing.assert_array_equal(y[_CHILD_COLUMNS], children)
