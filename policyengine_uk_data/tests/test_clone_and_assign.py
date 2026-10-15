"""Tests for the clone-and-assign module.

Validates that cloning preserves data integrity and correctly
assigns OA geography to cloned records.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.calibration.clone_and_assign import (
    clone_and_assign,
    _household_country_codes,
    _remap_ids,
)
from policyengine_uk_data.utils.stack import stack_datasets
from policyengine_uk_data.utils.subsample import subsample_dataset


@pytest.fixture(scope="module")
def small_crosswalk(tmp_path_factory) -> Path:
    """Create a small synthetic crosswalk for fast tests."""
    tmp_dir = tmp_path_factory.mktemp("crosswalk")
    path = tmp_dir / "test_crosswalk.csv.gz"

    rows = []
    for i in range(50):
        la = "E09000001" if i < 25 else "E09000002"
        const = "E14001063" if i < 25 else "E14001064"
        rows.append(
            {
                "oa_code": f"E00{i:06d}",
                "lsoa_code": f"E01{i // 5:05d}",
                "msoa_code": f"E02{i // 10:04d}0",
                "la_code": la,
                "constituency_code": const,
                "region_code": "E12000007",
                "country": "England",
                "population": str(100 + i),
            }
        )
    for i in range(20):
        rows.append(
            {
                "oa_code": f"W00{i:06d}",
                "lsoa_code": f"W01{i // 5:05d}",
                "msoa_code": f"W02{i // 10:04d}0",
                "la_code": "W06000001",
                "constituency_code": "W07000041",
                "region_code": "W99999999",
                "country": "Wales",
                "population": str(80 + i),
            }
        )
    for i in range(20):
        rows.append(
            {
                "oa_code": f"S00{i:06d}",
                "lsoa_code": f"S01{i // 5:05d}",
                "msoa_code": f"S02{i // 10:04d}0",
                "la_code": "S12000033",
                "constituency_code": "S14000001",
                "region_code": "S99999999",
                "country": "Scotland",
                "population": str(90 + i),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, compression="gzip")
    return path


@pytest.fixture
def toy_dataset() -> UKSingleYearDataset:
    """Create a minimal synthetic FRS dataset for testing."""
    n_hh = 10
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n_hh + 1),
            "household_weight": np.full(n_hh, 1000.0),
            "region": (["LONDON"] * 5 + ["WALES"] * 3 + ["SCOTLAND"] * 2),
        }
    )

    # Two persons per household
    person_rows = []
    for hh_id in range(1, n_hh + 1):
        for p_idx in range(1, 3):
            person_rows.append(
                {
                    "person_id": hh_id * 1000 + p_idx,
                    "person_household_id": hh_id,
                    "person_benunit_id": hh_id * 100 + 1,
                    "age": 30 + p_idx,
                    "gender": "MALE" if p_idx == 1 else "FEMALE",
                }
            )
    person = pd.DataFrame(person_rows)

    # One benunit per household
    benunit = pd.DataFrame(
        {
            "benunit_id": np.arange(1, n_hh + 1) * 100 + 1,
        }
    )

    return UKSingleYearDataset(
        person=person,
        benunit=benunit,
        household=household,
        fiscal_year=2023,
    )


class TestCloneAndAssign:
    @pytest.mark.parametrize("existing_source", [False, True])
    def test_source_household_id_survives_cloning(
        self, toy_dataset, small_crosswalk, existing_source
    ):
        source_ids = toy_dataset.household["household_id"].to_numpy().copy()
        if existing_source:
            source_ids = np.tile(source_ids[:5], 2)
            toy_dataset.household["source_household_id"] = source_ids
        result = clone_and_assign(
            toy_dataset, n_clones=3, crosswalk_path=str(small_crosswalk)
        )
        np.testing.assert_array_equal(
            result.household["source_household_id"], np.tile(source_ids, 3)
        )

    @pytest.mark.parametrize(
        "source_on_left,source_on_right",
        [(False, False), (True, False), (False, True), (True, True)],
    )
    def test_stacking_preserves_existing_sources(
        self, toy_dataset, source_on_left, source_on_right
    ):
        left, right = toy_dataset.copy(), toy_dataset.copy()
        original_ids = toy_dataset.household["household_id"].to_numpy()
        expected = []
        for dataset, has_source in ((left, source_on_left), (right, source_on_right)):
            source_ids = original_ids * 100 if has_source else original_ids
            if has_source:
                dataset.household["source_household_id"] = source_ids
            expected.extend(source_ids)
        result = stack_datasets(left, right)
        result.validate()
        np.testing.assert_array_equal(result.household["source_household_id"], expected)

    def test_stacked_copies_trace_to_survey_households_after_save(
        self, toy_dataset, small_crosswalk, tmp_path
    ):
        """SPI subsampling, CG doubling and OA cloning retain one source ID."""
        toy_dataset.household["household_is_spi_synthetic"] = False
        toy_dataset.household["household_is_capital_gains_clone"] = False
        spi_copy = subsample_dataset(toy_dataset, 4)
        spi_copy.household["household_is_spi_synthetic"] = True
        source_ids = np.concatenate(
            [toy_dataset.household["household_id"], spi_copy.household["household_id"]]
        )
        stacked = stack_datasets(toy_dataset, spi_copy)
        cg_copy = stacked.copy()
        cg_copy.household["household_is_capital_gains_clone"] = True
        stacked = stack_datasets(stacked, cg_copy)
        result = clone_and_assign(
            stacked, n_clones=3, crosswalk_path=str(small_crosswalk)
        )
        path = tmp_path / "synthetic_enhanced_frs.h5"
        result.save(path)
        household = UKSingleYearDataset(path).household
        original = ~(
            household["household_is_spi_synthetic"]
            | household["household_is_capital_gains_clone"]
        ) & (household["clone_index"] == 0)
        original_ids = household.loc[original, "household_id"]

        assert household["household_id"].is_unique
        assert household["source_household_id"].isin(original_ids).all()
        assert household["source_household_id"].nunique() == len(toy_dataset.household)
        np.testing.assert_array_equal(
            household["source_household_id"], np.tile(source_ids, 2 * 3)
        )

    def test_output_dimensions(self, toy_dataset, small_crosswalk):
        n_clones = 3
        result = clone_and_assign(
            toy_dataset,
            n_clones=n_clones,
            crosswalk_path=str(small_crosswalk),
        )

        n_hh = len(toy_dataset.household)
        n_person = len(toy_dataset.person)
        n_benunit = len(toy_dataset.benunit)

        assert len(result.household) == n_hh * n_clones
        assert len(result.person) == n_person * n_clones
        assert len(result.benunit) == n_benunit * n_clones

    def test_weight_preservation(self, toy_dataset, small_crosswalk):
        """Total population weight should be preserved."""
        original_total = toy_dataset.household["household_weight"].sum()

        result = clone_and_assign(
            toy_dataset,
            n_clones=5,
            crosswalk_path=str(small_crosswalk),
        )

        cloned_total = result.household["household_weight"].sum()
        np.testing.assert_allclose(cloned_total, original_total, rtol=1e-10)

    def test_unique_household_ids(self, toy_dataset, small_crosswalk):
        result = clone_and_assign(
            toy_dataset,
            n_clones=3,
            crosswalk_path=str(small_crosswalk),
        )
        assert result.household["household_id"].is_unique

    def test_unique_person_ids(self, toy_dataset, small_crosswalk):
        result = clone_and_assign(
            toy_dataset,
            n_clones=3,
            crosswalk_path=str(small_crosswalk),
        )
        assert result.person["person_id"].is_unique

    def test_unique_benunit_ids(self, toy_dataset, small_crosswalk):
        result = clone_and_assign(
            toy_dataset,
            n_clones=3,
            crosswalk_path=str(small_crosswalk),
        )
        assert result.benunit["benunit_id"].is_unique

    def test_foreign_key_integrity(self, toy_dataset, small_crosswalk):
        """Every person's household_id should exist in households."""
        result = clone_and_assign(
            toy_dataset,
            n_clones=3,
            crosswalk_path=str(small_crosswalk),
        )

        hh_ids = set(result.household["household_id"].values)
        person_hh_ids = set(result.person["person_household_id"].values)
        assert person_hh_ids.issubset(hh_ids)

        benunit_ids = set(result.benunit["benunit_id"].values)
        person_bu_ids = set(result.person["person_benunit_id"].values)
        assert person_bu_ids.issubset(benunit_ids)

    def test_geography_columns_present(self, toy_dataset, small_crosswalk):
        result = clone_and_assign(
            toy_dataset,
            n_clones=2,
            crosswalk_path=str(small_crosswalk),
        )

        for col in [
            "oa_code",
            "lsoa_code",
            "msoa_code",
            "la_code_oa",
            "constituency_code_oa",
            "region_code_oa",
            "clone_index",
        ]:
            assert col in result.household.columns, f"Missing column: {col}"

    def test_country_constraint(self, toy_dataset, small_crosswalk):
        """English households should get English OAs, etc."""
        result = clone_and_assign(
            toy_dataset,
            n_clones=2,
            crosswalk_path=str(small_crosswalk),
        )

        hh = result.household
        eng_mask = hh["region"].isin(
            [
                "LONDON",
                "NORTH_EAST",
                "SOUTH_EAST",
            ]
        )
        wales_mask = hh["region"] == "WALES"
        scot_mask = hh["region"] == "SCOTLAND"

        # English households get E-prefixed OAs
        assert hh.loc[eng_mask, "oa_code"].str.startswith("E").all()
        # Welsh get W-prefixed
        assert hh.loc[wales_mask, "oa_code"].str.startswith("W").all()
        # Scottish get S-prefixed
        assert hh.loc[scot_mask, "oa_code"].str.startswith("S").all()

    def test_clone_index_values(self, toy_dataset, small_crosswalk):
        n_clones = 4
        result = clone_and_assign(
            toy_dataset,
            n_clones=n_clones,
            crosswalk_path=str(small_crosswalk),
        )

        clone_indices = sorted(result.household["clone_index"].unique())
        assert clone_indices == list(range(n_clones))

    def test_data_preserved_across_clones(self, toy_dataset, small_crosswalk):
        """Non-ID columns should be identical across clones."""
        result = clone_and_assign(
            toy_dataset,
            n_clones=3,
            crosswalk_path=str(small_crosswalk),
        )

        hh = result.household
        # Check that region values are preserved
        for clone_idx in range(3):
            clone_hh = hh[hh["clone_index"] == clone_idx]
            regions = clone_hh["region"].values
            original_regions = toy_dataset.household["region"].values
            np.testing.assert_array_equal(regions, original_regions)

    def test_claimant_or_partner_follows_each_clone(self, toy_dataset, small_crosswalk):
        roles = np.tile([True, False], len(toy_dataset.household))
        toy_dataset.person["is_claimant_or_partner"] = roles
        result = clone_and_assign(
            toy_dataset,
            n_clones=3,
            crosswalk_path=str(small_crosswalk),
        )

        assert result.person["is_claimant_or_partner"].dtype == bool
        np.testing.assert_array_equal(
            result.person["is_claimant_or_partner"].values, np.tile(roles, 3)
        )

    def test_single_clone_is_near_identity(self, toy_dataset, small_crosswalk):
        """With n_clones=1, output should match input dimensions."""
        result = clone_and_assign(
            toy_dataset,
            n_clones=1,
            crosswalk_path=str(small_crosswalk),
        )

        assert len(result.household) == len(toy_dataset.household)
        assert len(result.person) == len(toy_dataset.person)

        # Weights should be identical (divided by 1)
        np.testing.assert_allclose(
            result.household["household_weight"].values,
            toy_dataset.household["household_weight"].values,
        )

    def test_ni_households_gracefully_skipped(self, small_crosswalk):
        """NI households should clone but get empty OA geography."""
        n_hh = 6
        household = pd.DataFrame(
            {
                "household_id": np.arange(1, n_hh + 1),
                "household_weight": np.full(n_hh, 1000.0),
                "region": ["LONDON"] * 3 + ["NORTHERN_IRELAND"] * 3,
            }
        )
        person_rows = []
        for hh_id in range(1, n_hh + 1):
            person_rows.append(
                {
                    "person_id": hh_id * 1000 + 1,
                    "person_household_id": hh_id,
                    "person_benunit_id": hh_id * 100 + 1,
                    "age": 35,
                    "gender": "MALE",
                }
            )
        person = pd.DataFrame(person_rows)
        benunit = pd.DataFrame({"benunit_id": np.arange(1, n_hh + 1) * 100 + 1})

        # small_crosswalk has no NI OAs
        ds = UKSingleYearDataset(
            person=person, benunit=benunit, household=household, fiscal_year=2023
        )

        result = clone_and_assign(ds, n_clones=2, crosswalk_path=str(small_crosswalk))

        # Should not crash — NI households get empty geography
        assert len(result.household) == n_hh * 2
        hh = result.household
        ni_mask = hh["region"] == "NORTHERN_IRELAND"
        assert (hh.loc[ni_mask, "oa_code"] == "").all()
        # English households should still have OA codes
        eng_mask = hh["region"] == "LONDON"
        assert hh.loc[eng_mask, "oa_code"].str.startswith("E").all()


class TestHelpers:
    def test_remap_ids_clone_zero(self):
        ids = np.array([100, 200, 300])
        result = _remap_ids(ids, clone_idx=0, id_multiplier=1000)
        np.testing.assert_array_equal(result, ids)

    def test_remap_ids_clone_nonzero(self):
        ids = np.array([100, 200, 300])
        result = _remap_ids(ids, clone_idx=2, id_multiplier=1000)
        expected = np.array([2100, 2200, 2300])
        np.testing.assert_array_equal(result, expected)

    def test_household_country_codes(self, toy_dataset):
        codes = _household_country_codes(toy_dataset)
        # 5 London (England=1), 3 Wales (2), 2 Scotland (3)
        expected = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 3])
        np.testing.assert_array_equal(codes, expected)
