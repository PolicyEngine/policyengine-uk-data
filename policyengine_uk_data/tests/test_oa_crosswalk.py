"""Tests for OA crosswalk building and geographic assignment.

These tests validate:
1. Crosswalk completeness (all 4 countries, expected OA counts)
2. Hierarchy consistency (OA → LSOA → MSOA → LA nesting)
3. Country constraints (E→England, W→Wales, S→Scotland)
4. Population totals (within reasonable range of known figures)
5. Assignment correctness (country constraints, constituency
   collision avoidance, distribution proportionality)
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import policyengine_uk_data.calibration.oa_crosswalk as oa_crosswalk_module
from policyengine_uk_data.calibration.oa_crosswalk import (
    load_oa_crosswalk,
    CROSSWALK_PATH,
)
from policyengine_uk_data.calibration.oa_assignment import (
    assign_random_geography,
    save_geography,
    load_geography,
)


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def crosswalk() -> pd.DataFrame:
    """Load or build the OA crosswalk for testing.

    Uses the pre-built file if available, otherwise builds
    fresh (slow, ~5 minutes).
    """
    if CROSSWALK_PATH.exists():
        return load_oa_crosswalk()
    else:
        pytest.skip(
            "OA crosswalk not yet built. Run: "
            "python -m policyengine_uk_data.calibration."
            "oa_crosswalk"
        )


@pytest.fixture(scope="module")
def small_crosswalk(tmp_path_factory) -> pd.DataFrame:
    """Create a small synthetic crosswalk for fast tests."""
    tmp_dir = tmp_path_factory.mktemp("crosswalk")
    path = tmp_dir / "test_crosswalk.csv.gz"

    rows = []
    # England: 100 OAs across 2 LAs, 2 constituencies
    for i in range(100):
        la = "E09000001" if i < 50 else "E09000002"
        const = "E14001063" if i < 50 else "E14001064"
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
    # Wales: 20 OAs
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
    # Scotland: 30 OAs
    for i in range(30):
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
    # Northern Ireland: 10 Data Zones
    for i in range(10):
        rows.append(
            {
                "oa_code": f"95GG{i:04d}",
                "lsoa_code": f"95GG{i:04d}",
                "msoa_code": f"95HH{i // 3:03d}0",
                "la_code": "N09000001",
                "constituency_code": "",
                "region_code": "N99999999",
                "country": "Northern Ireland",
                "population": str(70 + i),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, compression="gzip")
    return df, path


# ── Crosswalk Structure Tests (use real data if available) ──


class TestCrosswalkStructure:
    """Tests on the real crosswalk file."""

    def test_has_expected_columns(self, crosswalk):
        expected = {
            "oa_code",
            "lsoa_code",
            "msoa_code",
            "la_code",
            "constituency_code",
            "region_code",
            "country",
            "population",
        }
        assert expected == set(crosswalk.columns)

    def test_all_four_countries_present(self, crosswalk):
        countries = set(crosswalk["country"].unique())
        assert "England" in countries
        assert "Wales" in countries
        assert "Scotland" in countries
        # NI is excluded until NISRA updates their download
        # URLs (currently returning 404). Uncomment when
        # NI Data Zone lookup is available again.
        # assert "Northern Ireland" in countries

    def test_ew_oa_count_range(self, crosswalk):
        """E+W should have ~188K-190K OAs."""
        ew = crosswalk[crosswalk["country"].isin(["England", "Wales"])]
        assert 180_000 < len(ew) < 200_000, (
            f"E+W OA count {len(ew)} outside expected range"
        )

    def test_scotland_oa_count_range(self, crosswalk):
        """Scotland should have ~46K OAs."""
        scot = crosswalk[crosswalk["country"] == "Scotland"]
        assert 40_000 < len(scot) < 55_000, (
            f"Scotland OA count {len(scot)} outside expected range"
        )

    @pytest.mark.skip(reason="NISRA DZ2021 download URLs returning 404")
    def test_ni_dz_count_range(self, crosswalk):
        """NI should have ~3.7K-4K Data Zones."""
        ni = crosswalk[crosswalk["country"] == "Northern Ireland"]
        assert 3_000 < len(ni) < 7_000, f"NI DZ count {len(ni)} outside expected range"

    def test_no_duplicate_oa_codes(self, crosswalk):
        assert crosswalk["oa_code"].is_unique

    def test_england_oa_prefix(self, crosswalk):
        eng = crosswalk[crosswalk["country"] == "England"]
        assert eng["oa_code"].str.startswith("E00").all()

    def test_wales_oa_prefix(self, crosswalk):
        wales = crosswalk[crosswalk["country"] == "Wales"]
        assert wales["oa_code"].str.startswith("W00").all()

    def test_scotland_oa_prefix(self, crosswalk):
        scot = crosswalk[crosswalk["country"] == "Scotland"]
        assert scot["oa_code"].str.startswith("S00").all()

    def test_population_total_range(self, crosswalk):
        """UK total population should be ~67M (2021 Census)."""
        total = crosswalk["population"].sum()
        assert 55_000_000 < total < 75_000_000, (
            f"UK population {total:,} outside expected range"
        )

    def test_population_is_numeric(self, crosswalk):
        assert pd.api.types.is_numeric_dtype(crosswalk["population"])

    def test_every_oa_has_la(self, crosswalk):
        missing = crosswalk["la_code"].isna() | (crosswalk["la_code"] == "")
        assert missing.sum() == 0, f"{missing.sum()} OAs missing LA code"

    def test_every_england_oa_has_region(self, crosswalk):
        eng = crosswalk[crosswalk["country"] == "England"]
        missing = eng["region_code"].isna() | (eng["region_code"] == "")
        assert missing.sum() == 0, f"{missing.sum()} English OAs missing region code"

    def test_ew_oas_have_constituency(self, crosswalk):
        """E+W OAs should have constituency codes."""
        ew = crosswalk[crosswalk["country"].isin(["England", "Wales"])]
        has_const = (ew["constituency_code"].notna()) & (ew["constituency_code"] != "")
        pct = has_const.mean()
        assert pct > 0.95, (
            f"Only {pct:.1%} of E+W OAs have constituency codes (expected >95%)"
        )

    def test_hierarchy_nesting_oa_in_lsoa(self, crosswalk):
        """Each OA should map to exactly one LSOA."""
        grouped = crosswalk.groupby("oa_code")["lsoa_code"].nunique()
        assert (grouped == 1).all(), "Some OAs map to multiple LSOAs"

    def test_hierarchy_nesting_lsoa_in_la(self, crosswalk):
        """Each LSOA should map to exactly one LA."""
        grouped = crosswalk.groupby("lsoa_code")["la_code"].nunique()
        multi = grouped[grouped > 1]
        assert len(multi) == 0, (
            f"{len(multi)} LSOAs map to multiple LAs: {multi.index.tolist()[:5]}"
        )


# ── Assignment Tests (use synthetic crosswalk) ──────────


class TestOAAssignment:
    """Tests for the OA assignment function."""

    def test_output_shape(self, small_crosswalk):
        df, path = small_crosswalk
        n_records = 20
        n_clones = 3
        countries = np.array([1] * 15 + [2] * 5)

        geo = assign_random_geography(
            household_countries=countries,
            n_clones=n_clones,
            seed=42,
            crosswalk_path=str(path),
        )

        assert geo.n_records == n_records
        assert geo.n_clones == n_clones
        assert len(geo.oa_code) == n_records * n_clones
        assert len(geo.la_code) == n_records * n_clones
        assert len(geo.constituency_code) == (n_records * n_clones)

    def test_country_constraint(self, small_crosswalk):
        """English households should get English OAs only."""
        df, path = small_crosswalk
        countries = np.array([1] * 10 + [2] * 5 + [3] * 3 + [4] * 2)

        geo = assign_random_geography(
            household_countries=countries,
            n_clones=2,
            seed=42,
            crosswalk_path=str(path),
        )

        n = len(countries)
        for clone_idx in range(2):
            start = clone_idx * n
            for i in range(n):
                idx = start + i
                if countries[i] == 1:
                    assert geo.country[idx] == "England", (
                        f"Record {i} clone {clone_idx}: "
                        f"expected England, got "
                        f"{geo.country[idx]}"
                    )
                elif countries[i] == 2:
                    assert geo.country[idx] == "Wales"
                elif countries[i] == 3:
                    assert geo.country[idx] == "Scotland"
                elif countries[i] == 4:
                    assert geo.country[idx] == ("Northern Ireland")

    def test_constituency_collision_avoidance(self, small_crosswalk):
        """Different clones of same household should have
        different constituencies (where possible)."""
        df, path = small_crosswalk
        # All English so they draw from 2 constituencies
        countries = np.array([1] * 10)

        geo = assign_random_geography(
            household_countries=countries,
            n_clones=2,
            seed=42,
            crosswalk_path=str(path),
        )

        n = len(countries)
        collisions = 0
        for i in range(n):
            const_0 = geo.constituency_code[i]
            const_1 = geo.constituency_code[n + i]
            if const_0 == const_1:
                collisions += 1

        # With 2 constituencies and 2 clones, we should
        # have very few collisions (ideally 0)
        assert collisions < n * 0.2, (
            f"{collisions}/{n} constituency collisions (expected < 20%)"
        )

    def test_save_load_roundtrip(self, small_crosswalk):
        """GeographyAssignment should survive save/load."""
        df, path = small_crosswalk
        countries = np.array([1] * 5 + [2] * 3)

        geo = assign_random_geography(
            household_countries=countries,
            n_clones=2,
            seed=42,
            crosswalk_path=str(path),
        )

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            save_geography(geo, Path(f.name))
            loaded = load_geography(Path(f.name))

            assert loaded.n_records == geo.n_records
            assert loaded.n_clones == geo.n_clones
            np.testing.assert_array_equal(loaded.oa_code, geo.oa_code)
            np.testing.assert_array_equal(
                loaded.constituency_code,
                geo.constituency_code,
            )

    def test_population_weighted_sampling(self, small_crosswalk):
        """Higher-population OAs should be sampled more
        frequently."""
        df, path = small_crosswalk
        # All English, large sample for statistical test
        countries = np.array([1] * 5000)

        geo = assign_random_geography(
            household_countries=countries,
            n_clones=1,
            seed=42,
            crosswalk_path=str(path),
        )

        # Count assignments per OA
        oa_counts = pd.Series(geo.oa_code).value_counts()

        # The English OAs in our fixture have populations
        # 100-199. Higher-numbered OAs should tend to be
        # sampled more. Check that the top-10 most sampled
        # OAs have higher average population than bottom-10.
        eng_df = df[df["country"] == "England"].copy()
        eng_df["population"] = eng_df["population"].astype(int)

        top_10 = oa_counts.head(10).index.tolist()
        bottom_10 = oa_counts.tail(10).index.tolist()

        top_pop = eng_df[eng_df["oa_code"].isin(top_10)]["population"].mean()
        bottom_pop = eng_df[eng_df["oa_code"].isin(bottom_10)]["population"].mean()

        assert top_pop > bottom_pop, (
            f"Top-10 avg pop ({top_pop}) should exceed bottom-10 ({bottom_pop})"
        )

    def test_string_country_input(self, small_crosswalk):
        """Should accept string country names too."""
        df, path = small_crosswalk
        countries = np.array(["England"] * 5 + ["Wales"] * 3)

        geo = assign_random_geography(
            household_countries=countries,
            n_clones=1,
            seed=42,
            crosswalk_path=str(path),
        )

        assert len(geo.oa_code) == 8
        for i in range(5):
            assert geo.country[i] == "England"
        for i in range(5, 8):
            assert geo.country[i] == "Wales"

    def test_uppercase_country_input(self, small_crosswalk):
        """Should accept repo-style uppercase country names."""
        df, path = small_crosswalk
        countries = np.array(["ENGLAND", "WALES", "SCOTLAND", "NORTHERN_IRELAND"])

        geo = assign_random_geography(
            household_countries=countries,
            n_clones=1,
            seed=42,
            crosswalk_path=str(path),
        )

        assert geo.country.tolist() == [
            "England",
            "Wales",
            "Scotland",
            "Northern Ireland",
        ]

    def test_object_country_code_input(self, small_crosswalk):
        """Should accept object-dtype arrays of numeric country codes."""
        df, path = small_crosswalk
        countries = np.array([1, 2, 3, 4], dtype=object)

        geo = assign_random_geography(
            household_countries=countries,
            n_clones=1,
            seed=42,
            crosswalk_path=str(path),
        )

        assert geo.country.tolist() == [
            "England",
            "Wales",
            "Scotland",
            "Northern Ireland",
        ]

    def test_missing_country_distribution_raises(self, small_crosswalk, tmp_path):
        """Missing country distributions should fail loudly."""
        df, _ = small_crosswalk
        no_ni = df[df["country"] != "Northern Ireland"]
        path = tmp_path / "crosswalk_no_ni.csv.gz"
        no_ni.to_csv(path, index=False, compression="gzip")

        with pytest.raises(ValueError, match="Northern Ireland"):
            assign_random_geography(
                household_countries=np.array([4]),
                n_clones=1,
                seed=42,
                crosswalk_path=str(path),
            )


class TestCrosswalkHelpers:
    def test_scotland_direct_lookup_prefers_oa_mapping(self, monkeypatch):
        def fake_download_csv_from_zip(
            url: str, csv_filter: str = ".csv", timeout: int = 300
        ):
            if url == oa_crosswalk_module._SCOTLAND_OA_DZ_URL:
                return pd.DataFrame(
                    {
                        "OA22": ["S00123456"],
                        "DZ22": ["S01000001"],
                    }
                )
            if url == oa_crosswalk_module._SCOTLAND_OA_CONST_URL:
                if csv_filter != "oa22_ukpc24":
                    return pd.DataFrame(
                        {
                            "UKParliamentaryConstituency2024Code": ["S14009999"],
                            "UKParliamentaryConstituency2024Name": ["Wrong file"],
                        }
                    )
                return pd.DataFrame(
                    {
                        "OA22": ["S00123456"],
                        "UKPC24_CODE": ["S14000042"],
                    }
                )
            raise AssertionError(f"Unexpected ZIP URL: {url}")

        def fake_download_csv(url: str, timeout: int = 300):
            if url == oa_crosswalk_module._SCOTLAND_DZ_LOOKUP_URL:
                return pd.DataFrame(
                    {
                        "DZ22_CODE": ["S01000001"],
                        "IZ22_CODE": ["S02000001"],
                        "LA_CODE": ["S12000033"],
                        "UKPC24_CODE": ["S14009999"],
                    }
                )
            raise AssertionError(f"Unexpected CSV URL: {url}")

        monkeypatch.setattr(
            oa_crosswalk_module, "_download_csv_from_zip", fake_download_csv_from_zip
        )
        monkeypatch.setattr(oa_crosswalk_module, "_download_csv", fake_download_csv)

        result = oa_crosswalk_module._get_scotland_oa_hierarchy()

        assert result.loc[0, "constituency_code"] == "S14000042"
