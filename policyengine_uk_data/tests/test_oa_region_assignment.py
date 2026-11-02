"""Invariants for region-constrained Output Area assignment.

A household's OA is drawn from its own FRS region, so the OA's region,
LA and constituency can never contradict ``household.region``. The
property tests state that for every input; the example tests pin the
edge cases and the real crosswalk's shape.
"""

import itertools
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from policyengine_uk.data import UKSingleYearDataset

from policyengine_uk_data.calibration.clone_and_assign import (
    _household_country_codes,
    clone_and_assign,
)
from policyengine_uk_data.calibration.oa_assignment import (
    FRS_REGION_TO_CODE,
    _normalise_region,
    assign_random_geography,
)
from policyengine_uk_data.calibration.oa_crosswalk import (
    CROSSWALK_PATH,
    load_oa_crosswalk,
)
from policyengine_uk_data.storage import STORAGE_FOLDER

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

REGION_TO_COUNTRY = {
    region: {"E": "England", "W": "Wales", "S": "Scotland", "N": "Northern Ireland"}[
        code[0]
    ]
    for region, code in FRS_REGION_TO_CODE.items()
}
COUNTRY_TO_FRS_CODE = {"England": 1, "Wales": 2, "Scotland": 3, "Northern Ireland": 4}
GEOGRAPHY_FIELDS = {
    "lsoa_code": "lsoa_code",
    "msoa_code": "msoa_code",
    "la_code": "la_code",
    "constituency_code": "constituency_code",
    "region_code": "region_code",
}
_file_counter = itertools.count()


def _crosswalk_rows(region: str, constituency_populations: list[list[int]]) -> list:
    """OAs for one region: one LA per constituency, one OA per population."""
    code = FRS_REGION_TO_CODE[region]
    rows = []
    for c, populations in enumerate(constituency_populations):
        for o, population in enumerate(populations):
            rows.append(
                {
                    "oa_code": f"{code[0]}00{code[-2:]}{c:02d}{o:02d}",
                    "lsoa_code": f"{code[0]}01{code[-2:]}{c:02d}{o:02d}",
                    "msoa_code": f"{code[0]}02{code[-2:]}{c:02d}",
                    "la_code": f"{code[0]}06{code[-2:]}{c:02d}",
                    "constituency_code": f"{code[0]}14{code[-2:]}{c:02d}",
                    "region_code": code,
                    "country": REGION_TO_COUNTRY[region],
                    "population": population,
                }
            )
    return rows


def _write_crosswalk(directory: Path, rows: list) -> tuple[pd.DataFrame, str]:
    """Write to a fresh path: the loaders cache on the path string."""
    frame = pd.DataFrame(rows)
    path = Path(directory) / f"crosswalk_{next(_file_counter)}.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    return frame, str(path)


def _countries(countries: list[str]) -> np.ndarray:
    return np.array([COUNTRY_TO_FRS_CODE[c] for c in countries])


# Ways a region can arrive: FRS name, crosswalk code, bytes, untidy text.
REPRESENTATIONS = {
    "name": lambda region: region,
    "code": lambda region: FRS_REGION_TO_CODE[region],
    "bytes": lambda region: region.encode(),
    "untidy": lambda region: f"  {region.lower()} ",
}
UNKNOWN_VALUES = ["UNKNOWN", "", None, np.nan, b"UNKNOWN"]


def _toy_dataset(regions: list, weights: list[float]) -> UKSingleYearDataset:
    ids = np.arange(1, len(regions) + 1)
    return UKSingleYearDataset(
        person=pd.DataFrame(
            {
                "person_id": ids * 1000 + 1,
                "person_household_id": ids,
                "person_benunit_id": ids * 100 + 1,
                "age": 40,
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids * 100 + 1}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": weights,
                "region": pd.Series(regions, dtype=object),
            }
        ),
        fiscal_year=2024,
    )


@st.composite
def households(draw, crosswalk_regions: list[str], unknown_countries: list[str]):
    """Households as (FRS region or None, country, raw region value)."""
    known = [(region, REGION_TO_COUNTRY[region]) for region in crosswalk_regions]
    unknown = [(None, country) for country in unknown_countries]
    region, country = draw(st.sampled_from(known + unknown))
    if region is None:
        raw = draw(st.sampled_from(UNKNOWN_VALUES))
    else:
        raw = REPRESENTATIONS[draw(st.sampled_from(sorted(REPRESENTATIONS)))](region)
    return region, country, raw


@st.composite
def crosswalk_and_households(draw):
    """A synthetic crosswalk plus households the sampler can serve.

    A household with no region can belong to any country in the crosswalk.
    """
    crosswalk_regions = draw(
        st.lists(
            st.sampled_from(sorted(FRS_REGION_TO_CODE)),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    rows = []
    for region in crosswalk_regions:
        rows += _crosswalk_rows(
            region,
            draw(
                st.lists(
                    st.lists(st.integers(0, 500), min_size=1, max_size=4),
                    min_size=1,
                    max_size=4,
                )
            ),
        )
    countries = sorted({REGION_TO_COUNTRY[r] for r in crosswalk_regions})
    sample = draw(
        st.lists(households(crosswalk_regions, countries), min_size=1, max_size=12)
    )
    return rows, sample


def _as_tuple(geography) -> tuple:
    return tuple(
        tuple(getattr(geography, field))
        for field in ["oa_code", "country", *GEOGRAPHY_FIELDS]
    )


class TestRegionConstraintProperties:
    @given(
        case=crosswalk_and_households(),
        n_clones=st.integers(1, 4),
        seed=st.integers(0, 2**32 - 1),
    )
    @settings(max_examples=80, deadline=None)
    def test_assignment_invariants(self, case, n_clones, seed):
        rows, sample = case
        regions, countries, raw = map(list, zip(*sample))
        with tempfile.TemporaryDirectory() as directory:
            crosswalk, path = _write_crosswalk(directory, rows)
            kwargs = dict(
                household_countries=_countries(countries),
                household_regions=np.array(raw, dtype=object),
                n_clones=n_clones,
                seed=seed,
                crosswalk_path=path,
            )
            geography = assign_random_geography(**kwargs)
            repeat = assign_random_geography(**kwargs)

        # Determinism: same inputs and seed, same assignment.
        assert _as_tuple(geography) == _as_tuple(repeat)

        n = len(sample)
        assert len(geography.oa_code) == n * n_clones
        by_oa = crosswalk.set_index("oa_code")
        region_population = crosswalk.groupby("region_code")["population"].sum()
        country_population = crosswalk.groupby("country")["population"].sum()
        for i, oa in enumerate(geography.oa_code):
            region, country = regions[i % n], countries[i % n]
            row = by_oa.loc[oa]
            # The OA's region is the household's region; a household with
            # no region stays in its country.
            assert row["country"] == country == geography.country[i]
            if region is None:
                stratum_population = country_population[country]
            else:
                assert geography.region_code[i] == FRS_REGION_TO_CODE[region]
                stratum_population = region_population[FRS_REGION_TO_CODE[region]]
            # Every other code is the crosswalk's own row for that OA.
            for field, column in GEOGRAPHY_FIELDS.items():
                assert getattr(geography, field)[i] == row[column]
            # Unpopulated OAs are never drawn while the stratum has people.
            if stratum_population > 0:
                assert row["population"] > 0

    @given(
        case=crosswalk_and_households(),
        uncovered=st.lists(
            st.sampled_from(
                [("NORTHERN_IRELAND", "Northern Ireland"), (None, "England")]
            ),
            max_size=3,
        ),
        weights=st.lists(
            st.floats(0, 5_000, allow_nan=False), min_size=15, max_size=15
        ),
        n_clones=st.integers(1, 4),
        seed=st.integers(0, 2**32 - 1),
    )
    @settings(max_examples=50, deadline=None)
    def test_clone_and_assign_invariants(
        self, case, uncovered, weights, n_clones, seed
    ):
        rows, sample = case
        # clone_and_assign reads a household without a region as English.
        sample = [(r, "England" if r is None else c, raw) for r, c, raw in sample]
        covered_countries = {row["country"] for row in rows}
        sample += [
            (region, country, region or "UNKNOWN")
            for region, country in uncovered
            if country not in covered_countries
        ]
        regions, countries, raw = map(list, zip(*sample))
        dataset = _toy_dataset(raw, weights[: len(sample)])
        with tempfile.TemporaryDirectory() as directory:
            _, path = _write_crosswalk(directory, rows)
            kwargs = dict(n_clones=n_clones, seed=seed, crosswalk_path=path)
            household = clone_and_assign(dataset, **kwargs).household
            repeat = clone_and_assign(dataset, **kwargs).household

        pd.testing.assert_frame_equal(household, repeat)

        original = dataset.household
        # Weights are preserved in total and for every source household.
        np.testing.assert_allclose(
            household["household_weight"].sum(),
            original["household_weight"].sum(),
            rtol=1e-12,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            household.groupby("source_household_id")["household_weight"].sum().values,
            original.set_index("household_id")["household_weight"].values,
            rtol=1e-12,
            atol=1e-9,
        )

        # Cloning never rewrites the FRS region (a missing value may come
        # back as None or NaN; both are missing).
        def tidy(values):
            return [None if pd.isna(v) else v for v in values]

        assert tidy(household["region"]) == tidy(original["region"]) * n_clones

        for i, (region, country) in enumerate(
            zip(regions * n_clones, countries * n_clones)
        ):
            assigned = household.iloc[i]
            if country not in covered_countries:
                # A country the crosswalk does not cover gets no geography.
                for column in ["oa_code", "la_code_oa", "constituency_code_oa"]:
                    assert assigned[column] == ""
            elif region is None:
                assert assigned["region_code_oa"].startswith("E12")
            else:
                assert assigned["region_code_oa"] == FRS_REGION_TO_CODE[region]
                assert assigned["oa_code"] != ""

    @given(
        unknown_countries=st.lists(
            st.sampled_from(["England", "Wales", "Scotland", "Northern Ireland"]),
            max_size=6,
        ),
        known=st.lists(
            st.sampled_from(["LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND"]),
            max_size=6,
        ),
        order=st.randoms(use_true_random=False),
        n_clones=st.integers(1, 4),
        seed=st.integers(0, 2**32 - 1),
    )
    @settings(max_examples=60, deadline=None)
    def test_matches_country_sampling_when_region_is_the_country(
        self, unknown_countries, known, order, n_clones, seed
    ):
        """Differential: when each country is one region, giving regions
        changes nothing, even with households of unknown region mixed in."""
        sample = [(None, c) for c in unknown_countries] + [
            (r, REGION_TO_COUNTRY[r]) for r in known
        ] or [("LONDON", "England")]
        order.shuffle(sample)
        rows = []
        for region in ["LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND"]:
            rows += _crosswalk_rows(region, [[30, 10, 0], [5, 55], [20]])
        with tempfile.TemporaryDirectory() as directory:
            _, path = _write_crosswalk(directory, rows)
            kwargs = dict(
                household_countries=_countries([c for _, c in sample]),
                n_clones=n_clones,
                seed=seed,
                crosswalk_path=path,
            )
            by_country = assign_random_geography(**kwargs)
            by_region = assign_random_geography(
                household_regions=np.array(
                    [r or "UNKNOWN" for r, _ in sample], dtype=object
                ),
                **kwargs,
            )
        assert _as_tuple(by_country) == _as_tuple(by_region)


class TestRegionConstraintExamples:
    @pytest.fixture(scope="class")
    def two_region_crosswalk(self, tmp_path_factory):
        rows = _crosswalk_rows("LONDON", [[100, 300]]) + _crosswalk_rows(
            "NORTH_EAST", [[50_000], [50_000]]
        )
        return _write_crosswalk(tmp_path_factory.mktemp("two_region"), rows)

    def test_sampling_is_population_weighted_within_the_region(
        self, two_region_crosswalk
    ):
        """London's two OAs split 1:3 however large the North East is."""
        crosswalk, path = two_region_crosswalk
        n = 20_000
        geography = assign_random_geography(
            household_countries=np.ones(n, dtype=int),
            household_regions=np.array(["LONDON"] * n, dtype=object),
            n_clones=1,
            seed=0,
            crosswalk_path=path,
        )
        assert set(geography.region_code) == {"E12000007"}
        larger = crosswalk.loc[crosswalk["population"] == 300, "oa_code"].iloc[0]
        share = (geography.oa_code == larger).mean()
        # Binomial(20,000, 0.75): four standard errors is 0.012.
        assert abs(share - 0.75) < 0.012

    def test_country_sampling_without_regions_is_unchanged(self, two_region_crosswalk):
        """Callers that know only the country still sample country-wide."""
        _, path = two_region_crosswalk
        geography = assign_random_geography(
            household_countries=np.ones(2_000, dtype=int),
            n_clones=1,
            seed=0,
            crosswalk_path=path,
        )
        assert (geography.region_code == "E12000001").mean() > 0.9

    def test_unknown_region_falls_back_to_the_country(self, two_region_crosswalk):
        _, path = two_region_crosswalk
        for unknown in ["UNKNOWN", "", None, np.nan]:
            geography = assign_random_geography(
                household_countries=np.ones(500, dtype=int),
                household_regions=np.array([unknown] * 500, dtype=object),
                n_clones=1,
                seed=0,
                crosswalk_path=path,
            )
            assert set(geography.region_code) == {"E12000001", "E12000007"}

    def test_region_names_codes_and_bytes_are_equivalent(self, two_region_crosswalk):
        _, path = two_region_crosswalk
        results = [
            assign_random_geography(
                household_countries=np.ones(50, dtype=int),
                household_regions=np.array([value] * 50, dtype=object),
                n_clones=2,
                seed=3,
                crosswalk_path=path,
            )
            for value in ["LONDON", "E12000007", b"LONDON", " london "]
        ]
        assert len({_as_tuple(result) for result in results}) == 1

    def test_region_without_oas_raises(self, two_region_crosswalk):
        _, path = two_region_crosswalk
        with pytest.raises(ValueError, match=r"E12000009 \(England\)"):
            assign_random_geography(
                household_countries=np.array([1]),
                household_regions=np.array(["SOUTH_WEST"], dtype=object),
                n_clones=1,
                crosswalk_path=path,
            )

    def test_region_in_the_wrong_country_raises(self, two_region_crosswalk):
        _, path = two_region_crosswalk
        with pytest.raises(ValueError, match="disagree"):
            assign_random_geography(
                household_countries=np.array([2]),
                household_regions=np.array(["LONDON"], dtype=object),
                n_clones=1,
                crosswalk_path=path,
            )

    def test_unrecognised_region_raises(self, two_region_crosswalk):
        _, path = two_region_crosswalk
        with pytest.raises(ValueError, match="Unrecognised household region"):
            assign_random_geography(
                household_countries=np.array([1]),
                household_regions=np.array(["MERCIA"], dtype=object),
                n_clones=1,
                crosswalk_path=path,
            )

    def test_region_length_mismatch_raises(self, two_region_crosswalk):
        _, path = two_region_crosswalk
        with pytest.raises(ValueError, match="expected 2"):
            assign_random_geography(
                household_countries=np.array([1, 1]),
                household_regions=np.array(["LONDON"], dtype=object),
                n_clones=1,
                crosswalk_path=path,
            )

    def test_clones_take_different_constituencies_within_the_region(self, tmp_path):
        """Collision avoidance still works when the pool is one region."""
        rows = _crosswalk_rows("NORTH_EAST", [[100]] * 12) + _crosswalk_rows(
            "LONDON", [[100]] * 40
        )
        _, path = _write_crosswalk(tmp_path, rows)
        n, n_clones = 200, 10
        geography = assign_random_geography(
            household_countries=np.ones(n, dtype=int),
            household_regions=np.array(["NORTH_EAST"] * n, dtype=object),
            n_clones=n_clones,
            seed=42,
            crosswalk_path=path,
        )
        constituencies = geography.constituency_code.reshape(n_clones, n)
        assert set(geography.region_code) == {"E12000001"}
        assert all(len(set(constituencies[:, i])) == n_clones for i in range(n))

    def test_single_constituency_region_terminates(self, tmp_path):
        """Unavoidable collisions exhaust the retries and still assign."""
        _, path = _write_crosswalk(tmp_path, _crosswalk_rows("WALES", [[10, 20]]))
        geography = assign_random_geography(
            household_countries=np.full(5, 2),
            household_regions=np.array(["WALES"] * 5, dtype=object),
            n_clones=3,
            seed=1,
            crosswalk_path=path,
        )
        assert len(set(geography.constituency_code)) == 1

    def test_zero_population_region_samples_uniformly(self, tmp_path):
        """A region with no recorded population draws its OAs uniformly."""
        rows = _crosswalk_rows("WALES", [[0, 0], [0, 0]]) + _crosswalk_rows(
            "LONDON", [[100]]
        )
        crosswalk, path = _write_crosswalk(tmp_path, rows)
        n = 8_000
        geography = assign_random_geography(
            household_countries=np.full(n, 2),
            household_regions=np.array(["WALES"] * n, dtype=object),
            n_clones=1,
            seed=5,
            crosswalk_path=path,
        )
        shares = pd.Series(geography.oa_code).value_counts(normalize=True)
        welsh = set(crosswalk.loc[crosswalk["country"] == "Wales", "oa_code"])
        assert set(shares.index) == welsh
        # Binomial(8,000, 0.25): four standard errors is 0.019.
        assert (shares - 0.25).abs().max() < 0.02

    def test_mixed_unknown_and_sole_region_draw_like_the_country(self, tmp_path):
        """Review counterexample: an UNKNOWN and a London household, London
        being England's only region, draw exactly as a country-only call."""
        rows = (
            _crosswalk_rows("LONDON", [[100], [100]])
            + _crosswalk_rows("WALES", [[100]])
            + _crosswalk_rows("SCOTLAND", [[100]])
            + _crosswalk_rows("NORTHERN_IRELAND", [[100]])
        )
        _, path = _write_crosswalk(tmp_path, rows)
        kwargs = dict(
            household_countries=np.array([1, 1]),
            n_clones=1,
            seed=0,
            crosswalk_path=path,
        )
        by_country = assign_random_geography(**kwargs)
        by_region = assign_random_geography(
            household_regions=np.array(["UNKNOWN", "LONDON"], dtype=object),
            **kwargs,
        )
        assert _as_tuple(by_country) == _as_tuple(by_region)

    def test_clone_and_assign_reads_untidy_regions_like_the_sampler(self, tmp_path):
        """Review counterexample: bytes, codes and untidy text give the
        same country in cloning as in the sampler."""
        rows = _crosswalk_rows("WALES", [[100]]) + _crosswalk_rows("LONDON", [[100]])
        _, path = _write_crosswalk(tmp_path, rows)
        raw = [b"WALES", " wales ", "W99999999", b"NORTHERN_IRELAND", "UNKNOWN"]
        assert _household_country_codes(
            _toy_dataset(raw, [1.0] * len(raw))
        ).tolist() == [2, 2, 2, 4, 1]
        household = clone_and_assign(
            _toy_dataset(raw, [1.0] * len(raw)), n_clones=1, crosswalk_path=path
        ).household
        assert household["region_code_oa"].tolist() == [
            "W99999999",
            "W99999999",
            "W99999999",
            "",
            "E12000007",
        ]


@pytest.fixture(scope="module")
def real_crosswalk() -> pd.DataFrame:
    if not CROSSWALK_PATH.exists():
        pytest.skip("OA crosswalk not built")
    return load_oa_crosswalk()


class TestRealCrosswalk:
    def test_region_codes_name_the_right_places(self, real_crosswalk):
        """FRS_REGION_TO_CODE agrees with where well-known LAs are."""
        names = pd.read_csv(STORAGE_FOLDER / "local_authorities_2021.csv")
        la_region = (
            real_crosswalk[["la_code", "region_code"]]
            .drop_duplicates()
            .merge(names, left_on="la_code", right_on="code")
            .set_index("name")["region_code"]
        )
        expected = {
            "Hartlepool": "NORTH_EAST",
            "Manchester": "NORTH_WEST",
            "Leeds": "YORKSHIRE",
            "Nottingham": "EAST_MIDLANDS",
            "Birmingham": "WEST_MIDLANDS",
            "Cambridge": "EAST_OF_ENGLAND",
            "Westminster": "LONDON",
            "Brighton and Hove": "SOUTH_EAST",
            "Bristol, City of": "SOUTH_WEST",
            "Cardiff": "WALES",
            "Glasgow City": "SCOTLAND",
        }
        for name, region in expected.items():
            assert la_region[name] == FRS_REGION_TO_CODE[region], name

    def test_every_crosswalk_region_is_an_frs_region(self, real_crosswalk):
        assert set(real_crosswalk["region_code"]) <= set(FRS_REGION_TO_CODE.values())

    def test_local_areas_nest_in_regions(self, real_crosswalk):
        """No LA or constituency straddles a region, so sampling within the
        region can still reach every OA of every local area."""
        for column in ["la_code", "constituency_code"]:
            regions_per_area = real_crosswalk.groupby(column)["region_code"].nunique()
            assert (regions_per_area == 1).all(), column

    def test_every_region_has_a_constituency_per_production_clone(self, real_crosswalk):
        constituencies = real_crosswalk.groupby("region_code")[
            "constituency_code"
        ].nunique()
        assert constituencies.min() >= 10

    def test_assignment_on_the_real_crosswalk(self, real_crosswalk):
        covered = sorted(
            region
            for region, code in FRS_REGION_TO_CODE.items()
            if code in set(real_crosswalk["region_code"])
        )
        regions = np.array(covered * 40, dtype=object)
        n_clones = 10
        geography = assign_random_geography(
            household_countries=_countries([REGION_TO_COUNTRY[r] for r in regions]),
            household_regions=regions,
            n_clones=n_clones,
            seed=42,
        )
        expected = np.tile([FRS_REGION_TO_CODE[r] for r in regions], n_clones)
        assert (geography.region_code == expected).all()
        constituencies = geography.constituency_code.reshape(n_clones, len(regions))
        distinct = np.array(
            [len(set(constituencies[:, i])) for i in range(len(regions))]
        )
        assert (distinct == n_clones).all()


def test_built_dataset_oa_region_matches_frs_region(enhanced_frs):
    """Every household of the built enhanced FRS sits in its own region.

    A household with no region below the country is English to
    clone_and_assign, so its OA must be English.
    """
    household = enhanced_frs.household
    region_code = household["region_code_oa"].map(
        lambda value: value.decode() if isinstance(value, bytes) else str(value)
    )
    expected = household["region"].map(_normalise_region)
    has_oa = region_code != ""
    assert has_oa.any()
    known = has_oa & expected.notna()
    assert int((region_code[known] != expected[known]).sum()) == 0
    unknown = has_oa & expected.isna()
    assert region_code[unknown].str.startswith("E12").all()
