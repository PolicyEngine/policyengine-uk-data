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
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("policyengine_uk") is None:
    pytest.skip(
        "policyengine_uk not available in test environment",
        allow_module_level=True,
    )

from policyengine_core.errors import ParameterNotFoundError

from policyengine_uk_data.datasets.spi import model_simulates_unknown_region


SPI_COLUMNS = [
    "SEX",
    "SREF",
    "FACT",
    "DIVIDENDS",
    "GIFTAID",
    "GORCODE",
    "SCOT_TXP",
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


def test_current_spi_release_metadata_points_to_2022_23():
    from policyengine_uk_data.datasets.spi import (
        SPI_FISCAL_YEAR,
        SPI_H5_FILENAME,
        SPI_RELEASE_NAME,
        SPI_TAB_FILENAME,
    )

    assert SPI_RELEASE_NAME == "spi_2022_23"
    assert SPI_TAB_FILENAME == "put2223uk.tab"
    assert SPI_FISCAL_YEAR == 2022
    assert SPI_H5_FILENAME == "spi_2022_23.h5"


def test_income_spi_generation_handles_current_unknown_codes():
    from policyengine_uk_data.datasets.imputations.income import generate_spi_table

    data = {col: np.zeros(1, dtype=float) for col in SPI_COLUMNS}
    data["SREF"] = [1]
    data["FACT"] = [1]
    data["SEX"] = [1]
    data["GORCODE"] = [13]
    data["AGERANGE"] = [-1]
    spi = pd.DataFrame(data)

    out = generate_spi_table(spi, seed=0, sample_size=5)

    assert out["region"].tolist() == ["UNKNOWN"] * 5
    assert out["age"].between(16, 70, inclusive="left").all()


def test_income_model_cache_is_release_scoped():
    from policyengine_uk_data.datasets.imputations.income import (
        INCOME_MODEL_PATH,
    )
    from policyengine_uk_data.datasets.spi import SPI_RELEASE_NAME

    assert INCOME_MODEL_PATH.name == f"income_{SPI_RELEASE_NAME}.pkl"


def test_income_model_sample_size_is_reduced_in_testing(monkeypatch):
    from policyengine_uk_data.datasets.imputations import income as income_module

    monkeypatch.delenv("TESTING", raising=False)
    assert (
        income_module.get_income_model_sample_size()
        == income_module.INCOME_MODEL_SAMPLE_SIZE
    )
    assert (
        income_module.get_income_model_metadata()["sample_size"]
        == income_module.INCOME_MODEL_SAMPLE_SIZE
    )

    monkeypatch.setenv("TESTING", "1")
    assert (
        income_module.get_income_model_sample_size()
        == income_module.TESTING_INCOME_MODEL_SAMPLE_SIZE
    )
    assert (
        income_module.get_income_model_metadata()["sample_size"]
        == income_module.TESTING_INCOME_MODEL_SAMPLE_SIZE
    )


def test_income_projection_uses_current_spi_release():
    from policyengine_uk_data.utils import incomes_projection
    from policyengine_uk_data.datasets.spi import SPI_FISCAL_YEAR, SPI_H5_FILENAME

    assert incomes_projection.SPI_DATASET.endswith(SPI_H5_FILENAME)
    assert incomes_projection.SPI_FISCAL_YEAR == SPI_FISCAL_YEAR
    assert "savings_interest_income" in incomes_projection.ALL_INCOME_VARIABLES


def test_income_projection_reads_spi_dataset_year_read_only(monkeypatch):
    from policyengine_uk_data.utils import incomes_projection

    calls = {}

    class FakeStore:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return None

        def __getitem__(self, key):
            assert key == "time_period"
            return pd.Series([2022])

    def fake_hdf_store(path, mode=None):
        calls["path"] = path
        calls["mode"] = mode
        return FakeStore()

    monkeypatch.setattr(incomes_projection.pd, "HDFStore", fake_hdf_store)

    assert incomes_projection._read_spi_dataset_year("/readonly/spi_2022_23.h5") == 2022
    assert calls == {
        "path": "/readonly/spi_2022_23.h5",
        "mode": "r",
    }


def test_income_projection_builds_current_spi_dataset_when_missing(
    tmp_path,
    monkeypatch,
):
    from policyengine_uk_data.utils import incomes_projection

    tab_dir = tmp_path / "spi_2022_23"
    tab_dir.mkdir()
    tab_path = tab_dir / "put2223uk.tab"
    tab_path.write_text("fake tab")

    calls = {}

    class FakeDataset:
        def save(self, path):
            calls["saved_path"] = path
            path.write_text("fake h5")

    def fake_create_spi(path, fiscal_year):
        calls["tab_path"] = path
        calls["fiscal_year"] = fiscal_year
        return FakeDataset()

    monkeypatch.setattr(incomes_projection, "STORAGE_FOLDER", tmp_path)
    monkeypatch.setattr(incomes_projection, "SPI_RELEASE_NAME", "spi_2022_23")
    monkeypatch.setattr(incomes_projection, "SPI_TAB_FILENAME", "put2223uk.tab")
    monkeypatch.setattr(incomes_projection, "SPI_H5_FILENAME", "spi_2022_23.h5")
    monkeypatch.setattr(incomes_projection, "SPI_FISCAL_YEAR", 2022)
    monkeypatch.setattr(incomes_projection, "create_spi", fake_create_spi)
    monkeypatch.setattr(incomes_projection, "_read_spi_dataset_year", lambda path: 2022)

    dataset_path = incomes_projection.ensure_spi_dataset()

    assert dataset_path == str(tmp_path / "spi_2022_23.h5")
    assert calls == {
        "tab_path": tab_path,
        "fiscal_year": 2022,
        "saved_path": tmp_path / "spi_2022_23.h5",
    }


def test_income_projection_rebuilds_stale_spi_dataset_year(
    tmp_path,
    monkeypatch,
):
    from policyengine_uk_data.utils import incomes_projection

    tab_dir = tmp_path / "spi_2022_23"
    tab_dir.mkdir()
    (tab_dir / "put2223uk.tab").write_text("fake tab")
    dataset_path = tmp_path / "spi_2022_23.h5"
    dataset_path.write_text("stale h5")

    read_years = iter([2026, 2022])
    calls = {}

    class FakeDataset:
        def save(self, path):
            calls["saved_path"] = path
            path.write_text("rebuilt h5")

    monkeypatch.setattr(incomes_projection, "STORAGE_FOLDER", tmp_path)
    monkeypatch.setattr(incomes_projection, "SPI_RELEASE_NAME", "spi_2022_23")
    monkeypatch.setattr(incomes_projection, "SPI_TAB_FILENAME", "put2223uk.tab")
    monkeypatch.setattr(incomes_projection, "SPI_H5_FILENAME", "spi_2022_23.h5")
    monkeypatch.setattr(incomes_projection, "SPI_FISCAL_YEAR", 2022)
    monkeypatch.setattr(
        incomes_projection,
        "_read_spi_dataset_year",
        lambda path: next(read_years),
    )
    monkeypatch.setattr(
        incomes_projection,
        "create_spi",
        lambda path, fiscal_year: FakeDataset(),
    )

    assert incomes_projection.ensure_spi_dataset() == str(dataset_path)
    assert calls == {"saved_path": dataset_path}
    assert dataset_path.read_text() == "rebuilt h5"


@pytest.mark.parametrize(
    "model_simulates, expected",
    [
        (False, ["SOUTH_EAST", "LONDON", "SOUTH_EAST"]),
        (True, ["UNKNOWN", "LONDON", "SOUTH_EAST"]),
    ],
)
def test_income_projection_loads_local_h5_dataset(
    monkeypatch, model_simulates, expected
):
    """UNKNOWN becomes SOUTH_EAST only for a model that cannot simulate it."""
    from policyengine_uk_data.utils import incomes_projection

    calls = {}

    class FakeDataset:
        def __init__(self, path):
            calls["path"] = path
            self.household = pd.DataFrame(
                {"region": ["UNKNOWN", "LONDON", "SOUTH_EAST"]}
            )

    monkeypatch.setattr(
        incomes_projection,
        "ensure_spi_dataset",
        lambda: "/tmp/spi_2022_23.h5",
    )
    monkeypatch.setattr(incomes_projection, "UKSingleYearDataset", FakeDataset)
    monkeypatch.setattr(
        incomes_projection,
        "model_simulates_unknown_region",
        lambda: model_simulates,
    )

    dataset = incomes_projection.load_spi_dataset()

    assert isinstance(dataset, FakeDataset)
    assert calls == {"path": "/tmp/spi_2022_23.h5"}
    assert dataset.household["region"].tolist() == expected


def test_income_projection_rebuilds_spi_dataset_without_scottish_flag(
    tmp_path, monkeypatch
):
    """A cached H5 from before create_spi read SCOT_TXP is rebuilt; a current
    one is reused."""
    from policyengine_uk_data.datasets.spi import create_spi
    from policyengine_uk_data.utils import incomes_projection

    tab_dir = tmp_path / "spi_2022_23"
    tab_dir.mkdir()
    tab = tab_dir / "put2223uk.tab"
    _write_fake_spi(tab, gor_values=(11, 13), maind_values=(0, 0))
    dataset_path = tmp_path / "spi_2022_23.h5"
    stale = create_spi(tab, 2022)
    stale.person = stale.person.drop(columns="pays_scottish_income_tax")
    stale.save(dataset_path)

    builds = []

    def counting_create_spi(path, fiscal_year):
        builds.append(fiscal_year)
        return create_spi(path, fiscal_year)

    monkeypatch.setattr(incomes_projection, "STORAGE_FOLDER", tmp_path)
    monkeypatch.setattr(incomes_projection, "SPI_RELEASE_NAME", "spi_2022_23")
    monkeypatch.setattr(incomes_projection, "SPI_TAB_FILENAME", "put2223uk.tab")
    monkeypatch.setattr(incomes_projection, "SPI_H5_FILENAME", "spi_2022_23.h5")
    monkeypatch.setattr(incomes_projection, "SPI_FISCAL_YEAR", 2022)
    monkeypatch.setattr(incomes_projection, "create_spi", counting_create_spi)

    assert not incomes_projection._has_scottish_taxpayer_flag(dataset_path)
    assert incomes_projection.ensure_spi_dataset() == str(dataset_path)
    assert builds == [2022]
    assert incomes_projection._has_scottish_taxpayer_flag(dataset_path)
    assert incomes_projection.ensure_spi_dataset() == str(dataset_path)
    assert builds == [2022]


def test_income_model_cache_rejects_stale_spi_release(tmp_path, monkeypatch):
    from policyengine_uk_data.datasets.imputations import income as income_module

    cache = tmp_path / "income_spi_2022_23.pkl"
    stale_metadata = {
        **income_module.get_income_model_metadata(),
        "spi_release_name": "spi_2020_21",
        "spi_tab_filename": "put2021uk.tab",
    }
    with cache.open("wb") as f:
        pickle.dump(
            {
                "model": SimpleNamespace(
                    imputed_variables=list(income_module.IMPUTATIONS)
                ),
                "input_columns": income_module.PREDICTORS,
                "metadata": stale_metadata,
            },
            f,
        )

    sentinel = object()
    monkeypatch.setattr(income_module, "INCOME_MODEL_PATH", cache)
    monkeypatch.setattr(income_module, "save_imputation_models", lambda: sentinel)

    assert income_module.create_income_model() is sentinel


def test_income_model_cache_rejects_stale_sample_size(tmp_path, monkeypatch):
    from policyengine_uk_data.datasets.imputations import income as income_module

    monkeypatch.delenv("TESTING", raising=False)
    cache = tmp_path / "income_spi_2022_23.pkl"
    stale_metadata = {
        **income_module.get_income_model_metadata(),
        "sample_size": income_module.TESTING_INCOME_MODEL_SAMPLE_SIZE,
    }
    with cache.open("wb") as f:
        pickle.dump(
            {
                "model": SimpleNamespace(
                    imputed_variables=list(income_module.IMPUTATIONS)
                ),
                "input_columns": income_module.PREDICTORS,
                "metadata": stale_metadata,
            },
            f,
        )

    sentinel = object()
    monkeypatch.setattr(income_module, "INCOME_MODEL_PATH", cache)
    monkeypatch.setattr(income_module, "save_imputation_models", lambda: sentinel)

    assert income_module.create_income_model() is sentinel


def test_income_model_cache_accepts_current_spi_release(tmp_path, monkeypatch):
    from policyengine_uk_data.datasets.imputations import income as income_module

    cache = tmp_path / "income_spi_2022_23.pkl"
    current_metadata = income_module.get_income_model_metadata()
    with cache.open("wb") as f:
        pickle.dump(
            {
                "model": SimpleNamespace(
                    imputed_variables=list(income_module.IMPUTATIONS)
                ),
                "input_columns": income_module.PREDICTORS,
                "metadata": current_metadata,
            },
            f,
        )

    monkeypatch.setattr(income_module, "INCOME_MODEL_PATH", cache)
    monkeypatch.setattr(
        income_module,
        "save_imputation_models",
        lambda: pytest.fail("current SPI release cache should be reused"),
    )

    assert income_module.create_income_model().metadata == current_metadata


def _set_spi_columns(path, **columns):
    df = pd.read_csv(path, sep="\t")
    for col, values in columns.items():
        df[col] = list(values)
    df.to_csv(path, sep="\t", index=False)


# HMRC's GORCODE codes 1-12 (SN 9422, Annex A) as policyengine-uk regions,
# written out here rather than read from REGION_MAP so a wrong mapping fails.
HMRC_REGIONS = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    3: "YORKSHIRE",  # Yorkshire and the Humber
    4: "EAST_MIDLANDS",
    5: "WEST_MIDLANDS",
    6: "EAST_OF_ENGLAND",
    7: "LONDON",
    8: "SOUTH_EAST",
    9: "SOUTH_WEST",
    10: "WALES",
    11: "SCOTLAND",
    12: "NORTHERN_IRELAND",
}


@pytest.mark.parametrize("not_scottish", [0, ".", ""])
def test_create_spi_region_and_scottish_taxpayer_invariants(tmp_path, not_scottish):
    """For every GORCODE (documented 1-14, composite -1, undocumented 99) and
    SCOT_TXP value, the region follows GORCODE alone and is always a Region
    member, and Scottish taxpayer status follows SCOT_TXP alone. HMRC
    documents "not a Scottish taxpayer" as "."; the 2022-23 tape writes 0.
    """
    from itertools import product

    from policyengine_uk.variables.household.demographic.geography import Region

    from policyengine_uk_data.datasets.spi import create_spi

    cases = list(product([-1, *range(1, 15), 99], (not_scottish, 1)))
    gor = [g for g, _ in cases]
    scot = [s for _, s in cases]
    tab = tmp_path / "spi.tab"
    _write_fake_spi(tab, gor_values=gor, maind_values=[0] * len(cases))
    _set_spi_columns(tab, SCOT_TXP=scot)

    ds = create_spi(tab, 2022)

    regions = ds.household["region"].tolist()
    assert regions == [HMRC_REGIONS.get(g, "UNKNOWN") for g in gor]
    assert set(regions) <= {region.name for region in Region}
    assert ds.person["pays_scottish_income_tax"].tolist() == [s == 1 for s in scot]
    # Address abroad, address unknown and composite records stay UNKNOWN even
    # when they are Scottish taxpayers.
    assert {r for r, g in zip(regions, gor) if g in (-1, 13, 14)} == {"UNKNOWN"}


def test_create_spi_scottish_taxpayer_status_survives_h5_round_trip(tmp_path):
    from policyengine_uk.data import UKSingleYearDataset

    from policyengine_uk_data.datasets.spi import create_spi

    tab = tmp_path / "spi.tab"
    _write_fake_spi(tab, gor_values=(13, 11, 7), maind_values=(0, 0, 0))
    _set_spi_columns(tab, SCOT_TXP=(1, 0, 1))
    ds = create_spi(tab, 2022)
    ds.save(tmp_path / "spi.h5")

    loaded = UKSingleYearDataset(str(tmp_path / "spi.h5"))

    assert loaded.person["pays_scottish_income_tax"].tolist() == [True, False, True]
    assert loaded.household["region"].tolist() == ["UNKNOWN", "SCOTLAND", "LONDON"]


UNKNOWN_RENT_INDEX = (
    "gov.economic_assumptions.yoy_growth.ons.private_rental_prices.UNKNOWN"
)


@pytest.mark.parametrize(
    "missing, expected", [(None, True), (UNKNOWN_RENT_INDEX, False)]
)
def test_unknown_region_probe_reads_the_simulation(monkeypatch, missing, expected):
    import policyengine_uk

    def simulate(dataset):
        assert dataset.household["region"].tolist() == ["UNKNOWN"]
        if missing:
            raise ParameterNotFoundError(missing, "2023-01-01")

    monkeypatch.setattr(policyengine_uk, "Microsimulation", simulate)

    assert model_simulates_unknown_region.__wrapped__() is expected


def test_unknown_region_probe_raises_other_missing_parameters(monkeypatch):
    import policyengine_uk

    def simulate(dataset):
        raise ParameterNotFoundError("gov.hmrc.income_tax.rates.uk", "2023-01-01")

    monkeypatch.setattr(policyengine_uk, "Microsimulation", simulate)

    with pytest.raises(ParameterNotFoundError, match=r"rates\.uk'"):
        model_simulates_unknown_region.__wrapped__()


def test_unknown_region_probe_agrees_with_policyengine_uk_release():
    """2.104.5 is the first policyengine-uk release with
    PolicyEngine/policyengine-uk#1985. Where the imported model is the
    installed release, the probe agrees with its version."""
    from importlib.metadata import PackageNotFoundError, distribution
    from pathlib import Path

    import policyengine_uk
    from packaging.version import Version

    try:
        release = distribution("policyengine-uk")
    except PackageNotFoundError:
        pytest.skip("policyengine-uk is not installed as a distribution")
    installed = Path(release.locate_file("policyengine_uk/__init__.py"))
    if not installed.exists() or not installed.samefile(policyengine_uk.__file__):
        pytest.skip("the imported policyengine-uk is not the installed release")

    assert model_simulates_unknown_region() == (
        Version(release.version) >= Version("2.104.5")
    )


# GORCODE, SCOT_TXP: abroad, unknown, composite, London, Scotland, abroad and
# Scottish, Scotland but not Scottish.
SIMULATED_RECORDS = ((13, 0), (14, 0), (-1, 0), (7, 0), (11, 1), (13, 1), (11, 0))
# The data year, an uprated year and 2030, the last year policyengine-uk
# carries dataset inputs to.
YEARS = [2022, 2026, 2030]


def _spi_income_tax(tmp_path, region_rule=False, **kwargs):
    """Income tax on SIMULATED_RECORDS, each paid £60,000, in YEARS. With
    region_rule, the Scottish taxpayer flag is dropped, so policyengine-uk
    derives it from the region as it did before create_spi read SCOT_TXP."""
    from policyengine_uk import Microsimulation

    from policyengine_uk_data.datasets.spi import create_spi

    tab = tmp_path / "spi.tab"
    gor, scot = zip(*SIMULATED_RECORDS)
    _write_fake_spi(tab, gor_values=gor, maind_values=[0] * len(gor))
    _set_spi_columns(
        tab, SCOT_TXP=scot, PAY=[60_000] * len(gor), AGERANGE=[3] * len(gor)
    )
    dataset = create_spi(tab, 2022, **kwargs)
    if region_rule:
        dataset.person = dataset.person.drop(columns="pays_scottish_income_tax")
    sim = Microsimulation(dataset=dataset)
    return {year: sim.calculate("income_tax", year).values for year in YEARS}


def test_spi_income_tax_follows_scottish_taxpayer_flag(tmp_path):
    """Equal pay, so income tax depends only on SCOT_TXP, in the data year
    and after uprating. Uses the legacy SOUTH_EAST label so it runs on any
    policyengine-uk release."""
    for year, tax in _spi_income_tax(tmp_path, unknown_region="SOUTH_EAST").items():
        abroad, unknown, composite, london, scotland, abroad_scot, scotland_ruk = tax

        assert london > 0, year
        assert abroad == unknown == composite == scotland_ruk == london, year
        assert abroad_scot == scotland != london, year


def test_spi_income_tax_changes_only_where_scottish_flag_and_region_disagree(
    tmp_path,
):
    """Records whose SCOT_TXP agrees with their region get exactly the income
    tax they got when the region decided Scottish status. The two records
    where they disagree change."""
    label = "UNKNOWN" if model_simulates_unknown_region() else "SOUTH_EAST"
    flag = _spi_income_tax(tmp_path, unknown_region=label)
    region = _spi_income_tax(tmp_path, region_rule=True, unknown_region=label)
    agrees = np.array([(g == 11) == (s == 1) for g, s in SIMULATED_RECORDS])

    for year in YEARS:
        assert (flag[year][agrees] == region[year][agrees]).all(), year
        assert (flag[year][~agrees] != region[year][~agrees]).all(), year


def test_create_spi_output_with_unknown_region_can_be_simulated(tmp_path):
    """SPI records with an address abroad (13), an unknown address (14) or a
    composite record (-1) keep region UNKNOWN and still run through
    policyengine-uk, giving record for record the income tax of the legacy
    SOUTH_EAST relabelling: the region label does not move income tax.
    Models without PolicyEngine/policyengine-uk#1985 must fail on the missing
    rent index, and on nothing else.
    """
    if not model_simulates_unknown_region():
        with pytest.raises(
            ParameterNotFoundError, match=r"private_rental_prices\.UNKNOWN'"
        ):
            _spi_income_tax(tmp_path)
        pytest.xfail(
            "policyengine-uk before 2.104.5 (PolicyEngine/policyengine-uk#1985) "
            "has no rent index for Region.UNKNOWN"
        )

    tax = _spi_income_tax(tmp_path)
    legacy = _spi_income_tax(tmp_path, unknown_region="SOUTH_EAST")

    for year in YEARS:
        assert (tax[year] > 0).all(), year
        assert (tax[year] == legacy[year]).all(), year
