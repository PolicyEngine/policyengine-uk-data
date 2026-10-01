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


SPI_COLUMNS = [
    "SEX",
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
    "PAS",
    "TI",
]


def _write_fake_spi(path, gor_values=(1, 2, 3), maind_values=(1, 0, 1), **columns):
    """Write a minimal SPI-shaped tab file for tests.

    The real SPI file has dozens of columns; the test only needs them to
    exist with sensible types so ``create_spi`` can build dataframes.
    ``columns`` overrides any column with per-row values.
    """
    n = len(gor_values)
    data = {col: np.zeros(n, dtype=float) for col in SPI_COLUMNS}
    data["SREF"] = np.arange(1, n + 1)
    data["FACT"] = np.ones(n)
    data["GORCODE"] = list(gor_values)
    data["MAIND"] = list(maind_values)
    data["AGERANGE"] = [1] * n  # bucket (16, 25)
    data.update({col: list(values) for col, values in columns.items()})
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
    """The transferable amount follows the fiscal year: 10% of the personal
    allowance, rounded up to a multiple of £10 (ITA 2007 s. 55B(4)-(5)).
    That is £1,250 in 2020-21 (allowance £12,500) and £1,260 from 2021-22
    (allowance £12,570).
    """
    from policyengine_uk_data.datasets.spi import create_spi

    tab = tmp_path / "spi.tab"
    _write_fake_spi(tab, gor_values=(1, 2, 3), maind_values=(1, 0, 1))

    ds_2020 = create_spi(tab, 2020, seed=0)
    marriage_2020 = ds_2020.person["marriage_allowance"].to_numpy()
    # MAIND == 1 rows receive the transferable amount; others receive nothing.
    assert (marriage_2020[[0, 2]] == 1_250).all()
    assert marriage_2020[1] == 0

    ds_2025 = create_spi(tab, 2025, seed=0)
    marriage_2025 = ds_2025.person["marriage_allowance"].to_numpy()
    assert (marriage_2025[[0, 2]] == 1_260).all()
    assert marriage_2025[1] == 0


def test_create_spi_marriage_allowance_sides(tmp_path):
    """MAIND == 1 marks the recipient, who gets the transferable amount as
    `marriage_allowance`. The tape has no transferor flag, but a transferor's
    PAS is the personal allowance less the transferable amount, so that
    record gives the amount up through `marriage_allowance_relinquished`.
    A tapered allowance that happens to equal that value is not a transfer,
    and composite records (MAIND == -1) are neither side.
    """
    from policyengine_uk_data.datasets.spi import create_spi

    tab = tmp_path / "spi.tab"
    _write_fake_spi(
        tab,
        gor_values=(7, 7, 7, 7, 7),
        maind_values=(1, 0, 0, 0, -1),
        # recipient, transferor, no transfer, tapered, composite
        PAS=(13_830, 11_310, 12_570, 11_310, 0),
        TI=(30_000, 12_000, 30_000, 102_520, 600_000),
    )

    person = create_spi(tab, 2022, seed=0).person

    assert person["marriage_allowance"].tolist() == [1_260, 0, 0, 0, 0]
    assert person["marriage_allowance_relinquished"].tolist() == [0, 1_260, 0, 0, 0]
    # PAS already includes the transfer on both sides; it must stay out of
    # the dataset so the model does not count the transfer twice.
    assert "personal_allowance" not in person


def test_create_spi_marriage_allowance_invariants(tmp_path):
    """Over every combination of MAIND, PAS and income: each side is 0 or
    the transferable amount, no record is both sides, only MAIND == 1
    receives, and no one receiving or relinquishing lacks the matching
    tape evidence.
    """
    from itertools import product

    from policyengine_uk_data.datasets.spi import create_spi

    pa, transferable = 12_570, 1_260
    cases = list(
        product(
            (-1, 0, 1),
            (0, pa - transferable, pa, pa + transferable, 5_000),
            (8_000, 99_999, 100_000, 150_000),
        )
    )
    maind, pas, ti = zip(*cases)
    tab = tmp_path / "spi.tab"
    _write_fake_spi(
        tab, gor_values=[7] * len(cases), maind_values=maind, PAS=pas, TI=ti
    )

    person = create_spi(tab, 2022, seed=0).person
    received = person["marriage_allowance"].to_numpy()
    relinquished = person["marriage_allowance_relinquished"].to_numpy()

    assert set(received) <= {0, transferable}
    assert set(relinquished) <= {0, transferable}
    assert not ((received > 0) & (relinquished > 0)).any()
    assert ((received > 0) == (np.array(maind) == 1)).all()
    assert (np.array(pas)[relinquished > 0] == pa - transferable).all()
    assert (np.array(ti)[relinquished > 0] < 100_000).all()
    assert "personal_allowance" not in person


def test_spi_marriage_allowance_matches_statute_in_policyengine_uk(tmp_path):
    """Run SPI-shaped records through policyengine-uk for 2022-23.

    The recipient's tax falls by the basic rate, or for a Scottish taxpayer
    the Scottish basic rate, times the transferable amount (ITA 2007 s.
    55B(1), (3)): 20% x £1,260 = £252 in both cases, including a Scottish
    intermediate-rate payer. The recipient's own allowance is unchanged. The
    transferor's allowance falls by £1,260 (s. 55B(6)).
    """
    from policyengine_uk import Microsimulation
    from policyengine_uk.system import system
    from policyengine_uk_data.datasets.spi import create_spi

    if "marriage_allowance_relinquished" not in system.variables:
        pytest.skip(
            "needs a policyengine-uk release with the statutory Marriage "
            "Allowance (PolicyEngine/policyengine-uk#1963)"
        )

    tab = tmp_path / "spi.tab"
    _write_fake_spi(
        tab,
        # London recipient, London no transfer, Scottish recipient,
        # Scottish no transfer, London transferor
        gor_values=(7, 7, 11, 11, 7),
        maind_values=(1, 0, 1, 0, 0),
        PAY=(30_000, 30_000, 30_000, 30_000, 12_000),
        PAS=(13_830, 12_570, 13_830, 12_570, 11_310),
        TI=(30_000, 30_000, 30_000, 30_000, 12_000),
    )
    sim = Microsimulation(dataset=create_spi(tab, 2022, seed=0))
    tax = sim.calculate("income_tax", 2022).values
    allowance = sim.calculate("personal_allowance", 2022).values

    assert tax[1] - tax[0] == pytest.approx(252)
    assert tax[3] - tax[2] == pytest.approx(252)
    assert allowance.tolist() == [12_570, 12_570, 12_570, 12_570, 11_310]
    assert tax[4] == pytest.approx(0.2 * (12_000 - 11_310))


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


def test_income_projection_loads_local_h5_dataset(monkeypatch):
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

    dataset = incomes_projection.load_spi_dataset()

    assert isinstance(dataset, FakeDataset)
    assert calls == {"path": "/tmp/spi_2022_23.h5"}
    assert dataset.household["region"].tolist() == [
        "SOUTH_EAST",
        "LONDON",
        "SOUTH_EAST",
    ]


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
