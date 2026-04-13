import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

_WEALTH_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "imputations" / "wealth.py"
)
_WEALTH_SPEC = importlib.util.spec_from_file_location(
    "student_loan_balance_wealth_module",
    _WEALTH_PATH,
)
wealth = importlib.util.module_from_spec(_WEALTH_SPEC)
_WEALTH_SPEC.loader.exec_module(wealth)


def test_generate_was_table_derives_student_loan_balance():
    row = {column: 0 for column in wealth.WAS_RENAMES}
    row["R7xshhwgt"] = 1
    row["GORR7"] = 11
    row["DVPriRntW7"] = 1
    row["TotpenR7_aggr"] = 100
    row["DvvalDBTR7_aggr"] = 25
    row["Tot_LosR7_aggr"] = 20_000
    row["Tot_los_exc_SLCR7_aggr"] = 5_000

    was = wealth.generate_was_table(pd.DataFrame([row]))

    assert "student_loan_balance" in was.columns
    assert was.student_loan_balance.iloc[0] == 15_000
    assert "student_loan_balance" in wealth.IMPUTE_VARIABLES


def test_create_wealth_model_reuses_current_cached_model(tmp_path, monkeypatch):
    model_path = tmp_path / "wealth.pkl"
    model_path.write_bytes(b"placeholder")
    cached_model = SimpleNamespace(
        model=SimpleNamespace(imputed_variables=list(wealth.IMPUTE_VARIABLES))
    )

    class DummyQRF:
        def __init__(self, file_path=None):
            assert file_path == model_path
            self.model = cached_model.model

    monkeypatch.setattr(wealth, "STORAGE_FOLDER", tmp_path)
    monkeypatch.setattr(wealth, "QRF", DummyQRF)
    monkeypatch.setattr(
        wealth,
        "save_imputation_models",
        lambda: (_ for _ in ()).throw(AssertionError("should not retrain")),
    )

    model = wealth.create_wealth_model()
    assert model.model.imputed_variables == list(wealth.IMPUTE_VARIABLES)


def test_create_wealth_model_retrains_when_cached_outputs_stale(tmp_path, monkeypatch):
    model_path = tmp_path / "wealth.pkl"
    model_path.write_bytes(b"placeholder")

    class DummyQRF:
        def __init__(self, file_path=None):
            assert file_path == model_path
            self.model = SimpleNamespace(imputed_variables=["owned_land"])

    fresh_model = object()

    monkeypatch.setattr(wealth, "STORAGE_FOLDER", tmp_path)
    monkeypatch.setattr(wealth, "QRF", DummyQRF)
    monkeypatch.setattr(wealth, "save_imputation_models", lambda: fresh_model)

    assert wealth.create_wealth_model() is fresh_model
