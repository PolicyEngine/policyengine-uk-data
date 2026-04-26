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


def test_allocate_student_loan_balance_prefers_repayers_then_tertiary():
    person = pd.DataFrame(
        {
            "person_household_id": [1, 1, 2, 2],
            "age": [30, 28, 20, 32],
            "student_loan_repayments": [200, 0, 0, 0],
            "student_loans": [0, 0, 0, 0],
            "current_education": [
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "TERTIARY",
                "NOT_IN_EDUCATION",
            ],
            "highest_education": [
                "UPPER_SECONDARY",
                "UPPER_SECONDARY",
                "UPPER_SECONDARY",
                "TERTIARY",
            ],
        }
    )

    result = wealth._allocate_student_loan_balance_to_people(
        pd.Series({1: 1_000.0, 2: 600.0}),
        person,
    )

    assert result.tolist() == [1_000.0, 0.0, 0.0, 600.0]


def test_impute_wealth_routes_student_loan_balance_to_people(monkeypatch):
    class DummyDataset:
        def __init__(self):
            self.person = pd.DataFrame(
                {
                    "person_household_id": [1, 1, 2],
                    "age": [29, 27, 21],
                    "student_loan_repayments": [150, 0, 0],
                    "student_loans": [0, 0, 0],
                    "current_education": [
                        "NOT_IN_EDUCATION",
                        "NOT_IN_EDUCATION",
                        "TERTIARY",
                    ],
                    "highest_education": [
                        "TERTIARY",
                        "UPPER_SECONDARY",
                        "UPPER_SECONDARY",
                    ],
                }
            )
            self.household = pd.DataFrame(index=[1, 2])
            self.validated = False

        def copy(self):
            copied = DummyDataset()
            copied.person = self.person.copy()
            copied.household = self.household.copy()
            copied.validated = self.validated
            return copied

        def validate(self):
            self.validated = True

    class DummyModel:
        input_columns = ["household_net_income", "region"]

        @staticmethod
        def predict(_input_df):
            return pd.DataFrame(
                {
                    "owned_land": [10.0, 20.0],
                    "student_loan_balance": [900.0, 300.0],
                },
                index=[1, 2],
            )

    class DummyMicrosimulation:
        def __init__(self, dataset):
            self.dataset = dataset

        def calculate_dataframe(self, predictors, map_to):
            assert predictors == ["household_net_income", "region"]
            assert map_to == "household"
            return pd.DataFrame(
                {
                    "household_net_income": [1.0, 2.0],
                    "region": ["LONDON", "WALES"],
                },
                index=[1, 2],
            )

    monkeypatch.setattr(wealth, "create_wealth_model", lambda: DummyModel())
    monkeypatch.setattr(wealth, "Microsimulation", DummyMicrosimulation)

    imputed = wealth.impute_wealth(DummyDataset())

    assert imputed.household["owned_land"].tolist() == [10.0, 20.0]
    assert "student_loan_balance" not in imputed.household.columns
    assert imputed.person["student_loan_balance"].tolist() == [900.0, 0.0, 300.0]
    assert imputed.validated
