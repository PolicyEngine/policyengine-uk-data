import pandas as pd

from policyengine_uk_data.datasets.imputations import consumption
from policyengine_uk_data.datasets.frs import WEEKS_IN_YEAR


def test_generate_lcfs_table_accepts_current_lowercase_tab_headers(monkeypatch):
    def add_has_fuel(household):
        household = household.copy()
        household["has_fuel_consumption"] = 1.0
        return household

    monkeypatch.setattr(consumption, "impute_has_fuel_to_lcfs", add_has_fuel)

    household = pd.DataFrame(
        {
            "case": [1],
            "g018": [2],
            "g019": [1],
            "gorx": [7],
            "p389p": [1_000.0],
            "p344p": [1_500.0],
            "weighta": [0.5],
            "a121": [2],
            "a122": [5],
            "b226": [10.0],
            "b489": [0.0],
            "b490": [0.0],
            "p537": [20.0],
            **{f"p{code}": [1.0] for code in range(601, 613)},
            "c72211": [5.0],
            "c72212": [6.0],
        }
    )
    person = pd.DataFrame(
        {
            "case": [1, 1],
            "b303p": [100.0, 200.0],
            "b3262p": [10.0, 20.0],
            "b3381": [0.0, 0.0],
            "p049p": [5.0, 5.0],
        }
    )

    result = consumption.generate_lcfs_table(person, household)

    assert len(result) == 1
    assert result["region"].iloc[0] == "LONDON"
    assert result["tenure_type"].iloc[0] == "OWNED_WITH_MORTGAGE"
    assert result["accommodation_type"].iloc[0] == "HOUSE_SEMI_DETACHED"
    assert result["employment_income"].iloc[0] == 300.0 * WEEKS_IN_YEAR
    assert result["household_weight"].iloc[0] == 500
    assert (
        result["domestic_energy_consumption"].iloc[0]
        == result["electricity_consumption"].iloc[0] + result["gas_consumption"].iloc[0]
    )
    assert (
        result[consumption.PREDICTOR_VARIABLES + consumption.IMPUTATIONS]
        .notna()
        .all()
        .all()
    )
