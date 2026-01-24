"""Sanity checks for the enhanced FRS dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_PATH = (
    Path(__file__).parent.parent / "data" / "output" / "enhanced_frs_2023.h5"
)


@pytest.fixture(scope="module")
def dataset():
    """Load the enhanced FRS dataset."""
    if not DATA_PATH.exists():
        pytest.skip("Dataset not materialised yet")

    store = pd.HDFStore(DATA_PATH, "r")
    data = {
        "person": store["/person"],
        "benunit": store["/benunit"],
        "household": store["/household"],
    }
    store.close()
    return data


class TestDatasetStructure:
    """Basic structural checks."""

    def test_has_required_tables(self, dataset):
        assert "person" in dataset
        assert "benunit" in dataset
        assert "household" in dataset

    def test_person_count_reasonable(self, dataset):
        # UK has ~67m people, FRS sample should be 30k-50k
        n_people = len(dataset["person"])
        assert 20_000 < n_people < 100_000, f"Got {n_people} people"

    def test_household_count_reasonable(self, dataset):
        # Should be roughly 1/2 to 1/3 of person count
        n_hh = len(dataset["household"])
        n_people = len(dataset["person"])
        ratio = n_people / n_hh
        assert 1.5 < ratio < 4.0, f"Person/HH ratio: {ratio:.2f}"

    def test_benunit_count_reasonable(self, dataset):
        # Benefit units between household and person count
        n_bu = len(dataset["benunit"])
        n_hh = len(dataset["household"])
        n_people = len(dataset["person"])
        assert n_hh <= n_bu <= n_people


class TestIncomeVariables:
    """Income distribution sanity checks."""

    def test_employment_income_prevalence(self, dataset):
        # Roughly 50-75% of working-age adults have employment income
        person = dataset["person"]
        if "age" in person.columns and "employment_income" in person.columns:
            working_age = person[(person["age"] >= 18) & (person["age"] < 65)]
            has_emp_income = (working_age["employment_income"] > 0).mean()
            assert (
                0.35 < has_emp_income < 0.85
            ), f"Employment income rate: {has_emp_income:.1%}"

    def test_self_employment_income_prevalence(self, dataset):
        # Roughly 8-20% of workers are self-employed
        person = dataset["person"]
        if (
            "age" in person.columns
            and "self_employment_income" in person.columns
        ):
            working_age = person[(person["age"] >= 18) & (person["age"] < 65)]
            has_se_income = (working_age["self_employment_income"] > 0).mean()
            assert (
                0.03 < has_se_income < 0.25
            ), f"Self-employment rate: {has_se_income:.1%}"

    def test_pension_income_for_elderly(self, dataset):
        # Most pensioners should have some pension income
        person = dataset["person"]
        if (
            "age" in person.columns
            and "private_pension_income" in person.columns
        ):
            elderly = person[person["age"] >= 66]
            if len(elderly) > 0:
                has_pension = (elderly["private_pension_income"] > 0).mean()
                # Private pensions aren't universal, but should be common
                assert (
                    0.15 < has_pension < 0.80
                ), f"Private pension rate (66+): {has_pension:.1%}"

    def test_employment_income_mean_reasonable(self, dataset):
        # Mean employment income for earners should be £20k-£50k
        person = dataset["person"]
        if "employment_income" in person.columns:
            earners = person[person["employment_income"] > 0]
            if len(earners) > 0:
                mean_income = earners["employment_income"].mean()
                assert (
                    15_000 < mean_income < 60_000
                ), f"Mean employment income: £{mean_income:,.0f}"

    def test_no_negative_employment_income(self, dataset):
        person = dataset["person"]
        if "employment_income" in person.columns:
            assert (person["employment_income"] >= 0).all()

    def test_dividend_income_rarer_than_employment(self, dataset):
        person = dataset["person"]
        if (
            "dividend_income" in person.columns
            and "employment_income" in person.columns
        ):
            div_rate = (person["dividend_income"] > 0).mean()
            emp_rate = (person["employment_income"] > 0).mean()
            assert (
                div_rate < emp_rate
            ), "Dividend income more common than employment"


class TestWealthVariables:
    """Wealth imputation sanity checks."""

    def test_property_wealth_exists(self, dataset):
        hh = dataset["household"]
        assert "property_wealth" in hh.columns

    def test_main_residence_value_reasonable(self, dataset):
        hh = dataset["household"]
        if "main_residence_value" in hh.columns:
            owners = hh[hh["main_residence_value"] > 0]
            if len(owners) > 0:
                mean_value = owners["main_residence_value"].mean()
                # UK average house price roughly £250k-£350k
                assert (
                    100_000 < mean_value < 600_000
                ), f"Mean residence value: £{mean_value:,.0f}"

    def test_homeownership_rate(self, dataset):
        hh = dataset["household"]
        if "main_residence_value" in hh.columns:
            ownership_rate = (hh["main_residence_value"] > 0).mean()
            # UK homeownership roughly 60-70%
            assert (
                0.40 < ownership_rate < 0.80
            ), f"Homeownership rate: {ownership_rate:.1%}"

    def test_vehicle_ownership_rate(self, dataset):
        hh = dataset["household"]
        if "num_vehicles" in hh.columns:
            has_vehicle = (hh["num_vehicles"] > 0).mean()
            # Most UK households have a vehicle
            assert (
                0.50 < has_vehicle < 0.90
            ), f"Vehicle ownership: {has_vehicle:.1%}"

    def test_savings_non_negative(self, dataset):
        hh = dataset["household"]
        if "savings" in hh.columns:
            # Some households may have negative net savings, but gross should be positive
            assert (
                hh["savings"] >= -1
            ).all()  # Allow small floating point errors


class TestConsumptionVariables:
    """Consumption imputation sanity checks.

    Note: These tests only run when consumption model has been applied.
    With model=None, consumption variables are initialised to zero.
    """

    def test_food_consumption_exists(self, dataset):
        hh = dataset["household"]
        assert "food_and_non_alcoholic_beverages_consumption" in hh.columns

    def test_food_consumption_universal(self, dataset):
        # Everyone eats - almost all households should have food spending
        hh = dataset["household"]
        if "food_and_non_alcoholic_beverages_consumption" in hh.columns:
            has_food = (
                hh["food_and_non_alcoholic_beverages_consumption"] > 0
            ).mean()
            if has_food == 0:
                pytest.skip("Consumption model not applied (all zeros)")
            assert has_food > 0.90, f"Food consumption rate: {has_food:.1%}"

    def test_food_consumption_reasonable(self, dataset):
        hh = dataset["household"]
        if "food_and_non_alcoholic_beverages_consumption" in hh.columns:
            consumers = hh[
                hh["food_and_non_alcoholic_beverages_consumption"] > 0
            ]
            if len(consumers) == 0:
                pytest.skip("Consumption model not applied (all zeros)")
            mean_food = consumers[
                "food_and_non_alcoholic_beverages_consumption"
            ].mean()
            # Annual food spending roughly £3k-£8k per household
            assert (
                1_000 < mean_food < 15_000
            ), f"Mean food spending: £{mean_food:,.0f}"

    def test_transport_consumption_common(self, dataset):
        hh = dataset["household"]
        if "transport_consumption" in hh.columns:
            has_transport = (hh["transport_consumption"] > 0).mean()
            if has_transport == 0:
                pytest.skip("Consumption model not applied (all zeros)")
            assert (
                0.50 < has_transport < 0.95
            ), f"Transport spending rate: {has_transport:.1%}"

    def test_petrol_spending_linked_to_vehicles(self, dataset):
        hh = dataset["household"]
        if "petrol_spending" in hh.columns and "num_vehicles" in hh.columns:
            has_petrol = (hh["petrol_spending"] > 0).mean()
            if has_petrol == 0:
                pytest.skip("Consumption model not applied (all zeros)")
            has_vehicle = hh["num_vehicles"] > 0
            petrol_with_vehicle = (
                hh.loc[has_vehicle, "petrol_spending"] > 0
            ).mean()
            petrol_without_vehicle = (
                hh.loc[~has_vehicle, "petrol_spending"] > 0
            ).mean()
            # Those with vehicles more likely to buy petrol
            assert petrol_with_vehicle > petrol_without_vehicle


class TestDemographics:
    """Demographic distribution checks."""

    def test_age_distribution_reasonable(self, dataset):
        person = dataset["person"]
        if "age" in person.columns:
            mean_age = person["age"].mean()
            # UK mean age roughly 40
            assert 35 < mean_age < 50, f"Mean age: {mean_age:.1f}"

    def test_children_present(self, dataset):
        person = dataset["person"]
        if "age" in person.columns:
            child_rate = (person["age"] < 18).mean()
            # Roughly 20% of population are children
            assert 0.10 < child_rate < 0.30, f"Child rate: {child_rate:.1%}"

    def test_elderly_present(self, dataset):
        person = dataset["person"]
        if "age" in person.columns:
            elderly_rate = (person["age"] >= 65).mean()
            # Roughly 18% of UK population 65+
            assert (
                0.10 < elderly_rate < 0.30
            ), f"Elderly rate: {elderly_rate:.1%}"

    def test_regions_present(self, dataset):
        hh = dataset["household"]
        if "region" in hh.columns:
            n_regions = hh["region"].nunique()
            # UK has 12 regions
            assert n_regions >= 10, f"Only {n_regions} regions found"


class TestWeights:
    """Weight file sanity checks.

    Weights are stored as HDF5 with year keys, shape (n_areas, n_households).
    """

    def _load_weights(self, path):
        """Load weights from HDF5, handling year-keyed format."""
        import h5py

        with h5py.File(path, "r") as f:
            # Get first year key
            year = list(f.keys())[0]
            return np.array(f[year])

    def test_constituency_weights_exist(self):
        weights_path = (
            DATA_PATH.parent / "parliamentary_constituency_weights.h5"
        )
        if not weights_path.exists():
            pytest.skip("Constituency weights not materialised")

        weights = self._load_weights(weights_path)
        # Shape is (n_constituencies, n_households) - should have 650 constituencies
        assert (
            weights.shape[0] >= 600
        ), f"Only {weights.shape[0]} constituencies"

    def test_la_weights_exist(self):
        weights_path = DATA_PATH.parent / "local_authority_weights.h5"
        if not weights_path.exists():
            pytest.skip("LA weights not materialised")

        weights = self._load_weights(weights_path)
        # Should have ~360 local authorities
        assert (
            weights.shape[0] >= 300
        ), f"Only {weights.shape[0]} local authorities"

    def test_weights_positive(self):
        weights_path = (
            DATA_PATH.parent / "parliamentary_constituency_weights.h5"
        )
        if not weights_path.exists():
            pytest.skip("Constituency weights not materialised")

        weights = self._load_weights(weights_path)
        assert (weights >= 0).all(), "Negative weights found"

    def test_weights_sum_reasonable(self):
        weights_path = (
            DATA_PATH.parent / "parliamentary_constituency_weights.h5"
        )
        if not weights_path.exists():
            pytest.skip("Constituency weights not materialised")

        weights = self._load_weights(weights_path)
        # Each constituency (row) should have weights summing to something reasonable
        row_sums = weights.sum(axis=1)
        # Constituency populations roughly 70k-100k on average
        assert (
            row_sums.mean() > 1000
        ), f"Mean weight sum: {row_sums.mean():.0f}"


class TestBenefits:
    """Benefit receipt sanity checks."""

    def test_child_benefit_for_families(self, dataset):
        # Households with children should often receive child benefit
        person = dataset["person"]
        hh = dataset["household"]
        if "age" in person.columns and "child_benefit_reported" in hh.columns:
            # Count children per household
            person_hh = (
                person[["household_id", "age"]].copy()
                if "household_id" in person.columns
                else None
            )
            if person_hh is not None:
                children_per_hh = (
                    person_hh[person_hh["age"] < 16]
                    .groupby("household_id")
                    .size()
                )
                hh_with_children = hh["household_id"].isin(
                    children_per_hh.index
                )
                if hh_with_children.sum() > 0:
                    cb_rate = (
                        hh.loc[hh_with_children, "child_benefit_reported"] > 0
                    ).mean()
                    # Most families with children receive child benefit
                    assert (
                        cb_rate > 0.50
                    ), f"Child benefit rate (families): {cb_rate:.1%}"

    def test_state_pension_for_elderly(self, dataset):
        person = dataset["person"]
        if (
            "age" in person.columns
            and "state_pension_reported" in person.columns
        ):
            elderly = person[person["age"] >= 67]
            if len(elderly) > 0:
                sp_rate = (elderly["state_pension_reported"] > 0).mean()
                # Most over-67s receive state pension
                assert (
                    sp_rate > 0.70
                ), f"State pension rate (67+): {sp_rate:.1%}"


class TestImputedVariables:
    """Check imputed variables have reasonable distributions."""

    def test_wealth_inequality(self, dataset):
        # Wealth should be unequally distributed (Gini > 0.5)
        hh = dataset["household"]
        if "property_wealth" in hh.columns:
            wealth = hh["property_wealth"].values
            wealth = wealth[wealth > 0]  # Positive wealth only
            if len(wealth) > 100:
                # Simple Gini approximation
                sorted_wealth = np.sort(wealth)
                n = len(sorted_wealth)
                cumulative = np.cumsum(sorted_wealth)
                gini = (2 * np.sum((np.arange(1, n + 1) * sorted_wealth))) / (
                    n * cumulative[-1]
                ) - (n + 1) / n
                assert gini > 0.3, f"Property wealth Gini: {gini:.2f}"

    def test_num_bedrooms_reasonable(self, dataset):
        hh = dataset["household"]
        if "num_bedrooms" in hh.columns:
            mean_beds = hh["num_bedrooms"].mean()
            # UK average roughly 2.5-3.5 bedrooms
            assert 1.5 < mean_beds < 5.0, f"Mean bedrooms: {mean_beds:.1f}"

    def test_council_tax_bands(self, dataset):
        hh = dataset["household"]
        if "council_tax" in hh.columns:
            ct = hh["council_tax"]
            ct_payers = ct[ct > 0]
            if len(ct_payers) > 0:
                mean_ct = ct_payers.mean()
                # Annual council tax roughly £1.5k-£2.5k on average
                assert (
                    500 < mean_ct < 5000
                ), f"Mean council tax: £{mean_ct:,.0f}"


class TestDataQuality:
    """General data quality checks."""

    def test_no_all_null_columns(self, dataset):
        for table_name, df in dataset.items():
            all_null = df.columns[df.isnull().all()]
            assert (
                len(all_null) == 0
            ), f"{table_name} has all-null columns: {list(all_null)}"

    def test_no_duplicate_person_ids(self, dataset):
        person = dataset["person"]
        if "person_id" in person.columns:
            assert person["person_id"].is_unique

    def test_household_ids_in_person(self, dataset):
        person = dataset["person"]
        hh = dataset["household"]
        if "household_id" in person.columns and "household_id" in hh.columns:
            person_hh_ids = set(person["household_id"].unique())
            hh_ids = set(hh["household_id"].unique())
            assert person_hh_ids == hh_ids, "Household ID mismatch"

    def test_income_not_mostly_zero(self, dataset):
        person = dataset["person"]
        # Core income variables that should have meaningful non-zero rates
        core_incomes = [
            "employment_income",
            "self_employment_income",
            "private_pension_income",
        ]
        for col in core_incomes:
            if col in person.columns:
                zero_rate = (person[col] == 0).mean()
                # These should have at least 5% non-zero (workers, self-employed, pensioners)
                assert zero_rate < 0.98, f"{col} is {zero_rate:.1%} zeros"
