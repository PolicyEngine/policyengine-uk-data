from policyengine_uk_data.datasets.frs import FRS
from pathlib import Path
from policyengine_uk_data.impute import QRF
from policyengine_uk_data import data_folder
from policyengine_uk import Microsimulation
from policyengine_core.data import Dataset

class EFRS(FRS):
    def generate(self, year: int, frs_checkpoint: str | Path = None):
        self.year = year
        if frs_checkpoint is None:
            super().generate(year=year)
        else:
            self.load_from_h5(frs_checkpoint, year=year)

        self.add_other_imputations()
        self.add_income_imputations()
        self.save_to_h5("efrs_2022.h5")

    def add_other_imputations(self):
        consumption = QRF(file_path=data_folder / "models" / "consumption.pkl")
        vat = QRF(file_path=data_folder / "models" / "vat.pkl")
        wealth = QRF(file_path=data_folder / "models" / "wealth.pkl")

        self.save_to_h5(data_folder / "_tmp.h5")
        simulation = Microsimulation(dataset=Dataset.from_file(data_folder / "_tmp.h5", time_period=self.year))

        for imputation_model in [
            consumption,
            vat,
            wealth,
        ]:
            predictors = imputation_model.input_columns
            # Use PolicyEngine UK to work out the predictor values for this dataset (e.g. net income)
            input_df = simulation.calculate_dataframe(predictors)
            pred_values = imputation_model.predict(input_df)
            for col in pred_values.columns:
                self.household[col] = pred_values[col]

        (data_folder / "_tmp.h5").unlink(missing_ok=True)


    def add_income_imputations(self):
        copy = self.copy()
        copy.household.household_weight *= 0
        income_model_inputs = copy.person[["age", "gender"]]
        income_model_inputs["region"] = copy.household.set_index("household_id").loc[copy.person.person_household_id].region.values
        income_model = QRF(file_path=data_folder / "models/income.pkl")
        pred_income = income_model.predict(income_model_inputs)
        for col in pred_income.columns:
            copy.person[col] = pred_income[col]

        self.stack(copy)
