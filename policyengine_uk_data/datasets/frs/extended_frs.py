from policyengine_core.data import Dataset
from policyengine_uk_data.utils.imputations import *
from policyengine_uk_data.storage import STORAGE_FOLDER
from typing import Type
from policyengine_uk_data.datasets.frs.frs import FRS_2022_23
from tqdm import tqdm


class ExtendedFRS(Dataset):
    input_frs: Type[Dataset]

    def generate(self):
        from policyengine_uk import Microsimulation
        from policyengine_uk_data.utils.qrf import QRF

        create_consumption_model()
        create_vat_model()
        create_wealth_model()

        consumption = QRF(file_path=STORAGE_FOLDER / "consumption.pkl")
        vat = QRF(file_path=STORAGE_FOLDER / "vat.pkl")
        wealth = QRF(file_path=STORAGE_FOLDER / "wealth.pkl")

        data = self.input_frs().load_dataset()
        simulation = Microsimulation(dataset=self.input_frs)
        for imputation_model in tqdm(
            [consumption, vat, wealth], desc="Imputing data"
        ):
            predictors = imputation_model.input_columns

            X_input = simulation.calculate_dataframe(
                predictors, map_to="household"
            )
            if imputation_model == wealth:
                # WAS doesn't sample NI -> put NI households in Wales (closest aggregate)
                X_input.loc[
                    X_input["region"] == "NORTHERN_IRELAND", "region"
                ] = "WALES"
            Y_output = imputation_model.predict(X_input)

            for output_variable in Y_output.columns:
                values = Y_output[output_variable].values
                values[values < 0] = 0
                data[output_variable] = {self.time_period: values}

        # Clone the dataset for income imputation
        new_data = {}
        for variable in data:
            new_data[variable] = {}
            for time_period in data[variable]:
                if "_id" in variable:
                    # e.g. [1, 2, 3] -> [11, 12, 13, 21, 22, 23]
                    marker = 10 ** np.ceil(
                        max(np.log10(data[variable][time_period]))
                    )
                    values = list(data[variable][time_period] + marker) + list(
                        data[variable][time_period] + marker * 2
                    )
                    new_data[variable][time_period] = values
                elif "_weight" in variable:
                    new_data[variable][time_period] = list(
                        data[variable][time_period]
                    ) + list(data[variable][time_period] * 0)
                else:
                    new_data[variable][time_period] = (
                        list(data[variable][time_period]) * 2
                    )

        income_inputs = simulation.calculate_dataframe(
            ["age", "gender", "region"]
        )
        create_income_model()

        income = QRF(file_path=STORAGE_FOLDER / "income.pkl")
        full_imputations = income.predict(income_inputs)
        for variable in full_imputations.columns:
            # Assign over the second half of the dataset
            if variable in new_data.keys():
                new_data[variable][str(self.time_period)] = list(
                    data[variable][str(self.time_period)]
                ) + list(full_imputations[variable].values)
            else:
                new_data[variable] = {
                    str(self.time_period): list(
                        full_imputations[variable].values * 0
                    )
                    + list(full_imputations[variable].values)
                }

        self.save_dataset(new_data)


class ExtendedFRS_2022_23(ExtendedFRS):
    name = "extended_frs_2022_23"
    label = "Extended FRS (2022-23)"
    file_path = STORAGE_FOLDER / "extended_frs_2022_23.h5"
    data_format = Dataset.TIME_PERIOD_ARRAYS
    input_frs = FRS_2022_23
    time_period = 2022


if __name__ == "__main__":
    ExtendedFRS_2022_23().generate()
