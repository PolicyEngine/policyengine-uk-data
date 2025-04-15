import typing
if typing.TYPE_CHECKING:
    from policyengine_uk_data.datasets.enhanced_frs.main import EFRS
from policyengine_uk_data.impute import QRF
from policyengine_uk_data import data_folder
from policyengine_uk import Microsimulation
from policyengine_core.data import Dataset


def add_imputations(dataset: "EFRS"):
    consumption = QRF(file_path=data_folder / "models" / "consumption.pkl")
    vat = QRF(file_path=data_folder / "models" / "vat.pkl")
    wealth = QRF(file_path=data_folder / "models" / "wealth.pkl")

    dataset.save_to_h5(data_folder / "_tmp.h5")
    simulation = Microsimulation(dataset=Dataset.from_file(data_folder / "_tmp.h5", time_period=dataset.year))

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
            dataset.household[col] = pred_values[col]

    (data_folder / "_tmp.h5").unlink(missing_ok=True)