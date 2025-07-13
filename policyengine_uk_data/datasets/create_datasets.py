from policyengine_uk_data.datasets.frs import create_frs
from policyengine_uk_data.storage import STORAGE_FOLDER
import logging
from policyengine_uk.data import UKDataset

logging.basicConfig(level=logging.INFO)

# First, create the regular FRS dataset

logging.info("Creating FRS dataset")

frs = create_frs(
    raw_frs_folder=STORAGE_FOLDER / "frs_2022_23",
    year=2022,
)

frs.save(
    STORAGE_FOLDER / "frs_2022.h5",
)

frs = UKDataset(str(STORAGE_FOLDER / "frs_2022.h5"))

logging.info(
    f"FRS dataset created and saved to {STORAGE_FOLDER / 'frs_2022.h5'}"
)

# Add imputations of consumption, wealth, VAT, income and capital gains

from policyengine_uk_data.datasets.imputations import (
    impute_consumption,
    impute_wealth,
    impute_vat,
    impute_income,
    impute_capital_gains,
)

logging.info("Imputing consumption")
frs = impute_consumption(frs)
logging.info("Imputing wealth")
frs = impute_wealth(frs)
logging.info("Imputing VAT")
frs = impute_vat(frs)
logging.info("Imputing income")
frs = impute_income(frs)
logging.info("Imputing capital gains")
frs = impute_capital_gains(frs)

frs.save(STORAGE_FOLDER / "extended_frs_2022.h5")
logging.info(
    f"Extended FRS dataset created and saved to {STORAGE_FOLDER / 'extended_frs_2022.h5'}"
)
