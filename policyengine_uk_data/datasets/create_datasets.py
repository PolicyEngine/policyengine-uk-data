from policyengine_uk_data.datasets.frs import create_frs
from policyengine_uk_data.storage import STORAGE_FOLDER
import logging
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk_data.utils.uprating import uprate_dataset

logging.basicConfig(level=logging.INFO)

# First, create the regular FRS dataset

logging.info("Creating FRS dataset")

frs = create_frs(
    raw_frs_folder=STORAGE_FOLDER / "frs_2023_24",
    year=2023,
)

frs.save(
    STORAGE_FOLDER / "frs_2023_24.h5",
)

logging.info(f"FRS dataset created and saved.")

# Add imputations of consumption, wealth, VAT, income and capital gains

from policyengine_uk_data.datasets.imputations import (
    impute_consumption,
    impute_wealth,
    impute_vat,
    impute_income,
    impute_capital_gains,
    impute_services,
)

logging.info("Imputing consumption")
frs = impute_consumption(frs)
logging.info("Imputing wealth")
frs = impute_wealth(frs)
logging.info("Imputing VAT")
frs = impute_vat(frs)
logging.info("Imputing public service usage")
frs = impute_services(frs)
logging.info("Imputing income")
frs = impute_income(frs)
logging.info("Imputing capital gains")
frs = impute_capital_gains(frs)

# Uprate to 2025

logging.info("Uprating dataset to 2025")

frs = uprate_dataset(frs, 2025)

from policyengine_uk_data.datasets.local_areas.constituencies.calibrate import (
    calibrate,
)

logging.info("Calibrating dataset with national and constituency targets.")

frs_calibrated = calibrate(frs)

# Downrate back to 2023

frs_calibrated = uprate_dataset(frs_calibrated, 2023)

frs_calibrated.save(STORAGE_FOLDER / "enhanced_frs_2023_24.h5")
logging.info(f"Extended FRS dataset created and saved.")
