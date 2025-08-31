"""
Firm dataset for PolicyEngine UK.

This module processes synthetic firm data into PolicyEngine UK dataset format,
handling firm demographics, turnover, VAT, employment, and other business variables.
The synthetic firm data represents the UK business population for tax-benefit modelling.
"""

from policyengine_core.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from policyengine_uk_data.utils.datasets import STORAGE_FOLDER
import logging

logger = logging.getLogger(__name__)


def create_firm(year: int = 2023):
    """
    Process synthetic firm data into PolicyEngine UK dataset format.

    Generates synthetic firm microdata and transforms it into a structured
    PolicyEngine UK dataset with firm, sector, and employment-level variables
    mapped to the appropriate tax-benefit system variables.

    Args:
        year: Survey year for the dataset.

    Returns:
        Dataset with processed firm data ready for policy simulation.
    """
    # Always generate fresh synthetic data using generate_synthetic_data.py
    logger.info("Generating synthetic firm data...")
    import sys

    sys.path.append(str(Path(__file__).parent / "firm"))
    from generate_synthetic_data import SyntheticFirmGenerator

    generator = SyntheticFirmGenerator(device="cpu")
    synthetic_df = generator.generate_synthetic_firms()

    # Create entity DataFrames for firm structure
    pe_firm = pd.DataFrame()
    pe_sector = pd.DataFrame()
    pe_business_group = pd.DataFrame()

    # Add primary keys and identifiers
    pe_firm["firm_id"] = range(len(synthetic_df))
    pe_firm["firm_sector_id"] = synthetic_df["sic_code"].astype(int)
    pe_firm["firm_business_group_id"] = pe_firm["firm_id"] // 100

    # Create unique sectors
    unique_sectors = synthetic_df["sic_code"].astype(int).unique()
    pe_sector["sector_id"] = unique_sectors

    # Create business groups
    unique_groups = pe_firm["firm_business_group_id"].unique()
    pe_business_group["business_group_id"] = unique_groups

    # Add grossing weights
    pe_firm["firm_weight"] = synthetic_df["weight"].values

    # Add basic firm variables - exactly from synthetic data
    pe_firm["sic_code"] = synthetic_df["sic_code"]
    pe_firm["annual_turnover_k"] = synthetic_df["annual_turnover_k"].values
    pe_firm["annual_input_k"] = synthetic_df["annual_input_k"].values
    pe_firm["vat_liability_k"] = synthetic_df["vat_liability_k"].values
    pe_firm["employment"] = synthetic_df["employment"].astype(int).values
    pe_firm["vat_registered"] = (
        synthetic_df["vat_registered"].astype(bool).values
    )

    # Add derived variables that exist in generate_synthetic_data.py validation
    def map_to_hmrc_band(turnover_k):
        if turnover_k <= 0:
            return "Negative_or_Zero"
        elif turnover_k <= 85:
            return "£1_to_Threshold"
        elif turnover_k <= 150:
            return "£Threshold_to_£150k"
        elif turnover_k <= 300:
            return "£150k_to_£300k"
        elif turnover_k <= 500:
            return "£300k_to_£500k"
        elif turnover_k <= 1000:
            return "£500k_to_£1m"
        elif turnover_k <= 10000:
            return "£1m_to_£10m"
        else:
            return "Greater_than_£10m"

    def _map_employment_to_band(employment):
        if employment <= 4:
            return "0-4"
        elif employment <= 9:
            return "5-9"
        elif employment <= 19:
            return "10-19"
        elif employment <= 49:
            return "20-49"
        elif employment <= 99:
            return "50-99"
        elif employment <= 249:
            return "100-249"
        else:
            return "250+"

    pe_firm["hmrc_band"] = pe_firm["annual_turnover_k"].apply(map_to_hmrc_band)
    pe_firm["employment_band"] = pe_firm["employment"].apply(
        _map_employment_to_band
    )
    pe_firm["sic_numeric"] = pe_firm["sic_code"].astype(int)

    # Add year field
    pe_firm["year"] = year
    pe_sector["year"] = year
    pe_business_group["year"] = year

    # Create the dataset - use a simple object to hold the data
    class FirmDataset:
        def __init__(self):
            self.firm = pe_firm
            self.sector = pe_sector
            self.business_group = pe_business_group

        def save(self, path):
            # Save as HDF5 for compatibility
            self.firm.to_hdf(path, key="firm", mode="w")
            self.sector.to_hdf(path, key="sector", mode="a")
            self.business_group.to_hdf(path, key="business_group", mode="a")

    dataset = FirmDataset()

    # Add metadata about the dataset
    dataset.metadata = {
        "source": "synthetic_firm_generator",
        "year": year,
        "n_firms": len(pe_firm),
        "total_weighted_firms": pe_firm["firm_weight"].sum(),
        "vat_registered_firms": pe_firm[pe_firm["vat_registered"]][
            "firm_weight"
        ].sum(),
        "total_employment": (
            pe_firm["employment"] * pe_firm["firm_weight"]
        ).sum(),
        "total_turnover_billions": (
            pe_firm["annual_turnover_k"] * pe_firm["firm_weight"]
        ).sum()
        / 1e6,
        "total_vat_liability_billions": (
            pe_firm["vat_liability_k"] * pe_firm["firm_weight"]
        ).sum()
        / 1e6,
    }

    logger.info(f"Created firm dataset with {len(pe_firm):,} firms")
    logger.info(
        f"Total weighted population: {dataset.metadata['total_weighted_firms']:,.0f}"
    )
    logger.info(
        f"Total employment: {dataset.metadata['total_employment']:,.0f}"
    )
    logger.info(
        f"Total turnover: £{dataset.metadata['total_turnover_billions']:.1f}bn"
    )

    return dataset


# Dataset class for direct import like FRS
class firm_2023_24:
    """UK Firm dataset for 2023-24, following the FRS pattern."""

    def __init__(self):
        # Load the dataset from storage or create if needed
        dataset_path = STORAGE_FOLDER / "firm_2023_24.h5"

        if dataset_path.exists():
            self.firm = pd.read_hdf(dataset_path, key="firm")
            self.sector = pd.read_hdf(dataset_path, key="sector")
            self.business_group = pd.read_hdf(
                dataset_path, key="business_group"
            )
        else:
            # Create and save the dataset
            dataset = create_firm(year=2023)
            dataset.save(dataset_path)
            self.firm = dataset.firm
            self.sector = dataset.sector
            self.business_group = dataset.business_group


# Main execution for testing
if __name__ == "__main__":
    """Test the firm dataset creation."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("Creating firm dataset...")

    # Create the dataset
    firm_dataset = create_firm(year=2023)

    # Save to storage
    output_path = STORAGE_FOLDER / "firm_2023_24.h5"
    firm_dataset.save(output_path)

    logger.info(f"Saved firm dataset to {output_path}")

    # Display summary statistics
    print("\n" + "=" * 60)
    print("FIRM DATASET SUMMARY")
    print("=" * 60)

    for key, value in firm_dataset.metadata.items():
        if isinstance(value, (int, float)):
            if "billions" in key:
                print(f"{key}: £{value:.2f}bn")
            elif key in [
                "n_firms",
                "total_weighted_firms",
                "vat_registered_firms",
                "total_employment",
            ]:
                print(f"{key}: {value:,.0f}")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")

    print("=" * 60)
    print("Dataset creation complete!")
