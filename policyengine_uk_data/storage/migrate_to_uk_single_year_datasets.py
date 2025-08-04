from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk import Microsimulation
from policyengine_core.data import Dataset
from policyengine_uk_data.datasets import EnhancedFRS_2023_24, FRS_2023_24


def migrate_to_uk_single_year_dataset(file_path: str):
    sim = Microsimulation(dataset=Dataset.from_file(file_path))

    single_year_dataset = UKSingleYearDataset.from_simulation(
        sim, fiscal_year=2023
    )

    single_year_dataset.save(file_path)


if __name__ == "__main__":
    migrate_to_uk_single_year_dataset(FRS_2023_24.file_path)
    migrate_to_uk_single_year_dataset(EnhancedFRS_2023_24.file_path)
