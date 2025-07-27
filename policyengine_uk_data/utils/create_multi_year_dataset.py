from policyengine_uk.data import UKMultiYearDataset, UKSingleYearDataset
from policyengine_uk.data.economic_assumptions import apply_uprating
from policyengine_uk import Microsimulation
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_core.data import Dataset


def convert_legacy_to_multi_year_dataset(
    file_path: str,
    new_file_path: str,
    start_year: int = 2023,
    end_year: int = 2029,
) -> UKMultiYearDataset:
    """
    Convert a legacy single year dataset to a multi-year dataset.
    """
    sim = Microsimulation(dataset=Dataset.from_file(file_path))

    dataset = UKSingleYearDataset.from_simulation(sim, fiscal_year=start_year)
    dataset.time_period = str(start_year)

    datasets = [dataset]

    for year in range(start_year + 1, end_year + 1):
        dataset = dataset.copy()
        dataset.time_period = str(year)

        datasets.append(dataset.copy())

    multi_year_dataset = UKMultiYearDataset(
        datasets=datasets,
    )
    multi_year_dataset = apply_uprating(multi_year_dataset)

    multi_year_dataset.save(new_file_path)


if __name__ == "__main__":
    file_paths = [
        STORAGE_FOLDER / "frs_2023_24.h5",
        STORAGE_FOLDER / "enhanced_frs_2023_24.h5",
    ]
    out_file_paths = [
        STORAGE_FOLDER / "frs_2023_29.h5",
        STORAGE_FOLDER / "enhanced_frs_2023_29.h5",
    ]

    for file_path, new_file_path in zip(file_paths, out_file_paths):
        convert_legacy_to_multi_year_dataset(
            file_path=str(file_path),
            new_file_path=str(new_file_path),
            start_year=2023,
            end_year=2029,
        )
        print(f"Converted {file_path} to {new_file_path}")
        # Test here
        sim = Microsimulation(
            dataset=UKMultiYearDataset(file_path=str(new_file_path))
        )
