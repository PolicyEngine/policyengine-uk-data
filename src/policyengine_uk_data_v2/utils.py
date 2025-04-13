from pathlib import Path

import h5py
import pandas as pd


def save_dataframes_to_h5(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    state: pd.DataFrame,
    output_path: str | Path,
    year: int,
) -> None:
    data = {}
    for df in (person, benunit, household, state):
        for column in df.columns:
            data[column] = df[column].values

    output_path = Path(output_path)
    output_path.unlink(missing_ok=True)

    with h5py.File(output_path, "w") as f:
        for column, values in data.items():
            # if values is object, convert to "S"
            if values.dtype == "object":
                values = values.astype("S")
            f.create_dataset(f"{column}/{year}", data=values)


def load_dataframes_from_h5(input_path: str | Path) -> tuple[pd.DataFrame]:
    from policyengine_uk.system import system

    dataframes = {
        "person": pd.DataFrame(),
        "benunit": pd.DataFrame(),
        "household": pd.DataFrame(),
        "state": pd.DataFrame(),
    }

    input_path = Path(input_path)

    with h5py.File(input_path, "r") as f:
        for variable in f:
            if variable not in system.variables:
                continue
            entity = system.variables.get(variable).entity.key
            for time_period in f[variable]:
                dataframes[entity][time_period] = pd.DataFrame(
                    f[variable][time_period][:]
                )

    return dataframes
