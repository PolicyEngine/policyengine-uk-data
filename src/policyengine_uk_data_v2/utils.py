from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, TypedDict

import h5py
import numpy as np
import pandas as pd
from policyengine_core.parameters import ParameterNode


class Dataframes(TypedDict):
    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    state: pd.DataFrame


def load_parameters() -> ParameterNode:
    return ParameterNode("", directory_path=Path(__file__).parent / "parameters")


def save_dataframes_to_h5(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    state: pd.DataFrame,
    output_path: Union[str, Path],
    year: int,
) -> None:
    data: Dict[str, np.ndarray] = {}
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


def load_dataframes_from_h5(input_path: Union[str, Path]) -> Dataframes:
    from policyengine_uk.system import system

    dataframes: Dataframes = {
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
                dataframes[entity][variable] = pd.DataFrame(f[variable][time_period][:])

    return dataframes


max_ = np.maximum


def sum_positive_variables(
    variables: List[pd.Series],
) -> pd.Series:
    """
    Sums the given variables, replacing negative values with 0.
    """
    return sum([max_(0, variable) for variable in variables])


def sum_from_positive_fields(
    table: pd.DataFrame,
    fields: List[str],
) -> pd.Series:
    """
    Sums the given fields, replacing negative values with 0.
    """
    return sum_positive_variables([table[field] for field in fields])


def concat(*args: np.ndarray) -> np.ndarray:
    """
    Concatenate the given arrays along the first axis.
    """
    return np.concatenate(args, axis=0)


def sum_to_entity(
    values: pd.Series,
    foreign_key: pd.Series,
    primary_key: pd.Series,
) -> pd.Series:
    """Sums values by joining foreign and primary keys.

    Args:
        values (pd.Series): The values in the non-entity table.
        foreign_key (pd.Series): E.g. pension.person_id.
        primary_key (pd.Series): E.g. person.index.

    Returns:
        pd.Series: A value for each person.
    """
    return pd.Series(
        values.groupby(foreign_key).sum().reindex(primary_key).fillna(0).values
    )
