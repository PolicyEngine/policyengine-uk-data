import numpy as np
import h5py
from pathlib import Path
from typing import Union


def transform_2010_to_2024(
    weights_2010: np.ndarray,
    file_path: Union[str, Path] = Path(__file__).parent
    / "constituencies_2010_to_2024.h5",
) -> np.ndarray:
    """
    Transforms weights from 2010 constituencies to 2024 constituencies using the mapping matrix.

    Args:
        weights_2010 (np.ndarray): An N-dimensional array of weights corresponding to 2010 constituencies (first axis must be 650 in length)
        file_path (Union[str, Path]): Path to the h5 file containing the mapping matrix.
            Can be provided as string or Path object.

    Returns:
        np.ndarray: An N-dimensional array of weights corresponding to 2024 constituencies.

    Raises:
        ValueError: If the input weights array length doesn't match the mapping matrix dimensions.
    """
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as hf:
        mapping_matrix = hf["df"][:]

    transformed = mapping_matrix @ weights_2010
    return transformed
