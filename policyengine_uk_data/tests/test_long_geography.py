from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from policyengine_uk_data.calibration.long_geography import (
    LONG_GEOGRAPHY_COLUMNS,
    LONG_GEOGRAPHY_WEIGHTS_FILE,
    area_support_from_long_geography,
    build_area_household_indices,
    build_long_geography_frame,
    geography_support_report,
    write_long_geography_weights,
)


class MockDataset:
    def __init__(self):
        self.household = pd.DataFrame(
            {
                "household_id": [101, 102, 103, 104],
                "source_year": [2023, 2023, 2023, 2023],
                "source_household_id": [1, 2, 3, 1],
                "clone_index": [0, 0, 0, 1],
                "household_weight": [1.0, 2.0, 3.0, 1.0],
                "oa_code": [b"E00000001", "E00000002", "", "E00000003"],
                "constituency_code_oa": [
                    b"E14001001",
                    "E14001002",
                    "",
                    "E14001001",
                ],
                "la_code_oa": ["E09000001", "E09000001", "X99999999", None],
            }
        )


def _write_h5(path: Path, key: str, data: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=data)


def test_build_long_geography_frame_is_oa_first_and_keeps_schema():
    dataset = MockDataset()

    frame = build_long_geography_frame(
        dataset,
        area_types="oa",
        area_codes=["E00000001", "E00000002", "E00000003"],
    )

    assert frame.columns.tolist() == LONG_GEOGRAPHY_COLUMNS
    assert frame["area_code"].tolist() == [
        "E00000001",
        "E00000002",
        "E00000003",
    ]
    assert frame["household_index"].tolist() == [0, 1, 3]
    assert frame["household_id"].tolist() == [101, 102, 104]
    assert frame["source_year"].tolist() == [2023, 2023, 2023]
    assert frame["source_household_id"].tolist() == [1, 2, 1]
    assert frame["source_household_key"].tolist() == [
        "2023:1",
        "2023:2",
        "2023:1",
    ]
    assert frame["clone_index"].tolist() == [0, 0, 1]
    assert frame["weight"].tolist() == [1.0, 2.0, 1.0]
    assert frame["weight_source"].unique().tolist() == ["household_weight"]


def test_build_long_geography_frame_applies_1d_weights_to_each_area_type():
    dataset = MockDataset()
    weights = np.array([10.0, 0.0, 30.0, 40.0])

    frame = build_long_geography_frame(
        dataset,
        area_types=("constituency", "la"),
        area_codes={
            "constituency": ["E14001001", "E14001002"],
            "la": ["E09000001"],
        },
        weights=weights,
        weight_source="test_weights",
        drop_zero_weights=True,
    )

    assert set(frame["area_type"]) == {"constituency", "la"}
    assert frame["household_index"].tolist() == [0, 3, 0]
    assert frame["weight"].tolist() == [10.0, 40.0, 10.0]
    assert frame["weight_source"].unique().tolist() == ["test_weights"]


def test_build_long_geography_frame_converts_legacy_2d_weights_to_rows():
    dataset = MockDataset()
    weights = np.array(
        [
            [11.0, 12.0, 13.0, 14.0],
            [21.0, 22.0, 23.0, 24.0],
        ]
    )

    frame = build_long_geography_frame(
        dataset,
        area_types="constituency",
        area_codes=["E14001001", "E14001002"],
        weights=weights,
    )

    assert frame["area_index"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert frame["area_code"].tolist() == [
        "E14001001",
        "E14001001",
        "E14001001",
        "E14001001",
        "E14001002",
        "E14001002",
        "E14001002",
        "E14001002",
    ]
    assert frame["household_index"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert frame["weight"].tolist() == [
        11.0,
        12.0,
        13.0,
        14.0,
        21.0,
        22.0,
        23.0,
        24.0,
    ]


def test_2d_weights_can_only_build_one_area_type():
    dataset = MockDataset()
    weights = np.ones((2, 4))

    with pytest.raises(ValueError, match="one area_type"):
        build_long_geography_frame(
            dataset,
            area_types=("constituency", "la"),
            area_codes={
                "constituency": ["E14001001", "E14001002"],
                "la": ["E09000001"],
            },
            weights=weights,
        )


def test_write_long_geography_weights_refuses_large_dense_conversion(
    tmp_path,
    monkeypatch,
):
    dataset = MockDataset()
    (tmp_path / "constituencies_2024.csv").write_text(
        "code,name\nE14001001,A\nE14001002,B\n"
    )
    _write_h5(
        tmp_path / "constituency_weights.h5",
        "2025",
        np.ones((2, 4)),
    )

    import policyengine_uk_data.calibration.long_geography as mod

    monkeypatch.setattr(mod, "STORAGE_FOLDER", tmp_path)

    with pytest.raises(ValueError, match="Refusing to expand"):
        write_long_geography_weights(
            dataset=dataset,
            weight_files={"constituency": "constituency_weights.h5"},
            dataset_key="2025",
            output_path=tmp_path / LONG_GEOGRAPHY_WEIGHTS_FILE,
            area_types=("constituency",),
            max_dense_cells=4,
        )


def test_build_area_household_indices_returns_empty_arrays_for_empty_areas():
    dataset = MockDataset()

    indices = build_area_household_indices(
        dataset,
        area_type="constituency",
        area_codes=["E14001001", "E14001002", "E14001003"],
    )

    assert indices["E14001001"].tolist() == [0, 3]
    assert indices["E14001002"].tolist() == [1]
    assert indices["E14001003"].tolist() == []


def test_build_long_geography_frame_requires_geography_column():
    dataset = MockDataset()
    dataset.household = dataset.household.drop(columns=["la_code_oa"])

    with pytest.raises(ValueError, match="la_code_oa"):
        build_long_geography_frame(
            dataset,
            area_types="la",
            area_codes=["E09000001"],
        )


def test_area_support_tracks_unique_source_households_and_ess():
    dataset = MockDataset()
    frame = build_long_geography_frame(
        dataset,
        area_types="constituency",
        area_codes=["E14001001", "E14001002", "E14001003"],
    )

    support = area_support_from_long_geography(
        frame,
        area_codes={"constituency": ["E14001001", "E14001002", "E14001003"]},
    )

    area = support[support["area_code"] == "E14001001"].iloc[0]
    assert area["n_rows"] == 2
    assert area["n_source_households"] == 1
    assert area["effective_sample_size"] == pytest.approx(2.0)

    empty_area = support[support["area_code"] == "E14001003"].iloc[0]
    assert empty_area["n_rows"] == 0
    assert empty_area["effective_sample_size"] == 0


def test_area_support_distinguishes_same_household_id_from_different_years():
    dataset = MockDataset()
    dataset.household.loc[3, "source_year"] = 2024
    frame = build_long_geography_frame(
        dataset,
        area_types="constituency",
        area_codes=["E14001001"],
    )

    support = area_support_from_long_geography(frame)

    assert support["n_source_households"].iloc[0] == 2


def test_geography_support_report_summarizes_low_support_areas():
    dataset = MockDataset()

    _, summary = geography_support_report(
        dataset,
        area_types=("constituency", "la"),
        area_codes={
            "constituency": ["E14001001", "E14001002", "E14001003"],
            "la": ["E09000001"],
        },
        min_source_households=2,
        min_effective_sample_size=2,
    )

    constituency = summary[summary["area_type"] == "constituency"].iloc[0]
    assert constituency["n_areas"] == 3
    assert constituency["n_nonempty_areas"] == 2
    assert constituency["low_support_areas"] == 3


def test_write_long_geography_weights_combines_oa_and_derived_area_types(
    tmp_path,
    monkeypatch,
):
    dataset = MockDataset()
    pd.DataFrame({"oa_code": ["E00000001", "E00000002", "E00000003"]}).to_csv(
        tmp_path / "oa_crosswalk.csv.gz",
        index=False,
        compression="gzip",
    )
    (tmp_path / "constituencies_2024.csv").write_text(
        "code,name\nE14001001,A\nE14001002,B\n"
    )
    (tmp_path / "local_authorities_2021.csv").write_text("code,name\nE09000001,C\n")
    _write_h5(
        tmp_path / "constituency_weights.h5",
        "2025",
        np.array(
            [
                [11.0, 12.0, 13.0, 14.0],
                [21.0, 22.0, 23.0, 24.0],
            ]
        ),
    )
    _write_h5(
        tmp_path / "la_weights.h5",
        "2025",
        np.array([100.0, 0.0, 300.0, 400.0]),
    )

    import policyengine_uk_data.calibration.long_geography as mod

    monkeypatch.setattr(mod, "STORAGE_FOLDER", tmp_path)

    output_path = tmp_path / LONG_GEOGRAPHY_WEIGHTS_FILE
    frame = write_long_geography_weights(
        dataset=dataset,
        weight_files={
            "constituency": "constituency_weights.h5",
            "la": "la_weights.h5",
        },
        dataset_key="2025",
        output_path=output_path,
    )

    assert output_path.exists()
    loaded = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(loaded, frame, check_dtype=False)
    assert frame["area_type"].tolist() == [
        "oa",
        "oa",
        "oa",
        "constituency",
        "constituency",
        "constituency",
        "constituency",
        "constituency",
        "constituency",
        "constituency",
        "constituency",
        "la",
    ]
    assert frame["weight"].tolist() == [
        1.0,
        2.0,
        1.0,
        11.0,
        12.0,
        13.0,
        14.0,
        21.0,
        22.0,
        23.0,
        24.0,
        100.0,
    ]
