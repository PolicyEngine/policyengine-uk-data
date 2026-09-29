"""Validate the versioned UK local-authority lookup assets."""

from pathlib import Path

import pandas as pd


STORAGE_DIR = Path(__file__).parents[1] / "storage"
LAD22_PATH = STORAGE_DIR / "local_authorities_lad22.csv"
LAD23_PATH = STORAGE_DIR / "local_authorities_lad23.csv"

LAD22_PREDECESSOR_CODES = {
    "E07000026",
    "E07000027",
    "E07000028",
    "E07000029",
    "E07000030",
    "E07000031",
    "E07000163",
    "E07000164",
    "E07000165",
    "E07000166",
    "E07000167",
    "E07000168",
    "E07000169",
    "E07000187",
    "E07000188",
    "E07000189",
    "E07000246",
}
LAD23_SUCCESSOR_CODES = {
    "E06000063",
    "E06000064",
    "E06000065",
    "E06000066",
}


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    assert list(frame.columns) == ["code", "x", "y", "name"]
    return frame


def test_versioned_local_authority_assets_have_complete_rosters():
    lad22 = _load(LAD22_PATH)
    lad23 = _load(LAD23_PATH)

    assert len(lad22) == 374
    assert len(lad23) == 361
    assert lad22["code"].is_unique
    assert lad23["code"].is_unique

    lad22_codes = set(lad22["code"])
    lad23_codes = set(lad23["code"])
    assert lad22_codes - lad23_codes == LAD22_PREDECESSOR_CODES
    assert lad23_codes - lad22_codes == LAD23_SUCCESSOR_CODES
    assert "N09000011" in lad22_codes
    assert "N09000011" in lad23_codes


def test_versioned_local_authority_assets_have_complete_display_metadata():
    for path in (LAD22_PATH, LAD23_PATH):
        frame = _load(path)
        assert frame[["code", "name", "x", "y"]].notna().all().all()
        assert (frame["code"].str.len() == 9).all()
        assert (frame["name"].str.strip() != "").all()
        assert (frame["x"] == frame["x"].astype(int)).all()
        assert (frame["y"] == frame["y"].astype(int)).all()
        assert not frame.duplicated(subset=["x", "y"]).any()


def test_shared_authorities_keep_the_same_display_metadata():
    lad22 = _load(LAD22_PATH).set_index("code")
    lad23 = _load(LAD23_PATH).set_index("code")
    shared_codes = sorted(set(lad22.index) & set(lad23.index))

    pd.testing.assert_frame_equal(
        lad22.loc[shared_codes, ["x", "y", "name"]],
        lad23.loc[shared_codes, ["x", "y", "name"]],
    )
