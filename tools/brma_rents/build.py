#!/usr/bin/env python3
"""Rebuild ``storage/brma_private_rents.csv`` from pinned public sources.

One row per BRMA and LHA category: the median weekly rent on the BRMA's list
of rents, in the prices of the dataset year, and the standard deviation of
log rents on the list. See ``README.md`` for the method.

Usage (from the repository root, with openpyxl installed):
  python tools/brma_rents/build.py <cache dir> [--year 2024] [--check]
"""

import argparse
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
STORAGE = HERE.parents[1] / "policyengine_uk_data" / "storage"
OUTPUT = STORAGE / "brma_private_rents.csv"
sys.path.insert(0, str(HERE.parent / "brma_households"))
from fetch import fetch  # noqa: E402

CATEGORIES = list("ABCDE")
Z30 = NormalDist().inv_cdf(0.3)
IQR_SDS = NormalDist().inv_cdf(0.75) - NormalDist().inv_cdf(0.25)
SCOTTISH_SHEETS = (2019, 2020, 2021)  # years to September; the latest three released
PIPR_AREA = {
    "North East": "NORTH_EAST",
    "North West": "NORTH_WEST",
    "Yorkshire and The Humber": "YORKSHIRE",
    "East Midlands": "EAST_MIDLANDS",
    "West Midlands": "WEST_MIDLANDS",
    "East of England": "EAST_OF_ENGLAND",
    "London": "LONDON",
    "South East": "SOUTH_EAST",
    "South West": "SOUTH_WEST",
    "Wales": "WALES",
    "Scotland": "SCOTLAND",
    "Northern Ireland": "NORTHERN_IRELAND",
}


def brma_name(published: pd.Series) -> pd.Series:
    """policyengine-uk's ``BRMAName`` for a published BRMA name."""
    name = published.str.upper().str.replace(r"[^A-Z0-9]+", "_", regex=True)
    return name.str.strip("_").replace(
        {
            "HIGHLANDS_AND_ISLANDS": "HIGHLAND_AND_ISLANDS",
            "RENFREWSHIRE_INVERCLYDE": "RENFREWSHIRE_AND_INVERCLYDE",
        }
    )


def spread(log_rent: pd.Series) -> float:
    """Standard deviation of log rents implied by their interquartile range."""
    return (log_rent.quantile(0.75) - log_rent.quantile(0.25)) / IQR_SDS


def uprating(cache: Path, year: int) -> pd.DataFrame:
    """Log of each region's rent index in the dataset year over each list window.

    The list for April Y holds rents collected from October Y-2 to September
    Y-1. Columns are April determination years; the dataset year runs from
    April to March.
    """
    table = pd.read_excel(
        cache / "ons_price_index_of_private_rents_2026_09.xlsx",
        sheet_name="Table 1",
        header=2,
        usecols=["Time period", "Area name", "Index"],
    )
    table = table[table["Area name"].isin(PIPR_AREA)]
    index = table.pivot(index="Time period", columns="Area name", values="Index")
    # Months ONS has not yet published for an area are marked ``[x]``.
    index = index.rename(columns=PIPR_AREA).apply(pd.to_numeric, errors="coerce")
    months = pd.date_range(f"{year}-04-01", periods=12, freq="MS")
    assert months.isin(index.index).all() and index.loc[months].notna().all().all()
    target = index.loc[months].mean()
    windows = {}
    for determination in range(2017, int(index.index.max().year) + 1):
        window = pd.date_range(f"{determination - 2}-10-01", periods=12, freq="MS")
        if window.isin(index.index).all():
            windows[determination] = np.log(
                target / index.loc[window].mean(skipna=False)
            )
    return pd.DataFrame(windows)


def english_cells(cache: Path, factor: pd.DataFrame, region: pd.Series, year: int):
    """Median and spread of England's lists that overlap the dataset year."""
    rents = []
    for determination in (year + 1, year + 2):
        rows = pd.read_csv(
            cache / f"voa_list_of_rents_{determination}.csv",
            encoding="cp1252",
            dtype=str,
        ).dropna(how="all")
        assert (rows.PERIOD == "Week").all(), determination
        weekly = rows.NET_RENT.str.replace("[£,]", "", regex=True).astype(float)
        assert (weekly > 0).all(), determination
        brma = brma_name(rows.BRMA)
        rents.append(
            pd.DataFrame(
                {
                    "brma": brma,
                    "lha_category": rows.LHA_TYPE.str.removeprefix("Cat "),
                    "log_rent": np.log(weekly)
                    + brma.map(region).map(factor[determination]).to_numpy(),
                }
            )
        )
    rents = pd.concat(rents)
    assert rents.log_rent.notna().all()
    cell = rents.groupby(["brma", "lha_category"]).log_rent
    return pd.DataFrame(
        {
            "median_weekly_rent": np.exp(cell.median()),
            "log_sd": cell.apply(spread),
            "rents": cell.size(),
            "basis": "list",
        }
    ).reset_index()


def scottish_spreads(cache: Path) -> pd.DataFrame:
    """Spread of log rents on Scotland's lists, averaged over the latest sheets."""
    sheets = []
    for year in SCOTTISH_SHEETS:
        sheet = pd.read_excel(
            cache / "scottish_list_of_rents_foi_2016_2021.xlsx", sheet_name=str(year)
        ).dropna(how="all")
        sheet = sheet[sheet["BRMA"].notna() & (sheet["NET RENT"] > 0)]
        assert (sheet["FREQUENCY"].str.strip() == "Weekly").all(), year
        sheets.append(
            pd.DataFrame(
                {
                    "sheet": year,
                    "brma": brma_name(sheet["BRMA"].str.strip()),
                    "lha_category": sheet["LHA TYPE"]
                    .str.strip()
                    .str.removeprefix("Cat "),
                    "log_rent": np.log(sheet["NET RENT"].astype(float)),
                }
            )
        )
    by_sheet = pd.concat(sheets).groupby(["brma", "lha_category", "sheet"]).log_rent
    table = pd.DataFrame({"log_sd": by_sheet.apply(spread), "rents": by_sheet.size()})
    cell = table.reset_index().groupby(["brma", "lha_category"])
    return pd.DataFrame(
        {
            "log_sd": cell.apply(
                lambda c: np.average(c.log_sd, weights=c.rents), include_groups=False
            ),
            "rents": cell.rents.sum(),
        }
    ).reset_index()


def percentile_cells(published, factor, region, brmas, determinations, spreads, basis):
    """Median from published 30th percentiles, assuming log-normal rents.

    Each 30th percentile is uprated from its list's window to the dataset
    year; the median is the 30th percentile raised by ``-Z30`` spreads.
    """
    rows = published[
        published.brma.isin(brmas)
        & published.year.isin(determinations)
        & published.percentile_30.notna()
    ]
    log_p30 = np.log(rows.percentile_30) + [
        factor.loc[region[b], y] for b, y in zip(rows.brma, rows.year)
    ]
    cells = log_p30.groupby([rows.brma, rows.lha_category]).mean().rename("log_p30")
    cells = cells.reset_index().merge(
        spreads, on=spreads.columns.intersection(["brma", "lha_category"]).tolist()
    )
    cells["median_weekly_rent"] = np.exp(cells.log_p30 - Z30 * cells.log_sd)
    return cells.assign(basis=basis).drop(columns="log_p30")


def build(cache: Path, year: int) -> pd.DataFrame:
    households = pd.read_csv(STORAGE / "brma_private_rented_households.csv")
    # A BRMA that crosses a region boundary is uprated with the region that
    # holds most of its private renters.
    by_region = households.groupby(["brma", "region"]).households.sum()
    region = (
        by_region.reset_index().sort_values("households").groupby("brma").region.last()
    )
    factor = uprating(cache, year)  # rows: region; columns: determination
    published = pd.read_csv(cache / "lha_published_rates.csv.gzip", compression="gzip")

    def in_nation(nation):
        return set(region.index[region == nation])

    english = english_cells(cache, factor, region, year)
    scottish = scottish_spreads(cache)
    # Wales and Northern Ireland publish no lists: use the typical spread of
    # the English and Scottish lists for the category.
    typical = (
        pd.concat([english, scottish])
        .groupby("lha_category")
        .log_sd.median()
        .reset_index()
    )
    recent = [year + 1, year + 2]
    northern_irish = published[
        published.brma.isin(in_nation("NORTHERN_IRELAND"))
        & published.percentile_30.notna()
    ].year
    latest = [y for y in recent if y in set(northern_irish)] or [northern_irish.max()]
    table = pd.concat(
        [
            english,
            percentile_cells(
                published,
                factor,
                region,
                in_nation("SCOTLAND"),
                recent,
                scottish,
                "30th percentile and list spread",
            ),
            percentile_cells(
                published,
                factor,
                region,
                in_nation("WALES"),
                recent,
                typical.assign(rents=0),
                "30th percentile and typical spread",
            ),
            percentile_cells(
                published,
                factor,
                region,
                in_nation("NORTHERN_IRELAND"),
                latest,
                typical.assign(rents=0),
                "30th percentile and typical spread",
            ),
        ]
    )
    table = table[
        ["brma", "lha_category", "median_weekly_rent", "log_sd", "rents", "basis"]
    ]
    table = table.sort_values(["brma", "lha_category"]).reset_index(drop=True)
    expected = pd.MultiIndex.from_product([sorted(region.index), CATEGORIES])
    assert table.set_index(["brma", "lha_category"]).index.equals(expected)
    assert table[["median_weekly_rent", "log_sd"]].notna().all().all()
    table["median_weekly_rent"] = table.median_weekly_rent.round(2)
    table["log_sd"] = table.log_sd.round(4)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument(
        "--year", type=int, default=2024, help="dataset year (April start)"
    )
    parser.add_argument(
        "--check", action="store_true", help="compare with storage, do not write"
    )
    args = parser.parse_args()
    fetch(args.cache, yaml.safe_load((HERE / "sources.yaml").read_text()))
    table = build(args.cache, args.year)
    if args.check:
        pd.testing.assert_frame_equal(table, pd.read_csv(OUTPUT))
        print(f"{OUTPUT.name} matches the rebuild ({len(table)} rows).")
    else:
        table.to_csv(OUTPUT, index=False)
        print(f"Wrote {OUTPUT} ({len(table)} rows).")


if __name__ == "__main__":
    main()
