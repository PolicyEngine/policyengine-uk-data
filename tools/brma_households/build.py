#!/usr/bin/env python3
"""Rebuild the BRMA private-rented household table from verified cached originals."""

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon

from fetch import decoded_content, sources, verify_cache


# Overrides established on 2026-10-02; live reply pages are evidence, not inputs.
LSOA_OVERRIDES = {
    "E01014018": "BRECON_AND_RADNOR",  # Official VOA postcode lookup HR3 5TA, 2026-10-02.
    "E01022272": "MONMOUTHSHIRE",  # Official VOA postcode lookup NP16 7AG, 2026-10-02.
    "E01022273": "MONMOUTHSHIRE",  # Official VOA postcode lookup NP16 7RB, 2026-10-02.
    "E01022274": "MONMOUTHSHIRE",  # Official VOA postcode lookup NP16 7QB, 2026-10-02.
    "E01022275": "MONMOUTHSHIRE",  # Official VOA postcode lookup NP16 7EN, 2026-10-02.
    "W01002024": "CARDIFF",  # Official VOA lookup CF11 0AS (and 12 other postcodes), 2026-10-02.
}
SCOTTISH_OVERRIDES = {
    "S00179382": "WEST_DUNBARTONSHIRE",  # G83 8NQ; official VOA West Dunbartonshire council lookup, 2026-10-02.
    "S00179383": "WEST_DUNBARTONSHIRE",  # G83 8SD; official VOA West Dunbartonshire council lookup, 2026-10-02.
}


def read_json(path):
    return json.loads(decoded_content(path))


def brma_key(name):
    return re.sub(r"\W+", "_", name.replace("&", "").upper()).strip("_")


def voa_polygons(path):
    """Parse actual GML types: its XSD wrongly declares nine MultiPolygons."""
    ns = {"gml": "http://www.opengis.net/gml", "ogr": "http://ogr.maptools.org/"}

    def ring(node):
        return [tuple(map(float, xy.split(","))) for xy in node.text.split()]

    def polygon(node):
        shell = ring(
            node.find("gml:outerBoundaryIs/gml:LinearRing/gml:coordinates", ns)
        )
        holes = [
            ring(n)
            for n in node.findall(
                "gml:innerBoundaryIs/gml:LinearRing/gml:coordinates", ns
            )
        ]
        return Polygon(shell, holes)

    def parts(geom):
        if geom.geom_type == "Polygon":
            return [geom]
        if geom.geom_type in {"MultiPolygon", "GeometryCollection"}:
            return [p for child in geom.geoms for p in parts(child)]
        return []

    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("English BRMA(LHA) Layer.gml"))
    records = []
    for feature in root.findall("gml:featureMember/ogr:English_BRMA_LHA__Layer", ns):
        node = feature.find("ogr:geometryProperty", ns)[0]
        assert {n.attrib["srsName"] for n in node.iter() if "srsName" in n.attrib} == {
            "EPSG:27700"
        }
        if node.tag.endswith("}Polygon"):
            geom = polygon(node)
        else:
            assert node.tag.endswith("}MultiPolygon")
            geom = MultiPolygon(
                [polygon(n) for n in node.findall("gml:polygonMember/gml:Polygon", ns)]
            )
        repaired = parts(make_valid(geom))
        assert repaired
        geom = repaired[0] if len(repaired) == 1 else MultiPolygon(repaired)
        assert geom.is_valid and not geom.is_empty
        records.append(
            {
                "brma": brma_key(feature.findtext("ogr:Name", namespaces=ns)),
                "geometry": geom,
            }
        )
    assert len(records) == len({r["brma"] for r in records}) == 152
    return gpd.GeoDataFrame(records, crs=27700)


def lsoa_geography(cache):
    regions = dict(
        zip(
            [f"E1200000{i}" for i in range(1, 10)],
            [
                "NORTH_EAST",
                "NORTH_WEST",
                "YORKSHIRE",
                "EAST_MIDLANDS",
                "WEST_MIDLANDS",
                "EAST_OF_ENGLAND",
                "LONDON",
                "SOUTH_EAST",
                "SOUTH_WEST",
            ],
        )
    )
    hierarchy = pd.read_csv(
        cache / "ons_geo_oa21_lsoa21.csv", usecols=["OA21CD", "LSOA21CD"], dtype=str
    )
    hierarchy = hierarchy[hierarchy.LSOA21CD.str.startswith("E")]
    region = pd.read_csv(
        cache / "ons_geo_oa21_region.csv", usecols=["oa21cd", "rgn22cd"], dtype=str
    )
    hierarchy = hierarchy.merge(
        region, left_on="OA21CD", right_on="oa21cd", how="left", validate="one_to_one"
    )
    assert (
        hierarchy.rgn22cd.notna().all()
        and hierarchy.groupby("LSOA21CD").rgn22cd.nunique().eq(1).all()
    )
    points = gpd.read_file(f"zip://{cache / 'ons_geo_lsoa21_pwc.zip'}")
    points = points[points.LSOA21CD.str.startswith("E")].to_crs(27700)
    assert points.LSOA21CD.is_unique and len(points) == 33755
    eng = gpd.sjoin(
        points,
        voa_polygons(cache / "brma_england_voa_may2020.zip"),
        how="left",
        predicate="intersects",
    )
    assert eng.LSOA21CD.is_unique
    outside = eng[eng.brma.isna()].copy()
    assert set(outside.LSOA21CD) == {c for c in LSOA_OVERRIDES if c.startswith("E")}, (
        "English override set changed"
    )
    outside["brma"] = outside.LSOA21CD.map(LSOA_OVERRIDES)
    wal_points = gpd.read_file(cache / "geo_lsoa21_wales_pwc.geojson").to_crs(27700)
    assert wal_points.LSOA21CD.is_unique and len(wal_points) == 1917
    wal_polygons = gpd.read_file(
        f"zip://{cache / 'geo_brma_wales_2014.zip'}!Broad Rental Market Areas/wales_brma.shp"
    ).to_crs(27700)
    wal_polygons.geometry = wal_polygons.geometry.make_valid()
    wal = gpd.sjoin(
        wal_points,
        wal_polygons[["brma_name", "geometry"]],
        how="left",
        predicate="intersects",
    )
    assert wal.LSOA21CD.is_unique
    assert set(wal.loc[wal.brma_name.isna(), "LSOA21CD"]) == {
        c for c in LSOA_OVERRIDES if c.startswith("W")
    }, "Welsh override set changed"
    wal["brma"] = wal.brma_name.map(lambda n: brma_key(n) if pd.notna(n) else None)
    wal["brma"] = wal.brma.fillna(wal.LSOA21CD.map(LSOA_OVERRIDES))
    xw = pd.concat(
        [
            eng[eng.brma.notna()].sort_values("LSOA21CD"),
            wal.sort_values("LSOA21CD"),
            outside.sort_values("LSOA21CD"),
        ],
        ignore_index=True,
    )[["LSOA21CD", "brma"]]
    regional = (
        hierarchy[["LSOA21CD", "rgn22cd"]]
        .drop_duplicates()
        .set_index("LSOA21CD")
        .rgn22cd.map(regions)
    )
    xw["region"] = xw.LSOA21CD.map(regional)
    xw.loc[xw.LSOA21CD.str.startswith("W"), "region"] = "WALES"
    assert xw.region.notna().all() and xw.LSOA21CD.is_unique
    return xw.rename(columns={"LSOA21CD": "area_code"})


def england_and_wales(cache, entries):
    with zipfile.ZipFile(cache / "census_nomis_ts054.zip") as z:
        ts = pd.read_csv(io.BytesIO(z.read("census2021-ts054-lsoa.csv")))
    col = "Tenure of household: Private rented"
    assert (
        ts[col]
        .eq(
            ts[col + ": Private landlord or letting agency"]
            + ts[col + ": Other private rented"]
        )
        .all()
    )
    prs = ts.set_index("geography code")[col].astype(float)
    rows = []
    paths = sorted(
        cache / e["filename"]
        for e in entries
        if e["filename"].startswith(
            ("census_lsoa_tenure5a_bedrooms_", "census_wales_lsoa_tenure5a_bedrooms_")
        )
    )
    for path in paths:
        data = read_json(path)
        assert not data.get("blocked_areas"), path
        for observation in data["observations"]:
            area, tenure, beds = (d["option_id"] for d in observation["dimensions"])
            if tenure == "3" and beds in {"1", "2", "3", "4"}:
                rows.append(
                    (area, "4+" if beds == "4" else beds, observation["observation"])
                )
    mix = pd.DataFrame(rows, columns=["lsoa", "bedrooms", "n"]).pivot_table(
        index="lsoa", columns="bedrooms", values="n", aggfunc="sum"
    )
    mix = mix.div(mix.sum(axis=1), axis=0)
    xw = lsoa_geography(cache)
    assert set(xw.area_code) == set(prs.index) == set(mix.index)
    assert mix.notna().all().all() or prs[mix.isna().any(axis=1)].eq(0).all()
    cells = mix.fillna(0).mul(prs, axis=0).stack().rename("households").reset_index()
    cells.columns = ["area_code", "bedrooms", "households"]
    out = xw.merge(cells, on="area_code")
    return out.groupby(["region", "brma", "bedrooms"], as_index=False).households.sum()


def scotland(cache):
    """Allocate ward tenure × bedrooms using OA household-weighted BRMA shares."""

    def name(value):
        value = re.sub(r"\s*/\s*", " AND ", value.strip()).upper().replace(" ", "_")
        return "AYRSHIRES" if value == "AYRSHIRE" else value

    def postcode(value):
        key = re.sub(r"\s+", "", str(value).upper())
        match = re.fullmatch(r"([A-Z]{1,2}[0-9][0-9A-Z]?[0-9][A-Z]{2})[A-Z]?", key)
        if not match:
            raise ValueError(f"Invalid Scottish postcode: {value!r}")
        return match.group(1)

    brmas = gpd.read_file(f"zip://{cache / 'scot_brma_boundaries_2009.zip'}")
    assert brmas.geometry.is_valid.all() and len(brmas) == 18
    brmas["brma"] = brmas.BRMAName.map(name)
    oas = gpd.read_file(
        f"zip://{cache / 'scot_oa2022_population_weighted_centroids.zip'}!OutputArea2022_PWC/OutputArea2022_PWC.shp"
    )
    oas = oas.rename(columns={"code": "area_code", "HHcount": "households"}).to_crs(
        brmas.crs
    )
    assert len(oas) == 46363 and not oas.area_code.duplicated().any()
    matches = gpd.sjoin(
        oas, brmas[["brma", "geometry"]], how="left", predicate="within"
    )
    missing = matches.loc[matches.brma.isna(), "area_code"]
    boundary = gpd.sjoin(
        oas[oas.area_code.isin(missing)],
        brmas[["brma", "geometry"]],
        how="left",
        predicate="intersects",
    )
    matches = pd.concat(
        [matches[~matches.area_code.isin(missing)], boundary], ignore_index=True
    )
    single = matches[
        matches.area_code.map(matches.groupby("area_code").brma.count()).eq(1)
    ]
    oas = oas.drop(columns="geometry").merge(
        single[["area_code", "brma"]], on="area_code", how="left", validate="one_to_one"
    )
    original_brma = oas.brma.copy()
    official = pd.read_excel(
        cache / "scot_foi_postcodes_brma_2023.xlsx", usecols=[0, 1], dtype=str
    )
    official.columns = ["brma", "postcode"]
    official["postcode"] = official.postcode.map(postcode)
    official["brma"] = official.brma.map(name)
    candidates = official.groupby("postcode").brma.agg(lambda x: set(x))
    oas["candidates"] = oas.masterpc.map(postcode).map(candidates)
    unique = oas.candidates.map(
        lambda x: next(iter(x)) if isinstance(x, set) and len(x) == 1 else None
    )
    oas["brma"] = oas.brma.fillna(unique)
    assert oas.brma.notna().all(), "OA outside polygons without unique postcode BRMA"
    with zipfile.ZipFile(cache / "scot_census2022_geography_index.zip") as z:
        higher = pd.read_csv(
            z.open("Census_2022_Index/OA_TO_HIGHER_AREAS.csv"),
            encoding="utf-8-sig",
            usecols=["OA2022", "EW2022", "CA2019"],
        )
    oas = oas.merge(
        higher, left_on="area_code", right_on="OA2022", validate="one_to_one"
    )
    conflict = pd.Series(
        [
            isinstance(cs, set) and pd.notna(b) and b not in cs
            for b, cs in zip(original_brma, oas.candidates)
        ],
        index=oas.index,
    )
    oas.loc[conflict & unique.notna(), "brma"] = unique
    unresolved = conflict & unique.isna()
    assert set(oas.loc[unresolved, "area_code"]) == set(SCOTTISH_OVERRIDES), (
        "Scottish override set changed"
    )
    assert oas.loc[unresolved, "CA2019"].eq("S12000039").all()
    # Clear unresolved polygon assignments before applying the documented exceptions.
    oas.loc[unresolved, "brma"] = None
    assert set(oas.loc[oas.brma.isna(), "area_code"]) == set(SCOTTISH_OVERRIDES)
    oas["brma"] = oas.brma.fillna(oas.area_code.map(SCOTTISH_OVERRIDES))
    assert all(
        not isinstance(cs, set) or b in cs for b, cs in zip(oas.brma, oas.candidates)
    )
    shares = oas.groupby(["EW2022", "brma"], as_index=False).households.sum()
    shares["share"] = shares.households / shares.EW2022.map(
        oas.groupby("EW2022").households.sum()
    )
    shares = shares.rename(columns={"EW2022": "area_code"})
    assert shares.area_code.nunique() == 355
    assert shares.groupby("area_code").share.sum().sub(1).abs().max() < 1e-12

    with (cache / "census_2022_ward2022_tenure_bedrooms.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.reader(f))
    assert ["Counting: Households"] in rows and not any(
        r and r[0].strip() == "ERROR" for r in rows
    )
    tenures = {
        "Private rented: Private landlord or letting agency": "private_rented_landlord_or_agent",
        "Private rented: Other": "private_rented_other",
    }
    bands = {
        "One bedroom": "1",
        "Two bedrooms": "2",
        "Three bedrooms": "3",
        "Four bedrooms": "4+",
        "Five or more bedrooms": "4+",
    }
    records, tenure, headers = [], None, None
    for row in rows:
        if not row:
            continue
        label = row[0].strip()
        if len(row) == 1:
            tenure = tenures.get(label)
        elif label == "Number of bedrooms":
            headers = row[1:]
            assert {h for h in headers if h} == set(bands) | {"Total"}, headers
        elif tenure and re.fullmatch(r"S13\d{6}", label):
            for bedroom, value in zip(headers, row[1:], strict=True):
                if bedroom in bands:
                    records.append((label, bands[bedroom], tenure, int(value)))
    census = pd.DataFrame(
        records, columns=["area_code", "bedrooms", "tenure", "households"]
    )
    census = census.groupby(
        ["area_code", "bedrooms", "tenure"], as_index=False
    ).households.sum()
    assert len(census) == 355 * 4 * 2 and set(census.area_code) == set(shares.area_code)
    joined = census.merge(shares[["area_code", "brma", "share"]], on="area_code")
    joined["households"] *= joined.share
    output = joined.groupby(
        ["brma", "bedrooms", "tenure"], as_index=False
    ).households.sum()
    output = output.groupby(["brma", "bedrooms"], as_index=False).households.sum()
    output["region"] = "SCOTLAND"
    return output[["region", "brma", "bedrooms", "households"]]


def northern_ireland(cache):
    """Private-rented households by NIHE BRMA, from Open Government Licence data.

    NIHE defines each BRMA as a set of postcode districts, and NISRA publishes
    Census 2021 households for every postcode district (no suppression at that
    level). Tenure is published only for census areas, and linking those to
    postcode districts needs ONS Postcode Directory records whose Northern
    Ireland licence (LPS end user licence) does not clearly allow publishing
    derived figures. So each BRMA's private-rented households are its census
    households times Northern Ireland's private-rented share (NISRA tenure by
    Data Zone, summed).
    """
    names = {
        "Belfast": "BELFAST",
        "Lough Neagh Lower": "LOUGH_NEAGH_LOWER",
        "Lough Neagh Upper": "LOUGH_NEAGH_UPPER",
        "North": "NORTH_NI",
        "North West": "NORTH_WEST_NI",
        "South East": "SOUTH_EAST_NI",
        "South": "SOUTH_NI",
        "South West": "SOUTH_WEST_NI",
    }
    text = html.unescape(
        re.sub(
            r"<[^>]+>", " ", (cache / "nihe_current_lha_archive_2021.html").read_text()
        )
    )
    choices = "|".join(sorted(names, key=len, reverse=True))
    records = re.findall(rf"({choices}) BRMA:\s*((?:BT|[\d,\s\-])+)", text)
    assert len(records) == 8
    memberships = {}
    for name, districts in records:
        for first, last in re.findall(r"BT(\d+)(?:-BT(\d+))?", districts):
            for number in range(int(first), int(last or first) + 1):
                district = f"BT{number}"
                assert district not in memberships
                memberships[district] = names[name]
    assert len(memberships) == 80
    districts = pd.read_excel(
        cache / "nisra_census2021_postcode_households.xlsx",
        sheet_name="Postcode district",
        header=5,
    )
    districts["district"] = districts["Postcode district"].astype(str).str.strip()
    districts = districts[districts.district.str.fullmatch(r"BT\d+")]
    assert set(districts.district) == set(memberships)
    households = districts.groupby(districts.district.map(memberships)).Households.sum()
    tenure = pd.read_csv(cache / "tenure_7_data_zones.csv")
    tenure.columns = [
        "area_code",
        "area_name",
        "tenure_code",
        "tenure_label",
        "households",
    ]
    assert len(tenure) == 3780 * 7 and tenure.area_code.nunique() == 3780
    private_share = (
        tenure.loc[tenure.tenure_code.isin([5, 6]), "households"].sum()
        / tenure.households.sum()
    )
    out = (
        (households * private_share)
        .rename("households")
        .rename_axis("brma")
        .reset_index()
    )
    out["region"], out["bedrooms"] = "NORTHERN_IRELAND", "all"
    return out[["region", "brma", "bedrooms", "households"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("policyengine_uk_data/storage/brma_private_rented_households.csv"),
    )
    args = parser.parse_args()
    entries = sources()
    verify_cache(args.cache, entries)
    table = pd.concat(
        [
            england_and_wales(args.cache, entries),
            scotland(args.cache),
            northern_ireland(args.cache),
        ],
        ignore_index=True,
    )
    table = table[table.households > 0].copy()
    table["households"] = table.households.round().astype(int)
    table = table[table.households > 0].sort_values(["region", "brma", "bedrooms"])
    data = (
        table[["region", "brma", "bedrooms", "households"]]
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    digest = hashlib.sha256(data).hexdigest()
    expected = "fd40dae019e5eefb8c976873f69c66f9b74a0747505326e8098b0ad99cc2ae1f"
    if digest != expected:
        raise ValueError(f"Rebuilt table differs: expected {expected}, got {digest}")
    args.output.write_bytes(data)
    print(f"Wrote {len(table)} rows to {args.output}; SHA256 {digest}")


if __name__ == "__main__":
    main()
