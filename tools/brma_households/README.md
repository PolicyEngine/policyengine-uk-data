# BRMA private-rented households

`build.py` writes `policyengine_uk_data/storage/brma_private_rented_households.csv` (936 rows: `region,brma,bedrooms,households`). The table counts private-rented households (private landlord or letting agency, plus other private rented) by Broad Rental Market Area (BRMA), keeping each area's census region. Bedrooms are `1`, `2`, `3` or `4+`; Northern Ireland uses `all` because its census did not ask about bedrooms. Sources, method and validation are summarised in `policyengine_uk_data/storage/BRMA_DATA_SOURCES.md`.

## Method

- **England, Census 2021.**
  - Each LSOA's TS054 private-rented total is split by that LSOA's bedroom mix for ONS custom-API tenure category 3, "private rented or lives rent free".
  - The LSOA's ONS population-weighted centroid places it in a VOA May 2020 BRMA polygon.
  - Regions come from the ONS OA-to-LSOA and OA-to-region lookups.
  - The GML is parsed directly so that nine MultiPolygons declared as Polygons keep all their parts. `make_valid` repairs seven self-intersections.
- **Wales, Census 2021.**
  - Same method as England, using the archived Rent Officers Wales polygons (May 2012 geometry, September 2014 release).
  - Eleven Welsh LSOAs fall in West Cheshire and keep region `WALES`.
- **Scotland, Census 2022.**
  - NRS ward tenure × bedrooms tables are split across BRMAs by output-area household counts. Each output area is placed by its population-weighted centroid in the Scottish Government BRMA polygons.
  - Official postcode candidates assign nine output areas whose centroids fall outside every polygon.
  - Rent Service Scotland's postcode lookup moves 61 Balloch output areas to West Dunbartonshire: 59 on unique postcode matches and 2 via the council's own lookup.
  - Four and five-or-more bedrooms are combined into `4+`.
- **Northern Ireland, Census 2021.**
  - NISRA households by postcode district are summed into NIHE's postcode-district BRMAs.
  - The totals are scaled by Northern Ireland's private-rented share from NISRA tenure by Data Zone.
  - A private-rented split by area would need ONS Postcode Directory Northern Ireland records, which are under the LPS end user licence, so it is not attempted.
- **Overrides.** Eight area assignments come from official lookups rather than polygons: five English border LSOAs, Cardiff Bay and two Scottish output areas. Each carries a dated evidence comment in `build.py`. The build asserts the exact set of areas needing an override, so a boundary change fails loudly.
- **Output.** Cells are rounded to whole households, and empty cells are dropped. There is no national balancing adjustment.

## Inputs

`sources.yaml` lists the 87 inputs with their URLs, landing pages, licences and cache file names.
- The 72 ONS custom-API batches are pinned on the sha256 of the decoded JSON (`content_sha256`), because their gzip bytes vary between downloads.
- Every other input is pinned on the sha256 of its original bytes.

## Rebuild

Run from the repository root. Scotland's ward table is the one manual download: follow its `manual: true` instructions in `sources.yaml` and place the file in the cache. The scripts never log in or accept terms. Both scripts check every input's hash, and `build.py` reads only cached originals and checks the output's digest.

```sh
nice -n 10 uvx --with pandas --with geopandas --with shapely --with pyogrio --with openpyxl --with pyyaml python tools/brma_households/fetch.py .brma-cache
nice -n 10 uvx --with pandas --with geopandas --with shapely --with pyogrio --with openpyxl --with pyyaml python tools/brma_households/build.py .brma-cache
```

- `fetch.py --only SOURCE_ID ...` fetches or checks only the listed sources.
- `build.py --output PATH` writes the table elsewhere.

## Licences and limits

- **Licences.** Census and geography inputs are under the Open Government Licence. NIHE's archived BRMA page has no stated reuse licence; the build uses only its postcode-district lists.
- **Locations are approximate.** Centroids and ward shares approximate where households live.
- **Rent-free households.** England and Wales bedroom mixes include them.
- **Disclosure control.** Census cells are perturbed.
- **Geography vintages differ** between nations.
- **Bedrooms describe the home,** not the household's LHA entitlement.
- **Northern Ireland has one private-rented share for every BRMA.**

## Verification

On 2 October 2026:
- All inputs passed `fetch.py` against a cache filled from retained originals.
- Fresh downloads of eight originals, covering all seven publishers and including an English and a Welsh ONS batch, matched their pins.
- The rebuild has 936 rows, sha256 `fd40dae019e5eefb8c976873f69c66f9b74a0747505326e8098b0ad99cc2ae1f`.
