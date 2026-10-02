# BRMA private-rented households

File: `brma_private_rented_households.csv`, with columns `region, brma, bedrooms, households`. It holds private-rented households (private landlord or letting agency, plus other private rented) by region, Broad Rental Market Area (BRMA) and number of bedrooms (`1`, `2`, `3`, `4+`). Northern Ireland's 2021 census has no bedrooms question, so its rows have `bedrooms = all`. A BRMA that crosses a region boundary has one row per region, for the households on each side.

`datasets/brma.py` uses it to draw each FRS benefit unit's BRMA within its region. LHA categories A and B (shared and one-bedroom) use one-bedroom homes, C uses two-bedroom, D three-bedroom and E four-or-more.

## Sources

| Nation | Households | BRMA geography |
|---|---|---|
| England | Census 2021 (ONS). Private-rented households per LSOA from TS054, split by bedrooms using the LSOA's mix for "private rented or lives rent free" (custom table `hh_tenure_5a` × `number_bedrooms_5a`). Rent-free households are 0.3-1.2% of that category. | VOA BRMA boundary layer, May 2020. Each LSOA is assigned by its population-weighted centroid. Five border LSOAs fall outside the English layer and take the Welsh BRMA that VOA's LHA lookup returns at their centroids. |
| Wales | As for England. | Rent Officers Wales BRMA layer. 11 LSOAs fall in West Cheshire. |
| Scotland | Scotland's Census 2022 (NRS), tenure by bedrooms by 2022 electoral ward. This is the finest geography at which the table is not suppressed. | Scottish Government BRMA polygons. Each ward is split by its output areas' household-weighted centroids, corrected by Rent Service Scotland's postcode-to-BRMA lookup (FOI 202300368850). |
| Northern Ireland | NISRA Census 2021, tenure by Data Zone 2021. | NIHE postcode-district definition of BRMAs. Data Zones that span BRMAs are split by Census 2021 postcode household counts, with suppressed postcodes given NISRA's published district averages. Postcode-to-Data Zone links come from the ONS Postcode Directory, whose Northern Ireland records are under the LPS end user licence; only these BRMA aggregates are published. |

Totals reconcile with the published national private-rented counts to within 0.03%:

| Nation | Private-rented households in this file | Published |
|---|---|---|
| England | 4,795,158 | 4,794,889 |
| Wales | 228,601 | 228,642 |
| Scotland | 323,001 | 323,042 |
| Northern Ireland | 132,466 | 132,436 |

Small-area census counts are perturbed for disclosure control, so their sums differ slightly from national tables.

## Validation

The check below correlates, within each region, the BRMA shares of each candidate weight with DWP's count of Universal Credit households whose housing costs are assessed under LHA ("LHA covers rent" plus "does not cover rent"). The DWP figures are the mean of April 2019 to November 2020 (UC statistics supplementary table 3.2, February 2021).

| Nation | Census (this file) | `lha_list_of_rents.csv.gz` row counts (previous weights) |
|---|---|---|
| England | 0.94 | 0.82 |
| Wales | 0.95 | 0.23 |
| Scotland | 0.82 | 0.21 |

The previous weights' Scottish, Welsh and Northern Ireland lists were copies of English BRMAs' lists (issue #515).

Against the genuine list-of-rents category counts, the census bedroom bands correlate as follows (VOA 2019-20 for England; Scottish Government FOI 202200303624 for Scotland):
- categories B-E: 0.82 to 0.94 in England, 0.95 to 0.98 in Scotland;
- category A, shared accommodation: 0.51 in England, 0.95 in Scotland. The census has no measure of room lets, and no bedroom band does better than 0.59.

## Rebuilding

`tools/brma_households/` (from the repository root) downloads every source, checks each one against its pinned sha256 and rebuilds this file byte for byte; see its `README.md`.
