# BRMA private-rented households

File: `brma_private_rented_households.csv`, with columns `region, brma, bedrooms, households`. It holds private-rented households (private landlord or letting agency, plus other private rented) by region, Broad Rental Market Area (BRMA) and number of bedrooms (`1`, `2`, `3`, `4+`). Northern Ireland's 2021 census has no bedrooms question, so its rows have `bedrooms = all`. A BRMA that crosses a region boundary has one row per region, for the households on each side.

`datasets/brma.py` uses it to draw each FRS benefit unit's BRMA within its region. LHA categories A and B (shared and one-bedroom) use one-bedroom homes, C uses two-bedroom, D three-bedroom and E four-or-more.

## Sources

| Nation | Households | BRMA geography |
|---|---|---|
| England | Census 2021 (ONS). Private-rented households per LSOA from TS054, split by bedrooms using the LSOA's mix for "private rented or lives rent free" (custom table `hh_tenure_5a` × `number_bedrooms_5a`). This is the only tenure × bedrooms split ONS releases for every LSOA; rent-free households are 0.3-1.2% of the category. | VOA BRMA boundary layer, May 2020. Each LSOA is assigned by its population-weighted centroid. Five border LSOAs fall outside the English layer and take the Welsh BRMA that VOA's LHA lookup returns near their centroids. |
| Wales | As for England. | Rent Officers Wales BRMA layer (2012 boundaries, 2014 release). 11 LSOAs fall in West Cheshire. Cardiff Bay (W01002024) lies outside the layer and is assigned to Cardiff, as VOA's lookup returns for its postcodes. |
| Scotland | Scotland's Census 2022 (NRS), tenure by bedrooms by 2022 electoral ward. This is the finest geography at which the table is not suppressed. | Scottish Government BRMA polygons. Each ward is split across BRMAs by its output areas' household counts, placing each output area by its population-weighted centroid. Rent Service Scotland's postcode-to-BRMA lookup (FOI 202300368850) moves 61 Balloch output areas to West Dunbartonshire: 59 on unique postcode matches and 2 via the council's own lookup. |
| Northern Ireland | NISRA Census 2021 households by postcode district, times Northern Ireland's private-rented share (NISRA tenure by Data Zone, 17.2%). | NIHE's definition of each BRMA as a set of postcode districts. |

Northern Ireland's private-rented households are not split by area within the nation. Linking NISRA's tenure areas to postcode districts would need the ONS Postcode Directory's Northern Ireland records, whose licence (LPS end user licence) does not clearly allow publishing derived figures. Using every BRMA's all-tenure households instead moves BRMA shares by 1.5 percentage points on average; for example, Belfast gets 20.1% of NI's private renters where the private-rented split would give 23.5%.

Totals reconcile with the published national private-rented counts to within 0.03%:

| Nation | Private-rented households in this file | Published |
|---|---|---|
| England | 4,795,158 | 4,794,889 |
| Wales | 228,601 | 228,642 |
| Scotland | 323,001 | 323,042 |
| Northern Ireland | 132,449 | 132,436 |

Small-area census counts are perturbed for disclosure control, so their sums differ slightly from national tables.

## Validation

The check below correlates, within each region, the BRMA shares of each candidate weight with DWP's count of Universal Credit households whose housing costs are assessed under LHA ("LHA covers rent" plus "does not cover rent"). The DWP figures are the mean of April 2019 to November 2020 (UC statistics supplementary table 3.2, February 2021).

To make the comparison, each BRMA's households from this file are summed across bedroom bands and region parts, then placed in policyengine-uk's single region for that BRMA.

| Nation | This file | `lha_list_of_rents.csv.gz` row counts (previous weights) |
|---|---|---|
| England | 0.94 | 0.82 |
| Wales | 0.95 | 0.23 |
| Scotland | 0.82 | 0.21 |

The previous weights' Scottish, Welsh and Northern Ireland lists were copies of English BRMAs' lists (issue #515).

Against the genuine list-of-rents category counts, the census bedroom bands correlate as follows (VOA 2019-20 for England; Scottish Government FOI 202200303624 for Scotland):
- categories B-E: 0.82 to 0.94 in England, 0.95 to 0.98 in Scotland;
- category A, shared accommodation: 0.51 in England, 0.95 in Scotland. The census has no measure of room lets, and no bedroom band does better than 0.59.

## Rebuilding

`tools/brma_households/` (run from the repository root) downloads every source, checks each one against a pinned sha256 and rebuilds this file byte for byte; see its `README.md`.
