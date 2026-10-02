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

# BRMA private rents

File: `brma_private_rents.csv`, with columns `brma, lha_category, median_weekly_rent, log_sd, rents, basis`. For each BRMA and LHA category it holds the median weekly rent on the rent officers' list of rents, in 2024-25 prices, and the standard deviation of log rents on the list. `rents` is the number of list entries behind the spread, and `basis` says how the row was made.

`datasets/brma.py` uses it to draw a private renter's BRMA given its rent: the probability of each BRMA in the household's region is proportional to the census private-rented households with the home's number of bedrooms (the file above) times the density of the household's rent on the BRMA's list for homes of that size (log-normal, with this file's median and spread).

## Sources

| Nation | Median | Spread | `basis` |
|---|---|---|---|
| England | VOA lists of rents for April 2025 and April 2026 (rents collected October 2023 to September 2025), pooled. | The same lists: interquartile range of log rents ÷ 1.349. | `list` |
| Scotland | Published 30th percentiles for April 2025 and April 2026, raised to a median assuming log-normal rents. | Rent Service Scotland's lists for the years to September 2019-2021 (Scottish Government FOI 202200303624), the latest released. | `30th percentile and list spread` |
| Wales | Rent Officers Wales's lists for April 2024 and April 2025 (rents collected October 2022 to September 2024; Welsh Government FOI ATISN 25142), pooled. They reproduce the published April 2024 and April 2025 30th percentiles (median difference 0; 94% and 98% of cells within 1%). | The same lists. | `list` |
| Northern Ireland | As Scotland, from the Housing Executive's 30th percentiles for April 2024, the latest in policyengine-uk's table. | The median spread of the English, Welsh and Scottish cells of the same category; no Northern Ireland list is published. | `30th percentile and typical spread` |

The published 30th percentiles are those compiled in policyengine-uk's `lha_published_rates.csv.gz`. Every list and percentile is moved to 2024-25 with the ONS Price Index of Private Rents for its region or nation.

## Why reported rents need a model

The lists are the rent officers' lists of rents that set LHA rates. Private renters in the FRS report lower and more varied rents. Against FRS 2024-25, a model fitted by maximum likelihood (`fit_reported_rent_model`) finds:
- reported rents are 4% to 20% below the list median for the same region and bedrooms (a separate shift per region);
- about 84% of households follow the list's distribution plus a little noise (0.13 log points);
- about 16% pay far less (0.64 log points lower on average, with wide noise). Their rent says little about where they live, so they keep roughly the census shares.

Because every region has its own shift, a region-wide error in the level of the lists (for example, in uprating) changes nothing: only a BRMA's rents relative to others in its region matter.

## Validation

- **The rent kernel is calibrated out of sample.** Taking 150,000 April 2026 list entries in categories B-E as households whose BRMA is known, with each region's April 2026 entry counts as the prior and medians and spreads from the April 2025 list only, the mean probability given to the true BRMA rises from 0.085 (census shares) to 0.129 with rent, and the most likely BRMA is the true one for 21.8% of entries against 16.1%. Stated probabilities match observed frequencies: BRMAs given 20-30% are right 23.8% of the time (mean stated 24.2%), those given 50-60% are right 53.4% (54.2%), and those given 90%+ are right 91.3% (93.3%).
- **Published percentiles stand in for lists.** Treating England, Wales and Scotland as Northern Ireland is treated (30th percentile and typical spread) changes an FRS private renter's BRMA probabilities by 4.5 points of total variation on average in England, 7.4 in Wales and 6.6 in Scotland. Conditioning on rent at all moves them by 20.5, 21.3 and 32.8 points. Measured against the April 2024 LHA rate for the home's size (not the household's LHA category), the mean weekly rent within that rate is, with the lists, the percentile treatment and census shares only: England £197.24, £197.42 and £190.59; Wales £127.25, £128.17 and £121.53; Scotland £146.91, £148.16 and £133.63.
- **Scotland's medians agree with its own lists.** The 2021 list medians, uprated with each Scottish BRMA's own ONS index, correlate at 0.98 with the medians used here (mean gap 1.8%, standard deviation 8.4%). The ONS indices grew by between 10% and 36% across Scottish BRMAs, which is why the current 30th percentiles are used rather than one national uprating.
- **The census shares survive.** Within every region, FRS private renters' weighted mean BRMA probabilities stay within 3 points of total variation of the census shares for their homes' bedrooms. A test on the built dataset enforces 5 points.

## Limits

- The census band and the list category are chosen by the home's bedrooms. Shared accommodation (category A) is never used to place a household, because the FRS does not identify rooms in shared houses.
- A household's BRMA is still not tied to the output area or local authority given to it later in the build.
- Northern Ireland borrows the spread of rents, and its 30th percentiles are for April 2024, uprated. NIHE publishes mean advertised rents by BRMA and bedrooms, which could pin its spreads; they are not used.

## Rebuilding

`tools/brma_rents/` downloads every source, checks each against a pinned sha256 and rebuilds this file; see its `README.md`.
