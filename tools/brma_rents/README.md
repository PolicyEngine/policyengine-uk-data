# BRMA private rents

Rebuilds `policyengine_uk_data/storage/brma_private_rents.csv`: for each Broad Rental Market Area (BRMA) and LHA category, the median weekly rent on the rent officers' list of rents and the spread of log rents, in the prices of the dataset year. `datasets/brma.py` uses it to place private renters given their rent.

## Run

From the repository root, in an environment with `openpyxl`:

```
python tools/brma_rents/build.py <cache dir>            # download, verify, write the CSV
python tools/brma_rents/build.py <cache dir> --check    # compare the rebuild with the CSV
```

Every source in `sources.yaml` is downloaded once to the cache and checked against its pinned sha256 before anything is read. Fetching reuses `tools/brma_households/fetch.py`.

## Method

`--year` (default 2024) is the dataset year, April to March. The list of rents behind an April determination holds rents collected from October two years earlier to September of the year before.

1. **Uprating.** Each list is moved to the dataset year with the ONS Price Index of Private Rents for its region or nation: the mean index over the dataset year divided by the mean index over the list's collection months. A BRMA that crosses a region boundary uses the region holding most of its private renters.
2. **England (`basis = list`).** The two VOA lists whose collection months overlap the dataset year (for 2024: April 2025 and April 2026) are uprated and pooled. The median is the median of log rents; the spread is their interquartile range divided by 1.349, the interquartile range of a standard normal.
3. **Scotland (`30th percentile and list spread`).** The spread comes from Rent Service Scotland's lists released under FOI 202200303624 (years to September 2019, 2020 and 2021, the latest released), averaged over sheets by their number of rents. The median comes from the 30th percentiles published for the same two determinations as England's lists, uprated, assuming log-normal rents: median = 30th percentile × exp(0.5244 × spread).
4. **Wales and Northern Ireland (`30th percentile and typical spread`).** Neither nation publishes its lists. The median comes from published 30th percentiles as for Scotland; the spread is the median spread of the English and Scottish cells of the same category. Wales uses the same two determinations. Northern Ireland uses the latest determination with published 30th percentiles (April 2024).

`rents` is the number of list entries behind a cell's spread (0 where the spread is borrowed).

## When the dataset year changes

Add the newer VOA lists, the newer published-rates file and a newer ONS index to `sources.yaml`, run with the new `--year`, and commit the CSV. A region-wide error in the level of rents does not change any household's BRMA probabilities, because the model fitted in `datasets/brma.py` has a free shift per region; only rents of BRMAs relative to others in their region matter.
