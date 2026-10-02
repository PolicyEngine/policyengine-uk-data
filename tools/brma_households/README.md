# BRMA private-rented households

`policyengine_uk_data/storage/brma_private_rented_households.csv` (the default `--output`) contains 936 rows: `region,brma,bedrooms,households`.
It counts private landlord/letting agency and other private-rented households by
Broad Rental Market Area (BRMA), retaining each area's census region. Bedrooms are
`1`, `2`, `3`, `4+`; Northern Ireland uses `all` because its census did not ask bedrooms.

## Sources and method

- **England, Census 2021:** ONS/Nomis TS054 LSOA private-rented totals multiplied
  by each LSOA's ONS custom-API tenure5a code 3 bedroom proportions. ONS V4
  population-weighted centroids intersect VOA May 2020 BRMA polygons; regions
  come from ONS OA-to-LSOA and OA-to-region lookups. GML is parsed directly to
  preserve nine incorrectly declared MultiPolygons; `make_valid` repairs seven
  self-intersections. Five border LSOAs use documented official VOA lookup overrides:
  E01014018 → Brecon and Radnor; E01022272–E01022275 → Monmouthshire.
- **Wales, Census 2021:** the same TS054 × tenure5a method and Welsh ONS V4
  centroids, with the archived Rent Officers Wales polygons (May 2012 geometry,
  September 2014 release). Invalid geometry is repaired. Cardiff Bay W01002024
  uses a fixed override established by 13 official postcode replies on 2 October 2026.
  Eleven Welsh LSOAs in West Cheshire retain region `WALES`.
- **Scotland, Census 2022:** NRS ward tenure × bedrooms, allocated using OA
  household-weighted centroid shares in Scottish Government BRMA polygons.
  Official postcode candidates supply nine missing assignments and correct 61
  Balloch OAs; two ambiguous candidates use West Dunbartonshire council membership.
  Four and five-or-more bedrooms combine into `4+`.
- **Northern Ireland, Census 2021:** NISRA Data Zone tenure codes 5 and 6,
  allocated using NIHE postcode-district BRMAs and ONSPD postcode-to-zone records.
  Single-BRMA zones receive share 1; mixed zones use census postcode household
  counts, substituting NISRA district averages after stripping district-key whitespace.

BRMA cells are rounded to whole households with pandas; zero rows are dropped.
No national balancing adjustment is made. Each of 89 inputs has its exact URL and cache filename in `sources.yaml`.
The 72 ONS custom-API batches pin decoded JSON bytes with `content_sha256`;
other inputs pin original bytes with `sha256`. Gzip and plain JSON are accepted
without parsing or reserialising for verification. Live reply pages and postcode
extracts used only to establish overrides are excluded, as are research checks.
All eight overrides have dated evidence comments in `build.py`; exact unresolved
area sets are asserted before assignment to guard against boundary changes.

## Rebuild

Run from the repository root. The sole browser/session download is Scotland's
ward export; follow its `manual: true` manifest instructions and place it in the
cache. Scripts never log in or accept terms. Both commands verify all input pins;
the builder uses only cached originals and checks the expected output digest.

```sh
nice -n 10 uvx --with pandas --with geopandas --with shapely --with pyogrio --with openpyxl --with pyyaml python tools/brma_households/fetch.py .brma-cache
nice -n 10 uvx --with pandas --with geopandas --with shapely --with pyogrio --with openpyxl --with pyyaml python tools/brma_households/build.py .brma-cache
```

`fetch.py --only SOURCE_ID ...` supports selected download checks; `build.py
--output PATH` chooses the output destination. Missing files or changed hashes fail.

## Licences and limits

Census and most geography inputs use the Open Government Licence; retain ONS,
NRS, Scottish/Welsh Government and Ordnance Survey attributions in the manifest.
NIHE's archived page has no stated open reuse licence; live VOA replies are not inputs.
**NI postcode geography comes from ONSPD records under the LPS Northern Ireland
End User Licence; only aggregated BRMA counts are published, never raw BT records.**
Centroids and ward/postcode shares approximate household locations. England/Wales
bedroom proportions include rent-free households; census cells are perturbed.
Geography vintages differ and bedrooms describe accommodation, not LHA entitlement.
The 34 postcode records without Data Zone codes do not contribute to mixed-zone shares.

## Verification

On 2 October 2026, all 89 inputs copied from retained originals passed `fetch.py`
verification with no downloads, including all 72 decoded-content hashes.
Fresh `fetch.py` downloads of eight originals covered all seven publishers;
both an English and a Welsh custom-API batch passed decoded-content verification.
The corrected full rebuild was byte-identical to the updated reference, 936 rows:
`3ba714d4f5267656f19cd60785da379804420f633aa4c5745bdd0fba20473e72`.
The NI key fix changes Belfast by +1 and South East NI by −1 household.
Changed-content, missing-manual and override-set mismatch checks were rejected.
Scotland's manual export was verified from its retained original, not re-exported.
`nice -n 10` was invoked; the sandbox refused priority changes and Git index writes.
