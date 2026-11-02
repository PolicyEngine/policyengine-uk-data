# ONS age target sources

`policyengine_uk_data/storage/regional_age_80_plus.csv` contains public population
projection aggregates in persons. It contains no FRS data. Reproduce it from the
repository root with the locked environment:

```sh
uv run python tools/ons_age_targets/build.py
```

Downloads are cached in `.job/`; SHA-256 checks pin the source files. The generator
uses pandas, requests and openpyxl, which are already locked dependencies.

The English regions use the [ONS 2022-based single-year subnational
projection](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/datasets/localauthoritiesinenglandz1),
five-year migration variant, published 24 June 2025. The [ONS release bulletin](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/bulletins/subnationalpopulationprojectionsforengland/2022based)
identifies this variant as the one aligned with the 2022-based principal national
projection. For each of
the 309 English local authorities, the extract sums ages 80 through 89 and the
terminal `90 and over` group. The committed `storage/oa_crosswalk.csv.gz` maps all
309 authorities unambiguously to the nine English regions. Summing those regions
therefore reproduces the sum of all 309 authorities, without omitting or counting
an authority twice. Published counts have three decimal places; the output retains
that precision.

Scotland, Wales and Northern Ireland use the ONS **2022-based principal national
projections**, so all 12 regional 80+ targets have the same projection vintage and
cover 2022–2029. Each country total sums both sexes at ages 80–104 and the terminal
groups `105 - 109` and `110 and over`. The regional CSV includes the reference URL
for each row. This replaces the legacy regional top-band values; it is a source
refresh as well as a correction to the open-ended band definition. The lower
regional age bands remain in `demographics.csv`.

Pinned downloads, relative to
`https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationprojections/datasets/`:

| Source | Download suffix | SHA-256 |
| --- | --- | --- |
| English local authorities | `localauthoritiesinenglandz1/2022basedfiveyearmigrationvariantprojection/2022snpppopulationsyoa5yr.zip` | `171cd05fc753277388352c76abf8cc8b305ca537cd593a6e4467af7e8a8e4b5b` |
| Scotland | `z5zippedpopulationprojectionsdatafilesscotland/2022based/sc.zip` | `6e0dc9250a9890680754ab501be56973c722a57950e4535c364fbc4494372b3d` |
| Wales | `z4zippedpopulationprojectionsdatafileswales/2022based/wa.zip` | `32a7c2aacf9b3c945abeaa19a6d6d7aa62133f491bfca5a9700a393244a7b051` |
| Northern Ireland | `z6zippedpopulationprojectionsdatafilesnorthernireland/2022based/ni.zip` | `69b006c2741943d97d153cd024e4b21ea94b52ef766efcde6bd737a400d95489` |

The national targets continue to use the existing **2024-based UK principal
projection**. The test fixture `policyengine_uk_data/tests/fixtures/ons_uk_population_2025.csv` preserves
its complete `Sex`, `Age` and `2025` population columns, including `105 - 109` and
`110 and over`. Its source ZIP is
`z1zippedpopulationprojectionsdatafilesuk/2024based/uk.zip`, SHA-256
`dea79703b9caddd560f1958b5a8ca12aa817b2396d244e6a2c775b1af7f48ab1`.

The old regional rows first appeared in commit `c98c0ccf` (9 September 2024);
`63dfba48` (10 June 2025) reformatted them without changing their values. Their
reference `ons_age_sex_region` does not identify a projection vintage or extraction
method. Their finite `80_89` names alone do not prove that their published values
excluded 90+: the replacement uses directly reproducible sums rather than adding
an estimated 90+ population to those legacy values.
