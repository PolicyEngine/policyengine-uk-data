# Consuming panel snapshots from `policyengine-uk`

This is the coordination note for the sibling [`policyengine-uk`](https://github.com/PolicyEngine/policyengine-uk) repository when per-year panel snapshots from step 2 of [#345](https://github.com/PolicyEngine/policyengine-uk-data/issues/345) start being used in simulations. Nothing here changes the single-year behaviour — it describes what the *consumer* side has to know when someone passes it an `enhanced_frs_<year>.h5` built by `create_yearly_snapshots`.

## The short version

- A panel snapshot is just a `UKSingleYearDataset` stored at `enhanced_frs_<year>.h5`. The `time_period` attribute on the file is that year in monetary terms.
- Person, benefit unit and household IDs are stable across the full set of yearly files in a panel build — they are the join keys documented in [README.md § Panel ID contract](../README.md) and enforced by `policyengine_uk_data.utils.panel_ids.assert_panel_id_consistency`.
- The smoothness-calibrated weights (see #345 step 5) are expected to evolve smoothly — no 10× jumps year on year.

## Decisions `policyengine-uk` needs to make

### 1. Runtime uprating vs. stored-year uprating

Today `policyengine-uk` re-uprates at simulation time: the consumer calls `Microsimulation(dataset=..., time_period=2027)` against a 2023-valued file and the framework scales variables forward.

Once a 2027 snapshot exists (already uprated and demographically aged), this re-uprating becomes **double counting**. Two sensible options, tracked in #345 step 6:

- **A. Skip runtime uprating when the requested year matches `dataset.time_period`.** Cheapest change — single conditional in `Microsimulation`. Non-matching years still get uprated as today.
- **B. Tag panel snapshots with a flag** that turns off runtime uprating entirely (`dataset.is_panel = True`). More explicit but requires a schema bump.

Option A is backwards-compatible. Option B is tidier long-term. My current view: ship A first, revisit once panel consumption patterns shake out.

### 2. Fixture selection in downstream tests

If `policyengine-uk` tests want to cover panel behaviour, they should:

- Accept the `enhanced_frs_for_year(year)` factory fixture pattern (see `policyengine_uk_data/tests/conftest.py`).
- Not hard-code `enhanced_frs_2023_24.h5`; read `dataset.time_period` off the dataset instead.

### 3. Which years to build by default

The `create_yearly_snapshots` helper is year-range agnostic. The data repo currently ships `uprating_factors.csv` covering 2020-2034, so that is the natural envelope. The downstream repo's CI should pick a small representative subset (e.g. 2023, 2025, 2030) rather than all eleven years, to keep build times reasonable.

## Out of scope for #345

- Behavioural responses (labour supply, migration) in reforms across panel years.
- Cross-year output aggregation (e.g. "lifetime income tax paid by a household").
- Integration with microsimulation-level panel joins (joining simulation output dataframes by `person_id` across year snapshots).

Each is a legitimate follow-up once the data side is stable.

## Related code

- `policyengine_uk_data/datasets/yearly_snapshots.py` — producer.
- `policyengine_uk_data/utils/panel_ids.py` — ID contract enforcement.
- `policyengine_uk_data/utils/demographic_ageing.py` — the other side of step 3.
- `policyengine_uk_data/storage/upload_yearly_snapshots.py` — upload to private repo.
- `policyengine_uk_data/tests/conftest.py` — `enhanced_frs_for_year` factory.
