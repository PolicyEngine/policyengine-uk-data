# Output Area Calibration Pipeline

This document describes the plan to port the US-side clone-and-prune calibration methodology to the UK data, going down to Output Area (OA) level — the UK equivalent of the US Census Block.

## Background

The US pipeline (policyengine-us-data) uses an L0-regularized clone-and-prune approach:
1. Clone each CPS household N times
2. Assign each clone a different Census Block (population-weighted)
3. Build a sparse calibration matrix (targets x records)
4. Run L0-regularized optimization to drop most clones, keeping only the best-fitting records per area
5. Publish per-area H5 files from the sparse weight vector

The UK pipeline currently uses standard PyTorch Adam gradient descent on a dense weight matrix (n_areas x n_households) at constituency (650) and local authority (360) level. We want to bring the US approach to the UK at Output Area (~180K OAs) granularity.

## Implementation Phases

### Phase 1: Output Area Crosswalk & Geographic Assignment
**Status: Complete**

Build the OA crosswalk and population-weighted assignment function.

**Deliverables:**
- `policyengine_uk_data/calibration/oa_crosswalk.py` — downloads/builds the OA → LSOA → MSOA → LA → constituency → region → country crosswalk
- `policyengine_uk_data/storage/oa_crosswalk.csv.gz` — compressed crosswalk file
- `policyengine_uk_data/calibration/oa_assignment.py` — assigns cloned records to OAs (population-weighted, country-constrained)
- Tests validating crosswalk completeness and assignment correctness

**Data sources:**
- ONS Open Geography Portal: OA → LSOA → MSOA → LA lookup
- ONS mid-year population estimates at OA level
- ONS OA → constituency lookup (2024 boundaries)

**US reference:** PR #484 (census-block-assignment)

---

### Phase 2: Clone-and-Assign
**Status: Complete**

Clone each FRS household N times and assign each clone a different OA.

**Deliverables:**
- `policyengine_uk_data/calibration/clone_and_assign.py` — clones all three entity tables (household, person, benunit), remaps IDs, divides weights by N, attaches OA geography columns
- `datasets/create_datasets.py` — clone step inserted after imputations, before uprating/calibration (N=10 production, N=2 testing)
- `tests/test_clone_and_assign.py` — 14 tests covering dimensions, weight preservation, ID uniqueness, FK integrity, country constraints, data preservation

**Key design:**
- N=10 clones in production, N=2 in testing mode
- Constituency collision avoidance: each clone gets a different constituency where possible
- Country constraint preserved: English households → English OAs only
- Weights divided by N so population totals are preserved
- Pure pandas/numpy operations — no simulation required, fast execution

**US reference:** PR #457 (district-h5) + PR #531 (census-block-calibration)

---

### Phase 3: L0 Calibration Engine
**Status: Not Started**

Port L0-regularized optimization from US side.

**Deliverables:**
- `policyengine_uk_data/utils/calibrate_l0.py`
- Add `l0-python` dependency to `pyproject.toml`

**Key design:**
- HardConcrete gates for continuous L0 relaxation
- Relative squared error loss
- L0 + L2 regularization with presets (local vs national)
- Keep existing `calibrate.py` as fallback during validation

**US reference:** PR #364 (bogorek-l0) + PR #365

---

### Phase 4: Sparse Matrix Builder
**Status: Not Started**

Build sparse calibration matrix from cloned dataset.

**Deliverables:**
- `policyengine_uk_data/calibration/matrix_builder.py`
- Wire existing `targets/sources/` definitions into sparse matrix rows

**US reference:** PR #456 + PR #489

---

### Phase 5: SQLite Target Database
**Status: Not Started**

Hierarchical target storage: UK → country → region → LA → constituency → MSOA → LSOA → OA.

**Deliverables:**
- `policyengine_uk_data/db/` directory with ETL scripts
- Migrate existing CSV/Excel targets into SQLite

**US reference:** PR #398 (treasury) + PR #488 (db-work)

---

### Phase 6: Local Area Publishing
**Status: Not Started**

Generate per-area H5 files from sparse weights. Modal integration for scale.

**Deliverables:**
- `policyengine_uk_data/calibration/publish_local_h5s.py`

**US reference:** PR #465 (modal)
