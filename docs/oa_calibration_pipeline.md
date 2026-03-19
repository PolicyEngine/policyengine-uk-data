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
**Status: Complete**

L0-regularised calibration using HardConcrete gates from the `l0-python` package.

**Deliverables:**
- `policyengine_uk_data/utils/calibrate_l0.py` — wraps `SparseCalibrationWeights` with the existing target matrix interface; builds a sparse `(n_targets, n_records)` matrix with country masking baked into sparsity; supports target grouping for balanced metric weighting
- `l0-python>=0.4.0` added to dev dependencies in `pyproject.toml`
- `tests/test_calibrate_l0.py` — 6 tests covering sparse matrix construction, country masking, zero-target filtering, group structure, error reduction, and sparsity behaviour
- Existing `calibrate.py` preserved as fallback

**Key design:**
- HardConcrete gates for continuous L0 relaxation
- Relative squared error loss with target group weighting
- L0 + L2 regularisation (configurable strengths)
- Sparse matrix representation — country masking baked into sparsity pattern rather than dense mask multiplication
- Same `matrix_fn` / `national_matrix_fn` interface as existing calibration

**US reference:** PR #364 (bogorek-l0) + PR #365

---

### Phase 4: Sparse Matrix Builder
**Status: Complete**

Build sparse calibration matrix from cloned dataset, bridging Phase 2 (clone-and-assign) and Phase 3 (L0 calibration).

**Deliverables:**
- `policyengine_uk_data/calibration/matrix_builder.py` — sparse assignment matrix builder, consolidated metric computation, target loading for both constituency and LA levels
- `tests/test_matrix_builder.py` — 10 tests covering assignment matrix shape, sparsity, binary values, area types, unassigned households

**Key design:**
- `build_assignment_matrix()`: builds sparse `(n_areas, n_households)` binary matrix from OA geography columns — each household in exactly one area
- `create_cloned_target_matrix()`: backward-compatible `(metrics, targets, country_mask)` interface for use as `matrix_fn` in both dense Adam and L0 calibrators
- `build_sparse_calibration_matrix()`: direct sparse path producing `(M_csr, y, group_ids)` — skips dense country_mask entirely, O(n_households × n_metrics) non-zeros
- `_compute_household_metrics()`: consolidates metric computation duplicated between constituency and LA loss files
- `_load_area_targets()`: consolidates target loading with national consistency adjustments, boundary mapping, and LA extras
- Supports both constituency (650 areas) and LA (360 areas) geography types

**US reference:** PR #456 + PR #489

---

### Phase 5: SQLite Target Database
**Status: Complete**

Hierarchical target storage with two parallel geographic branches:
- Administrative: country → region → LA → MSOA → LSOA → OA
- Parliamentary: country → constituency

LA and constituency are parallel — a constituency can span multiple LAs and vice versa.

**Deliverables:**
- `policyengine_uk_data/db/schema.py` — SQLite schema: `areas` (geographic hierarchy), `targets` (definitions), `target_values` (year-indexed values)
- `policyengine_uk_data/db/etl.py` — ETL loading areas from OA crosswalk + area code CSVs, targets from registry + local CSV/XLSX sources
- `policyengine_uk_data/db/query.py` — query API: `get_targets()`, `get_area_targets()`, `get_area_children()`, `get_area_hierarchy()`
- `tests/test_target_db.py` — tests covering schema creation, area hierarchy, target loading, queries

**Key design:**
- Areas table with `parent_code` encoding hierarchy; LAs parent to regions, constituencies parent to countries
- Targets loaded from two sources: registry (national/country/region via `get_all_targets()`) and local CSVs (constituency/LA age, income, UC, LA extras)
- Query API supports filtering by geographic level, area code, variable, source, year
- `get_area_hierarchy()` walks up the tree from any code (e.g. OA → LSOA → MSOA → LA → region → country)
- Full rebuild via `python -m policyengine_uk_data.db.etl`

**US reference:** PR #398 (treasury) + PR #488 (db-work)

---

### Phase 6: Local Area Publishing
**Status: Not Started**

Generate per-area H5 files from sparse weights. Modal integration for scale.

**Deliverables:**
- `policyengine_uk_data/calibration/publish_local_h5s.py`

**US reference:** PR #465 (modal)
