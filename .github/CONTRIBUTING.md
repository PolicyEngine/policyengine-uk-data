# Contributing to policyengine-uk-data

See the [shared PolicyEngine contribution guide](https://github.com/PolicyEngine/.github/blob/main/CONTRIBUTING.md) for cross-repo conventions (towncrier changelog fragments, `uv run`, PR description format, anti-patterns). This file covers policyengine-uk-data specifics.

## Commands

```bash
make install            # install deps (uv)
make format             # format (required)
make download           # download raw FRS + SPI inputs from HF (needs HUGGING_FACE_TOKEN)
make data               # full dataset build into policyengine_uk_data/storage/ (impute, calibrate); uploads nothing
make test               # test suite
uv run pytest policyengine_uk_data/tests/path/to/test.py -v
```

Releases are published by the Push workflow (`.github/workflows/push.yaml`), which runs `make upload` only after `make data` and `make test` pass.

Python 3.13+. Default branch: `main`. Raw FRS / SPI microdata live on HuggingFace; set `HUGGING_FACE_TOKEN` before running anything that touches the dataset build.

## What lives here

This repo builds the `.h5` files that feed `policyengine-uk`:

- `datasets/frs.py` — raw FRS → PolicyEngine variable mapping
- `datasets/imputations/` — QRF / other imputations layered on top (income, wealth, consumption, etc.)
- `datasets/local_areas/` — constituency and local-authority calibration
- `targets/` — calibration target sources (OBR, DWP, HMRC, ONS, SLC, etc.)
- `utils/calibrate.py` — the reweighting optimiser
- `storage/` — raw inputs, intermediate artefacts, published outputs

## Data-protection rules — no exceptions

The enhanced FRS dataset is licensed under strict UK Data Service terms. Violating them risks losing access, which would end PolicyEngine UK.

- **Never expose data contents publicly.** The HuggingFace repo `policyengine/policyengine-uk-data-private` is public + **gated** (manual approval): file contents are restricted to the org and to UKDS-licensed users approved through the gate. Never disable or weaken the gate, never grant access without UKDS proof, and never upload UKDS-derived data to any ungated public location. Do not flip the repo back to private — that voids all approved external grants (see PolicyEngine/policyengine-uk#1816).
- **Never modify `upload_completed_datasets.py` or `utils/data_upload.py`** to change upload destinations without explicit confirmation from the data controller (currently Nikhil Woodruff).
- **Never print, log, or output individual-level records.** Aggregates (sums, means, counts, weighted totals) are fine; individual rows are not.
- **If you see a private/public repo split, assume it is intentional** — ask why before changing it.

## Updating datasets

If your change is a non-bugfix update to a cloud-hosted dataset (FRS, enhanced FRS), bump both the filename and URL in the class definition and in `storage/upload_completed_datasets.py`. That lets us store historical dataset versions separately and reproducibly.

## Repo-specific anti-patterns

- **Don't** hardcode dataset years in variable transforms; use `dataset.time_period` and the uprating pipeline.
- **Don't** commit large binary artefacts — use HuggingFace storage.
- **Don't** skip `make test` when touching the imputation or calibration pipeline. PR CI's Test job rebuilds the datasets at reduced fidelity (`TESTING=1`) and runs the tests, which takes about 25–50 minutes; the full build on `main` takes about 75–110 minutes.
