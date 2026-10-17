# #507 reproducibility build plan

Status: **PENDING HUB INSTALLATION; no builds or microsimulations started by this job.** This workspace cannot write `~/reviews/uk-hub`. The hub must install these artifacts in `~/reviews/uk-hub/jobs/adopt-ud507/` and add the proposed row from `QUEUE_ROW.md` to `~/reviews/uk-hub/rebuild-queue.md`. Run only after the hub replaces that row's Status with a grant containing **`GO mm/dd hh:mmZ`**, for example `**GO 10/12 12:00Z (UK hub).**` The driver requires exactly one `ud#507` queue row, checks its Status before waiting and again before launch, and rejects `NO GO`, `LAPSED`, `EXPIRED` or `REVOKED` even if another GO token is present.

Run four fresh builds, one at a time: `branch1`, `branch2`, `main1`, `main2`. The branch uses the final reviewed head of `origin/deterministic-calibration`; resolve it once before setup and preserve it in `expected_refs.json`. Both main controls use **`2b9c1c2c64ced6e1d95fed1ed398a22627233669`**. The saved pre-resume merge was `d9d13a182a7fdaa8f38a27f0c19985639e2c5888`; use the final head including the new tests and L0 fix.

All four use the same frozen HTTP responses and cached imputation models, Python 3.13, the committed `uv.lock` (policyengine-uk **2.122.2**, policyengine-core **3.32.13**), **512 epochs**, **one OA clone**, and **one CPU thread**. `TESTING` and `PYTHONPATH` are unset. No wrapper seeds the process-wide RNG: the branch owns calibration seed 0, while main retains its unseeded weight dropout.

`setup_build_worktree.sh` derives a temporary copy of the requested harness at `~/reviews/uk-data-spi-reported-2026-10-01/setup_build_worktree.sh`. Its required SHA256 is `3207a17bdfc1e441cba58ba34c063e2413ba7fc2242dcacff25c338c3fcfb1de`. The only functional change removes that harness's automatic `cherry-pick -x 348dd49`: the branch already contains #507, and applying it to main would invalidate the controls. It also adjusts the success message. Setup aborts if the source harness changes, the destination exists, the requested ref changes, the branch lacks the original #507 commit, or main's dropout is seeded.

Frozen models come from the harness's `prereqs/*.pkl`, except `income_spi_2022_23.pkl`, taken from the 7 October `uk-data-combined-1008` build to match #544's updated metadata. Frozen HTTP responses come from the harness's cache, plus its one missing post-batch HMRC salary-sacrifice CSV. `seed_http_cache.py` checks that CSV against SHA256 `04e0dab6354b01c96d04f9e379f237ea2083476bd9e920c882a236cdcf88101d` and uses the combined build's saved response metadata. No live target fetches are allowed. Any missing cache response, changed model, or changed cache manifest fails the build.

The hub installs the committed scripts and freezes the refs with these commands. Preserve the existing job's progress and scripts first; keep its other evidence files.

```zsh
set -euo pipefail
REPO=~/PolicyEngine/policyengine-uk-data
J=~/reviews/uk-hub/jobs/adopt-ud507
git -C "$REPO" fetch origin deterministic-calibration main
BRANCH_REF=$(git -C "$REPO" rev-parse origin/deterministic-calibration)
MAIN_REF=2b9c1c2c64ced6e1d95fed1ed398a22627233669
mkdir -p "$J/previous-attempt-20261012"
if [[ -f "$J/PROGRESS.md" ]]; then cp -c "$J/PROGRESS.md" "$J/previous-attempt-20261012/PROGRESS.md"; fi
for FILE in BUILD_PLAN.md QUEUE_ROW.md setup_build_worktree.sh freeze_inputs.sh run_build.sh cached_build.py seed_http_cache.py compare.py totals.py verify_historical_spread.py; do
  if [[ -f "$J/$FILE" ]]; then cp -c "$J/$FILE" "$J/previous-attempt-20261012/$FILE"; fi
  git -C "$REPO" show "$BRANCH_REF:jobs/adopt-ud507/$FILE" > "$J/$FILE"
done
printf '\n- 2026-10-12: Installed #507 build artifacts at branch %s; code and tests ready, impact pending the four queued builds. Existing progress preserved.\n' "$BRANCH_REF" >> "$J/PROGRESS.md"
python3 - "$J" "$BRANCH_REF" "$MAIN_REF" <<'PY'
import json
import sys
from pathlib import Path
job, branch, main = sys.argv[1:]
Path(job, "expected_refs.json").write_text(json.dumps({"branch": branch, "main": main}, indent=2) + "\n")
PY
```

After installing the queue row and receiving GO, use these exact setup and build commands. `freeze_inputs.sh` runs once and refuses to overwrite an existing frozen directory. The four worktrees must be new; cached models are copied from the single frozen snapshot into each. Setup runs the pinned harness's `uv sync --frozen --all-extras --python 3.13`.

```zsh
zsh "$J/freeze_inputs.sh" "$REPO"
zsh "$J/setup_build_worktree.sh" ud507-branch1 "$BRANCH_REF" branch
zsh "$J/setup_build_worktree.sh" ud507-branch2 "$BRANCH_REF" branch
zsh "$J/setup_build_worktree.sh" ud507-main1 "$MAIN_REF" main
zsh "$J/setup_build_worktree.sh" ud507-main2 "$MAIN_REF" main
zsh "$J/run_build.sh" ud507-branch1 branch1
zsh "$J/run_build.sh" ud507-branch2 branch2
zsh "$J/run_build.sh" ud507-main1 main1
zsh "$J/run_build.sh" ud507-main2 main2
```

Before each build, the driver requires statvfs free disk above 100 GB, compressor below 30 GB, load5 below 36 on three checks 60 seconds apart, no other data build, and `reservation.py allowed uk-hub` exit 0. The hub must retain its GO and report starts and finishes under its queue rules. A task-specific monitor checks free disk every 30 seconds while the build runs; below 30 GB, or if the disk check fails, it sends TERM only to that build's captured child PID and logs the event in `disk_floor.log`. Traps stop the monitor and its own build if the driver exits early. The driver gives up after six hours of waiting without starting a build. Expect approximately 1.2 GB of retained artifacts per build; gzip logs and delete storage build copies after each build. Raw survey prerequisites remain the same shared harness inputs.

Then run four microsimulations sequentially in the same locked environment, followed by comparison. `totals.py` checks the thread caps, compressor and engine versions before constructing a microsimulation. Re-read the hub's reservation and running-process state before this finisher stage.

```zsh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE PYTHONHASHSEED=0
unset TESTING PYTHONPATH
W=~/PolicyEngine/_worktrees/ud507-branch1
for LABEL in branch1 branch2 main1 main2; do
  "$W/.venv/bin/python" "$J/totals.py" "$J/builds/$LABEL/enhanced_frs_2024_25.h5" "$J/impact/$LABEL.json"
done
"$W/.venv/bin/python" "$J/compare.py"
```

`compare.py` writes `RESULT.md` and `RESULT.json`. PASS requires exact recorded refs and committed calibration source, matching locks/settings/frozen manifests, successful builds, no target-cache misses or OBR receipts fallback, and complete nonempty weight arrays. Branch constituency and local-authority weight arrays must have identical shapes, dtypes and bytes, and branch dataset tables and decompressed long-geography content must match. Main's weight arrays must differ. All four pre-calibration FRS tables and eFRS person/benefit-unit tables must agree; other input noise fails the comparison. Aggregate JSONs must match their dataset hashes and engine versions. The report gives both builds' income tax and household net income totals and their absolute spread for 2025 and 2026. Impact remains pending these queued builds; do not present the historical measurements as the controlled experiment's result.

The PR's rounded historical figures are verified by reading saved aggregate JSONs only:

```zsh
python3 "$J/verify_historical_spread.py"
```

Sources are `~/reviews/frs-cvpay-property-income-2026-09-30/impact/efrs_{main_matched,main_matched_2,main_a,main_b}.json`. All report policyengine-uk head `44240bd8d7d7e97f2980ccc8134e76e61541966f`. Across those four outputs, 2026 income tax ranges from £312.411684864bn (`main_b`) to £313.196281856bn (`main_matched`): **£0.784596992bn**. Household net income ranges from £1756.085878784bn (`main_a`) to £1759.036964864bn (`main_b`): **£2.951086080bn**. These support the quoted roughly £0.8bn and £3bn maxima; the maxima use different pairs. The verifier asserts the shared engine head and both spreads without running a build or simulation.
