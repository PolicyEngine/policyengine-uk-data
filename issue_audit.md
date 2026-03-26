# Open issue audit (2026-03-18)

## Closed (addressed and closed on 2026-03-18)

| Issue | Title | Resolution |
|-------|-------|------------|
| ~~#108~~ | Versioning action runs too broadly | Fixed. Commit `441f552` (PR #287) switched from watching `changelog_entry.yaml` to `changelog.d/**` with a `!pyproject.toml` exclusion, plus a commit-message guard. |
| ~~#107~~ | Use more secure GitHub token in GH action | Fixed. Commit `20e36c9` (PR #293) replaced the expired PAT with a GitHub App token (`actions/create-github-app-token@v1`). |
| ~~#106~~ | Move away from PyPI token usage | Fixed. The PyPI publishing step was removed entirely (commit `dc7c7f4`). No `secrets.PYPI` reference remains in any workflow. GCP uses OIDC via `google-github-actions/auth@v2`. |
| ~~#285~~ | Switch from black to ruff format | Fixed. Commit `cc4a884` (PR #287) switched pyproject.toml, Makefile, and both CI workflows to `ruff format`. Black is fully removed. |

## Quick wins (easy to close, minimal work needed)

| Issue | Title | Effort | What's needed |
|-------|-------|--------|---------------|
| #278 | Fix ESA calibration target entity mapping | Trivial | **In progress.** Changelog fragment added and pushed. Draft PR exists, needs marking as ready and merging. |
| #69 | Clarify documentation of total vs full-time worker local targets | Docs only | **PR #297 open.** Clarified "all workers (part-time and full-time)" across notebooks, code comments, and READMEs. |
| #67 | Better document NOMIS employment income data | Docs only | **Blocked.** Issue claimed data is resident analysis, but NOMIS screenshots confirm workplace analysis. Comment posted asking for clarification. |
| #71 | Include original local areas age files | Docs only | **PR #298 open.** Constituency script now reads `raw_age.csv` (matching LA pattern). READMEs updated with correct source links (verified against methodology notebooks). Raw data was never committed; fix is forward-looking. |
| #73 | Assign `is_parent` from FRS microdata | Small code change | **Blocked.** Issue is ambiguous — FRS adult table ≠ parent. Comment posted asking for clarification. |
| #118 | Improve methodology documentation page | Docs only | **PR #300 open.** Fixed typos, updated stale time period reference, added note explaining why calibration plots appear identical. |

## Partially addressed (not closeable yet)

| Issue | Title | Status |
|-------|-------|--------|
| #295 | Land value calibration targets not converging | Targets added (PR #292), but all three tests are still `@pytest.mark.xfail` — actual convergence requires a recalibration run. |
| #217 | Calibration inflates UK population to 74M | Acknowledged only. Population test tolerance was relaxed from 2% to 7% as a temporary workaround. The root cause (population is 1 of 556 equally-weighted targets) is unresolved. |

## Not addressed (remaining open issues)

| Issue | Title |
|-------|-------|
| #296 | Add adversarial weight regularisation pipeline |
| #291 | Add Output Area crosswalk and geographic assignment (Phase 1) |
| #281 | Impute loan-holder-but-not-repaying status to FRS base dataset |
| #279 | Add Modal GPU calibration |
| #263 | Add policyengine-claude plugin auto-install — **PR #263 rebased and conflict-resolved, ready for review.** |
| #241 | Adjust VAT methodology to include flat % assumptions on VAT band shares |
| #238 | Impute student loan balance from WAS to FRS |
| #237 | Add student loan repayment calibration targets |
| #230 | Property income may be underestimated |
| #218 | Income projections inflated ~2.5x vs SPI targets |
| #206 | Wales constituency results seem a little low |
| #199 | Reinstate DfE education spending test |
| #186 | Add firm data |
| #174 | Separate out Council Tax imputation into its own function |
| #158 | Calibrate 2024-29 inclusive years |
| #146 | Add UK entity table datasets |
| #145 | Impute non-dom status |
| #132 | Add synthetic FRS |
| #113 | Documentation fails to deploy — linked to PR #297, deploy job has been passing consistently |
| #97 | Impute index of multiple deprivation |
| #88 | Add National Insurance targeting |
| #86 | Add public services imputation model |
| #83 | Ensure imputed capital gains CDF is valid (monotonic) |
| #68 | Investigate averaging for missing percentiles in local area earnings |
| #64 | Improve demographic imputation for Scotland and Northern Ireland |
| #35 | Move random variable logic from PolicyEngine UK |
| #28 | Extend capital gains distribution for lower income centiles |
| #27 | Use more detailed capital gains distributions for top income centiles |
