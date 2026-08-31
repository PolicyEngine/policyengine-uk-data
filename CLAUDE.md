# Claude notes

The purpose of this repo is to build the .h5 files that feed as input into the policyengine-uk tax-benefit microsimulation model.

## DATA PROTECTION — READ THIS FIRST

**The enhanced FRS dataset contains individual-level microdata from the UK Family Resources Survey, licensed under strict UK Data Service terms. Violating these terms could result in losing access to the data entirely, which would end PolicyEngine UK.**

### Rules — no exceptions

1. **NEVER expose data contents publicly.** The HuggingFace repo `policyengine/policyengine-uk-data-private` is **public + gated** (manual approval) as of 31 July 2026: file names are world-visible, but file contents are restricted to the org and to external users approved through the gate after showing UKDS FRS access. NEVER disable or weaken the gate, NEVER grant gate access without UKDS proof, and NEVER upload UKDS-derived data to any ungated public location. Do NOT flip the repo back to private to "fix" CI auth errors — private repos silently void every approved external grant (see PolicyEngine/policyengine-uk#1816); grant the CI account through the gate instead. The separate public repo (`policyengine/policyengine-uk-data-public`) is maintained through a separate process — do NOT modify the upload pipeline to push data there.
2. **NEVER modify `upload_completed_datasets.py` or `data_upload.py` to change upload destinations** without explicit confirmation from the data controller (currently Nikhil Woodruff).
3. **NEVER print, log, or output individual-level records** from the dataset. Aggregates (sums, means, counts, weighted totals) are fine; individual rows are not.
4. **If you see a private/public repo split, assume it is intentional** — ask why before changing it.

## General principles

Claude, please follow these always. These principles are aimed at preventing you from producing AI slop.

1. British English, sentence case
2. No excessive duplication, keep code files as concise as possible to produce the same meaningful value. No excessive printing
3. Don't create multiple files for successive versions. Keep checking: have I added lots of intermediate files which are deprecated? Delete them if so, but ideally don't create them in the first place
