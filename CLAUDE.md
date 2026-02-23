# Claude notes

The purpose of this repo is to build the .h5 files that feed as input into the policyengine-uk tax-benefit microsimulation model.

## DATA PROTECTION — READ THIS FIRST

**The enhanced FRS dataset contains individual-level microdata from the UK Family Resources Survey, licensed under strict UK Data Service terms. Violating these terms could result in losing access to the data entirely, which would end PolicyEngine UK.**

### Rules — no exceptions

1. **NEVER upload data to any public location.** The HuggingFace repo `policyengine/policyengine-uk-data-private` is private and authenticated. The separate public repo (`policyengine/policyengine-uk-data`) is maintained through a separate process — do NOT modify the upload pipeline to push data there.
2. **NEVER modify `upload_completed_datasets.py` or `data_upload.py` to change upload destinations** without explicit confirmation from the data controller (currently Nikhil Woodruff).
3. **NEVER print, log, or output individual-level records** from the dataset. Aggregates (sums, means, counts, weighted totals) are fine; individual rows are not.
4. **If you see a private/public repo split, assume it is intentional** — ask why before changing it.

## General principles

Claude, please follow these always. These principles are aimed at preventing you from producing AI slop.

1. British English, sentence case
2. No excessive duplication, keep code files as concise as possible to produce the same meaningful value. No excessive printing
3. Don't create multiple files for successive versions. Keep checking: have I added lots of intermediate files which are deprecated? Delete them if so, but ideally don't create them in the first place
