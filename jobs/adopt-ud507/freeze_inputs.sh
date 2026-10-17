#!/bin/zsh
# Hub only, before its four builds: freeze_inputs.sh <branch checkout>
set -eu
J=${0:A:h}
E=~/reviews/uk-data-spi-reported-2026-10-01
C=~/PolicyEngine/_worktrees/uk-data-combined-1008/policyengine_uk_data/storage
[ ! -e "$J/frozen" ] || { echo 'Frozen inputs already exist; preserve them.' >&2; exit 1; }
mkdir -p "$J/frozen/models"
for MODEL in "$E"/prereqs/*.pkl; do cp -c "$MODEL" "$J/frozen/models/${MODEL:t}"; done
# #544 changed the income-model metadata. Use the model retrained on 7 Oct
# with exactly current main's inputs, instead of silently retraining four times.
cp -c -f "$C/income_spi_2022_23.pkl" "$J/frozen/models/income_spi_2022_23.pkl"
(cd "$J/frozen/models" && shasum -a 256 *.pkl) > "$J/frozen/models.sha256"
python3 "$J/seed_http_cache.py" "$1"
(cd "$J/frozen/http_cache" && shasum -a 256 *) > "$J/frozen/targets.sha256"
echo 'Frozen cached models and targets ready. No dataset build ran.'
