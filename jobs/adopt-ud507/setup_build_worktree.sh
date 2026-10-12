#!/bin/zsh
# Hub only, after GO: setup_build_worktree.sh <name> <exact-ref> <branch|main>
# Reuse the pinned hub harness, omitting its seed cherry-pick: the PR already
# has that commit, while controls must retain main's unseeded dropout.
set -eu
J=${0:A:h}
E=~/reviews/uk-data-spi-reported-2026-10-01
W=~/PolicyEngine/_worktrees/$1
REF=$2
KIND=$3
HARNESS=$E/setup_build_worktree.sh
EXPECT=3207a17bdfc1e441cba58ba34c063e2413ba7fc2242dcacff25c338c3fcfb1de
[ "$(shasum -a 256 "$HARNESS" | awk '{print $1}')" = "$EXPECT" ] || {
  echo 'The source harness changed; review its delta before building.' >&2
  exit 1
}
[ -e "$J/frozen/models.sha256" ] || { echo 'Freeze inputs first.' >&2; exit 1; }
[ ! -e "$W" ] || { echo "Use a fresh worktree: $W already exists." >&2; exit 1; }
DERIVED=$(mktemp)
trap 'rm -f "$DERIVED"' EXIT
python3 - "$HARNESS" "$DERIVED" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
seed_line = 'git -c user.name=build -c user.email=build@localhost cherry-pick -x 348dd49 >/dev/null\n'
assert source.count(seed_line) == 1
source = source.replace(seed_line, '# #507 controls build exact refs; skip seed cherry-pick.\n')
source = source.replace(' (base $(git rev-parse --short HEAD~1))', ' (exact requested ref)')
Path(sys.argv[2]).write_text(source)
PY
zsh "$DERIVED" "$1" "$REF"
cd "$W"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$REF^{commit}")" ]
case "$KIND" in
  branch)
    git merge-base --is-ancestor 348dd493c7cba66b329a384410e4bb21de97cd4d HEAD
    rg -q 'dropout_generator = torch.Generator\(\).*manual_seed\(seed\)' policyengine_uk_data/utils/calibrate.py
    ;;
  main)
    [ "$(git rev-parse HEAD)" = "$(git rev-parse 2b9c1c2c64ce)" ]
    rg -q 'mask = torch.rand_like\(weights\) < p' policyengine_uk_data/utils/calibrate.py
    ;;
  *) echo 'kind must be branch or main' >&2; exit 1 ;;
esac
S=policyengine_uk_data/storage
for MODEL in "$J"/frozen/models/*.pkl; do cp -c -f "$MODEL" "$S/${MODEL:t}"; done
(cd "$S" && shasum -a 256 *.pkl) > "$W/models_setup.sha256"
cmp "$J/frozen/models.sha256" "$W/models_setup.sha256"
.venv/bin/python -c 'import importlib.metadata as m; assert m.version("policyengine-uk") == "2.122.2"; assert m.version("policyengine-core") == "3.32.13"'
git diff --exit-code
echo "$1 ready at $(git rev-parse --short HEAD) ($KIND; pristine exact ref)"
