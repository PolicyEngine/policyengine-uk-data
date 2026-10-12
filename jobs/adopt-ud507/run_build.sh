#!/bin/zsh
# Usage: run_build.sh <worktree name> <label>
# One production-settings build (512 epochs, PE_UK_DATA_OA_CLONES=1, TESTING unset) of the worktree's commit, with every
# download served from this job's frozen cache. Waits for the hub's gates (statvfs free > 100 GB, compressor < 30 GB,
# load5 < 36 on three checks 60 s apart, no other uk-data build, reservation.py allowed uk-hub exit 0); gives up after
# GATE_WAIT_HOURS (default 6) with exit 3. From the harness's run_gated_build.sh (sha256 4a3b4b0a...): same build command
# and cleanup, with the hub's gates, the thread caps, and outputs (data, weights, download log, model hashes) in builds/<label>/.
set -euo pipefail
J=${0:A:h}
W=~/PolicyEngine/_worktrees/$1
L=$2
B=$J/builds/$L
QUEUE=~/reviews/uk-hub/rebuild-queue.md
build_pid=''
monitor_pid=''
cleanup_children() {
  [ -z "$monitor_pid" ] || kill -TERM "$monitor_pid" 2>/dev/null || true
  [ -z "$build_pid" ] || kill -TERM "$build_pid" 2>/dev/null || true
}
trap cleanup_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
require_go() {
  python3 - "$QUEUE" <<'PY'
import re
import sys
from pathlib import Path

rows = [line for line in Path(sys.argv[1]).read_text().splitlines()
        if line.startswith("|") and re.search(r"\bud#507\b", line)]
if len(rows) != 1:
    raise SystemExit("Expected exactly one ud#507 queue row")
status = rows[0].split("|")[-2]
if re.search(r"\b(?:NO GO|LAPSED|EXPIRED|REVOKED)\b", status, re.I):
    raise SystemExit("ud#507 has no current hub GO")
if not re.search(r"\bGO \d{2}/\d{2} \d{2}:\d{2}Z\b", status):
    raise SystemExit("ud#507 requires GO mm/dd hh:mmZ in its queue Status")
print(rows[0])
PY
}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE PYTHONHASHSEED=0
unset TESTING PYTHONPATH
[ -e $B/finished ] && [ "$(gzip -cd $B/build.log.gz | tail -n 1)" = "exit 0" ] && { echo "$L already built"; exit 0; }
mkdir -p $B
require_go > $B/queue_go.txt
rm -f $B/finished $B/http.log
gate() {
  local free comp load5 others res
  require_go >/dev/null || return 1
  free=$(python3 -c 'import os;s=os.statvfs("/System/Volumes/Data");print(int(s.f_bavail*s.f_frsize/1e9))')
  comp=$(vm_stat | awk 'NR==1 {match($0, /[0-9]+ bytes/); page=substr($0,RSTART,RLENGTH)+0} /occupied by compressor/{gsub(/\./,"",$5); print int($5*page/1e9)}')
  load5=$(sysctl -n vm.loadavg | awk '{print int($3)}')
  others=$(ps -axo args= | grep -cE '[c]reate_datasets|[c]ached_build\.py|[b]uild_harness\.py' || true)
  if python3 ~/reviews/fleet-ops/tools/reservation.py allowed uk-hub >/dev/null 2>&1; then res=0; else res=$?; fi
  echo "$(date -u +%FT%TZ) free=${free}GB compressor=${comp}GB load5=$load5 other_builds=$others reservation=$res" >> $B/gates.log
  (( free > 100 && comp < 30 && load5 < 36 && others == 0 && res == 0 ))
}
deadline=$(( $(date +%s) + ${GATE_WAIT_HOURS:-6} * 3600 ))
until gate && sleep 60 && gate && sleep 60 && gate; do
  (( $(date +%s) > deadline )) && { echo "$(date -u +%FT%TZ) gates never held; no build ran" >> $B/gates.log; exit 3; }
  sleep 300
done
cd $W
S=policyengine_uk_data/storage
rm -f $S/enhanced_frs_2024_25.h5 $S/frs_2024_25.h5 $S/*_tiny.h5(N) $S/parliamentary_constituency_weights.h5 $S/local_authority_weights.h5 $S/local_geography_weights.csv.gz
git rev-parse HEAD > $B/commit
git status --porcelain --untracked-files=no > $B/status
[ ! -s $B/status ] || { echo 'Tracked build sources changed.' >&2; exit 1; }
grep -nE "dropout_generator = torch.Generator\(\)|mask = torch.rand_like\(weights\) < p" policyengine_uk_data/utils/calibrate.py > $B/calibrate_source
shasum -a 256 policyengine_uk_data/utils/calibrate.py | awk '{print $1}' > $B/calibrate.sha256
git show HEAD:policyengine_uk_data/utils/calibrate.py | shasum -a 256 | awk '{print $1}' > $B/calibrate_committed.sha256
cmp $B/calibrate.sha256 $B/calibrate_committed.sha256
(cd $S && shasum -a 256 *.pkl) > $B/models_before.sha256
cmp $J/frozen/models.sha256 $B/models_before.sha256 || exit 1
(cd $J/frozen/http_cache && shasum -a 256 *) > $B/targets_before.sha256
cmp $J/frozen/targets.sha256 $B/targets_before.sha256 || exit 1
shasum -a 256 uv.lock | awk '{print $1}' > $B/lock.sha256
{ env | grep -E '^(OMP|OPENBLAS|MKL|VECLIB|NUMEXPR)_|^PYTHONHASHSEED=' | sort; .venv/bin/python -c 'import importlib.metadata as m, torch; print("torch_threads", torch.get_num_threads()); print("torch", m.version("torch")); print("policyengine-uk", m.version("policyengine-uk")); print("policyengine-core", m.version("policyengine-core")); print("epochs", 512); print("oa_clones", 1); assert torch.get_num_threads() == 1; assert m.version("policyengine-uk") == "2.122.2"; assert m.version("policyengine-core") == "3.32.13"'; } > $B/env
date -u +%FT%TZ > $B/started
require_go > $B/queue_go.txt
rc=0
PE_UK_DATA_OA_CLONES=1 .venv/bin/python $J/cached_build.py $J/frozen/http_cache $B/http.log > $B/build.log 2>&1 &
build_pid=$!
(
  while kill -0 "$build_pid" 2>/dev/null; do
    free_bytes=$(python3 -c 'import os;s=os.statvfs("/System/Volumes/Data");print(s.f_bavail*s.f_frsize)') || free_bytes=0
    if (( free_bytes < 30000000000 )); then
      echo "$(date -u +%FT%TZ) disk floor reached (${free_bytes} bytes); terminating this build PID $build_pid" >> $B/disk_floor.log
      kill -TERM "$build_pid" 2>/dev/null || true
      exit 0
    fi
    sleep 30
  done
) &
monitor_pid=$!
wait "$build_pid" || rc=$?
build_pid=''
kill -TERM "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=''
(cd $S && shasum -a 256 *.pkl) > $B/models_after.sha256
(cd $J/frozen/http_cache && shasum -a 256 *) > $B/targets_after.sha256
cmp $J/frozen/models.sha256 $B/models_after.sha256 || rc=1
cmp $J/frozen/targets.sha256 $B/targets_after.sha256 || rc=1
if [ $rc -eq 0 ]; then
  for f in enhanced_frs_2024_25.h5 frs_2024_25.h5 parliamentary_constituency_weights.h5 local_authority_weights.h5 local_geography_weights.csv.gz; do cp -c $S/$f $B/$f; done
  shasum -a 256 $B/enhanced_frs_2024_25.h5 | awk '{print $1}' > $B/dataset.sha256
fi
for f in constituency_calibration_log.csv la_calibration_log.csv; do [ -e $f ] && gzip -c $f > $B/$f.gz; done
rm -f constituency_calibration_log.csv la_calibration_log.csv
# Retain only the APFS-cloned aggregate artifacts in this job's builds folder.
rm -f $S/enhanced_frs_2024_25.h5 $S/frs_2024_25.h5 $S/*_tiny.h5(N) $S/parliamentary_constituency_weights.h5 $S/local_authority_weights.h5 $S/local_geography_weights.csv.gz
echo "exit $rc" >> $B/build.log
gzip -f $B/build.log
date -u +%FT%TZ > $B/finished
exit $rc
