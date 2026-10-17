"""Make this job's frozen download cache: the hub harness's cache plus the one URL current main adds.

The harness cache (~/reviews/uk-data-spi-reported-2026-10-01/http_cache, 12 responses) predates the batch, which
moved the HMRC salary-sacrifice source to a new URL. The combined-1008 build of 57ce68a fetched that URL live once
on 7 Oct (12,742 bytes, sha256 04e0dab6...), and the response is byte-identical to the tracked
policyengine_uk_data/storage/hmrc_pension_relief_tables_6_1_6_2_2024_25.csv. This adds it under cached_build.py's
key, so all four builds are served every target download from the cache and none fetches live.

Usage: python3 seed_http_cache.py <a uk-data checkout at 2b9c1c2c or later>
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

JOB = Path(__file__).resolve().parent
HARNESS_CACHE = Path.home() / "reviews/uk-data-spi-reported-2026-10-01/http_cache"
COMBINED_META = (
    Path.home()
    / "PolicyEngine/_worktrees/uk-data-combined-1008/_out/hmrc_salary_sacrifice_download.json"
)
HMRC_CSV = "policyengine_uk_data/storage/hmrc_pension_relief_tables_6_1_6_2_2024_25.csv"
HMRC_SHA256 = "04e0dab6354b01c96d04f9e379f237ea2083476bd9e920c882a236cdcf88101d"


def key(url):
    # As cached_build.py: requests.get(url, ...) with no params.
    return hashlib.sha256(
        json.dumps([url, None], sort_keys=True, default=str).encode()
    ).hexdigest()[:24]


cache = JOB / "frozen" / "http_cache"
if cache.exists():
    sys.exit(f"{cache} exists; preserve the frozen cache")
subprocess.run(["cp", "-c", "-R", str(HARNESS_CACHE), str(cache)], check=True)

body = (Path(sys.argv[1]) / HMRC_CSV).read_bytes()
assert hashlib.sha256(body).hexdigest() == HMRC_SHA256, "tracked HMRC CSV changed"
meta = json.loads(COMBINED_META.read_text())
assert meta["status"] == 200 and len(body) == int(meta["headers"]["Content-Length"])
k = key(meta["requested"])
(cache / f"{k}.body").write_bytes(body)
shutil.copyfile(COMBINED_META, cache / f"{k}.json")

entries = sorted(cache.glob("*.json"))
for m in entries:
    requested = json.loads(m.read_text())["requested"]
    assert m.stem == key(requested), f"{m.name} is not keyed by its URL"
    assert (cache / f"{m.stem}.body").exists(), f"{m.stem}.body missing"
print(
    f"{cache}: {len(entries)} responses, all keyed by URL; added {k} for {meta['requested']}"
)
