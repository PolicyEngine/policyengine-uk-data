"""Run the hub's dataset command with a frozen, read-only HTTP cache.

Unlike the original harness, a missing response terminates the build instead
of fetching a live target. No random generator is seeded by this wrapper.
Usage, from a build worktree: python cached_build.py <cache> <http.log>
"""

import hashlib
import json
import runpy
import sys
from pathlib import Path

import requests
from requests.structures import CaseInsensitiveDict


def main():
    cache, http_log = Path(sys.argv[1]), Path(sys.argv[2])

    def cached_get(url, **kwargs):
        key = hashlib.sha256(
            json.dumps(
                [url, kwargs.get("params")], sort_keys=True, default=str
            ).encode()
        ).hexdigest()[:24]
        meta_path, body_path = cache / f"{key}.json", cache / f"{key}.body"
        if not meta_path.exists() or not body_path.exists():
            with http_log.open("a") as log:
                log.write(f"MISS frozen cache {url}\n")
            # Target loaders catch ordinary exceptions and silently fall back.
            # SystemExit stops the build before that can change its target set.
            raise SystemExit(f"Frozen target cache has no response for {url}")
        meta, body = json.loads(meta_path.read_text()), body_path.read_bytes()
        response = requests.Response()
        response.status_code, response._content, response.url = (
            meta["status"],
            body,
            meta["url"],
        )
        response.headers = CaseInsensitiveDict(meta["headers"])
        response.encoding, response.reason = meta["encoding"], meta["reason"]
        with http_log.open("a") as log:
            log.write(
                f"cache {meta['status']} {len(body)}b "
                f"{hashlib.sha256(body).hexdigest()} {url}\n"
            )
        return response

    requests.get = cached_get
    runpy.run_path(
        "policyengine_uk_data/datasets/create_datasets.py", run_name="__main__"
    )


if __name__ == "__main__":
    main()
