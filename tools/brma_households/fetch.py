#!/usr/bin/env python3
"""Fetch pinned originals without a login, browser session or accepting terms."""

import argparse
import gzip
import hashlib
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

import yaml


def sources():
    entries = yaml.safe_load(Path(__file__).with_name("sources.yaml").read_text())
    required = {
        "id",
        "publisher",
        "title",
        "url",
        "landing_page",
        "licence",
        "notes",
        "filename",
    }
    ids, filenames = set(), set()
    for entry in entries:
        if not required <= entry.keys() or not (
            {"sha256", "content_sha256"} & entry.keys()
        ):
            raise ValueError(f"Incomplete source: {entry.get('id')}")
        name = entry["filename"]
        if entry["id"] in ids or name in filenames or Path(name).name != name:
            raise ValueError(f"Duplicate source or unsafe cache filename: {name}")
        ids.add(entry["id"])
        filenames.add(name)
    return entries


def decoded_content(path):
    data = path.read_bytes()
    return gzip.decompress(data) if data.startswith(b"\x1f\x8b") else data


def verify(path, entry):
    if not path.is_file():
        instructions = f"\n{entry['notes']}" if entry.get("manual") else ""
        raise FileNotFoundError(f"Missing source {entry['id']}: {path}{instructions}")
    if "content_sha256" in entry:
        actual = hashlib.sha256(decoded_content(path)).hexdigest()
        expected = entry["content_sha256"]
    else:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        expected = entry["sha256"]
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch for {entry['id']}: expected {expected}, got {actual}"
        )


def verify_cache(cache, entries):
    """Check every original before the builder interprets any source."""
    for entry in entries:
        verify(cache / entry["filename"], entry)


def fetch(cache, entries):
    cache.mkdir(parents=True, exist_ok=True)
    missing = []
    for entry in entries:
        path = cache / entry["filename"]
        if not path.exists() and entry.get("manual"):
            print(f"Manual download required: {path}\n{entry['notes']}", flush=True)
            missing.append(entry["id"])
            continue
        if not path.exists():
            print(f"Downloading {entry['id']}", flush=True)
            request = Request(
                entry["url"],
                headers={
                    "User-Agent": "BRMA-household-pipeline/1.0",
                    "Accept-Encoding": "gzip" if path.suffix == ".gz" else "identity",
                },
            )
            temporary = path.with_name(path.name + ".part")
            try:
                # Keep entity bytes; custom-API pins verify decoded content instead.
                with (
                    urlopen(request, timeout=180) as response,
                    temporary.open("wb") as stream,
                ):
                    shutil.copyfileobj(response, stream)
                verify(temporary, entry)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        verify(path, entry)
        print(f"Verified {entry['id']}", flush=True)
    if missing:
        raise FileNotFoundError(
            "Supply the manual downloads above: " + ", ".join(missing)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="SOURCE_ID",
        help="Fetch/verify selected sources (default: all)",
    )
    args = parser.parse_args()
    entries = sources()
    if args.only:
        unknown = set(args.only) - {entry["id"] for entry in entries}
        if unknown:
            parser.error("Unknown source IDs: " + ", ".join(sorted(unknown)))
        entries = [entry for entry in entries if entry["id"] in args.only]
    fetch(args.cache, entries)


if __name__ == "__main__":
    main()
