# PolicyEngine UK Data

PolicyEngine's project to build accurate UK household survey data.

## Local dataset builds

For a full local dataset build:

1. Ensure the private prerequisite folders exist under `policyengine_uk_data/storage/`.
2. Use Python 3.13. Python 3.14 currently fails while loading PyTables/Blosc2 in this repo.
3. Prefer the sibling `policyengine-uk` checkout when building locally, because the published wheel in your active environment may not expose all variables required by the data pipeline.

If `../policyengine-uk` exists, you can run:

```sh
make data-local
```

## TRACE provenance output

Each UK data release now publishes both:

- `release_manifest.json`
- `trace.tro.jsonld`

The release manifest remains the operational source of truth for:

- published artifact paths and checksums
- build IDs and timestamps
- build-time `policyengine-uk` provenance

`trace.tro.jsonld` is a generated TRACE declaration built from that manifest. It gives a
standards-based provenance export over the same release artifacts, including a
composition fingerprint across the release manifest and the artifacts it describes.

Important boundary:

- the TRACE file does not replace the release manifest
- the TRACE file does not decide model/data compatibility

For the broader certified-bundle architecture, see
[`policyengine.py` release bundles](https://github.com/PolicyEngine/policyengine.py/blob/main/docs/release-bundles.md)
and the official [TRACE specification](https://transparency-certified.github.io/trace-specification/docs/specifications/).

Because UK data is private, the TRACE declaration is especially useful: it can record
hashes and artifact locations without exposing the underlying microdata bytes.
 
