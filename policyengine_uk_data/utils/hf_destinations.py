"""HuggingFace destination constants for policyengine-uk-data.

The repo uploads data to two distinct HuggingFace model repos:

- ``PRIVATE_REPO = "policyengine/policyengine-uk-data-private"`` holds every
  artefact that is derived from UKDS-licensed microdata — raw FRS/LCFS/WAS/
  ETB/SPI zips, the enhanced FRS h5 files built on top of them, and any
  weights produced by calibrating against those datasets. Access is
  restricted to PolicyEngine collaborators who have accepted the UK Data
  Service End User Licence.
- ``PUBLIC_REPO = "policyengine/policyengine-uk-data"`` is a separate,
  publicly readable repo that is populated via a different process and is
  **NOT** a place to push FRS-derived microdata. If in doubt about whether
  an artefact may go here, check with the data controller (currently
  Nikhil Woodruff).

This module exposes the destinations as module-level constants so callers
can reference them by name instead of duplicating string literals across
the codebase. It intentionally does NOT change any existing upload
destinations — the PR that introduces this module only adds detection
scaffolding. Existing destinations are preserved per repo policy and
CLAUDE.md rule 1 ("NEVER upload data to any public location"), and any
resolution of the naming inconsistency should happen explicitly in a
separate PR signed off by the data controller.
"""

from __future__ import annotations

from typing import Final


PRIVATE_REPO: Final[str] = "policyengine/policyengine-uk-data-private"
"""HuggingFace repo for UKDS-licensed FRS-derived artefacts.

Every upload of FRS, LCFS, WAS, ETB, SPI or enhanced FRS data — plus any
weights or manifests derived from them — MUST land here.
"""

PUBLIC_REPO: Final[str] = "policyengine/policyengine-uk-data"
"""HuggingFace repo for the separately-maintained public mirror.

Publicly readable. Populated through a distinct process and not a valid
destination for FRS-derived microdata. Referenced here so we can
distinguish intentional public reads (e.g. loading a non-UKDS sample
dataset) from accidental public writes.
"""


ALLOWED_REPOS: Final[frozenset[str]] = frozenset({PRIVATE_REPO, PUBLIC_REPO})
"""The only HF repo names code in this package should reference.

Used by `tests/test_hf_destinations.py` to AST-scan every `upload(...)`,
`upload_file(...)`, `upload_files_to_hf(...)` and `upload_data_files(...)`
call site. A destination outside this set is a code error; a destination
in this set that is the wrong choice for the data at hand is a policy
decision that must be reviewed by the data controller.
"""
