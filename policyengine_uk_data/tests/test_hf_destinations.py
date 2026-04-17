"""AST-based guard that every HF upload call routes through the shared
`PRIVATE_REPO` / `PUBLIC_REPO` constants in
:mod:`policyengine_uk_data.utils.hf_destinations`.

Motivation (bug-hunt finding S1):

- ``storage/upload_private_prerequisites.py`` uploads UKDS-licensed FRS/LCFS/
  WAS/ETB/SPI zips with a literal ``repo="policyengine/policyengine-uk-data"``
  argument — i.e. the PUBLIC repo.
- ``utils/data_upload.py::upload_data_files`` defaults ``hf_repo_name`` to the
  PUBLIC repo, while the sibling ``upload_files_to_hf`` defaults to the
  PRIVATE repo.
- Mixed literals across the codebase mean one typo in a future script could
  silently leak microdata.

Approach:

- Parse every ``.py`` file in this package with :mod:`ast`.
- For every call to one of the upload entry points (``upload``,
  ``upload_file``, ``upload_files_to_hf``, ``upload_data_files``), look for
  the ``repo=`` / ``hf_repo_name=`` keyword argument.
- If the argument is a string literal that isn't accessed via
  ``hf_destinations.PRIVATE_REPO`` / ``PUBLIC_REPO`` (or the module-level
  ``ALLOWED_REPOS`` set), record it as a violation.

The test is marked ``xfail`` until every call site is migrated. It will
begin failing (i.e. "pass unexpectedly") as a signal that the clean-up is
complete, at which point the ``xfail`` decorator should be removed.

**Do NOT silence this test by changing the destinations in place.** The
existing destinations are preserved by repo policy (see CLAUDE.md rule 1
and the policyengine-uk-data private/public split). Resolving the naming
inconsistency requires a data-controller decision — either rename the HF
repos or migrate each script individually with sign-off — not a blanket
string swap in this PR.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


UPLOAD_CALL_NAMES: set[str] = {
    "upload",
    "upload_file",
    "upload_files_to_hf",
    "upload_data_files",
}

KEYWORD_ARGS: set[str] = {"repo", "hf_repo_name", "repo_id"}

# Files that this test intentionally does NOT scan for violations:
# - The constants module itself (defines the literals).
# - The test file that validates those literals.
# - The xfail guard below.
IGNORED_FILENAMES: set[str] = {
    "hf_destinations.py",
    "test_hf_destinations.py",
}


def _iter_py_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if path.name in IGNORED_FILENAMES:
            continue
        if "__pycache__" in path.parts:
            continue
        files.append(path)
    return files


def _call_name(call: ast.Call) -> str | None:
    """Return the simple name of a call target, if recognisable.

    Handles both ``upload(...)`` and ``hf.upload(...)`` styles.
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _kwarg_value(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_allowed_reference(node: ast.AST) -> bool:
    """Check whether a kwarg value routes through the shared constants."""
    if isinstance(node, ast.Attribute):
        # hf_destinations.PRIVATE_REPO / .PUBLIC_REPO
        if node.attr in {"PRIVATE_REPO", "PUBLIC_REPO"}:
            return True
    if isinstance(node, ast.Name):
        # Imported as `from ...hf_destinations import PRIVATE_REPO`.
        if node.id in {"PRIVATE_REPO", "PUBLIC_REPO"}:
            return True
    return False


def _collect_violations() -> list[str]:
    violations: list[str] = []
    for path in _iter_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            # Don't let a single syntax-invalid file abort the scan.
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name not in UPLOAD_CALL_NAMES:
                continue
            for kwarg in KEYWORD_ARGS:
                value = _kwarg_value(node, kwarg)
                if value is None:
                    continue
                if isinstance(value, ast.Constant) and isinstance(
                    value.value, str
                ):
                    # A raw string literal — flag it.
                    violations.append(
                        f"{path}:{node.lineno} "
                        f"{name}(..., {kwarg}={value.value!r}, ...)"
                    )
                elif not _is_allowed_reference(value):
                    # Any other expression (variable, call, f-string, etc.)
                    # that isn't demonstrably the shared constant.
                    violations.append(
                        f"{path}:{node.lineno} {name}(..., {kwarg}=<expr>)"
                    )
    return violations


@pytest.mark.xfail(
    reason=(
        "Known naming inconsistency; existing destinations intentionally "
        "preserved per repo policy. Resolve by renaming the HF repo or "
        "migrating scripts — NOT by changing code in place without owner "
        "approval."
    ),
    strict=False,
)
def test_every_hf_upload_routes_through_guard_constants() -> None:
    violations = _collect_violations()
    if violations:
        formatted = "\n  ".join(violations)
        pytest.fail(
            "The following upload-site call arguments bypass the shared "
            "PRIVATE_REPO / PUBLIC_REPO constants in "
            "policyengine_uk_data.utils.hf_destinations:\n  "
            + formatted
        )


def test_hf_destinations_constants_are_distinct_and_well_formed() -> None:
    """Sanity: the two constants are different and look like HF repo ids."""
    from policyengine_uk_data.utils.hf_destinations import (
        ALLOWED_REPOS,
        PRIVATE_REPO,
        PUBLIC_REPO,
    )

    assert PRIVATE_REPO != PUBLIC_REPO
    assert PRIVATE_REPO == "policyengine/policyengine-uk-data-private"
    assert PUBLIC_REPO == "policyengine/policyengine-uk-data"
    assert ALLOWED_REPOS == {PRIVATE_REPO, PUBLIC_REPO}
    for repo in ALLOWED_REPOS:
        assert repo.startswith("policyengine/"), repo
