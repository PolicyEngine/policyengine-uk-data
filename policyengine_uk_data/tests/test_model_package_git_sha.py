"""The release manifest names a policyengine-uk commit only when it can trust it.

policyengine-uk releases before ``FIRST_MODEL_VERSION_WITH_OWN_GIT_SHA`` report
the HEAD of whichever git repository encloses the installed package
(PolicyEngine/policyengine-uk#2191). The release build installs policyengine-uk
into ``.venv`` inside this checkout, so those releases report
policyengine-uk-data's own commit (PolicyEngine/policyengine-uk-data#548).
"""

import logging
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from packaging.version import InvalidVersion, Version

from policyengine_uk_data.utils.data_upload import (
    FIRST_MODEL_VERSION_WITH_OWN_GIT_SHA,
    GIT_COMMIT_ID,
    _get_model_package_build_metadata,
    _trusted_model_package_git_sha,
)

FIXED_VERSION = str(FIRST_MODEL_VERSION_WITH_OWN_GIT_SHA)
_MAJOR, _MINOR, _MICRO = FIRST_MODEL_VERSION_WITH_OWN_GIT_SHA.release
MODEL_COMMIT = "0123456789abcdef0123456789abcdef01234567"
DATA_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
# Published manifests whose policyengine-uk git_sha equals
# build.metadata.data_package_git_sha, the uk-data release commit (read 2026-10-08).
PUBLISHED_RELEASE_COMMITS_AS_MODEL_GIT_SHA = [
    pytest.param(
        "2.122.2", "4cbedbecca352f52dffe47e849553c2759a00711", id="uk-data-1.58.0"
    ),
    pytest.param(
        "2.89.2", "12a1e028afeef08d8b2d74ee03fd9de3a78b2dd3", id="uk-data-1.56.16"
    ),
]
# 2.123.7 is the last release whose build_metadata walks up to any enclosing
# repository.
VERSIONS_BEFORE_FIX = [
    "2.89.2",
    "2.122.2",
    "2.123.7",
    f"{_MAJOR}.{_MINOR - 1}.99",
    f"{FIXED_VERSION}rc1",
    f"{FIXED_VERSION}.dev0",
    *([f"{_MAJOR}.{_MINOR}.{_MICRO - 1}"] if _MICRO else []),
]
VERSIONS_WITH_FIX = [
    FIXED_VERSION,
    f"{FIXED_VERSION}.post1",
    f"{FIXED_VERSION}+local",
    f"{_MAJOR}.{_MINOR}.{_MICRO + 1}",
    f"{_MAJOR}.{_MINOR + 1}.0",
    f"{_MAJOR + 1}.0.0",
]


def _trusted(git_sha, model_version, data_sha):
    return _trusted_model_package_git_sha(
        git_sha,
        model_package_version=model_version,
        data_package_git_sha=data_sha,
    )


@pytest.mark.parametrize(
    ("model_version", "release_commit"), PUBLISHED_RELEASE_COMMITS_AS_MODEL_GIT_SHA
)
def test_published_release_commits_are_not_recorded(model_version, release_commit):
    assert _trusted(release_commit, model_version, release_commit) is None
    # Each rule rejects them on its own.
    assert _trusted(release_commit, model_version, None) is None
    assert _trusted(release_commit, FIXED_VERSION, release_commit) is None


@pytest.mark.parametrize("model_version", VERSIONS_BEFORE_FIX)
def test_releases_before_the_fix_record_no_git_sha(model_version):
    assert _trusted(MODEL_COMMIT, model_version, DATA_COMMIT) is None


@pytest.mark.parametrize("model_version", VERSIONS_WITH_FIX)
def test_releases_with_the_fix_record_their_commit(model_version):
    assert _trusted(MODEL_COMMIT, model_version, DATA_COMMIT) == MODEL_COMMIT
    assert _trusted(MODEL_COMMIT, model_version, None) == MODEL_COMMIT


@pytest.mark.parametrize("commit", [MODEL_COMMIT.upper(), "ab" * 32])
def test_full_commit_ids_in_either_case_and_length_are_recorded(commit):
    assert _trusted(commit, FIXED_VERSION, DATA_COMMIT) == commit


@pytest.mark.parametrize(
    ("git_sha", "model_version"),
    [
        (DATA_COMMIT, FIXED_VERSION),
        (DATA_COMMIT.upper(), FIXED_VERSION),
        ("deadbeef", FIXED_VERSION),
        (MODEL_COMMIT + "\n", FIXED_VERSION),
        (MODEL_COMMIT[:-1] + "g", FIXED_VERSION),
        ("a" * 41, FIXED_VERSION),
        (int(MODEL_COMMIT, 16), FIXED_VERSION),
        (MODEL_COMMIT, None),
        (MODEL_COMMIT, "unknown"),
    ],
)
def test_untrustworthy_git_sha_is_not_recorded(git_sha, model_version):
    assert _trusted(git_sha, model_version, DATA_COMMIT) is None


def test_dropping_a_git_sha_logs_why(caplog):
    with caplog.at_level(logging.WARNING):
        _trusted(DATA_COMMIT, FIXED_VERSION, DATA_COMMIT)
        _trusted(MODEL_COMMIT, "2.122.2", DATA_COMMIT)

    assert "policyengine-uk-data's own commit" in caplog.text
    assert f"policyengine-uk 2.122.2 predates {FIXED_VERSION}" in caplog.text


def _installed_model_package(runtime_metadata: dict) -> dict:
    package = ModuleType("policyengine_uk")
    build_metadata = ModuleType("policyengine_uk.build_metadata")
    build_metadata.get_runtime_metadata = lambda: runtime_metadata
    package.build_metadata = build_metadata
    return {
        "policyengine_uk": package,
        "policyengine_uk.build_metadata": build_metadata,
    }


@pytest.mark.parametrize(
    ("model_version", "model_git_sha", "recorded_git_sha"),
    [
        pytest.param("2.122.2", DATA_COMMIT, None, id="uk-data-1.58.0-build"),
        pytest.param("2.122.2", MODEL_COMMIT, None, id="before-fix"),
        pytest.param(FIXED_VERSION, DATA_COMMIT, None, id="own-commit"),
        pytest.param(FIXED_VERSION, MODEL_COMMIT, MODEL_COMMIT, id="git-install"),
        pytest.param(FIXED_VERSION, None, None, id="wheel-install"),
    ],
)
def test_model_build_metadata_keeps_only_a_trustworthy_git_sha(
    model_version, model_git_sha, recorded_git_sha
):
    core = {"name": "policyengine-core", "version": "3.32.13"}
    runtime_metadata = {
        "name": "policyengine-uk",
        "version": model_version,
        "git_sha": model_git_sha,
        "data_build_fingerprint": "sha256:fingerprint",
        "core": core,
    }

    with (
        patch.dict(sys.modules, _installed_model_package(runtime_metadata)),
        patch(
            "policyengine_uk_data.utils.data_upload._get_model_package_version",
            return_value=None,
        ),
    ):
        build_metadata = _get_model_package_build_metadata(
            data_package_git_sha=DATA_COMMIT
        )

    assert build_metadata == {
        "version": model_version,
        "git_sha": recorded_git_sha,
        "data_build_fingerprint": "sha256:fingerprint",
        "core": core,
    }


HEX_COMMITS = st.one_of(
    st.text("0123456789abcdef", min_size=40, max_size=40),
    st.text("0123456789abcdef", min_size=64, max_size=64),
)
GIT_SHA_INPUTS = st.one_of(
    st.none(),
    HEX_COMMITS,
    HEX_COMMITS.map(str.upper),
    st.text("0123456789abcdefABCDEF", max_size=70),
    st.text(max_size=70),
    st.integers(),
)
MODEL_VERSIONS = st.one_of(
    st.none(),
    st.sampled_from(VERSIONS_BEFORE_FIX + VERSIONS_WITH_FIX),
    st.builds(
        lambda release, suffix: ".".join(map(str, release)) + suffix,
        st.tuples(st.integers(0, 3), st.integers(0, 200), st.integers(0, 20)),
        st.sampled_from(["", ".dev0", "rc1", ".post1", "+local"]),
    ),
    st.text(max_size=12),
)


def _data_sha(source: str, git_sha, other_commit: str):
    if source == "missing":
        return None
    if source == "other" or not isinstance(git_sha, str):
        return other_commit
    return git_sha.upper() if source == "same_upper" else git_sha


@settings(max_examples=1000, deadline=None, derandomize=True)
@given(
    git_sha=GIT_SHA_INPUTS,
    model_version=MODEL_VERSIONS,
    source=st.sampled_from(["missing", "other", "same", "same_upper"]),
    other_commit=HEX_COMMITS,
)
def test_a_recorded_git_sha_is_always_trustworthy(
    git_sha, model_version, source, other_commit
):
    data_sha = _data_sha(source, git_sha, other_commit)
    recorded = _trusted(git_sha, model_version, data_sha)

    assert _trusted(recorded, model_version, data_sha) == recorded
    if recorded is None:
        return
    assert recorded == git_sha
    assert GIT_COMMIT_ID.fullmatch(recorded)
    assert data_sha is None or recorded.lower() != data_sha.lower()
    assert Version(model_version) >= FIRST_MODEL_VERSION_WITH_OWN_GIT_SHA


@settings(max_examples=500, deadline=None, derandomize=True)
@given(commit=HEX_COMMITS, other_commit=HEX_COMMITS, model_version=MODEL_VERSIONS)
def test_only_releases_before_the_fix_lose_a_distinct_commit(
    commit, other_commit, model_version
):
    assume(commit != other_commit)
    try:
        before_fix = Version(model_version) < FIRST_MODEL_VERSION_WITH_OWN_GIT_SHA
    except (InvalidVersion, TypeError):
        before_fix = True

    recorded = _trusted(commit, model_version, other_commit)

    assert recorded == (None if before_fix else commit)
