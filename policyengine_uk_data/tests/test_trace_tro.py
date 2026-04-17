import hashlib
import json
from pathlib import Path

from policyengine_uk_data.utils.release_manifest import build_release_manifest
from policyengine_uk_data.utils.trace_tro import (
    TRACE_TRO_FILENAME,
    build_trace_tro_from_release_manifest,
    compute_trace_composition_fingerprint,
    serialize_trace_tro,
)


def _write_file(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_compute_trace_composition_fingerprint_is_order_independent():
    a = "b" * 64
    b = "a" * 64

    assert compute_trace_composition_fingerprint([a, b]) == (
        compute_trace_composition_fingerprint([b, a])
    )


def test_build_trace_tro_from_release_manifest_tracks_artifacts(tmp_path):
    enhanced_path = _write_file(
        tmp_path / "enhanced_frs_2023_24.h5",
        b"enhanced-frs",
    )
    baseline_path = _write_file(tmp_path / "frs_2023_24.h5", b"baseline-frs")

    manifest = build_release_manifest(
        files_with_repo_paths=[
            (enhanced_path, "enhanced_frs_2023_24.h5"),
            (baseline_path, "frs_2023_24.h5"),
        ],
        version="1.40.4",
        repo_id="policyengine/policyengine-uk-data-private",
        model_package_version="2.74.0",
        model_package_git_sha="deadbeef",
        model_package_data_build_fingerprint="sha256:fingerprint",
        created_at="2026-04-10T12:00:00Z",
    )

    tro = build_trace_tro_from_release_manifest(manifest)
    root = tro["@graph"][0]
    composition = root["trov:hasComposition"]
    artifacts = composition["trov:hasArtifact"]
    arrangement = root["trov:hasArrangement"][0]
    performance = root["trov:hasPerformance"][0]

    assert tro["@context"][0]["trov"] == "https://w3id.org/trace/trov/0.1#"
    assert root["trov:vocabularyVersion"] == "0.1"
    assert root["schema:name"] == "policyengine-uk-data 1.40.4 release TRO"
    assert root["trov:createdWith"]["schema:softwareVersion"] == "1.40.4"
    assert root["trov:wasAssembledBy"]["schema:name"] == (
        "PolicyEngine UK data release pipeline"
    )
    assert len(artifacts) == 3
    assert (
        arrangement["trov:hasArtifactLocation"][-1]["trov:path"]
        == "release_manifest.json"
    )
    assert performance["trov:contributedToArrangement"]["trov:arrangement"] == {
        "@id": "arrangement/0"
    }

    manifest_hash = hashlib.sha256(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    assert artifacts[-1]["trov:hash"]["trov:hashValue"] == manifest_hash

    expected_fingerprint = compute_trace_composition_fingerprint(
        [artifact["trov:hash"]["trov:hashValue"] for artifact in artifacts]
    )
    assert (
        composition["trov:hasFingerprint"]["trov:hash"]["trov:hashValue"]
        == expected_fingerprint
    )


def test_serialize_trace_tro_is_deterministic(tmp_path):
    dataset_path = _write_file(tmp_path / "enhanced_frs_2023_24.h5", b"enhanced-frs")
    manifest = build_release_manifest(
        files_with_repo_paths=[(dataset_path, "enhanced_frs_2023_24.h5")],
        version="1.40.4",
        repo_id="policyengine/policyengine-uk-data-private",
        created_at="2026-04-10T12:00:00Z",
    )

    tro = build_trace_tro_from_release_manifest(manifest)

    assert serialize_trace_tro(tro) == serialize_trace_tro(tro)
    assert TRACE_TRO_FILENAME == "trace.tro.jsonld"
