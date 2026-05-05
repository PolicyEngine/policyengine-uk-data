from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

RELEASE_MANIFEST_SCHEMA_VERSION = 1


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _compute_file_checksum(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _artifact_key(path_in_repo: str) -> str:
    return str(PurePosixPath(path_in_repo).with_suffix(""))


def _artifact_kind(path_in_repo: str) -> str:
    path = PurePosixPath(path_in_repo)
    suffix = path.suffix.lower()
    if suffix == ".h5":
        if "weight" in path.stem:
            return "weights"
        return "microdata"
    if suffix == ".db":
        return "database"
    return "auxiliary"


def _artifact_uri(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    path_in_repo: str,
) -> str:
    return f"hf://{repo_type}/{repo_id}@{revision}/{path_in_repo}"


def _artifact_visibility(repo_id: str) -> str:
    return "private" if repo_id.endswith("-private") else "public"


def _without_none_values(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _runtime_component_metadata(
    *,
    name: str,
    version: str | None,
    git_sha: str | None = None,
    data_build_fingerprint: str | None = None,
    core_package_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if version is None:
        return None

    metadata = _without_none_values(
        {
            "name": name,
            "version": version,
            "git_sha": git_sha,
            "data_build_fingerprint": data_build_fingerprint,
        }
    )
    if core_package_metadata is not None:
        metadata["core"] = dict(core_package_metadata)
    return metadata


def _build_metadata(
    *,
    data_package_git_sha: str | None,
) -> Dict[str, Any]:
    return _without_none_values(
        {
            "data_package_git_sha": data_package_git_sha,
        }
    )


def _core_version(core_package_metadata: Mapping[str, Any] | None) -> str | None:
    if core_package_metadata is None:
        return None
    version = core_package_metadata.get("version")
    return version if isinstance(version, str) and version else None


def _base_manifest(
    *,
    version: str,
    data_package_name: str,
    model_package_name: str,
    model_package_version: str | None,
    model_package_git_sha: str | None,
    model_package_data_build_fingerprint: str | None,
    core_package_metadata: Mapping[str, Any] | None,
    data_package_git_sha: str | None,
    repo_id: str,
    repo_type: str,
    build_id: str,
    created_at: str,
) -> Dict:
    build_metadata = _build_metadata(data_package_git_sha=data_package_git_sha)
    manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {
            "name": data_package_name,
            "version": version,
        },
        "compatible_model_packages": [],
        "compatible_core_packages": [],
        "default_datasets": {},
        "build": {
            "build_id": build_id,
            "built_at": created_at,
        },
        "artifacts": {},
        "metadata": {
            "artifact_release": {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "version": version,
                "visibility": _artifact_visibility(repo_id),
            }
        },
    }
    if build_metadata:
        manifest["build"]["metadata"] = build_metadata
    model_package_metadata = _runtime_component_metadata(
        name=model_package_name,
        version=model_package_version,
        git_sha=model_package_git_sha,
        data_build_fingerprint=model_package_data_build_fingerprint,
        core_package_metadata=core_package_metadata,
    )
    if model_package_metadata is not None:
        manifest["build"]["built_with_model_package"] = model_package_metadata
    if core_package_metadata is not None:
        manifest["build"]["built_with_core_package"] = dict(core_package_metadata)
    if model_package_version:
        manifest["compatible_model_packages"].append(
            {
                "name": model_package_name,
                "specifier": f"=={model_package_version}",
            }
        )
    core_version = _core_version(core_package_metadata)
    if core_version:
        manifest["compatible_core_packages"].append(
            {
                "name": core_package_metadata.get("name", "policyengine-core"),
                "specifier": f"=={core_version}",
            }
        )
    return manifest


def _normalize_existing_manifest(
    existing_manifest: Mapping | None,
    *,
    version: str,
    data_package_name: str,
) -> Dict | None:
    if existing_manifest is None:
        return None
    package = existing_manifest.get("data_package", {})
    if package.get("name") != data_package_name or package.get("version") != version:
        return None
    manifest = deepcopy(dict(existing_manifest))
    manifest.pop("created_at", None)
    return manifest


def build_release_manifest(
    *,
    files_with_repo_paths: Sequence[Tuple[Path | str, str]],
    version: str,
    repo_id: str,
    repo_type: str = "model",
    data_package_name: str = "policyengine-uk-data",
    model_package_name: str = "policyengine-uk",
    model_package_version: str | None = None,
    model_package_git_sha: str | None = None,
    model_package_data_build_fingerprint: str | None = None,
    core_package_metadata: Optional[Mapping[str, Any]] = None,
    data_package_git_sha: str | None = None,
    build_id: str | None = None,
    existing_manifest: Mapping | None = None,
    default_datasets: Optional[Mapping[str, str]] = None,
    created_at: str | None = None,
) -> Dict:
    manifest = _normalize_existing_manifest(
        existing_manifest,
        version=version,
        data_package_name=data_package_name,
    )
    manifest_timestamp = created_at or _utc_timestamp()
    resolved_build_id = build_id or f"{data_package_name}-{version}"

    if manifest is None:
        manifest = _base_manifest(
            version=version,
            data_package_name=data_package_name,
            model_package_name=model_package_name,
            model_package_version=model_package_version,
            model_package_git_sha=model_package_git_sha,
            model_package_data_build_fingerprint=model_package_data_build_fingerprint,
            core_package_metadata=core_package_metadata,
            data_package_git_sha=data_package_git_sha,
            repo_id=repo_id,
            repo_type=repo_type,
            build_id=resolved_build_id,
            created_at=manifest_timestamp,
        )
    else:
        manifest["schema_version"] = RELEASE_MANIFEST_SCHEMA_VERSION
        manifest.setdefault("compatible_core_packages", [])
        manifest.setdefault("metadata", {})["artifact_release"] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "version": version,
            "visibility": _artifact_visibility(repo_id),
        }
        manifest.setdefault("build", {})
        manifest["build"].setdefault("build_id", resolved_build_id)
        manifest["build"].setdefault("built_at", manifest_timestamp)
        build_metadata = _build_metadata(data_package_git_sha=data_package_git_sha)
        if build_metadata:
            manifest["build"].setdefault("metadata", {}).update(build_metadata)
        model_package_metadata = _runtime_component_metadata(
            name=model_package_name,
            version=model_package_version,
            git_sha=model_package_git_sha,
            data_build_fingerprint=model_package_data_build_fingerprint,
            core_package_metadata=core_package_metadata,
        )
        if model_package_metadata is not None:
            manifest["build"]["built_with_model_package"] = model_package_metadata
        if core_package_metadata is not None:
            manifest["build"]["built_with_core_package"] = dict(core_package_metadata)
        if model_package_version:
            manifest["compatible_model_packages"] = [
                {
                    "name": model_package_name,
                    "specifier": f"=={model_package_version}",
                }
            ]
        core_version = _core_version(core_package_metadata)
        if core_version:
            manifest["compatible_core_packages"] = [
                {
                    "name": core_package_metadata.get("name", "policyengine-core"),
                    "specifier": f"=={core_version}",
                }
            ]

    if default_datasets:
        manifest.setdefault("default_datasets", {}).update(default_datasets)

    for local_path, path_in_repo in files_with_repo_paths:
        local_path = Path(local_path)
        manifest["artifacts"][_artifact_key(path_in_repo)] = {
            "kind": _artifact_kind(path_in_repo),
            "uri": _artifact_uri(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=version,
                path_in_repo=path_in_repo,
            ),
            "path": path_in_repo,
            "repo_id": repo_id,
            "revision": version,
            "sha256": _compute_file_checksum(local_path),
            "size_bytes": local_path.stat().st_size,
            "metadata": {
                "repo_type": repo_type,
                "visibility": _artifact_visibility(repo_id),
            },
        }

    defaults = manifest["default_datasets"]
    if "national" not in defaults and "enhanced_frs_2023_24" in manifest["artifacts"]:
        defaults["national"] = "enhanced_frs_2023_24"
    if "baseline" not in defaults and "frs_2023_24" in manifest["artifacts"]:
        defaults["baseline"] = "frs_2023_24"

    return manifest


def serialize_release_manifest(manifest: Mapping) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
