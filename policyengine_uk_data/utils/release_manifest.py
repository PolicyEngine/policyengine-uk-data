from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from policyengine_uk_data.utils.hf_destinations import PRIVATE_REPO, PUBLIC_REPO

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
    if repo_id == PRIVATE_REPO:
        return "private"
    if repo_id == PUBLIC_REPO:
        return "public"
    raise ValueError(
        f"Unknown UK data Hugging Face repo {repo_id!r}; use "
        "PRIVATE_REPO or PUBLIC_REPO."
    )


def _artifact_release_metadata(
    *,
    repo_id: str,
    repo_type: str,
    version: str,
) -> Dict[str, str]:
    # UK data uses one release coordinate across package code, HF tags, and
    # published dataset artifacts. Do not treat this as a separate artifact version.
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "version": version,
        "visibility": _artifact_visibility(repo_id),
    }


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


def _model_package_compatibility(
    *,
    model_package_name: str,
    model_package_version: str | None,
) -> list[Dict[str, str]]:
    if not model_package_version:
        return []
    return [
        {
            "name": model_package_name,
            "specifier": f"=={model_package_version}",
        }
    ]


def _core_package_compatibility(
    *,
    core_package_metadata: Mapping[str, Any] | None,
) -> list[Dict[str, str]]:
    core_version = _core_version(core_package_metadata)
    if not core_version or core_package_metadata is None:
        return []
    return [
        {
            "name": core_package_metadata.get("name", "policyengine-core"),
            "specifier": f"=={core_version}",
        }
    ]


def _new_release_manifest(
    *,
    version: str,
    data_package_name: str,
) -> Dict:
    return {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {
            "name": data_package_name,
            "version": version,
        },
        "compatible_model_packages": [],
        "compatible_core_packages": [],
        "default_datasets": {},
        "build": {},
        "artifacts": {},
        "metadata": {},
    }


def _update_manifest_metadata(
    manifest: Dict,
    *,
    repo_id: str,
    repo_type: str,
    version: str,
) -> None:
    manifest["schema_version"] = RELEASE_MANIFEST_SCHEMA_VERSION
    manifest.setdefault("metadata", {})["artifact_release"] = (
        _artifact_release_metadata(
            repo_id=repo_id,
            repo_type=repo_type,
            version=version,
        )
    )


def _update_build_section(
    manifest: Dict,
    *,
    build_id: str,
    created_at: str,
    data_package_git_sha: str | None,
    model_package_name: str,
    model_package_version: str | None,
    model_package_git_sha: str | None,
    model_package_data_build_fingerprint: str | None,
    core_package_metadata: Mapping[str, Any] | None,
) -> None:
    build = manifest.setdefault("build", {})
    build.setdefault("build_id", build_id)
    build.setdefault("built_at", created_at)

    build_metadata = _build_metadata(data_package_git_sha=data_package_git_sha)
    if build_metadata:
        build.setdefault("metadata", {}).update(build_metadata)

    model_package_metadata = _runtime_component_metadata(
        name=model_package_name,
        version=model_package_version,
        git_sha=model_package_git_sha,
        data_build_fingerprint=model_package_data_build_fingerprint,
        core_package_metadata=core_package_metadata,
    )
    if model_package_metadata is not None:
        build["built_with_model_package"] = model_package_metadata
    if core_package_metadata is not None:
        build["built_with_core_package"] = dict(core_package_metadata)


def _update_compatibility(
    manifest: Dict,
    *,
    model_package_name: str,
    model_package_version: str | None,
    core_package_metadata: Mapping[str, Any] | None,
) -> None:
    manifest.setdefault("compatible_model_packages", [])
    model_package_compatibility = _model_package_compatibility(
        model_package_name=model_package_name,
        model_package_version=model_package_version,
    )
    if model_package_compatibility:
        manifest["compatible_model_packages"] = model_package_compatibility

    manifest.setdefault("compatible_core_packages", [])
    core_package_compatibility = _core_package_compatibility(
        core_package_metadata=core_package_metadata,
    )
    if core_package_compatibility:
        manifest["compatible_core_packages"] = core_package_compatibility


def _update_artifacts(
    manifest: Dict,
    *,
    files_with_repo_paths: Sequence[Tuple[Path | str, str]],
    repo_id: str,
    repo_type: str,
    version: str,
) -> None:
    artifacts = manifest.setdefault("artifacts", {})
    for local_path, path_in_repo in files_with_repo_paths:
        local_path = Path(local_path)
        artifacts[_artifact_key(path_in_repo)] = {
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


def _update_default_datasets(
    manifest: Dict,
    *,
    default_datasets: Optional[Mapping[str, str]],
) -> None:
    defaults = manifest.setdefault("default_datasets", {})
    if default_datasets:
        defaults.update(default_datasets)
    if "national" not in defaults and "enhanced_frs_2023_24" in manifest.get(
        "artifacts", {}
    ):
        defaults["national"] = "enhanced_frs_2023_24"
    if "baseline" not in defaults and "frs_2023_24" in manifest.get("artifacts", {}):
        defaults["baseline"] = "frs_2023_24"


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
        manifest = _new_release_manifest(
            version=version,
            data_package_name=data_package_name,
        )

    _update_manifest_metadata(
        manifest,
        repo_id=repo_id,
        repo_type=repo_type,
        version=version,
    )
    _update_build_section(
        manifest,
        build_id=resolved_build_id,
        created_at=manifest_timestamp,
        data_package_git_sha=data_package_git_sha,
        model_package_name=model_package_name,
        model_package_version=model_package_version,
        model_package_git_sha=model_package_git_sha,
        model_package_data_build_fingerprint=model_package_data_build_fingerprint,
        core_package_metadata=core_package_metadata,
    )
    _update_compatibility(
        manifest,
        model_package_name=model_package_name,
        model_package_version=model_package_version,
        core_package_metadata=core_package_metadata,
    )
    _update_artifacts(
        manifest,
        files_with_repo_paths=files_with_repo_paths,
        repo_id=repo_id,
        repo_type=repo_type,
        version=version,
    )
    _update_default_datasets(
        manifest,
        default_datasets=default_datasets,
    )
    return manifest


def serialize_release_manifest(manifest: Mapping) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
