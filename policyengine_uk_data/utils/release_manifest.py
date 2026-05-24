from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from policyengine_uk_data.datasets.frs_release import CURRENT_FRS_RELEASE
from policyengine_uk_data.utils.hf_destinations import PRIVATE_REPO, PUBLIC_REPO

RELEASE_MANIFEST_SCHEMA_VERSION = 1
LEGACY_DEFAULT_FRS_RELEASES = ("frs_2023_24",)


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


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"release_manifest.{field} must be an object.")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"release_manifest.{field} must be a non-empty string.")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"release_manifest.{field} must be a positive integer.")
    return value


def _require_exact_compatibility(
    entries: Any,
    *,
    package_name: str,
    package_version: str,
    field: str,
) -> None:
    if not isinstance(entries, list):
        raise ValueError(f"release_manifest.{field} must be a list.")
    expected = {"name": package_name, "specifier": f"=={package_version}"}
    if expected not in entries:
        raise ValueError(
            f"release_manifest.{field} must include exact specifier "
            f"{package_name}=={package_version}."
        )


def _validate_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    version: str,
    data_package_name: str = "policyengine-uk-data",
) -> None:
    if manifest.get("schema_version") != RELEASE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "release_manifest.schema_version must equal "
            f"{RELEASE_MANIFEST_SCHEMA_VERSION}."
        )

    data_package = _require_mapping(manifest.get("data_package"), "data_package")
    if data_package.get("name") != data_package_name:
        raise ValueError(
            f"release_manifest.data_package.name must equal {data_package_name}."
        )
    if data_package.get("version") != version:
        raise ValueError(f"release_manifest.data_package.version must equal {version}.")


def _validate_artifact_release_metadata(
    manifest: Mapping[str, Any],
    *,
    version: str,
    repo_id: str | None,
    repo_type: str | None,
) -> None:
    metadata = _require_mapping(manifest.get("metadata"), "metadata")
    artifact_release = _require_mapping(
        metadata.get("artifact_release"),
        "metadata.artifact_release",
    )
    if repo_id is not None and artifact_release.get("repo_id") != repo_id:
        raise ValueError(
            f"release_manifest.metadata.artifact_release.repo_id must equal {repo_id}."
        )
    if repo_type is not None and artifact_release.get("repo_type") != repo_type:
        raise ValueError(
            f"release_manifest.metadata.artifact_release.repo_type must equal {repo_type}."
        )
    if artifact_release.get("version") != version:
        raise ValueError(
            f"release_manifest.metadata.artifact_release.version must equal {version}."
        )
    if repo_id is not None:
        expected_visibility = _artifact_visibility(repo_id)
        if artifact_release.get("visibility") != expected_visibility:
            raise ValueError(
                "release_manifest.metadata.artifact_release.visibility must equal "
                f"{expected_visibility}."
            )


def _validate_build_and_compatibility(
    manifest: Mapping[str, Any],
    *,
    model_package_name: str,
    core_package_name: str,
) -> None:
    build = _require_mapping(manifest.get("build"), "build")
    build_metadata = _require_mapping(build.get("metadata"), "build.metadata")
    _require_string(
        build_metadata.get("data_package_git_sha"),
        "build.metadata.data_package_git_sha",
    )

    model_package = _require_mapping(
        build.get("built_with_model_package"),
        "build.built_with_model_package",
    )
    model_name = _require_string(
        model_package.get("name"),
        "build.built_with_model_package.name",
    )
    if model_name != model_package_name:
        raise ValueError(
            "release_manifest.build.built_with_model_package.name must equal "
            f"{model_package_name}."
        )
    model_version = _require_string(
        model_package.get("version"),
        "build.built_with_model_package.version",
    )
    _require_string(
        model_package.get("data_build_fingerprint"),
        "build.built_with_model_package.data_build_fingerprint",
    )

    core_package = _require_mapping(
        build.get("built_with_core_package"),
        "build.built_with_core_package",
    )
    core_name = _require_string(
        core_package.get("name"),
        "build.built_with_core_package.name",
    )
    if core_name != core_package_name:
        raise ValueError(
            "release_manifest.build.built_with_core_package.name must equal "
            f"{core_package_name}."
        )
    core_version = _require_string(
        core_package.get("version"),
        "build.built_with_core_package.version",
    )

    _require_exact_compatibility(
        manifest.get("compatible_model_packages"),
        package_name=model_name,
        package_version=model_version,
        field="compatible_model_packages",
    )
    _require_exact_compatibility(
        manifest.get("compatible_core_packages"),
        package_name=core_name,
        package_version=core_version,
        field="compatible_core_packages",
    )


def _validate_artifacts(
    manifest: Mapping[str, Any],
    *,
    version: str,
    repo_id: str | None,
    repo_type: str | None,
) -> Mapping[str, Any]:
    artifacts = _require_mapping(manifest.get("artifacts"), "artifacts")
    if not artifacts:
        raise ValueError("release_manifest.artifacts must not be empty.")
    for artifact_key, artifact in artifacts.items():
        field_prefix = f"artifacts.{artifact_key}"
        artifact = _require_mapping(artifact, field_prefix)
        artifact_uri = _require_string(artifact.get("uri"), f"{field_prefix}.uri")
        artifact_path = _require_string(artifact.get("path"), f"{field_prefix}.path")
        artifact_repo_id = _require_string(
            artifact.get("repo_id"), f"{field_prefix}.repo_id"
        )
        if repo_id is not None and artifact_repo_id != repo_id:
            raise ValueError(
                f"release_manifest.{field_prefix}.repo_id must equal {repo_id}."
            )
        expected_repo_id = repo_id or artifact_repo_id
        if artifact.get("revision") != version:
            raise ValueError(
                f"release_manifest.{field_prefix}.revision must equal {version}."
            )
        if repo_type is not None:
            expected_uri = _artifact_uri(
                repo_id=expected_repo_id,
                repo_type=repo_type,
                revision=version,
                path_in_repo=artifact_path,
            )
            if artifact_uri != expected_uri:
                raise ValueError(
                    f"release_manifest.{field_prefix}.uri must equal {expected_uri}."
                )
        _require_string(artifact.get("sha256"), f"{field_prefix}.sha256")
        _require_positive_int(artifact.get("size_bytes"), f"{field_prefix}.size_bytes")
        artifact_metadata = _require_mapping(
            artifact.get("metadata"),
            f"{field_prefix}.metadata",
        )
        if repo_type is not None and artifact_metadata.get("repo_type") != repo_type:
            raise ValueError(
                f"release_manifest.{field_prefix}.metadata.repo_type must equal {repo_type}."
            )
        if repo_id is not None:
            expected_visibility = _artifact_visibility(repo_id)
            if artifact_metadata.get("visibility") != expected_visibility:
                raise ValueError(
                    f"release_manifest.{field_prefix}.metadata.visibility must equal "
                    f"{expected_visibility}."
                )

    return artifacts


def _validate_default_datasets(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> None:
    default_datasets = _require_mapping(
        manifest.get("default_datasets"),
        "default_datasets",
    )
    if not default_datasets:
        raise ValueError("release_manifest.default_datasets must not be empty.")
    for default_key, artifact_key in default_datasets.items():
        if artifact_key not in artifacts:
            raise ValueError(
                "release_manifest.default_datasets "
                f"{default_key!r} points to missing artifact {artifact_key!r}."
            )


def validate_release_manifest(
    manifest: Mapping[str, Any],
    *,
    version: str,
    repo_id: str | None = None,
    repo_type: str | None = None,
    data_package_name: str = "policyengine-uk-data",
    model_package_name: str = "policyengine-uk",
    core_package_name: str = "policyengine-core",
) -> None:
    """Validate the bundle-facing UK data release contract.

    ``release_manifest.json`` is the authoritative signal that a release can be
    consumed by bundles. This validation is intentionally stricter than the
    general schema: a finalized release must include concrete artifact hashes,
    sizes, runtime package metadata, and exact compatibility specifiers.
    """
    _validate_manifest_identity(
        manifest,
        version=version,
        data_package_name=data_package_name,
    )
    _validate_artifact_release_metadata(
        manifest,
        version=version,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    _validate_build_and_compatibility(
        manifest,
        model_package_name=model_package_name,
        core_package_name=core_package_name,
    )
    artifacts = _validate_artifacts(
        manifest,
        version=version,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    _validate_default_datasets(manifest, artifacts)


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
    artifacts = manifest.get("artifacts", {})
    frs_releases = (CURRENT_FRS_RELEASE.name, *LEGACY_DEFAULT_FRS_RELEASES)
    for frs_release in frs_releases:
        enhanced_frs_release = f"enhanced_{frs_release}"
        if "national" not in defaults and enhanced_frs_release in artifacts:
            defaults["national"] = enhanced_frs_release
        if "baseline" not in defaults and frs_release in artifacts:
            defaults["baseline"] = frs_release
        if "national" in defaults and "baseline" in defaults:
            break


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
