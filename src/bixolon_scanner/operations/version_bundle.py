from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..contracts.artifact import directory_content_manifest
from ..contracts.catalog import SEMVER, SHA256, load_store_catalog_package, sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2

FINAL_BUNDLE_REQUIRED_FILES = (
    "product_scanner.exe",
    "worker/bixolon-worker.exe",
    "worker/model-package/metadata.json",
    "worker/store-catalog/catalog.json",
    "worker/store-catalog/checksums.json",
    "worker/model-package/licenses/APACHE-2.0.txt",
    "worker/model-package/licenses/DINOV3-LICENSE.md",
    "worker/model-package/licenses/THIRD_PARTY_MODELS.md",
    "worker/cuda-runtime/cudart64_13.dll",
    "worker/cuda-runtime/cublas64_13.dll",
    "worker/cuda-runtime/cudnn64_9.dll",
)


class ArtifactLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    manifest_sha256: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("version artifact paths must stay inside the repository")
        return path.as_posix()

    @field_validator("manifest_sha256")
    @classmethod
    def validate_manifest_sha256(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("artifact manifest must use lowercase SHA-256")
        return value


class CatalogLock(ArtifactLock):
    store_id: str = Field(min_length=1)


class CudaRuntimeLock(ArtifactLock):
    files: list[str] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("CUDA runtime files must be unique")
        if any(Path(name).name != name for name in value):
            raise ValueError("CUDA runtime entries must be file names")
        return value


class EvidenceLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("evidence paths must stay inside the repository")
        return path.as_posix()

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("evidence must use lowercase SHA-256")
        return value


class VersionBundleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^1\.0$")
    version: str
    app_build: int = Field(ge=1)
    source_date_epoch: int = Field(default=0, ge=0)
    source_candidate: str = Field(min_length=1)
    runtime: ArtifactLock
    catalog: CatalogLock
    cuda_runtime: CudaRuntimeLock
    evaluation_evidence: list[EvidenceLock] = Field(min_length=1)
    output_root: str = Field(min_length=1)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not SEMVER.fullmatch(value) or "-" in value:
            raise ValueError("product version must be a stable semantic version")
        return value

    @field_validator("output_root")
    @classmethod
    def validate_output_root(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("version output must stay inside the repository")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_evidence_paths(self) -> "VersionBundleConfig":
        paths = [row.path for row in self.evaluation_evidence]
        if len(paths) != len(set(paths)):
            raise ValueError("evaluation evidence paths must be unique")
        return self


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _repository_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"path escapes the repository: {value}")
    return root / relative


def load_version_config(path: Path) -> VersionBundleConfig:
    return VersionBundleConfig.model_validate(_read_json(path))


def _verify_directory_lock(root: Path, lock: ArtifactLock) -> tuple[Path, dict[str, Any]]:
    path = _repository_path(root, lock.path)
    manifest = directory_content_manifest(path)
    if manifest["manifest_sha256"] != lock.manifest_sha256:
        raise ValueError(f"artifact manifest mismatch: {lock.path}")
    return path, manifest


def _verify_evidence(root: Path, rows: list[EvidenceLock]) -> None:
    for row in rows:
        path = _repository_path(root, row.path)
        if not path.is_file() or sha256_file(path) != row.sha256:
            raise ValueError(f"evaluation evidence mismatch: {row.path}")


def _rewrite_runtime(source: Path, target: Path, version: str) -> None:
    shutil.copytree(source, target)
    metadata_path = target / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata.pop("promotion_status", None)
    metadata.pop("promotion", None)
    metadata["worker_version"] = version
    metadata["detector_policy_version"] = version
    metadata["detector"]["version"] = version
    metadata["embedder"]["version"] = version
    metadata["classifier_policy"]["version"] = version
    legacy_detector_name = "detector-production.onnx"
    versioned_detector_name = "detector-reference.onnx"
    legacy_detector = target / legacy_detector_name
    if legacy_detector.exists():
        legacy_detector.rename(target / versioned_detector_name)
        if metadata["detector"]["filename"] == legacy_detector_name:
            metadata["detector"]["filename"] = versioned_detector_name
        ensemble = metadata["detector"].get("ensemble")
        if isinstance(ensemble, dict):
            for member in ensemble.get("members", []):
                if member.get("filename") == legacy_detector_name:
                    member["filename"] = versioned_detector_name
        metadata["checksums"][versioned_detector_name] = metadata["checksums"].pop(
            legacy_detector_name
        )
    _write_json(metadata_path, metadata)


def _rewrite_catalog(source: Path, target: Path, version: str) -> None:
    shutil.copytree(source, target)
    metadata_path = target / "catalog.json"
    metadata = _read_json(metadata_path)
    metadata["authentication"] = "CHECKSUM-SHA256"
    metadata["catalog_version"] = version
    metadata["embedder_version"] = version
    metadata["classifier_policy_version"] = version
    _write_json(metadata_path, metadata)
    checksums_path = target / "checksums.json"
    checksums = _read_json(checksums_path)
    checksums["catalog.json"] = sha256_file(metadata_path)
    _write_json(checksums_path, dict(sorted(checksums.items())))
    signature_path = target / "signature.json"
    if signature_path.exists():
        signature_path.unlink()


def _immutable_payload_hashes(root: Path, excluded: set[str]) -> list[str]:
    return sorted(
        sha256_file(path)
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        if path.relative_to(root).as_posix() not in excluded
    )


def _assert_immutable_payloads(
    runtime_source: Path,
    runtime_target: Path,
    catalog_source: Path,
    catalog_target: Path,
) -> None:
    if _immutable_payload_hashes(runtime_source, {"metadata.json"}) != _immutable_payload_hashes(
        runtime_target, {"metadata.json"}
    ):
        raise ValueError("runtime model payload changed while assigning the product version")
    excluded = {"catalog.json", "checksums.json", "signature.json"}
    if _immutable_payload_hashes(catalog_source, excluded) != _immutable_payload_hashes(
        catalog_target, excluded
    ):
        raise ValueError("Catalog payload changed while assigning the product version")


def _validate_composition(
    config: VersionBundleConfig,
    runtime_path: Path,
    catalog_path: Path,
) -> None:
    runtime = load_runtime_package_v2(runtime_path)
    catalog = load_store_catalog_package(
        catalog_path,
        expected_store_id=config.catalog.store_id,
    )
    versions = (
        runtime.metadata.worker_version,
        runtime.metadata.detector.version,
        runtime.metadata.embedder.version,
        runtime.metadata.detector_policy_version,
        runtime.metadata.classifier_policy.version,
        catalog.metadata.catalog_version,
        catalog.metadata.embedder_version,
        catalog.metadata.classifier_policy_version,
    )
    if any(value != config.version for value in versions):
        raise ValueError("prepared bundle contains mixed product versions")
    runtime_metadata = _read_json(runtime_path / "metadata.json")
    if "promotion_status" in runtime_metadata or "promotion" in runtime_metadata:
        raise ValueError("prepared runtime contains a lifecycle field")
    if catalog.metadata.authentication != "CHECKSUM-SHA256":
        raise ValueError("prepared Catalog must use checksum-only validation")
    if (catalog_path / "signature.json").exists():
        raise ValueError("prepared Catalog must not contain a signature")


def prepare_version_bundle(
    config: VersionBundleConfig,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    root = repository_root.absolute()
    runtime_source, runtime_source_manifest = _verify_directory_lock(root, config.runtime)
    catalog_source, catalog_source_manifest = _verify_directory_lock(root, config.catalog)
    cuda_source, cuda_source_manifest = _verify_directory_lock(root, config.cuda_runtime)
    actual_cuda_files = sorted(
        path.relative_to(cuda_source).as_posix()
        for path in cuda_source.rglob("*")
        if path.is_file()
    )
    if sorted(config.cuda_runtime.files) != actual_cuda_files:
        raise ValueError("CUDA runtime file list does not match the version config")
    _verify_evidence(root, config.evaluation_evidence)

    version_root = _repository_path(root, f"{config.output_root}/{config.version}")
    version_root.mkdir(parents=True, exist_ok=True)
    target = version_root / "staging"
    temporary = Path(tempfile.mkdtemp(prefix=".staging-", dir=version_root))
    try:
        runtime_target = temporary / "runtime"
        catalog_target = temporary / "catalog"
        cuda_target = temporary / "cuda-runtime"
        _rewrite_runtime(runtime_source, runtime_target, config.version)
        _rewrite_catalog(catalog_source, catalog_target, config.version)
        shutil.copytree(cuda_source, cuda_target)
        _assert_immutable_payloads(
            runtime_source,
            runtime_target,
            catalog_source,
            catalog_target,
        )
        _validate_composition(config, runtime_target, catalog_target)
        version_payload = {
            "schema_version": "1.0",
            "version": config.version,
            "app_build": config.app_build,
        }
        provenance = {
            "schema_version": "1.0",
            "version": config.version,
            "source_candidate": config.source_candidate,
            "source_artifacts": {
                "runtime": {
                    "path": config.runtime.path,
                    "manifest_sha256": runtime_source_manifest["manifest_sha256"],
                },
                "catalog": {
                    "path": config.catalog.path,
                    "manifest_sha256": catalog_source_manifest["manifest_sha256"],
                },
            },
            "evaluation_evidence": [
                row.model_dump(mode="json") for row in config.evaluation_evidence
            ],
            "transformation": {
                "model_graph_or_weight_changed": False,
                "decision_policy_changed": False,
                "lifecycle_fields_emitted": False,
                "catalog_authentication": "CHECKSUM-SHA256",
            },
            "artifacts": {
                "runtime": directory_content_manifest(runtime_target),
                "catalog": directory_content_manifest(catalog_target),
                "cuda_runtime": cuda_source_manifest,
            },
        }
        _write_json(temporary / "version.json", version_payload)
        _write_json(temporary / "provenance.json", provenance)
        if target.exists():
            shutil.rmtree(target)
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "version": config.version,
        "staging_path": target.as_posix(),
        "runtime_manifest_sha256": provenance["artifacts"]["runtime"]["manifest_sha256"],
        "catalog_manifest_sha256": provenance["artifacts"]["catalog"]["manifest_sha256"],
        "cuda_manifest_sha256": provenance["artifacts"]["cuda_runtime"]["manifest_sha256"],
        "passed": True,
    }


def verify_prepared_version(
    config: VersionBundleConfig,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    root = repository_root.absolute()
    staging = _repository_path(root, f"{config.output_root}/{config.version}/staging")
    version_payload = _read_json(staging / "version.json")
    provenance = _read_json(staging / "provenance.json")
    if version_payload != {
        "schema_version": "1.0",
        "version": config.version,
        "app_build": config.app_build,
    }:
        raise ValueError("prepared version identity does not match the version config")
    if (
        provenance.get("version") != config.version
        or provenance.get("source_candidate") != config.source_candidate
    ):
        raise ValueError("prepared provenance does not match the version config")
    for name, directory in (
        ("runtime", staging / "runtime"),
        ("catalog", staging / "catalog"),
        ("cuda_runtime", staging / "cuda-runtime"),
    ):
        if directory_content_manifest(directory) != provenance["artifacts"][name]:
            raise ValueError(f"prepared {name} manifest mismatch")
    _validate_composition(config, staging / "runtime", staging / "catalog")
    result: dict[str, Any] = {
        "version": config.version,
        "staging_path": staging.as_posix(),
        "passed": True,
    }
    bundle = staging.parent / f"bixolon-scanner-{config.version}"
    if bundle.exists():
        result.update(_verify_final_bundle(config, staging=staging, bundle=bundle))
    return result


def _final_bundle_file_records(bundle: Path) -> list[dict[str, Any]]:
    manifest_path = bundle / "bundle-manifest.json"
    return [
        {
            "path": path.relative_to(bundle).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(candidate for candidate in bundle.rglob("*") if candidate.is_file())
        if path != manifest_path
    ]


def _validate_final_bundle_content(
    config: VersionBundleConfig,
    *,
    staging: Path,
    bundle: Path,
) -> None:
    if _read_json(bundle / "version.json") != _read_json(staging / "version.json"):
        raise ValueError("final bundle version identity mismatch")
    if _read_json(bundle / "provenance.json") != _read_json(staging / "provenance.json"):
        raise ValueError("final bundle provenance mismatch")
    runtime = bundle / "worker" / "model-package"
    catalog = bundle / "worker" / "store-catalog"
    _validate_composition(config, runtime, catalog)
    for relative in FINAL_BUNDLE_REQUIRED_FILES:
        if not (bundle / relative).is_file():
            raise ValueError(f"final bundle is missing a required file: {relative}")


def write_final_bundle_manifest(
    config: VersionBundleConfig,
    *,
    repository_root: Path,
    bundle: Path,
) -> dict[str, Any]:
    """Validate a temporary Windows bundle and write its canonical file manifest."""

    root = repository_root.absolute()
    version_root = _repository_path(root, f"{config.output_root}/{config.version}").resolve()
    staging = version_root / "staging"
    target = bundle.resolve()
    if target.parent != version_root or not target.is_dir():
        raise ValueError("final bundle must be a direct child of the configured version root")
    manifest_path = target / "bundle-manifest.json"
    if manifest_path.exists():
        raise ValueError("final bundle manifest already exists")
    _validate_final_bundle_content(config, staging=staging, bundle=target)
    files = _final_bundle_file_records(target)
    manifest = {
        "schema_version": "1.0",
        "version": config.version,
        "app_build": config.app_build,
        "file_count": len(files),
        "files": files,
    }
    _write_json(manifest_path, manifest)
    return {
        "version": config.version,
        "bundle_path": target.as_posix(),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "file_count": len(files),
        "passed": True,
    }


def _verify_final_bundle(
    config: VersionBundleConfig,
    *,
    staging: Path,
    bundle: Path,
) -> dict[str, Any]:
    manifest_path = bundle / "bundle-manifest.json"
    manifest = _read_json(manifest_path)
    actual_files = _final_bundle_file_records(bundle)
    recorded_files = manifest.get("files")
    if not isinstance(recorded_files, list):
        raise ValueError("final bundle manifest has no file list")
    recorded_by_path = {row.get("path"): row for row in recorded_files}
    actual_by_path = {row["path"]: row for row in actual_files}
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("version") != config.version
        or manifest.get("app_build") != config.app_build
        or manifest.get("file_count") != len(actual_files)
        or len(recorded_by_path) != len(recorded_files)
        or recorded_by_path != actual_by_path
    ):
        raise ValueError("final bundle manifest mismatch")
    _validate_final_bundle_content(config, staging=staging, bundle=bundle)
    return {
        "bundle_path": bundle.as_posix(),
        "bundle_manifest_sha256": sha256_file(manifest_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare or verify one BIXOLON version bundle")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "verify"):
        command = subparsers.add_parser(action)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--repository-root", type=Path, default=Path("."))
    manifest_command = subparsers.add_parser("manifest")
    manifest_command.add_argument("--config", type=Path, required=True)
    manifest_command.add_argument("--repository-root", type=Path, default=Path("."))
    manifest_command.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_version_config(args.config)
    if args.action == "prepare":
        result = prepare_version_bundle(config, repository_root=args.repository_root)
    elif args.action == "verify":
        result = verify_prepared_version(config, repository_root=args.repository_root)
    else:
        result = write_final_bundle_manifest(
            config,
            repository_root=args.repository_root,
            bundle=args.bundle,
        )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


def verify_main() -> None:
    main(["verify", *sys.argv[1:]])


if __name__ == "__main__":
    main()
