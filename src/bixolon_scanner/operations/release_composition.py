from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .. import __version__
from ..contracts.model_package import SEMVER, load_model_package, sha256_file

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class LockedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("release artifacts require lowercase SHA-256")
        return value


class ReleaseVersions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python: str
    worker: str
    detector: str
    classifier: str
    detector_training_pipeline: str | None = None
    classifier_training_pipeline: str | None = None
    dataset: str = Field(min_length=1)
    app: str = Field(pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\+[1-9]\d*$")

    @field_validator(
        "python",
        "worker",
        "detector",
        "classifier",
        "detector_training_pipeline",
        "classifier_training_pipeline",
    )
    @classmethod
    def validate_semver(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not SEMVER.fullmatch(value):
            raise ValueError("release component versions must use semantic versioning")
        return value


class TrainingContractComposition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector: LockedArtifact
    classifier: LockedArtifact


class ReleaseComposition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^1\.[01]$")
    release: str = Field(min_length=1)
    lifecycle: str = Field(pattern=r"^(locked|attested)$")
    versions: ReleaseVersions
    model_package: LockedArtifact
    model_package_metadata: LockedArtifact
    training_contracts: TrainingContractComposition | None = None
    manual_waiver: LockedArtifact | None = None
    windows_bundle_manifest: LockedArtifact | None = None
    independent_test_status: str = Field(pattern=r"^pending_user_images$")

    @model_validator(mode="after")
    def validate_attestation_mode(self) -> "ReleaseComposition":
        if self.schema_version == "1.0":
            if self.training_contracts is None or self.manual_waiver is not None:
                raise ValueError("schema 1.0 requires training contracts without a manual waiver")
        elif (self.training_contracts is None) == (self.manual_waiver is None):
            raise ValueError("schema 1.1 requires exactly one provenance attestation mode")
        pipeline_versions = (
            self.versions.detector_training_pipeline,
            self.versions.classifier_training_pipeline,
        )
        if self.training_contracts is not None and any(
            version is None for version in pipeline_versions
        ):
            raise ValueError("training pipeline versions must match the attestation mode")
        if self.training_contracts is None and any(
            version is not None for version in pipeline_versions
        ):
            raise ValueError("training pipeline versions must match the attestation mode")
        return self


def canonical_composition_sha256(composition: ReleaseComposition) -> str:
    body = json.dumps(
        composition.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _path(root: Path, value: str) -> Path:
    root = root.resolve()
    result = (root / value).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"release path escapes repository: {value}") from exc
    return result


def _verify_artifact(root: Path, lock: LockedArtifact, *, directory: bool = False) -> Path:
    path = _path(root, lock.path)
    if directory:
        if not path.is_dir():
            raise ValueError(f"release directory is missing: {lock.path}")
        if sha256_directory(path) != lock.sha256:
            raise ValueError("model package directory checksum mismatch")
    elif not path.is_file() or sha256_file(path) != lock.sha256:
        raise ValueError(f"release artifact checksum mismatch: {lock.path}")
    return path


def load_release_composition(path: Path) -> ReleaseComposition:
    return ReleaseComposition.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _verify_bundle_manifest(path: Path, composition: ReleaseComposition) -> dict[str, Any]:
    root = path.parent.resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload.get("files")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("release") != composition.release
        or payload.get("app_version") != composition.versions.app
        or not isinstance(rows, list)
        or payload.get("file_count") != len(rows)
    ):
        raise ValueError("Windows bundle manifest identity is invalid")
    declared: set[str] = set()
    for row in rows:
        relative = str(row.get("path", ""))
        file_path = (root / relative).resolve()
        try:
            file_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Windows bundle manifest path escapes the bundle") from exc
        if relative in declared or not file_path.is_file():
            raise ValueError("Windows bundle manifest contains a duplicate or missing file")
        declared.add(relative)
        if file_path.stat().st_size != row.get("size_bytes") or sha256_file(file_path) != row.get(
            "sha256"
        ):
            raise ValueError(f"Windows bundle file checksum mismatch: {relative}")
    actual = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate != path
    }
    if actual != declared:
        raise ValueError("Windows bundle manifest does not cover every bundle file")
    required = {
        "product_scanner.exe",
        "worker/bixolon-worker.exe",
        "worker/model-package/metadata.json",
        "worker/cuda-runtime/cudart64_13.dll",
        "worker/cuda-runtime/cublas64_13.dll",
        "worker/cuda-runtime/cudnn64_9.dll",
        "worker/cuda-runtime/nvJitLink_130_0.dll",
        "worker/cuda-runtime/nvrtc64_130_0.dll",
        "worker/cuda-runtime/nvrtc-builtins64_130.dll",
    }
    metadata_path = root / "worker/model-package/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_checksums = metadata.get("checksums")
    if not isinstance(model_checksums, dict) or not model_checksums:
        raise ValueError("Windows bundle model package has no locked model files")
    required.update(f"worker/model-package/{filename}" for filename in model_checksums)
    if not required <= declared:
        raise ValueError("Windows Release bundle is missing Worker, models, or CUDA runtime")
    if sha256_file(metadata_path) != composition.model_package_metadata.sha256:
        raise ValueError("Windows bundle model package does not match release composition")
    return {"file_count": len(rows), "required_files_present": True}


def _verify_manual_waiver(
    root: Path,
    composition: ReleaseComposition,
    *,
    metadata_path: Path,
    package_path: Path,
) -> dict[str, Any]:
    lock = composition.manual_waiver
    if lock is None:
        raise ValueError("manual waiver lock is missing")
    waiver_path = _verify_artifact(root, lock)
    waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    promotion = metadata.get("promotion", {})
    package_waivers = promotion.get("waivers", [])
    record_waivers = waiver.get("waivers", [])
    package_gates = {
        str(row.get("gate")) for row in package_waivers if isinstance(row, dict) and row.get("gate")
    }
    record_gates = {
        str(row.get("gate")) for row in record_waivers if isinstance(row, dict) and row.get("gate")
    }
    failed_checks = {
        str(name) for name, passed in waiver.get("checks", {}).items() if passed is False
    }
    versions = composition.versions
    record_versions = waiver.get("versions", {})
    expected_package_path = package_path.relative_to(root).as_posix()
    if (
        composition.schema_version != "1.1"
        or metadata.get("promotion_status") != "production"
        or promotion.get("decision") != "approved"
        or promotion.get("method") != "manual_waiver"
        or waiver.get("promotion_method") != "owner_approved_known_limitations"
        or waiver.get("release") != composition.release
        or waiver.get("dataset_version") != versions.dataset
        or waiver.get("production_metadata_sha256") != sha256_file(metadata_path)
        or waiver.get("production_package", {}).get("path") != expected_package_path
        or waiver.get("production_package", {}).get("sha256_directory")
        != sha256_directory(package_path)
        or record_versions.get("worker_version") != versions.worker
        or record_versions.get("detector_version") != versions.detector
        or record_versions.get("classifier_version") != versions.classifier
        or not package_gates
        or package_waivers != record_waivers
        or package_gates != record_gates
        or package_gates != set(waiver.get("failures", []))
        or package_gates != failed_checks
    ):
        raise ValueError("manual waiver attestation does not match the production package")
    return {
        "path": waiver_path.relative_to(root).as_posix(),
        "waived_gates": sorted(package_gates),
        "passed": True,
    }


def verify_release_composition(
    composition: ReleaseComposition,
    *,
    repository_root: Path,
    require_bundle: bool = False,
) -> dict[str, Any]:
    root = repository_root.resolve()
    package_path = _verify_artifact(root, composition.model_package, directory=True)
    metadata_path = _verify_artifact(root, composition.model_package_metadata)
    if metadata_path != package_path / "metadata.json":
        raise ValueError("composition package metadata must belong to the package directory")
    package = load_model_package(package_path)
    versions = composition.versions
    if __version__ != versions.python:
        raise ValueError("Python package version does not match release composition")
    if (
        package.metadata.worker_version != versions.worker
        or package.metadata.detector.version != versions.detector
        or package.metadata.classifier.version != versions.classifier
        or package.metadata.dataset_version != versions.dataset
    ):
        raise ValueError("model package versions do not match release composition")
    if composition.training_contracts is not None:
        for component, lock in (
            ("detector", composition.training_contracts.detector),
            ("classifier", composition.training_contracts.classifier),
        ):
            contract_path = _verify_artifact(root, lock)
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
            if payload.get("component") != component:
                raise ValueError("training contract component mismatch")
            expected = getattr(versions, f"{component}_training_pipeline")
            if payload.get("pipeline_version") != expected:
                raise ValueError("training pipeline version mismatch")
            contract_sha256 = hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            source = package.metadata.sources.get(component)
            dataset = payload.get("dataset")
            if not isinstance(dataset, dict):
                raise ValueError("training contract dataset lock is missing")
            manifest_path = _path(root, str(dataset.get("manifest_path", "")))
            if not manifest_path.is_file():
                raise ValueError("training contract manifest is missing")
            if (
                source is None
                or source.training_pipeline_version != expected
                or source.training_contract_sha256 != contract_sha256
                or source.training_dataset_version != dataset.get("dataset_version")
                or source.training_manifest_sha256 != sha256_file(manifest_path)
            ):
                raise ValueError("model package training provenance does not match composition")
        provenance = {"mode": "training_contracts", "passed": True}
    else:
        provenance = _verify_manual_waiver(
            root,
            composition,
            metadata_path=metadata_path,
            package_path=package_path,
        ) | {"mode": "manual_waiver"}
    pubspec = (root / "apps" / "product_scanner" / "pubspec.yaml").read_text(encoding="utf-8")
    if not re.search(rf"^version:\s*{re.escape(versions.app)}\s*$", pubspec, flags=re.MULTILINE):
        raise ValueError("Flutter app version does not match release composition")
    bundle = composition.windows_bundle_manifest
    if require_bundle and bundle is None:
        raise ValueError("Windows bundle manifest is required")
    if bundle is not None:
        bundle_path = _verify_artifact(root, bundle)
        bundle_result = _verify_bundle_manifest(bundle_path, composition)
    else:
        bundle_result = None
    return {
        "release": composition.release,
        "composition_sha256": canonical_composition_sha256(composition),
        "worker_version": versions.worker,
        "detector_version": versions.detector,
        "classifier_version": versions.classifier,
        "app_version": versions.app,
        "provenance": provenance,
        "bundle_locked": bundle is not None,
        "bundle": bundle_result,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a BIXOLON release composition")
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--require-bundle", action="store_true")
    args = parser.parse_args()
    result = verify_release_composition(
        load_release_composition(args.composition),
        repository_root=args.repository_root,
        require_bundle=args.require_bundle,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
