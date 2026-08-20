from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .. import __version__
from ..contracts.artifact import canonical_sha256, directory_content_manifest
from ..contracts.catalog import CatalogState, load_store_catalog_package, sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\]+)")


def _canonical_sha256(value: object) -> str:
    return canonical_sha256(value)


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"release evidence is outside the repository: {resolved}") from exc


def _file_lock(root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"release evidence file is missing: {path}")
    return {
        "path": _relative(root, path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _directory_lock(root: Path, path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"release artifact directory is missing: {path}")
    files = [
        _file_lock(root, candidate)
        for candidate in sorted(item for item in path.rglob("*") if item.is_file())
    ]
    if not files:
        raise ValueError(f"release artifact directory is empty: {path}")
    content = directory_content_manifest(path)
    return {
        "path": _relative(root, path),
        "file_count": len(files),
        "files": files,
        "manifest_sha256": _canonical_sha256(files),
        "content_manifest_sha256": content["manifest_sha256"],
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid release evidence JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"release evidence must be a JSON object: {path}")
    return payload


def parse_requirement_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _PIN.match(line)
        if match is None:
            continue
        name, version = match.groups()
        normalized = name.lower().replace("_", "-")
        if normalized in pins:
            raise ValueError(f"duplicate dependency pin: {normalized}")
        pins[normalized] = version
    if not pins:
        raise ValueError("dependency lock contains no exact pins")
    return pins


def validate_isolated_build_environment(
    distributions: dict[str, str], requirement_pins: dict[str, str]
) -> dict[str, str]:
    normalized = {
        name.lower().replace("_", "-"): version for name, version in distributions.items()
    }
    mismatches = {
        name: {"required": version, "observed": normalized.get(name)}
        for name, version in requirement_pins.items()
        if normalized.get(name) != version
    }
    if mismatches:
        raise ValueError(
            f"isolated Worker build environment differs from runtime lock: {mismatches}"
        )
    build_only = {
        "altgraph",
        "pefile",
        "pip",
        "pyinstaller",
        "pyinstaller-hooks-contrib",
        "pywin32-ctypes",
        "setuptools",
    }
    unexpected = set(normalized) - set(requirement_pins) - build_only
    if unexpected:
        raise ValueError(f"isolated Worker build environment has unexpected packages: {unexpected}")
    if (
        normalized.get("pyinstaller") != "6.11.1"
        or normalized.get("pyinstaller-hooks-contrib") != "2026.6"
    ):
        raise ValueError("Worker build requires the locked PyInstaller toolchain")
    return {name: normalized[name] for name in sorted(build_only) if name in normalized}


def inspect_build_environment(python_executable: Path) -> dict[str, str]:
    script = (
        "import importlib.metadata,json;"
        "print(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions() "
        "if d.metadata['Name']}))"
    )
    completed = subprocess.run(
        [str(python_executable.resolve()), "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("could not inspect the isolated Worker build environment")
    return {str(name): str(version) for name, version in value.items()}


def _validate_evidence(
    *,
    version: str,
    runtime_metadata_sha256: str,
    catalog_metadata_sha256: str,
    development: dict[str, Any],
    parity: dict[str, Any],
    embedder_parity: dict[str, Any],
    worker_smoke: dict[str, Any],
    packaged_worker_smoke: dict[str, Any],
    worker_artifact_content_manifest_sha256: str,
    reliability: dict[str, Any],
    vulnerability_scan: dict[str, Any],
    requirement_pins: dict[str, str],
    sbom: dict[str, Any],
) -> dict[str, bool]:
    versions = development.get("versions", {})
    checks = {
        "development_300_gates": (
            development.get("evaluation") == "scanner_2_0_development_300"
            and development.get("promotion_evidence") is False
            and development.get("dataset", {}).get("image_count") == 300
            and development.get("development_gates", {}).get("all_met") is True
            and all(
                versions.get(component) == version
                for component in (
                    "worker",
                    "detector",
                    "embedder",
                    "classifier_policy",
                    "catalog",
                )
            )
        ),
        "cpu_cuda_parity": (
            parity.get("evaluation") == "scanner_2_0_runtime_cpu_cuda_parity"
            and parity.get("image_count") == 300
            and parity.get("final_status_class_rank_parity_exact") is True
            and parity.get("passes") is True
        ),
        "embedder_parity": (
            embedder_parity.get("evaluation") == "scanner_2_0_embedder_parity"
            and embedder_parity.get("passes") is True
        ),
        "real_worker_smoke": (
            worker_smoke.get("evaluation") == "scanner_2_0_real_worker_smoke"
            and worker_smoke.get("runtime_metadata_sha256") == runtime_metadata_sha256
            and worker_smoke.get("catalog_metadata_sha256") == catalog_metadata_sha256
            and worker_smoke.get("passes") is True
        ),
        "packaged_worker_smoke": (
            packaged_worker_smoke.get("evaluation") == "scanner_2_0_packaged_worker_smoke"
            and packaged_worker_smoke.get("worker_artifact_content_manifest_sha256")
            == worker_artifact_content_manifest_sha256
            and packaged_worker_smoke.get("prohibited_runtime_module_path_count") == 0
            and packaged_worker_smoke.get("unlocked_bundled_distribution_count") == 0
            and packaged_worker_smoke.get("passes") is True
        ),
        "accelerated_reliability": (
            reliability.get("evaluation") == "scanner_2_0_accelerated_reliability_gate"
            and reliability.get("runtime_metadata_sha256") == runtime_metadata_sha256
            and reliability.get("catalog_metadata_sha256") == catalog_metadata_sha256
            and reliability.get("request_count", 0) >= 10_000
            and reliability.get("decision_mismatch_image_count") == 0
            and reliability.get("non_200_count") == 0
            and reliability.get("passes") is True
        ),
        "dependency_vulnerability_scan": all(
            not dependency.get("vulns") for dependency in vulnerability_scan.get("dependencies", [])
        )
        and len(vulnerability_scan.get("dependencies", [])) == len(requirement_pins),
        "sbom_coverage": len(sbom.get("components", [])) == len(requirement_pins),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"pre-private release evidence failed: {', '.join(failed)}")
    return checks


def build_pre_private_release_lock(
    *,
    repository_root: Path,
    runtime_dir: Path,
    catalog_dir: Path,
    development_report: Path,
    parity_report: Path,
    embedder_parity_report: Path,
    worker_smoke_report: Path,
    worker_artifact_dir: Path,
    packaged_worker_smoke_report: Path,
    reliability_report: Path,
    requirements_lock: Path,
    vulnerability_scan_path: Path,
    sbom_path: Path,
    license_paths: list[Path],
    gate_tool_paths: list[Path],
    development_identity_manifest_paths: list[Path],
    worker_build_recipe_path: Path,
    worker_build_python: Path,
    output: Path,
    signing_key: bytes,
    store_id: str,
    key_id: str,
) -> dict[str, Any]:
    root = repository_root.resolve()
    runtime = load_runtime_package_v2(runtime_dir)
    catalog = load_store_catalog_package(
        catalog_dir,
        signing_key=signing_key,
        expected_store_id=store_id,
        expected_key_id=key_id,
    )
    version = runtime.metadata.worker_version
    if runtime.metadata.promotion_status != "independent_test_pending":
        raise ValueError("runtime is not locked for the independent private test")
    if not all(
        component == version
        for component in (
            runtime.metadata.detector.version,
            runtime.metadata.embedder.version,
            runtime.metadata.detector_policy_version,
            runtime.metadata.classifier_policy.version,
            catalog.metadata.catalog_version,
            catalog.metadata.embedder_version,
            catalog.metadata.classifier_policy_version,
        )
    ):
        raise ValueError("runtime and Catalog component versions are not composed atomically")
    if catalog.activation.state not in {CatalogState.ACTIVE, CatalogState.ACTIVE_RESTRICTED}:
        raise ValueError("Catalog is not active")

    development = _read_json(development_report)
    parity = _read_json(parity_report)
    embedder_parity = _read_json(embedder_parity_report)
    worker_smoke = _read_json(worker_smoke_report)
    packaged_worker_smoke = _read_json(packaged_worker_smoke_report)
    reliability = _read_json(reliability_report)
    vulnerability_scan = _read_json(vulnerability_scan_path)
    sbom = _read_json(sbom_path)
    requirement_pins = parse_requirement_pins(requirements_lock)
    build_tool_versions = validate_isolated_build_environment(
        inspect_build_environment(worker_build_python), requirement_pins
    )
    runtime_metadata_sha256 = sha256_file(runtime_dir / "metadata.json")
    catalog_metadata_sha256 = sha256_file(catalog_dir / "catalog.json")
    worker_artifact_lock = _directory_lock(root, worker_artifact_dir)
    checks = _validate_evidence(
        version=version,
        runtime_metadata_sha256=runtime_metadata_sha256,
        catalog_metadata_sha256=catalog_metadata_sha256,
        development=development,
        parity=parity,
        embedder_parity=embedder_parity,
        worker_smoke=worker_smoke,
        packaged_worker_smoke=packaged_worker_smoke,
        worker_artifact_content_manifest_sha256=worker_artifact_lock["content_manifest_sha256"],
        reliability=reliability,
        vulnerability_scan=vulnerability_scan,
        requirement_pins=requirement_pins,
        sbom=sbom,
    )
    checks["isolated_worker_build_environment"] = True
    licenses = [_file_lock(root, path) for path in license_paths]
    if len(licenses) < 2:
        raise ValueError("release lock requires the license text and third-party model notice")
    gate_tools = [_file_lock(root, path) for path in gate_tool_paths]
    if len(gate_tools) < 3:
        raise ValueError("release lock requires private preflight, evaluation, and promotion tools")
    development_identities = [
        _file_lock(root, path) for path in development_identity_manifest_paths
    ]
    if len(development_identities) < 3:
        raise ValueError(
            "release lock requires the complete Scanner 2.0 development identity lineage"
        )
    identity_hashes = [row["sha256"] for row in development_identities]
    if len(identity_hashes) != len(set(identity_hashes)):
        raise ValueError("development identity manifests must be unique")
    evidence_paths = {
        "development_300": development_report,
        "cpu_cuda_parity": parity_report,
        "embedder_parity": embedder_parity_report,
        "real_worker_smoke": worker_smoke_report,
        "packaged_worker_smoke": packaged_worker_smoke_report,
        "accelerated_reliability": reliability_report,
    }
    body: dict[str, Any] = {
        "schema_version": "2.0",
        "release_candidate": version,
        "status": "owner_private_test_pending",
        "production_eligible": False,
        "python_distribution_version": __version__,
        "components": {
            "worker": version,
            "detector": runtime.metadata.detector.version,
            "embedder": runtime.metadata.embedder.version,
            "classifier_policy": runtime.metadata.classifier_policy.version,
            "catalog": catalog.metadata.catalog_version,
        },
        "artifacts": {
            "worker": worker_artifact_lock,
            "runtime": _directory_lock(root, runtime_dir),
            "catalog": _directory_lock(root, catalog_dir),
        },
        "evidence": {name: _file_lock(root, path) for name, path in evidence_paths.items()},
        "supply_chain": {
            "python_version": "3.11.9",
            "target_platform": "windows-x86_64-cuda",
            "dependency_count": len(requirement_pins),
            "requirements": _file_lock(root, requirements_lock),
            "vulnerability_scan": _file_lock(root, vulnerability_scan_path),
            "known_vulnerability_count": 0,
            "sbom": _file_lock(root, sbom_path),
            "licenses": licenses,
            "private_gate_tools": gate_tools,
            "worker_build": {
                "recipe": _file_lock(root, worker_build_recipe_path),
                "runtime_dependencies_match_exact_lock": True,
                "build_tool_versions": build_tool_versions,
            },
        },
        "checks": checks,
        "remaining_gates": ["owner_private_locked_production_test"],
        "private_test_policy": {
            "images_are_owner_supplied_and_unseen": True,
            "no_training_calibration_or_threshold_selection": True,
            "production_decision_requires_statistical_gates": True,
            "customer_onboarding_images_are_not_a_promotion_prerequisite": True,
            "required_development_identity_manifests": development_identities,
        },
    }
    body["lock_sha256"] = _canonical_sha256(body)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return body


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lock the Scanner 2.0 pre-private RC")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--embedder-parity-report", type=Path, required=True)
    parser.add_argument("--worker-smoke-report", type=Path, required=True)
    parser.add_argument("--worker-artifact", type=Path, required=True)
    parser.add_argument("--packaged-worker-smoke-report", type=Path, required=True)
    parser.add_argument("--reliability-report", type=Path, required=True)
    parser.add_argument("--requirements-lock", type=Path, required=True)
    parser.add_argument("--vulnerability-scan", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--license", type=Path, action="append", required=True)
    parser.add_argument("--gate-tool", type=Path, action="append", required=True)
    parser.add_argument(
        "--development-identity-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--worker-build-recipe", type=Path, required=True)
    parser.add_argument("--worker-build-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--signing-key-env", default="BIXOLON_CATALOG_SIGNING_KEY")
    args = parser.parse_args(argv)
    signing_key = os.environ.get(args.signing_key_env, "").encode()
    if len(signing_key) < 16:
        raise ValueError("Catalog signing key must contain at least 16 bytes")
    result = build_pre_private_release_lock(
        repository_root=args.repository_root,
        runtime_dir=args.runtime,
        catalog_dir=args.catalog,
        development_report=args.development_report,
        parity_report=args.parity_report,
        embedder_parity_report=args.embedder_parity_report,
        worker_smoke_report=args.worker_smoke_report,
        worker_artifact_dir=args.worker_artifact,
        packaged_worker_smoke_report=args.packaged_worker_smoke_report,
        reliability_report=args.reliability_report,
        requirements_lock=args.requirements_lock,
        vulnerability_scan_path=args.vulnerability_scan,
        sbom_path=args.sbom,
        license_paths=args.license,
        gate_tool_paths=args.gate_tool,
        development_identity_manifest_paths=args.development_identity_manifest,
        worker_build_recipe_path=args.worker_build_recipe,
        worker_build_python=args.worker_build_python,
        output=args.output,
        signing_key=signing_key,
        store_id=args.store_id,
        key_id=args.key_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
