from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..contracts.artifact import canonical_sha256, directory_content_manifest
from ..contracts.catalog import load_store_catalog_package, sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2

PRODUCTION_VERSION = "2.0.1"
EXPECTED_WAIVED_GATES = {
    "accelerated_reliability",
    "approved_misrecognition_statistical_upper_bound",
    "catalog_hmac_authentication",
    "cpu_cuda_full_decision_parity",
    "dependency_vulnerability_scan",
    "development_evidence_independence",
    "one_ips_cadence_performance",
    "owner_private_locked_production_test",
    "sbom_coverage",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_owner_waiver(
    waiver: dict[str, Any],
    *,
    release_candidate: str,
    runtime_manifest_sha256: str,
    catalog_manifest_sha256: str,
) -> None:
    rows = waiver.get("waivers")
    waived_gates = (
        {str(row.get("gate")) for row in rows if isinstance(row, dict)}
        if isinstance(rows, list)
        else set()
    )
    source = waiver.get("source_artifacts", {})
    if (
        waiver.get("schema_version") != "2.0"
        or waiver.get("release") != PRODUCTION_VERSION
        or waiver.get("source_release_candidate") != release_candidate
        or waiver.get("decision") != "owner_approved_manual_waiver"
        or waiver.get("approved") is not True
        or waiver.get("production_eligible_by_owner_waiver") is not True
        or waiver.get("independent_certified") is not False
        or waiver.get("catalog_authentication") != "CHECKSUM-SHA256"
        or source.get("runtime_manifest_sha256") != runtime_manifest_sha256
        or source.get("catalog_manifest_sha256") != catalog_manifest_sha256
        or set(waiver.get("failures", [])) != EXPECTED_WAIVED_GATES
        or waived_gates != EXPECTED_WAIVED_GATES
    ):
        raise ValueError("Scanner 2.0.1 owner waiver is incomplete or targets another candidate")


def promote_runtime_checksum_only(source: Path, target: Path, release_candidate: str) -> None:
    shutil.copytree(source, target)
    metadata_path = target / "metadata.json"
    metadata = _read_json(metadata_path)
    if (
        metadata.get("worker_version") != release_candidate
        or metadata.get("promotion_status") != "development"
    ):
        raise ValueError("runtime source is not the owner-approved development candidate")
    metadata["worker_version"] = PRODUCTION_VERSION
    metadata["promotion_status"] = "production"
    metadata["detector_policy_version"] = PRODUCTION_VERSION
    metadata["detector"]["version"] = PRODUCTION_VERSION
    metadata["embedder"]["version"] = PRODUCTION_VERSION
    metadata["classifier_policy"]["version"] = PRODUCTION_VERSION
    _write_json(metadata_path, metadata)


def promote_catalog_checksum_only(source: Path, target: Path, release_candidate: str) -> None:
    shutil.copytree(source, target)
    metadata_path = target / "catalog.json"
    metadata = _read_json(metadata_path)
    if metadata.get("catalog_version") != release_candidate:
        raise ValueError("Catalog source is not the owner-approved development candidate")
    metadata["authentication"] = "CHECKSUM-SHA256"
    metadata["catalog_version"] = PRODUCTION_VERSION
    metadata["embedder_version"] = PRODUCTION_VERSION
    metadata["classifier_policy_version"] = PRODUCTION_VERSION
    _write_json(metadata_path, metadata)
    checksums_path = target / "checksums.json"
    checksums = _read_json(checksums_path)
    checksums["catalog.json"] = sha256_file(metadata_path)
    _write_json(checksums_path, dict(sorted(checksums.items())))
    signature_path = target / "signature.json"
    if signature_path.exists():
        signature_path.unlink()


def promote(args: argparse.Namespace) -> dict[str, Any]:
    runtime_source_manifest = directory_content_manifest(args.runtime)
    catalog_source_manifest = directory_content_manifest(args.catalog)
    runtime_source = load_runtime_package_v2(args.runtime)
    release_candidate = runtime_source.metadata.worker_version
    waiver = _read_json(args.owner_waiver)
    validate_owner_waiver(
        waiver,
        release_candidate=release_candidate,
        runtime_manifest_sha256=runtime_source_manifest["manifest_sha256"],
        catalog_manifest_sha256=catalog_source_manifest["manifest_sha256"],
    )

    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"production output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".scanner-2.0.1-owner-waiver-", dir=output.parent))
    try:
        runtime_target = staging / "runtime"
        catalog_target = staging / "catalog"
        promote_runtime_checksum_only(args.runtime, runtime_target, release_candidate)
        promote_catalog_checksum_only(args.catalog, catalog_target, release_candidate)
        runtime = load_runtime_package_v2(runtime_target)
        catalog = load_store_catalog_package(
            catalog_target,
            expected_store_id=args.store_id,
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
        if (
            runtime.metadata.promotion_status != "production"
            or catalog.metadata.authentication != "CHECKSUM-SHA256"
            or any(version != PRODUCTION_VERSION for version in versions)
        ):
            raise ValueError("owner-waiver finalization produced an invalid version composition")
        shutil.copy2(args.owner_waiver, staging / "owner-waiver.json")
        attestation = {
            "schema_version": "2.0",
            "release": PRODUCTION_VERSION,
            "status": "production",
            "production_eligible": True,
            "promotion_basis": "owner_manual_waiver",
            "independent_certified": False,
            "source_release_candidate": release_candidate,
            "owner_waiver_sha256": sha256_file(args.owner_waiver),
            "waived_gates": sorted(EXPECTED_WAIVED_GATES),
            "catalog_authentication": "CHECKSUM-SHA256",
            "transformation": {
                "model_graph_or_weight_changed": False,
                "decision_policy_changed": False,
                "allowed_changes": [
                    "component semantic versions",
                    "runtime promotion status",
                    "Catalog metadata checksum",
                    "Catalog authentication from HMAC to checksum-only",
                ],
            },
            "source_artifacts": {
                "runtime_manifest_sha256": runtime_source_manifest["manifest_sha256"],
                "catalog_manifest_sha256": catalog_source_manifest["manifest_sha256"],
            },
            "artifacts": {
                "runtime": directory_content_manifest(runtime_target),
                "catalog": directory_content_manifest(catalog_target),
            },
            "waiver_tool_sha256": sha256_file(Path(__file__)),
        }
        attestation["attestation_sha256"] = canonical_sha256(attestation)
        _write_json(staging / "promotion-attestation.json", attestation)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(attestation, ensure_ascii=False, indent=2))
    return attestation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Promote Scanner 2.0.1 RC.3 with the explicit owner waiver"
    )
    parser.add_argument("--owner-waiver", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--store-id", required=True)
    promote(parser.parse_args(argv))


if __name__ == "__main__":
    main()
