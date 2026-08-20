from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..contracts.artifact import (
    assert_release_not_revoked,
    canonical_sha256,
    directory_content_manifest,
)
from ..contracts.catalog import load_store_catalog_package, sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2

PRODUCTION_VERSION = "2.0.0"
EXPECTED_WAIVED_GATES = {
    "owner_private_locked_production_test",
    "catalog_hmac_authentication",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _release_lock_sha256(payload: dict[str, Any]) -> str:
    expected = payload.get("lock_sha256")
    body = {key: value for key, value in payload.items() if key != "lock_sha256"}
    actual = canonical_sha256(body)
    if expected != actual:
        raise ValueError("pre-private release lock self-checksum is invalid")
    return actual


def _verify_source(directory: Path, lock: dict[str, Any]) -> None:
    observed = directory_content_manifest(directory)
    if observed["manifest_sha256"] != lock.get("content_manifest_sha256"):
        raise ValueError("promotion source differs from the pre-private release lock")


def _verify_original_finalizer(release_lock: dict[str, Any], repository_root: Path) -> None:
    expected_path = "src/bixolon_scanner/operations/scanner_v2_promote.py"
    matches = [
        row
        for row in release_lock.get("supply_chain", {}).get("private_gate_tools", [])
        if row.get("path") == expected_path
    ]
    path = repository_root / expected_path
    if len(matches) != 1 or matches[0].get("sha256") != sha256_file(path):
        raise ValueError("the locked production finalizer changed after RC.8 was frozen")


def validate_owner_waiver(
    waiver: dict[str, Any], *, release_lock_sha256: str, release_candidate: str
) -> None:
    waived = waiver.get("waivers")
    waived_gates = {str(row.get("gate")) for row in waived} if isinstance(waived, list) else set()
    if (
        waiver.get("schema_version") != "2.0"
        or waiver.get("release") != PRODUCTION_VERSION
        or waiver.get("source_release_candidate") != release_candidate
        or waiver.get("source_release_lock_sha256") != release_lock_sha256
        or waiver.get("decision") != "owner_approved_manual_waiver"
        or waiver.get("approved") is not True
        or waiver.get("production_eligible_by_owner_waiver") is not True
        or waiver.get("independent_certified") is not False
        or waiver.get("catalog_authentication") != "CHECKSUM-SHA256"
        or set(waiver.get("failures", [])) != EXPECTED_WAIVED_GATES
        or waived_gates != EXPECTED_WAIVED_GATES
    ):
        raise ValueError("Scanner 2.0 owner waiver is incomplete or targets another candidate")


def promote_runtime_checksum_only(source: Path, target: Path, release_candidate: str) -> None:
    shutil.copytree(source, target)
    metadata_path = target / "metadata.json"
    metadata = _read_json(metadata_path)
    if (
        metadata.get("worker_version") != release_candidate
        or metadata.get("promotion_status") != "independent_test_pending"
    ):
        raise ValueError("runtime source is not the evaluated pending release candidate")
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
        raise ValueError("Catalog source is not the evaluated release candidate")
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
    release_lock = _read_json(args.release_lock)
    lock_sha256 = _release_lock_sha256(release_lock)
    assert_release_not_revoked(args.release_lock, lock_sha256)
    if (
        release_lock.get("status") != "owner_private_test_pending"
        or release_lock.get("production_eligible") is not False
        or release_lock.get("remaining_gates") != ["owner_private_locked_production_test"]
    ):
        raise ValueError("release lock is not awaiting exactly the owner-private gate")
    repository_root = args.repository_root.resolve()
    _verify_original_finalizer(release_lock, repository_root)
    waiver = _read_json(args.owner_waiver)
    validate_owner_waiver(
        waiver,
        release_lock_sha256=lock_sha256,
        release_candidate=release_lock["release_candidate"],
    )
    _verify_source(args.runtime, release_lock["artifacts"]["runtime"])
    _verify_source(args.catalog, release_lock["artifacts"]["catalog"])

    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"production output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".scanner-2.0-owner-waiver-", dir=output.parent))
    try:
        runtime_target = staging / "runtime"
        catalog_target = staging / "catalog"
        promote_runtime_checksum_only(
            args.runtime, runtime_target, release_lock["release_candidate"]
        )
        promote_catalog_checksum_only(
            args.catalog, catalog_target, release_lock["release_candidate"]
        )
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
            "source_release_candidate": release_lock["release_candidate"],
            "source_release_lock_sha256": lock_sha256,
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
        description="Promote the exact Scanner RC with explicit owner and checksum-only waivers"
    )
    parser.add_argument("--release-lock", type=Path, required=True)
    parser.add_argument("--owner-waiver", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    promote(parser.parse_args(argv))


if __name__ == "__main__":
    main()
