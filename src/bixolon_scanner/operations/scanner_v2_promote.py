from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..contracts.artifact import canonical_sha256, directory_content_manifest
from ..contracts.catalog import load_store_catalog_package, sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2

PRODUCTION_VERSION = "2.0.1"
EXPECTED_POINT_GATES = {
    "segmentation_rate",
    "correct_approved_rate",
    "wrong_approved_over_all_gt",
    "wrong_approved_over_approved_output",
    "segmentation_image_false_negative_rate",
    "segmentation_image_false_positive_rate",
    "forced_top3_candidate_out_rate",
    "ood_false_approval_rate",
    "image_recapture_recall",
    "unnecessary_image_recapture_rate",
    "invalid_roi_correct_action_recall",
}
EXPECTED_STATISTICAL_GATES = {
    "approval_safety",
    "detector_fn",
    "detector_fp",
    "top3_safety",
    "ood_false_approval",
    "image_recapture_recall",
    "unnecessary_image_recapture",
    "invalid_roi_action",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


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


def _verify_gate_tool(release_lock: dict[str, Any], path: Path) -> None:
    matches = [
        row
        for row in release_lock.get("supply_chain", {}).get("private_gate_tools", [])
        if Path(str(row.get("path", ""))).name == path.name
    ]
    if len(matches) != 1 or matches[0].get("sha256") != sha256_file(path):
        raise ValueError("production finalizer differs from the pre-private release lock")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _promote_runtime(source: Path, target: Path, release_candidate: str) -> None:
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


def _promote_catalog(
    source: Path,
    target: Path,
    *,
    release_candidate: str,
    production_key_id: str,
    production_signing_key: bytes,
) -> None:
    shutil.copytree(source, target)
    metadata_path = target / "catalog.json"
    metadata = _read_json(metadata_path)
    if metadata.get("catalog_version") != release_candidate:
        raise ValueError("Catalog source is not the evaluated release candidate")
    metadata["catalog_version"] = PRODUCTION_VERSION
    metadata["embedder_version"] = PRODUCTION_VERSION
    metadata["classifier_policy_version"] = PRODUCTION_VERSION
    _write_json(metadata_path, metadata)
    checksums_path = target / "checksums.json"
    checksums = _read_json(checksums_path)
    checksums["catalog.json"] = sha256_file(metadata_path)
    _write_json(checksums_path, dict(sorted(checksums.items())))
    digest = hmac.new(
        production_signing_key, checksums_path.read_bytes(), hashlib.sha256
    ).hexdigest()
    _write_json(
        target / "signature.json",
        {
            "algorithm": "HMAC-SHA256",
            "key_id": production_key_id,
            "signed_file": "checksums.json",
            "digest": digest,
        },
    )


def promote(args: argparse.Namespace) -> dict[str, Any]:
    release_lock = _read_json(args.release_lock)
    lock_sha256 = _release_lock_sha256(release_lock)
    _verify_gate_tool(release_lock, Path(__file__))
    private_report = _read_json(args.private_report)
    point_gates = private_report.get("point_gates", {})
    statistical = private_report.get("statistical_certification", {})
    if (
        release_lock.get("status") != "owner_private_test_pending"
        or release_lock.get("production_eligible") is not False
        or release_lock.get("remaining_gates") != ["owner_private_locked_production_test"]
    ):
        raise ValueError("release lock is not awaiting exactly the owner-private gate")
    if (
        private_report.get("evaluation") != "scanner_2_0_owner_private_production_gate"
        or private_report.get("release_lock_sha256") != lock_sha256
        or private_report.get("production_eligible") is not True
        or private_report.get("decision") != "promote_exact_locked_candidate"
        or set(point_gates) != EXPECTED_POINT_GATES
        or not all(value is True for value in point_gates.values())
        or set(statistical) != EXPECTED_STATISTICAL_GATES
        or not all(result.get("passes") is True for result in statistical.values())
    ):
        raise ValueError("owner-private production gate did not authorize promotion")
    signing_key = os.environ.get(args.production_signing_key_env, "").encode()
    if len(signing_key) < 32:
        raise ValueError("production Catalog signing key must contain at least 32 bytes")
    _verify_source(args.runtime, release_lock["artifacts"]["runtime"])
    _verify_source(args.catalog, release_lock["artifacts"]["catalog"])

    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"production output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".scanner-2.0-promote-", dir=output.parent))
    try:
        runtime_target = staging / "runtime"
        catalog_target = staging / "catalog"
        _promote_runtime(args.runtime, runtime_target, release_lock["release_candidate"])
        _promote_catalog(
            args.catalog,
            catalog_target,
            release_candidate=release_lock["release_candidate"],
            production_key_id=args.production_key_id,
            production_signing_key=signing_key,
        )
        runtime = load_runtime_package_v2(runtime_target)
        catalog = load_store_catalog_package(
            catalog_target,
            signing_key=signing_key,
            expected_store_id=args.store_id,
            expected_key_id=args.production_key_id,
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
        if runtime.metadata.promotion_status != "production" or any(
            version != PRODUCTION_VERSION for version in versions
        ):
            raise ValueError("production finalization produced a non-atomic version composition")
        attestation = {
            "schema_version": "2.0",
            "release": PRODUCTION_VERSION,
            "status": "production",
            "production_eligible": True,
            "source_release_candidate": release_lock["release_candidate"],
            "source_release_lock_sha256": lock_sha256,
            "owner_private_report_sha256": sha256_file(args.private_report),
            "transformation": {
                "model_graph_or_weight_changed": False,
                "decision_policy_changed": False,
                "allowed_changes": [
                    "component semantic versions",
                    "runtime promotion status",
                    "Catalog metadata checksum",
                    "Catalog production signature",
                ],
            },
            "artifacts": {
                "runtime": directory_content_manifest(runtime_target),
                "catalog": directory_content_manifest(catalog_target),
            },
            "production_key_id": args.production_key_id,
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
        description="Finalize the exact private-gate-passing Scanner RC as production 2.0.1"
    )
    parser.add_argument("--release-lock", type=Path, required=True)
    parser.add_argument("--private-report", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--production-key-id", required=True)
    parser.add_argument(
        "--production-signing-key-env", default="BIXOLON_PRODUCTION_CATALOG_SIGNING_KEY"
    )
    promote(parser.parse_args(argv))


if __name__ == "__main__":
    main()
