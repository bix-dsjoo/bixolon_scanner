"""Promote a locked Worker candidate with an explicit, auditable owner decision."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..contracts.model_package import load_model_package, sha256_file

WAIVED_CHECK = "approved_misrecognition_risk_upper_95"
WAIVER_GATE = "approved_misrecognition_rate_upper_95"
OWNER_WAIVER_GATES = {
    "classifier_training_source_restriction",
    "evaluation_set_independence",
}


def production_metadata(
    candidate_metadata: dict[str, Any],
    report: dict[str, Any],
    *,
    decided_on: str,
) -> dict[str, Any]:
    """Build production metadata only for the narrowly approved risk exception."""
    if candidate_metadata.get("promotion_status") != "development":
        raise ValueError("only a development candidate can be promoted")
    checks = report.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("release report checks are missing")
    failed_checks = sorted(name for name, passed in checks.items() if passed is not True)
    if failed_checks != [WAIVED_CHECK]:
        raise ValueError(f"unexpected release gate failures: {failed_checks}")
    if report.get("failures") != [WAIVED_CHECK]:
        raise ValueError("release report failure list does not match the approved exception")

    risk = report.get("risk_evidence")
    if not isinstance(risk, dict):
        raise ValueError("release report risk evidence is missing")
    approved_count = int(risk.get("approved_sample_count", 0))
    observed_errors = int(risk.get("observed_error_count", -1))
    upper_95 = float(risk.get("upper_95", 1.0))
    if approved_count < 1 or observed_errors != 0:
        raise ValueError("statistical-risk promotion requires zero observed approved errors")

    targets = report.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("release report targets are missing")
    target = float(targets.get("maximum_misrecognition_rate", -1.0))
    if target <= 0.0 or upper_95 <= target:
        raise ValueError("the statistical exception is absent or no longer needed")

    if report.get("dataset_version") != candidate_metadata.get("dataset_version"):
        raise ValueError("release report and package dataset versions differ")
    versions = report.get("versions")
    if not isinstance(versions, dict):
        raise ValueError("release report versions are missing")
    expected_versions = {
        "worker_version": candidate_metadata.get("worker_version"),
        "detector_version": candidate_metadata.get("detector", {}).get("version"),
        "classifier_version": candidate_metadata.get("classifier", {}).get("version"),
    }
    if versions != expected_versions:
        raise ValueError("release report and package model versions differ")
    if report.get("gate_dataset") != "multi_object_scenes":
        raise ValueError("production decision must use multi_object_scenes")
    effective = report.get("effective_configuration")
    if not isinstance(effective, dict) or effective.get("jpeg_draft_size_overridden") is not False:
        raise ValueError("release report used an unsupported JPEG draft override")
    if effective.get("approval_threshold_overridden") is not False:
        raise ValueError("release report used an unsupported approval threshold override")

    metadata = copy.deepcopy(candidate_metadata)
    metadata["promotion_status"] = "production"
    metadata["promotion"] = {
        "decision": "approved",
        "method": "manual_waiver",
        "decided_on": decided_on,
        "waivers": [
            {
                "gate": WAIVER_GATE,
                "observed": upper_95,
                "target": target,
                "sample_count": approved_count,
                "correct_count": approved_count,
                "reason": (
                    "The project owner explicitly approved production deployment after "
                    "0 observed errors and all point, segmentation, recapture, and latency "
                    "gates passed; independent post-deployment validation remains required."
                ),
            }
        ],
        "remaining_limitations": [
            (
                f"The one-sided 95% upper bound is {upper_95:.8f}, above the "
                f"{target:.8f} target because only {approved_count} approved samples are available."
            ),
            "multi_object_scenes is derived from available bread_dataset source imagery.",
            "Independent user-image validation is still required after deployment.",
        ],
    }
    return metadata


def owner_waiver_metadata(
    candidate_metadata: dict[str, Any],
    report: dict[str, Any],
    *,
    decided_on: str,
) -> dict[str, Any]:
    """Build production metadata for the explicit Bread 1.1.0 bridge waiver."""
    if candidate_metadata.get("promotion_status") != "development":
        raise ValueError("only a development candidate can be promoted")
    if report.get("promotion_method") != "owner_approved_known_limitations":
        raise ValueError("owner waiver promotion method is missing")
    if report.get("source_candidate_id") != "bread-zero-error-1.1.0-domain-lda-fixed-four-v3":
        raise ValueError("owner waiver is restricted to the Bread 1.1.0 v3 candidate")

    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise ValueError("owner waiver checks are missing")
    if any(not isinstance(passed, bool) for passed in checks.values()):
        raise ValueError("owner waiver checks must be boolean")
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    if set(failed_checks) != OWNER_WAIVER_GATES:
        raise ValueError(f"unexpected owner waiver gate failures: {failed_checks}")
    if sorted(report.get("failures", [])) != failed_checks:
        raise ValueError("owner waiver failure list does not match checks")

    if report.get("dataset_version") != candidate_metadata.get("dataset_version"):
        raise ValueError("release report and package dataset versions differ")
    versions = report.get("versions")
    expected_versions = {
        "worker_version": candidate_metadata.get("worker_version"),
        "detector_version": candidate_metadata.get("detector", {}).get("version"),
        "classifier_version": candidate_metadata.get("classifier", {}).get("version"),
    }
    if versions != expected_versions:
        raise ValueError("release report and package model versions differ")
    if set(versions.values()) != {"1.1.0"}:
        raise ValueError("owner waiver is restricted to Worker, Detector, and Classifier 1.1.0")

    waivers = report.get("waivers")
    if not isinstance(waivers, list) or {row.get("gate") for row in waivers} != set(failed_checks):
        raise ValueError("owner waiver records do not match failed checks")
    limitations = report.get("remaining_limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(value, str) or not value.strip() for value in limitations)
    ):
        raise ValueError("owner waiver remaining limitations are missing")

    metadata = copy.deepcopy(candidate_metadata)
    metadata["promotion_status"] = "production"
    metadata["promotion"] = {
        "decision": "approved",
        "method": "manual_waiver",
        "decided_on": decided_on,
        "waivers": copy.deepcopy(waivers),
        "remaining_limitations": list(limitations),
    }
    return metadata


def promote_package(
    candidate_dir: Path,
    report_path: Path,
    output_dir: Path,
    *,
    decided_on: str,
    approve_statistical_risk: bool,
    approve_known_limitations: bool = False,
) -> dict[str, Any]:
    if approve_statistical_risk == approve_known_limitations:
        raise ValueError("exactly one explicit promotion approval is required")
    if candidate_dir.resolve() == output_dir.resolve():
        raise ValueError("production output must differ from the candidate directory")
    if output_dir.exists():
        raise FileExistsError(f"production version directory already exists: {output_dir}")

    candidate = load_model_package(candidate_dir)
    candidate_metadata = json.loads((candidate_dir / "metadata.json").read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if approve_known_limitations:
        expected_metadata_sha256 = report.get("source_candidate_metadata_sha256")
        if expected_metadata_sha256 != sha256_file(candidate_dir / "metadata.json"):
            raise ValueError("owner waiver candidate metadata checksum differs")
        metadata = owner_waiver_metadata(candidate_metadata, report, decided_on=decided_on)
    else:
        metadata = production_metadata(candidate_metadata, report, decided_on=decided_on)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.promoting-", dir=output_dir.parent)
    )
    try:
        package_files = [*candidate.detector_paths, candidate.classifier_path]
        if candidate.count_verifier_path is not None:
            package_files.append(candidate.count_verifier_path)
        for source in dict.fromkeys(package_files):
            shutil.copy2(source, temporary / source.name)
        metadata_path = temporary / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        expected_production_sha256 = report.get("production_metadata_sha256")
        if (
            expected_production_sha256 is not None
            and sha256_file(metadata_path) != expected_production_sha256
        ):
            raise ValueError("promoted metadata checksum differs from release decision")
        promoted = load_model_package(temporary)
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    promoted = load_model_package(output_dir)
    waived_gates = [waiver["gate"] for waiver in metadata["promotion"]["waivers"]]
    result = {
        "package_dir": str(output_dir),
        "promotion_status": promoted.metadata.promotion_status,
        "worker_version": promoted.metadata.worker_version,
        "detector_version": promoted.metadata.detector.version,
        "classifier_version": promoted.metadata.classifier.version,
        "dataset_version": promoted.metadata.dataset_version,
        "waived_gates": waived_gates,
    }
    if len(waived_gates) == 1:
        result["waived_gate"] = waived_gates[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a locked Worker package with an explicit owner approval record"
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--release-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--decided-on",
        default=datetime.now(UTC).date().isoformat(),
    )
    approvals = parser.add_mutually_exclusive_group()
    approvals.add_argument("--approve-statistical-risk", action="store_true")
    approvals.add_argument("--approve-known-limitations", action="store_true")
    args = parser.parse_args()
    result = promote_package(
        args.candidate_dir,
        args.release_report,
        args.output_dir,
        decided_on=args.decided_on,
        approve_statistical_risk=args.approve_statistical_risk,
        approve_known_limitations=args.approve_known_limitations,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
