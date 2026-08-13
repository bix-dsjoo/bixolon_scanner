from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..imaging import decode_image
from ..inference import build_onnx_adapters
from ..package import load_model_package
from ..pipeline import DecisionPipeline
from .data import read_manifest


def _run_package(
    package_dir: Path,
    records: list[dict[str, Any]],
    dataset_root: Path,
    provider: str,
    cuda_dll_dir: Path | None,
) -> dict[str, Any]:
    package = load_model_package(package_dir)
    detector, classifier, selected_provider = build_onnx_adapters(
        package, provider, cuda_dll_dir=cuda_dll_dir
    )
    pipeline = DecisionPipeline(
        detector,
        classifier,
        package.metadata.classifier,
        package.metadata.quality,
        package.metadata.count_verifier,
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        image = decode_image(
            (dataset_root / record["image_path"]).read_bytes(),
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        response = pipeline.scan(image, request_id=f"operational-{record['scan_id'][:8]}")
        expected_continue = record["expected_detector_action"] == "CONTINUE"
        classifier_executed = response.model_versions.classifier is not None
        continued = response.status.value != "RECAPTURE" and classifier_executed
        rows.append(
            {
                "sequence_index": record["sequence_index"],
                "scan_id": record["scan_id"],
                "expected_detector_action": record["expected_detector_action"],
                "expected_reason_codes": record["expected_reason_codes"],
                "actual_status": response.status.value,
                "actual_reason_codes": response.reason_codes,
                "classifier_executed": classifier_executed,
                "decision_correct": continued
                if expected_continue
                else response.status.value == "RECAPTURE",
                "expected_reason_matched": (
                    not record["expected_reason_codes"]
                    or bool(set(record["expected_reason_codes"]) & set(response.reason_codes))
                ),
            }
        )
    normal = [row for row in rows if row["expected_detector_action"] == "CONTINUE"]
    recapture = [row for row in rows if row["expected_detector_action"] == "RECAPTURE"]
    continued_count = sum(row["decision_correct"] for row in normal)
    recaptured_count = sum(row["decision_correct"] for row in recapture)
    return {
        "package_version": package.metadata.package_version,
        "detector_version": package.metadata.detector.version,
        "classifier_version": package.metadata.classifier.version,
        "provider": selected_provider,
        "normal_count": len(normal),
        "normal_continued_count": continued_count,
        "normal_continue_rate": continued_count / len(normal) if normal else None,
        "false_recapture_count": len(normal) - continued_count,
        "recapture_count": len(recapture),
        "recapture_correct_count": recaptured_count,
        "recapture_recall": recaptured_count / len(recapture) if recapture else None,
        "expected_reason_match_count": sum(row["expected_reason_matched"] for row in recapture),
        "reason_counts": dict(
            sorted(Counter(reason for row in rows for reason in row["actual_reason_codes"]).items())
        ),
        "rows": rows,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    metadata = json.loads((args.manifest.parent / "metadata.json").read_text(encoding="utf-8"))
    records = [
        record
        for record in read_manifest(args.manifest)
        if record.get("source") == "operational_scan_log_v2"
    ]
    if not records:
        raise ValueError("manifest contains no operational scan-log records")
    candidate = _run_package(
        args.package_dir, records, args.dataset_root, args.provider, args.cuda_dll_dir
    )
    baseline = (
        _run_package(
            args.baseline_package_dir,
            records,
            args.dataset_root,
            args.provider,
            args.cuda_dll_dir,
        )
        if args.baseline_package_dir
        else None
    )
    sessions = {record["capture_session_id"] for record in records}
    physical_groups = {record["physical_target_group_id"] for record in records}
    is_fit = args.evidence_role == "fit"
    required_continue_rate = 1.0 if is_fit else 0.9
    strict_reduction = baseline is not None and (
        candidate["false_recapture_count"] < baseline["false_recapture_count"]
    )
    decision_gate = (
        candidate["normal_continue_rate"] is not None
        and candidate["normal_continue_rate"] >= required_continue_rate
        and candidate["recapture_recall"] is not None
        and candidate["recapture_recall"] >= 0.99
        and (is_fit or strict_reduction)
    )
    independent_data_gate = (
        not is_fit
        and candidate["normal_count"] >= 100
        and candidate["recapture_count"] >= 100
        and len(sessions) >= 3
        and len(physical_groups) >= 3
    )
    report = {
        "schema_version": "1.0",
        "dataset_version": metadata["dataset_version"],
        "evidence_role": args.evidence_role,
        "promotion_evidence": not is_fit,
        "sample_count": len(records),
        "capture_session_count": len(sessions),
        "physical_target_group_count": len(physical_groups),
        "candidate": candidate,
        "baseline": baseline,
        "gates": {
            "required_normal_continue_rate": required_continue_rate,
            "normal_continue_gate_satisfied": (
                candidate["normal_continue_rate"] is not None
                and candidate["normal_continue_rate"] >= required_continue_rate
            ),
            "recapture_recall_gate_satisfied": (
                candidate["recapture_recall"] is not None and candidate["recapture_recall"] >= 0.99
            ),
            "strict_false_recapture_reduction": strict_reduction,
            "decision_gate_satisfied": decision_gate,
            "independent_data_gate_satisfied": independent_data_gate,
            "production_promotion_eligible": decision_gate and independent_data_gate,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate operational false-RECAPTURE fit or test data"
    )
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--baseline-package-dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-role", choices=("fit", "independent"), required=True)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
