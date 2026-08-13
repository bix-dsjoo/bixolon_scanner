from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .calibration import binomial_rate_upper_bound
from .ten_shot_evaluation import (
    PromotionGates,
    evaluate_promotion_gates,
    write_promotion_report,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _development_metrics(decision: dict[str, Any]) -> dict[str, Any]:
    metrics = decision["metrics"]
    sample_count = int(metrics["sample_count"])
    approved_count = int(metrics["approval_count"])
    approved_correct = round(float(metrics["approved_precision"]) * approved_count)
    top3_correct = round(float(metrics["overall_top3_accuracy"]) * sample_count)
    unknown_count = sample_count - approved_count
    if approved_correct != approved_count or unknown_count <= 0:
        raise ValueError(
            "development UNKNOWN Top-3 can only be derived when every approved sample "
            "is correct and at least one unknown sample is present"
        )
    unknown_top3_correct = top3_correct - approved_count
    return {
        "top1_accuracy": metrics["top1_accuracy"],
        "overall_top3_accuracy": metrics["overall_top3_accuracy"],
        "approved_count": approved_count,
        "approved_precision": metrics["approved_precision"],
        "false_approval_rate_upper_95": metrics["false_approval_rate_upper_95"],
        "approval_coverage": metrics["approval_coverage"],
        "unknown_top3_accuracy": unknown_top3_correct / unknown_count,
        "recapture_recall": 1.0,
        "risk_certification_maximum_count": sample_count,
    }


def _test_94_metrics(report: dict[str, Any]) -> dict[str, Any]:
    classifier = report["classifier_on_detector_crops"]
    return {
        "top1_accuracy": classifier["overall_top1_accuracy"],
        "overall_top3_accuracy": classifier["overall_top3_accuracy"],
        "approved_count": classifier["approved_count"],
        "approved_precision": classifier["approved_precision"],
        "false_approval_rate_upper_95": classifier["approved_false_rate_upper_95"],
        "approval_coverage": classifier["approval_coverage_of_classified_detections"],
        "unknown_top3_accuracy": classifier["unknown_top3_accuracy"],
        "recapture_recall": report["detector"]["recall"],
        "risk_certification_maximum_count": classifier["matched_sample_count"],
    }


def _bread_project_2_metrics(report: dict[str, Any]) -> dict[str, Any]:
    overall = report["overall"]
    rates = overall["rates"]
    classified = int(overall["classified_matched_boxes"])
    approved = int(overall["approved_boxes"])
    approved_correct = int(overall["approved_correct"])
    approved_errors = approved - approved_correct
    # APPROVED rows do not retain Top-3 in this diagnostic schema. Counting only
    # correct Top-1 approvals is therefore a conservative Top-3 lower bound.
    conservative_top3_correct = approved_correct + int(overall["unknown_top3_correct"])
    return {
        "top1_accuracy": rates["classifier_top1_accuracy_excluding_recapture"],
        "overall_top3_accuracy": conservative_top3_correct / classified,
        "approved_count": approved,
        "approved_precision": rates["approved_accuracy"],
        "false_approval_rate_upper_95": binomial_rate_upper_bound(approved_errors, approved),
        "approval_coverage": approved / classified,
        "unknown_top3_accuracy": rates["unknown_top3_accuracy"],
        "recapture_recall": rates["detector_box_success_rate"],
        "risk_certification_maximum_count": classified,
    }


def build_final_report(
    *,
    development_decision: dict[str, Any],
    parity_report: dict[str, Any],
    benchmark_report: dict[str, Any],
    test_94_report: dict[str, Any],
    bread_project_2_report: dict[str, Any],
    baseline_metadata: dict[str, Any],
    candidate_metadata: dict[str, Any],
    baseline_detector_sha256: str,
    candidate_detector_sha256: str,
) -> dict[str, Any]:
    parity_checks = parity_report["checks"]
    exact_outputs = all(
        parity_checks[name]
        for name in ("top1_equal", "top3_set_and_order_equal", "final_state_equal")
    )
    report = evaluate_promotion_gates(
        candidate=_development_metrics(development_decision),
        baseline={"approval_coverage": baseline_metadata["calibration"]["approval_coverage"]},
        parity={
            "pytorch_cpu_pass": bool(parity_checks["pytorch_cpu_tolerance"] and exact_outputs),
            "pytorch_cuda_pass": bool(parity_checks["pytorch_cuda_tolerance"] and exact_outputs),
            "cross_provider_pass": bool(parity_checks["cpu_cuda_tolerance"] and exact_outputs),
        },
        benchmark={"full_path_p95_ms": benchmark_report["by_path"]["full_path"]["p95_ms"]},
        detector={
            "onnx_frozen": baseline_detector_sha256 == candidate_detector_sha256,
            "metadata_frozen": all(
                baseline_metadata[name] == candidate_metadata[name]
                for name in ("input", "detector", "quality")
            ),
        },
        regression_sets={
            "test_94": _test_94_metrics(test_94_report),
            "bread_project_2_300": _bread_project_2_metrics(bread_project_2_report),
        },
        gates=PromotionGates(),
    )
    report["evidence"] = {
        "pretest_lock": parity_report["pretest_lock"],
        "package_version": benchmark_report["package_version"],
        "classifier_onnx_sha256": benchmark_report["package_artifact_sha256"]["classifier.onnx"],
        "baseline_detector_onnx_sha256": baseline_detector_sha256,
        "candidate_detector_onnx_sha256": candidate_detector_sha256,
        "parity_sample_count": parity_report["sample_count"],
        "benchmark_sample_count": benchmark_report["sample_count"],
        "regression_sets": {
            "test_94": _test_94_metrics(test_94_report),
            "bread_project_2_300": _bread_project_2_metrics(bread_project_2_report),
        },
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize a locked strict 10-shot classifier promotion decision"
    )
    parser.add_argument("--development-decision", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--test-94-report", type=Path, required=True)
    parser.add_argument("--bread-project-2-report", type=Path, required=True)
    parser.add_argument("--baseline-package", type=Path, required=True)
    parser.add_argument("--candidate-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    baseline_metadata = _load(args.baseline_package / "metadata.json")
    candidate_metadata = _load(args.candidate_package / "metadata.json")
    report = build_final_report(
        development_decision=_load(args.development_decision),
        parity_report=_load(args.parity_report),
        benchmark_report=_load(args.benchmark_report),
        test_94_report=_load(args.test_94_report),
        bread_project_2_report=_load(args.bread_project_2_report),
        baseline_metadata=baseline_metadata,
        candidate_metadata=candidate_metadata,
        baseline_detector_sha256=_sha256(args.baseline_package / "detector.onnx"),
        candidate_detector_sha256=_sha256(args.candidate_package / "detector.onnx"),
    )
    write_promotion_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
