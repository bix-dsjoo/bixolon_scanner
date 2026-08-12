from __future__ import annotations

from bixolon_scanner.training.ten_shot_finalize import build_final_report


def test_locked_regression_and_latency_failures_prevent_promotion():
    development = {
        "metrics": {
            "sample_count": 100,
            "top1_accuracy": 0.98,
            "overall_top3_accuracy": 0.99,
            "approval_count": 85,
            "approval_coverage": 0.85,
            "approved_precision": 1.0,
            "false_approval_rate_upper_95": 0.004,
        }
    }
    parity = {
        "checks": {
            "pytorch_cpu_tolerance": True,
            "pytorch_cuda_tolerance": True,
            "cpu_cuda_tolerance": True,
            "top1_equal": True,
            "top3_set_and_order_equal": True,
            "final_state_equal": True,
        },
        "pretest_lock": "lock.json",
        "sample_count": 100,
    }
    benchmark = {
        "package_version": "0.2.3",
        "package_artifact_sha256": {"classifier.onnx": "classifier"},
        "sample_count": 1000,
        "by_path": {"full_path": {"p95_ms": 118.0}},
    }
    test_94 = {
        "detector": {"recall": 0.99},
        "classifier_on_detector_crops": {
            "matched_sample_count": 100,
            "overall_top1_accuracy": 0.93,
            "overall_top3_accuracy": 0.98,
            "approved_count": 70,
            "approved_precision": 0.99,
            "approved_false_rate_upper_95": 0.02,
            "approval_coverage_of_classified_detections": 0.78,
            "unknown_top3_accuracy": 0.94,
        },
    }
    bread_2 = {
        "overall": {
            "classified_matched_boxes": 100,
            "approved_boxes": 85,
            "approved_correct": 85,
            "unknown_top3_correct": 15,
            "rates": {
                "classifier_top1_accuracy_excluding_recapture": 0.96,
                "approved_accuracy": 1.0,
                "unknown_top3_accuracy": 1.0,
                "detector_box_success_rate": 0.995,
            },
        }
    }
    baseline = {
        "input": {},
        "detector": {},
        "quality": {},
        "calibration": {"approval_coverage": 0.97},
    }
    report = build_final_report(
        development_decision=development,
        parity_report=parity,
        benchmark_report=benchmark,
        test_94_report=test_94,
        bread_project_2_report=bread_2,
        baseline_metadata=baseline,
        candidate_metadata=baseline,
        baseline_detector_sha256="detector",
        candidate_detector_sha256="detector",
    )

    assert report["promotion_status"] == "experiment_only"
    assert report["checks"]["full_path_latency"] is False
    assert report["checks"]["test_94:top1_floor"] is False
    assert report["checks"]["test_94:approval_risk_certification_feasible"] is False
    assert report["evidence"]["regression_sets"]["test_94"][
        "approval_coverage"
    ] == 0.78
