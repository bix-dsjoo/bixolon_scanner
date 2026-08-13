from __future__ import annotations

import json

import pytest

from bixolon_scanner.training.calibration import binomial_rate_upper_bound
from bixolon_scanner.training.selective_detector import (
    DetectorPolicy,
    PolicyEvaluationCache,
    _auc,
    apply_policy,
    assert_no_split_leakage,
    curve_metrics,
    detector_image_diagnostics,
    evaluate_policy,
    match_boxes,
    policy_grid,
    select_candidate,
)


def test_failure_auc_matches_pairwise_wins_and_half_credit_for_ties():
    labels = [True, True, False, False]
    scores = [0.9, 0.5, 0.5, 0.1]

    assert _auc(labels, scores) == pytest.approx(0.875)
    assert _auc([True, True], [0.1, 0.2]) is None


def _record(image_id: int = 1, *, split: str = "development"):
    return {
        "image_id": image_id,
        "width": 100,
        "height": 100,
        "split": split,
        "annotations": [
            {
                "annotation_id": 1,
                "category_id": 1,
                "bbox_xywh": [10.0, 10.0, 40.0, 40.0],
            }
        ],
        "groups": {"difficulty": "E"},
    }


def _prediction(*, box=None, score: float = 0.9, class_id: str = "bread_01"):
    box = box or [10.0, 10.0, 50.0, 50.0]
    return {
        "boxes_xyxy": [box],
        "scores": [score],
        "classifications": {
            "0": {
                "top1_class_id": class_id,
                "top3_class_ids": [class_id, "bread_02", "bread_03"],
                "confidence": 0.99,
                "recapture": False,
            }
        },
    }


def _policy(**updates):
    values = {
        "score_threshold": 0.5,
        "nms_iou_threshold": 0.7,
        "uncertainty_score_threshold": 0.2,
        "uncertainty_min_area_ratio": 0.039,
        "uncertainty_match_iou_threshold": 0.5,
        "min_object_area_ratio": 0.005,
        "max_queries": 300,
    }
    values.update(updates)
    return DetectorPolicy(**values)


@pytest.mark.parametrize(
    "policy",
    [
        _policy(),
        _policy(nms_iou_threshold=0.5),
        _policy(score_threshold=0.8, uncertainty_score_threshold=None),
        _policy(
            score_threshold=0.8,
            uncertainty_score_threshold=0.2,
            uncertainty_min_area_ratio=0.02,
        ),
    ],
)
def test_policy_evaluation_cache_is_exactly_equivalent(policy):
    records = [_record(1), _record(2)]
    predictions = [
        _prediction(),
        {
            "boxes_xyxy": [[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0]],
            "scores": [0.9, 0.3],
            "classifications": {
                "0": {
                    "top1_class_id": "bread_01",
                    "top3_class_ids": ["bread_01", "bread_02", "bread_03"],
                    "confidence": 0.99,
                    "recapture": False,
                },
                "1": {
                    "top1_class_id": "bread_02",
                    "top3_class_ids": ["bread_02", "bread_01", "bread_03"],
                    "confidence": 0.8,
                    "recapture": False,
                },
            },
        },
    ]

    expected = evaluate_policy(records, predictions, policy, approval_threshold=0.95)
    actual = evaluate_policy(
        records,
        predictions,
        policy,
        approval_threshold=0.95,
        cache=PolicyEvaluationCache(predictions),
    )

    assert actual == expected


def test_exact_image_assignment_and_iou_sensitivity():
    applied = apply_policy(_record(), _prediction(), _policy())
    diagnostics = detector_image_diagnostics(applied["detections"], _record()["annotations"])

    assert diagnostics["exact_iou_50"] is True
    assert diagnostics["exact_iou_75"] is True
    assert diagnostics["true_positive"] == 1
    assert diagnostics["false_positive"] == 0
    assert diagnostics["false_negative"] == 0

    shifted = _prediction(box=[10.0, 10.0, 50.0, 40.0])
    diagnostics = detector_image_diagnostics(
        apply_policy(_record(), shifted, _policy())["detections"],
        _record()["annotations"],
    )
    assert diagnostics["exact_iou_50"] is True
    assert diagnostics["exact_iou_75"] is True

    localization = _prediction(box=[10.0, 10.0, 50.0, 35.0])
    diagnostics = detector_image_diagnostics(
        apply_policy(_record(), localization, _policy())["detections"],
        _record()["annotations"],
    )
    assert diagnostics["exact_iou_50"] is True
    assert diagnostics["exact_iou_75"] is False
    assert diagnostics["error_types"]["localization"] == 1


def test_maximum_cardinality_assignment_avoids_greedy_failure():
    record = _record()
    record["annotations"].append(
        {
            "annotation_id": 2,
            "category_id": 2,
            "bbox_xywh": [40.0, 10.0, 40.0, 40.0],
        }
    )
    prediction = {
        "boxes_xyxy": [
            [20.0, 10.0, 70.0, 50.0],
            [10.0, 10.0, 50.0, 50.0],
        ],
        "scores": [0.9, 0.8],
    }
    detections = apply_policy(record, prediction, _policy())["detections"]
    matches = match_boxes(detections, record["annotations"], 0.5)
    assert len(matches) == 2


def test_uncertainty_current_relaxed_and_disabled_policies_match_worker_rules():
    prediction = _prediction()
    prediction["boxes_xyxy"].append([60.0, 60.0, 90.0, 90.0])
    prediction["scores"].append(0.25)

    current = apply_policy(_record(), prediction, _policy())
    relaxed = apply_policy(
        _record(),
        prediction,
        _policy(uncertainty_score_threshold=0.3),
    )
    disabled = apply_policy(_record(), prediction, _policy(uncertainty_score_threshold=None))

    assert current["reason_codes"] == ["DETECTOR_UNCERTAIN_OBJECT"]
    assert relaxed["pass"] is True
    assert disabled["pass"] is True


def test_gate_table_and_e2e_safe_auto_pass_are_image_level():
    records = [_record(index) for index in range(4)]
    predictions = [
        _prediction(),
        _prediction(class_id="bread_02"),
        _prediction(),
        {"boxes_xyxy": [], "scores": [], "classifications": {}},
    ]
    predictions[2]["fixed_hard_reason_codes"] = ["DETECTOR_BLUR"]
    report = evaluate_policy(records, predictions, _policy(), approval_threshold=0.95)["metrics"]

    assert report["gate_table"] == {
        "useful_reject": 1,
        "wasted_reject": 1,
        "silent_failure": 0,
        "safe_pass": 2,
    }
    assert report["approved_count"] == 2
    assert report["approved_error_count"] == 1
    assert report["safe_auto_pass_count"] == 1
    assert report["object_diagnostics"]["error_types"]["border_related"] == 0
    assert report["object_diagnostics"]["error_types"]["size_related"] == 0
    group = report["groups"]["difficulty=E"]
    assert group["detector_pass_count"] == 2
    assert group["detector_pass_risk_upper_95"] == pytest.approx(binomial_rate_upper_bound(0, 2))
    assert group["approved_count"] == 2
    assert group["approved_error_count"] == 1
    assert group["e2e_approved_risk_upper_95"] == pytest.approx(binomial_rate_upper_bound(1, 2))


def test_object_diagnostics_record_border_and_size_related_failures():
    record = _record()
    record["groups"].update({"border_contact": True, "small_object": True})

    metrics = evaluate_policy(
        [record],
        [_prediction(box=[60.0, 60.0, 90.0, 90.0])],
        _policy(),
        approval_threshold=0.95,
    )["metrics"]

    assert metrics["object_diagnostics"]["error_types"]["border_related"] == 1
    assert metrics["object_diagnostics"]["error_types"]["size_related"] == 1


def test_exact_one_sided_binomial_zero_error_sample_sizes():
    assert binomial_rate_upper_bound(0, 300) == pytest.approx(0.009936, rel=1e-3)
    assert binomial_rate_upper_bound(0, 598) <= 0.005
    assert binomial_rate_upper_bound(0, 2995) == pytest.approx(0.001, rel=1e-2)


def _candidate(
    name: str,
    *,
    silent: int,
    detector_u95: float,
    e2e_u95: float,
    safe_rate: float,
    hard_recall: float = 1.0,
    group_coverage: float = 0.8,
    augrc: float = 0.01,
):
    policy = _policy()
    natural_metrics = {
        "gate_table": {"silent_failure": silent},
        "detector_pass_risk_upper_95": detector_u95,
        "e2e_approved_risk_upper_95": e2e_u95,
        "safe_auto_pass_rate": safe_rate,
        "groups": {
            "difficulty=H": {
                "sample_count": 30,
                "approval_coverage": group_coverage,
            }
        },
    }
    return {
        "model_id": name,
        "seed": 20260812,
        "natural": {
            "policy": asdict_for_test(policy),
            "policy_key": policy.key,
            "metrics": natural_metrics,
        },
        "hard": {"metrics": {"error_catch_recall": hard_recall}},
        "augrc": augrc,
    }


def asdict_for_test(policy: DetectorPolicy):
    return json.loads(policy.key)


def test_candidate_selection_rejects_unsafe_high_coverage_then_uses_ties():
    unsafe = _candidate("unsafe", silent=1, detector_u95=0.006, e2e_u95=0.004, safe_rate=0.99)
    lower_worst_group = _candidate(
        "lower-group",
        silent=0,
        detector_u95=0.004,
        e2e_u95=0.004,
        safe_rate=0.9,
        group_coverage=0.7,
        augrc=0.001,
    )
    selected = _candidate(
        "selected",
        silent=0,
        detector_u95=0.004,
        e2e_u95=0.004,
        safe_rate=0.9,
        group_coverage=0.8,
        augrc=0.02,
    )

    decision = select_candidate([unsafe, lower_worst_group, selected])

    assert decision["promotion_status"] == "locked_candidate"
    assert decision["selected"]["model_id"] == "selected"


def test_no_eligible_candidate_is_experiment_only_and_minimizes_silent_failures():
    decision = select_candidate(
        [
            _candidate("two", silent=2, detector_u95=0.1, e2e_u95=0.1, safe_rate=0.9),
            _candidate("one", silent=1, detector_u95=0.2, e2e_u95=0.2, safe_rate=0.8),
        ]
    )
    assert decision["promotion_status"] == "experiment_only"
    assert decision["selected"]["model_id"] == "one"


def test_curve_reports_augrc_and_legacy_aurc():
    curve = curve_metrics(
        [
            {
                "score_threshold": 0.9,
                "sample_count": 100,
                "detector_coverage": 0.5,
                "detector_pass_risk": 0.0,
                "gate_table": {"silent_failure": 0},
            },
            {
                "score_threshold": 0.5,
                "sample_count": 100,
                "detector_coverage": 1.0,
                "detector_pass_risk": 0.1,
                "gate_table": {"silent_failure": 10},
            },
        ]
    )
    assert curve["aurc"] == pytest.approx(0.025)
    assert curve["augrc"] == pytest.approx(0.025)


def test_policy_grid_skips_invalid_uncertainty_thresholds():
    policies = policy_grid(
        {
            "score_thresholds": [0.2, 0.5],
            "nms_iou_thresholds": [0.7],
            "uncertainty_score_thresholds": [None, 0.3],
            "uncertainty_min_area_ratios": [0.039],
            "uncertainty_match_iou_threshold": 0.5,
            "min_object_area_ratio": 0.005,
            "max_queries": 300,
        }
    )
    assert len(policies) == 3
    assert all(
        policy.uncertainty_score_threshold is None
        or policy.uncertainty_score_threshold < policy.score_threshold
        for policy in policies
    )


def test_split_leakage_checks_sha_session_and_physical_target():
    records = [
        {
            "split": "development",
            "image_sha256": "a",
            "capture_session_id": "session-a",
            "physical_target_group_id": "target-a",
        },
        {
            "split": "test",
            "image_sha256": "b",
            "capture_session_id": "session-a",
            "physical_target_group_id": "target-b",
        },
    ]
    with pytest.raises(ValueError, match="split leakage"):
        assert_no_split_leakage(records)


def test_fold_leakage_checks_capture_session_grouping():
    records = [
        {
            "split": "development",
            "fold": 0,
            "capture_session_id": "session-a",
        },
        {
            "split": "development",
            "fold": 1,
            "capture_session_id": "session-a",
        },
    ]
    with pytest.raises(ValueError, match="fold leakage"):
        assert_no_split_leakage(records)
