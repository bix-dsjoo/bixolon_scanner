from __future__ import annotations

from bixolon_scanner.evaluation.recapture_separability import (
    best_detector_policy,
    best_monotonic_or_policy,
    best_single_feature_rules,
    detection_geometry_features,
    normal_envelope_outlier_diagnostic,
)


def test_best_single_feature_rule_obeys_false_recapture_constraint() -> None:
    rows = [
        {
            "is_recapture": False,
            "reason_codes": [],
            "features": {"score": 0.8, "sharpness": 10.0},
        },
        {
            "is_recapture": False,
            "reason_codes": [],
            "features": {"score": 0.9, "sharpness": 11.0},
        },
        {
            "is_recapture": True,
            "reason_codes": ["DETECTOR_NO_OBJECT"],
            "features": {"score": 0.2, "sharpness": 12.0},
        },
        {
            "is_recapture": True,
            "reason_codes": ["DETECTOR_BLUR"],
            "features": {"score": 0.3, "sharpness": 1.0},
        },
    ]

    rules = best_single_feature_rules(rows, maximum_false_recapture_rate=0.01)

    score_rule = next(rule for rule in rules if rule["feature"] == "score")
    assert score_rule["direction"] == "at_or_below"
    assert score_rule["true_recapture_count"] == 2
    assert score_rule["false_recapture_count"] == 0
    assert score_rule["caught_by_reason"]["DETECTOR_BLUR"]["caught"] == 1

    envelope = normal_envelope_outlier_diagnostic(rows)
    assert envelope["true_recapture_count"] == 2
    assert envelope["false_recapture_count"] == 0


def test_detector_policy_returns_none_when_false_recapture_constraint_is_infeasible() -> None:
    rows = [
        {
            "is_recapture": False,
            "reason_codes": [],
            "image_area": 100.0,
        }
    ]

    result = best_detector_policy(
        rows,
        {0.1: [[]], 0.5: [[]]},
        main_thresholds=[0.5],
        shadow_thresholds=[0.1],
        minimum_area_ratios=[0.0],
        match_iou_thresholds=[0.5],
        maximum_false_recapture_rate=0.01,
    )

    assert result is None


def test_detection_geometry_features_capture_overlap_border_and_query_duplicates() -> None:
    prediction = {
        "boxes_xyxy": [
            [0.0, 10.0, 50.0, 60.0],
            [1.0, 11.0, 51.0, 61.0],
            [40.0, 40.0, 90.0, 90.0],
        ],
        "scores": [0.9, 0.8, 0.7],
    }

    features = detection_geometry_features(
        prediction,
        width=100,
        height=100,
        nms_iou_threshold=0.99,
        maximum_aspect_ratio=5.0,
    )

    assert features["nms_0_485_count"] == 3.0
    assert features["nms_0_485_border_fraction"] > 0.0
    assert features["nms_0_485_maximum_pair_iou"] > 0.9
    assert features["query_0_485_maximum_cluster_size"] == 2.0


def test_monotonic_or_policy_combines_complementary_zero_false_rules() -> None:
    rows = [
        {"is_recapture": False, "reason_codes": [], "features": {"score": 0.8, "blur": 9.0}},
        {"is_recapture": False, "reason_codes": [], "features": {"score": 0.9, "blur": 10.0}},
        {
            "is_recapture": True,
            "reason_codes": ["DETECTOR_NO_OBJECT"],
            "features": {"score": 0.2, "blur": 9.0},
        },
        {
            "is_recapture": True,
            "reason_codes": ["DETECTOR_BLUR"],
            "features": {"score": 0.8, "blur": 1.0},
        },
    ]

    policy = best_monotonic_or_policy(rows, maximum_false_recapture_rate=0.0)

    assert len(policy["rules"]) == 2
    assert policy["true_recapture_count"] == 2
    assert policy["false_recapture_count"] == 0
