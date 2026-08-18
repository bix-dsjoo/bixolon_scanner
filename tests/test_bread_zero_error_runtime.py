from __future__ import annotations

import numpy as np

from bixolon_scanner.runtime.bread_zero_error import (
    _filter_prediction_by_area,
    _group_suppress,
    _maximum_aspect_ratio_extremity,
    _predictions_agree,
    consensus_agreement_count,
    consensus_is_ambiguous,
    detector_output_to_prediction,
    fuse_prediction_rows,
)


def test_aspect_ratio_extremity_is_orientation_invariant() -> None:
    assert _maximum_aspect_ratio_extremity({"boxes_xyxy": [[0, 0, 20, 10], [0, 0, 10, 25]]}) == 2.5


def test_detector_output_converts_normalized_boxes_and_class_scores() -> None:
    prediction = detector_output_to_prediction(
        np.asarray([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32),
        np.asarray([[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.5, 0.5]]),
        image_width=200,
        image_height=100,
    )

    assert prediction["boxes_xyxy"] == [
        [50.0, 25.0, 150.0, 75.0],
        [0.0, 0.0, 100.0, 50.0],
    ]
    assert prediction["class_ids"] == [1, 0]


def test_prediction_area_filter_preserves_aligned_fields() -> None:
    filtered = _filter_prediction_by_area(
        {
            "boxes_xyxy": [[0, 0, 20, 20], [0, 0, 90, 90]],
            "scores": [0.9, 0.8],
            "class_ids": [1, 2],
            "support_counts": [4, 3],
        },
        image_area=10_000,
        maximum_area_ratio=0.3,
    )

    assert filtered == {
        "boxes_xyxy": [[0, 0, 20, 20]],
        "scores": [0.9],
        "class_ids": [1],
        "support_counts": [4],
    }


def test_fusion_records_cross_model_support() -> None:
    rows = [
        {
            "boxes_xyxy": [[0.0, 0.0, 10.0, 10.0]],
            "scores": [0.9],
            "class_ids": [1],
        },
        {
            "boxes_xyxy": [[1.0, 0.0, 11.0, 10.0]],
            "scores": [0.8],
            "class_ids": [1],
        },
    ]

    fused = fuse_prediction_rows(
        rows,
        model_weights=[1.0, 1.0],
        score_thresholds=[0.02, 0.02],
        pre_nms_iou_threshold=1.0,
        maximum_candidates_per_model=300,
        cluster_iou_threshold=0.5,
    )

    assert fused["support_counts"] == [2]
    assert fused["scores"] == [0.9]


def test_fusion_uses_weighted_class_vote() -> None:
    rows = [
        {"boxes_xyxy": [[0, 0, 10, 10]], "scores": [0.9], "class_ids": [2]},
        {"boxes_xyxy": [[0, 0, 10, 10]], "scores": [0.8], "class_ids": [1]},
        {"boxes_xyxy": [[0, 0, 10, 10]], "scores": [0.7], "class_ids": [1]},
    ]

    fused = fuse_prediction_rows(
        rows,
        model_weights=[1.0, 1.0, 1.0],
        score_thresholds=[0.02, 0.02, 0.02],
        pre_nms_iou_threshold=1.0,
        maximum_candidates_per_model=300,
        cluster_iou_threshold=0.5,
    )

    assert fused["class_ids"] == [1]


def test_prediction_agreement_requires_one_to_one_geometry_matching() -> None:
    primary = {
        "boxes_xyxy": [[0, 0, 10, 10], [20, 0, 30, 10]],
        "scores": [0.9, 0.8],
    }
    recovery = {
        "boxes_xyxy": [[1, 0, 11, 10], [19, 0, 29, 10]],
        "scores": [0.7, 0.6],
    }

    assert _predictions_agree(primary, recovery, iou_threshold=0.4)
    assert not _predictions_agree(
        primary,
        {"boxes_xyxy": [[1, 0, 11, 10]], "scores": [0.7]},
        iou_threshold=0.4,
    )


def test_consensus_flags_a_primary_policy_without_member_agreement() -> None:
    from bixolon_scanner.contracts.model_package import DetectorPolicyConsensusMetadata

    policy = DetectorPolicyConsensusMetadata.model_validate(
        {
            "policies": [
                {
                    "member_filename": "fold0.onnx",
                    "score_threshold": 0.2,
                    "nms_iou_threshold": 0.5,
                    "containment_threshold": 0.95,
                    "group_minimum": 2,
                }
            ],
            "agreement_iou_threshold": 0.4,
            "minimum_agreeing_policy_count": 2,
        }
    )
    base = {
        "boxes_xyxy": [[0, 0, 10, 10], [20, 0, 30, 10]],
        "scores": [0.9, 0.8],
        "class_ids": [1, 2],
    }
    disagreeing = {
        "boxes_xyxy": [[0, 0, 10, 10]],
        "scores": [0.9],
        "class_ids": [1],
    }

    assert consensus_is_ambiguous(base, {"fold0.onnx": disagreeing}, policy)
    assert consensus_agreement_count(base, {"fold0.onnx": disagreeing}, policy) == 1
    assert not consensus_is_ambiguous(base, {"fold0.onnx": base}, policy)
    assert consensus_agreement_count(base, {"fold0.onnx": base}, policy) == 2


def test_group_suppression_counts_only_distinct_inner_candidates() -> None:
    outer = {"box": np.asarray([0, 0, 100, 100]), "score": 0.9, "class_id": 0}
    first = {"box": np.asarray([10, 10, 30, 30]), "score": 0.8, "class_id": 1}
    duplicate = {"box": np.asarray([11, 11, 31, 31]), "score": 0.7, "class_id": 1}
    second = {"box": np.asarray([60, 60, 80, 80]), "score": 0.6, "class_id": 2}

    duplicate_only = _group_suppress(
        [outer, first, duplicate], containment_threshold=0.9, group_minimum=2
    )
    distinct = _group_suppress(
        [outer, first, duplicate, second], containment_threshold=0.9, group_minimum=2
    )

    assert any(candidate is outer for candidate in duplicate_only)
    assert all(candidate is not outer for candidate in distinct)
