from __future__ import annotations

import numpy as np

from bixolon_scanner.evaluation.bread_locked_candidate import build_final_metrics


def test_final_metrics_use_all_gt_denominator_and_inclusive_thresholds() -> None:
    records = [
        {
            "image_id": 1,
            "annotations": [
                {"bbox_xywh": [0.0, 0.0, 10.0, 10.0], "category_id": 1},
                {"bbox_xywh": [20.0, 20.0, 10.0, 10.0], "category_id": 2},
            ],
        }
    ]
    predictions = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0.0, 0.0, 10.0, 10.0]],
            "scores": [0.9],
            "class_ids": [0],
        }
    ]
    score_rows = [{"image_id": 1, "detection_index": 0}]
    logits = np.asarray([[2.0, 1.0, 0.0]], dtype=np.float32)

    metrics, decisions, errors = build_final_metrics(
        records,
        predictions,
        score_rows,
        logits,
        approval_thresholds=[1.0, None, None],
        approval_default_threshold=0.0,
    )

    assert metrics["counts"]["judgeable_ground_truth_object_count"] == 2
    assert metrics["counts"]["approved_count"] == 1
    assert metrics["rates"]["end_to_end_approved_object_rate"] == 0.5
    assert metrics["rates"]["segmentation_image_false_negative_rate"] == 1.0
    assert decisions[0]["status"] == "APPROVED"
    assert errors[0]["false_negative_count"] == 1


def test_final_metrics_count_only_unknown_top3_candidate_out() -> None:
    records = [
        {
            "image_id": 2,
            "annotations": [
                {"bbox_xywh": [0.0, 0.0, 10.0, 10.0], "category_id": 4},
            ],
        }
    ]
    predictions = [
        {
            "image_id": 2,
            "boxes_xyxy": [[0.0, 0.0, 10.0, 10.0]],
            "scores": [0.9],
            "class_ids": [0],
        }
    ]
    score_rows = [{"image_id": 2, "detection_index": 0}]
    logits = np.asarray([[4.0, 3.9, 3.8, 0.0]], dtype=np.float32)

    metrics, decisions, _ = build_final_metrics(
        records,
        predictions,
        score_rows,
        logits,
        approval_thresholds=[0.2, None, None, None],
        approval_default_threshold=0.0,
    )

    assert decisions[0]["status"] == "UNKNOWN"
    assert metrics["counts"]["unknown_top3_candidate_out_count"] == 1
    assert metrics["rates"]["unknown_top3_candidate_out_rate"] == 1.0
