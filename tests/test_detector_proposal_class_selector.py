from __future__ import annotations

import numpy as np

from bixolon_scanner.experiments.bread.detector_proposal_class_selector import (
    candidate_mask_context,
    filtered_proposal_indices,
    select_class_verified_prediction,
)


def test_filtered_proposal_indices_apply_score_and_support() -> None:
    prediction = {
        "scores": [0.9, 0.01, 0.8],
        "support_counts": [4, 4, 2],
    }
    assert filtered_proposal_indices(prediction, minimum_score=0.02, minimum_support=3) == [0]


def test_candidate_mask_context_replaces_near_duplicate_base_box() -> None:
    boxes, target_index = candidate_mask_context(
        np.asarray([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        np.asarray([0.0, 0.0, 10.0, 10.0]),
    )
    assert boxes == [[20.0, 20.0, 30.0, 30.0], [0.0, 0.0, 10.0, 10.0]]
    assert target_index == 1


def test_select_class_verified_prediction_removes_unsupported_base() -> None:
    base = {
        "image_id": 1,
        "boxes_xyxy": [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
        "scores": [0.9, 0.8],
        "class_ids": [0, 0],
    }
    raw = {
        "image_id": 1,
        "boxes_xyxy": base["boxes_xyxy"],
        "scores": base["scores"],
        "support_counts": [4, 1],
    }
    entries = [
        {
            "proposal_index": 0,
            "box": np.asarray(base["boxes_xyxy"][0], dtype=np.float32),
            "detector_score": 0.9,
            "support_count": 4,
            "predicted_class": 3,
            "class_margin": 100.0,
        }
    ]
    selected, diagnostics = select_class_verified_prediction(
        base,
        raw,
        entries,
        minimum_support=3,
        base_match_iou=0.9,
        group_relation_iou=0.3,
        group_area_ratio=0.8,
        group_margin_ratio=1.5,
        group_novel_margin=1500.0,
        group_minimum_score=0.04,
        independent_maximum_iou=0.3,
        independent_margin=4000.0,
        independent_minimum_score=0.05,
    )
    assert len(selected["scores"]) == 1
    assert diagnostics["unsupported_base_removed_count"] == 1
