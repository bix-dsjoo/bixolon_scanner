from __future__ import annotations

import numpy as np
import pytest

from bixolon_scanner.experiments.bread.detector_class_repair import (
    box_iou_xyxy,
    candidate_is_novel,
    select_non_overlapping_candidate,
)


def test_box_iou_xyxy_reports_overlap_and_disjoint_boxes() -> None:
    values = box_iou_xyxy(
        np.asarray([0.0, 0.0, 10.0, 10.0]),
        np.asarray([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
    )
    assert values.tolist() == pytest.approx([1.0, 0.0])


def test_select_non_overlapping_candidate_uses_highest_eligible_score() -> None:
    base = {
        "boxes_xyxy": [[0.0, 0.0, 10.0, 10.0]],
        "scores": [0.9],
        "class_ids": [0],
    }
    ranked = {
        "boxes_xyxy": [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
            [40.0, 40.0, 50.0, 50.0],
        ],
        "scores": [0.99, 0.7, 0.8],
        "class_ids": [0, 0, 0],
    }
    assert select_non_overlapping_candidate(base, ranked, maximum_iou=0.3) == 2


def test_select_non_overlapping_candidate_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        select_non_overlapping_candidate(
            {"boxes_xyxy": [], "scores": [], "class_ids": []},
            {"boxes_xyxy": [], "scores": [], "class_ids": []},
            maximum_iou=1.1,
        )


def test_candidate_is_novel_enforces_unique_class_contract() -> None:
    assert candidate_is_novel(np.asarray([1, 4, 7]), 3)
    assert not candidate_is_novel(np.asarray([1, 4, 7]), 4)
