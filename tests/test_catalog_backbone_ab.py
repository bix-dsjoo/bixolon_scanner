import numpy as np
import pytest

from bixolon_scanner.experiments.bread.catalog_backbone_ab import (
    _metrics,
    _policy_metrics,
    _select_top3_safety_threshold,
)


def test_metrics_selects_largest_prefix_with_allowed_error() -> None:
    logits = np.asarray(
        [
            [4.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 0.9, 0.8],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([0, 1, 1, 3], dtype=np.int64)

    result = _metrics(logits, targets, allowed_errors=1)

    assert result["safe_approved_count"] == 3
    assert result["safe_approved_error_count"] == 1
    assert result["top3_correct_count"] == 4


def test_policy_uses_segmentation_denominator_and_all_gt_safety_rates() -> None:
    logits = np.asarray(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 2.0, 1.0],
            [0.0, 1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([0, 1, 0], dtype=np.int64)
    threshold = _metrics(logits, targets, allowed_errors=0)["approval_threshold"]
    assert threshold is not None
    safety = _select_top3_safety_threshold(
        logits,
        targets,
        approval_threshold=float(threshold),
    )

    result = _policy_metrics(
        logits,
        targets,
        approval_threshold=float(threshold),
        top3_safety_threshold=safety,
        all_ground_truth_count=4,
    )

    assert result["approved_rate_over_segmentation"] == pytest.approx(1 / 3)
    assert result["approved_rate_over_all_ground_truth"] == pytest.approx(0.25)
    assert result["unknown_top3_candidate_out_count"] == 0
    assert result["segment_recapture_count"] == 2
