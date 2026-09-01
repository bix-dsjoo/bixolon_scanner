import numpy as np
import pytest

from bixolon_scanner.experiments.bread.dinov3_catalog_proposal_logits import (
    _positive_metrics,
    catalog_target_indices,
    clipped_crop_box,
)


def test_catalog_target_indices_follow_catalog_order() -> None:
    assert catalog_target_indices(
        ["bread_02", "bread_01", "bread_03"],
        np.asarray([1, 3, 2]),
    ).tolist() == [1, 2, 0]


def test_clipped_crop_box_rounds_outward_and_rejects_empty() -> None:
    assert clipped_crop_box(np.asarray([-1.2, 3.4, 12.1, 25.8]), width=10, height=20) == (
        0,
        3,
        10,
        20,
    )
    with pytest.raises(ValueError, match="empty"):
        clipped_crop_box(np.asarray([5.0, 5.0, 5.0, 8.0]), width=10, height=10)


def test_positive_metrics_exclude_background_proposals() -> None:
    logits = np.asarray([[0.9, 0.1, 0.0], [0.1, 0.8, 0.2], [0.8, 0.1, 0.2]])
    retrieval = np.asarray([[0.8, 0.2, 0.0], [0.2, 0.7, 0.1], [0.9, 0.0, 0.1]])
    metrics = _positive_metrics(
        logits,
        retrieval,
        np.asarray([0.8, 0.6, 0.2]),
        np.asarray([0, 1, 2]),
    )

    assert metrics["positive_iou_0_5_count"] == 2
    assert metrics["adapter_top1_accuracy"] == 1.0
    assert metrics["adapter_top3_accuracy"] == 1.0
    assert metrics["retrieval_top1_accuracy"] == 1.0
