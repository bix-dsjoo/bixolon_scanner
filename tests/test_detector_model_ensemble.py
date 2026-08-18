import pytest

from bixolon_scanner.experiments.bread.detector_model_ensemble import (
    fuse_prediction_rows,
    fuse_prediction_sets,
)


def _row(image_id, boxes, scores, classes=None):
    return {
        "image_id": image_id,
        "boxes_xyxy": boxes,
        "scores": scores,
        "class_ids": classes or [0] * len(boxes),
    }


def test_fusion_combines_overlapping_models_and_preserves_complementary_box():
    result = fuse_prediction_rows(
        [
            _row(1, [[0, 0, 10, 10]], [0.8], [2]),
            _row(1, [[1, 0, 11, 10], [20, 20, 30, 30]], [0.6, 0.7], [2, 3]),
        ],
        model_weights=[1.0, 1.0],
        score_thresholds=[0.1, 0.1],
        cluster_iou_threshold=0.5,
        score_mode="mean_all",
    )

    assert result["support_counts"] == [2, 1]
    assert result["source_masks"] == [3, 2]
    assert result["scores"] == pytest.approx([0.7, 0.35])
    assert result["boxes_xyxy"][0][0] == pytest.approx(3.0 / 7.0)


def test_fusion_does_not_put_two_queries_from_same_model_in_one_cluster():
    result = fuse_prediction_rows(
        [
            _row(1, [[0, 0, 10, 10], [0.2, 0, 10.2, 10]], [0.9, 0.8]),
            _row(1, [], []),
        ],
        model_weights=[1.0, 1.0],
        score_thresholds=[0.1, 0.1],
        cluster_iou_threshold=0.5,
        score_mode="maximum",
    )

    assert len(result["scores"]) == 2
    assert result["support_counts"] == [1, 1]


def test_fusion_rejects_image_order_mismatch():
    with pytest.raises(ValueError, match="image ids differ"):
        fuse_prediction_sets(
            [[_row(1, [], [])], [_row(2, [], [])]],
            model_weights=[1.0, 1.0],
            score_thresholds=[0.1, 0.1],
            cluster_iou_threshold=0.5,
            score_mode="maximum",
        )


def test_support_adjusted_maximum_penalizes_single_model_cluster():
    result = fuse_prediction_rows(
        [
            _row(1, [[0, 0, 10, 10]], [0.8]),
            _row(1, [], []),
        ],
        model_weights=[1.0, 1.0],
        score_thresholds=[0.1, 0.1],
        cluster_iou_threshold=0.5,
        score_mode="support_adjusted_maximum",
    )

    assert result["scores"] == pytest.approx([0.4])


def test_pre_nms_removes_same_model_duplicate_before_fusion():
    result = fuse_prediction_rows(
        [
            _row(1, [[0, 0, 10, 10], [0.2, 0, 10.2, 10]], [0.9, 0.8]),
            _row(1, [], []),
        ],
        model_weights=[1.0, 1.0],
        score_thresholds=[0.1, 0.1],
        pre_nms_iou_threshold=0.5,
        cluster_iou_threshold=0.5,
        score_mode="maximum",
    )

    assert len(result["scores"]) == 1
