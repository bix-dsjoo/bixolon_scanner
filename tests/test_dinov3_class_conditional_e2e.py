import numpy as np

from bixolon_scanner.experiments.bread.dinov3_class_conditional_e2e import (
    ClassSelection,
    class_conditional_score,
    fuse_class_conditional_boxes,
    select_class_conditional_top_k,
)


def test_class_conditional_selector_chooses_one_box_per_sku() -> None:
    selection = select_class_conditional_top_k(
        np.asarray([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]]),
        np.asarray([[0, 1], [0, 2], [1, 2]]),
        np.asarray([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]),
        count=2,
        nms_iou=0.5,
    )

    assert selection.proposal_indices.tolist() == [0, 2]
    assert selection.class_indices.tolist() == [0, 1]


def test_class_conditional_product_score_is_geometric_mean() -> None:
    score = class_conditional_score(
        np.asarray([[0.25, 1.0]]),
        np.asarray([[1.0, 0.25]]),
        kind="product",
    )

    assert score.tolist() == [[0.5, 0.5]]


def test_class_conditional_box_fusion_uses_only_local_same_sku_support() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [2.0, 0.0, 12.0, 10.0],
            [30.0, 30.0, 40.0, 40.0],
        ]
    )
    selection = ClassSelection(
        proposal_indices=np.asarray([0]),
        class_indices=np.asarray([4]),
        scores=np.asarray([0.9]),
        residual_ratio=0.0,
    )

    fused = fuse_class_conditional_boxes(
        boxes,
        np.asarray([[4, 1], [4, 2], [4, 3]]),
        np.asarray([[0.9, 0.1], [0.9, 0.1], [1.0, 0.1]]),
        selection,
        top_n=3,
        cluster_iou=0.5,
        temperature=0.1,
    )

    np.testing.assert_allclose(fused, [[1.0, 0.0, 11.0, 10.0]], atol=1e-6)


def test_class_conditional_box_fusion_can_use_pair_specific_refinement() -> None:
    boxes = np.asarray([[0.0, 0.0, 10.0, 10.0], [2.0, 0.0, 12.0, 10.0]])
    selection = ClassSelection(
        proposal_indices=np.asarray([0]),
        class_indices=np.asarray([4]),
        scores=np.asarray([0.9]),
        residual_ratio=0.0,
    )
    refined = np.asarray(
        [
            [[5.0, 5.0, 15.0, 15.0], [0.0, 0.0, 10.0, 10.0]],
            [[7.0, 5.0, 17.0, 15.0], [2.0, 0.0, 12.0, 10.0]],
        ]
    )

    fused = fuse_class_conditional_boxes(
        boxes,
        np.asarray([[4, 1], [4, 2]]),
        np.asarray([[0.9, 0.1], [0.9, 0.1]]),
        selection,
        top_n=2,
        cluster_iou=0.5,
        temperature=0.1,
        refined_pair_boxes=refined,
    )

    np.testing.assert_allclose(fused, [[6.0, 5.0, 16.0, 15.0]], atol=1e-6)
