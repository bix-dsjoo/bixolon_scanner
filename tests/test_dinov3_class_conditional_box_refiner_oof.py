import numpy as np

from bixolon_scanner.experiments.bread.dinov3_class_conditional_box_refiner_oof import (
    apply_box_offsets,
    box_regression_targets,
)


def test_box_offsets_round_trip_for_matching_class() -> None:
    boxes = np.asarray([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    candidates = np.asarray([[3, 7], [3, 8]])
    annotations = [{"category_id": 4, "bbox_xywh": [2.0, 1.0, 12.0, 8.0]}]

    offsets, valid = box_regression_targets(boxes, candidates, annotations)
    refined = apply_box_offsets(boxes, offsets[:, 0])

    assert valid.tolist() == [[True, False], [True, False]]
    np.testing.assert_allclose(refined, [[2.0, 1.0, 14.0, 9.0], [2.0, 1.0, 14.0, 9.0]], atol=1e-6)
