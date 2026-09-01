import numpy as np
import pytest

from bixolon_scanner.experiments.bread.dinov3_proposal_features import (
    proposal_geometry,
    proposal_targets,
)


def test_proposal_targets_assign_maximum_iou_class() -> None:
    iou, target_class = proposal_targets(
        np.asarray([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float32),
        [
            {"bbox_xywh": [0.0, 0.0, 10.0, 10.0], "category_id": 3},
            {"bbox_xywh": [18.0, 18.0, 12.0, 12.0], "category_id": 7},
        ],
    )

    assert iou.tolist() == pytest.approx([1.0, 100.0 / 144.0])
    assert target_class.tolist() == [3, 7]


def test_proposal_geometry_is_normalized() -> None:
    values = proposal_geometry(
        np.asarray([[10.0, 20.0, 30.0, 60.0]], dtype=np.float32),
        np.asarray([0.75], dtype=np.float32),
        100,
        200,
    )

    assert values.shape == (1, 8)
    assert values[0, 0] == 0.75
    assert values[0, 3] == pytest.approx(0.2)
    assert values[0, 4] == pytest.approx(0.2)
