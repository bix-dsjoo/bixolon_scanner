import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bixolon_scanner.training.dinov3_center_localizer import (  # noqa: E402
    center_localizer_loss,
    center_targets,
    decode_center_predictions,
    flip_detection_records,
)


def test_center_targets_decode_and_loss_are_finite() -> None:
    records = [
        {
            "width": 100,
            "height": 100,
            "annotations": [{"bbox_xywh": [40.0, 40.0, 20.0, 20.0]}],
        }
    ]
    targets = center_targets(records, height=10, width=10, device="cpu")
    logits = torch.full((1, 1, 10, 10), -10.0)
    logits[0, 0, 5, 5] = 10.0
    offsets = torch.full((1, 2, 10, 10), 0.5)
    sizes = torch.full((1, 2, 10, 10), 0.2)

    loss, _ = center_localizer_loss((logits, offsets, sizes), targets)
    scores, boxes = decode_center_predictions((logits, offsets, sizes), maximum_objects=1)

    assert torch.isfinite(loss)
    assert scores[0, 0] > 0.99
    np.testing.assert_allclose(boxes[0, 0].numpy(), [0.55, 0.55, 0.2, 0.2])


def test_detection_record_horizontal_flip() -> None:
    rows = flip_detection_records(
        [
            {
                "width": 100,
                "height": 50,
                "annotations": [{"bbox_xywh": [10.0, 5.0, 20.0, 10.0]}],
            }
        ],
        np.asarray([True]),
    )

    assert rows[0]["annotations"][0]["bbox_xywh"] == [70.0, 5.0, 20.0, 10.0]
