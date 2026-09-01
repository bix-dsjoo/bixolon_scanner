import numpy as np
import pytest

from bixolon_scanner.training.dinov3_set_localizer import (
    build_dinov3_set_localizer,
    cxcywh_to_xyxy,
    hungarian_set_loss,
    normalized_target_boxes,
)

torch = pytest.importorskip("torch")


def test_normalized_target_boxes_and_conversion() -> None:
    target = normalized_target_boxes(
        {
            "width": 100,
            "height": 50,
            "annotations": [{"bbox_xywh": [10.0, 5.0, 20.0, 10.0]}],
        }
    )

    np.testing.assert_allclose(target, [[0.2, 0.2, 0.2, 0.2]])
    np.testing.assert_allclose(
        cxcywh_to_xyxy(torch.from_numpy(target)).numpy(), [[0.1, 0.1, 0.3, 0.3]]
    )


def test_set_localizer_loss_is_finite() -> None:
    model = build_dinov3_set_localizer(maximum_objects=2, hidden_dimension=32, decoder_layers=1)
    logits, boxes = model(torch.randn(1, 384, 4, 4), torch.randn(1, 768, 2, 2))
    loss, parts = hungarian_set_loss(logits, boxes, [torch.tensor([[0.5, 0.5, 0.2, 0.2]])])

    assert torch.isfinite(loss)
    assert set(parts) == {"objectness", "l1", "giou"}
