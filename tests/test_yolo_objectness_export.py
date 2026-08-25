import numpy as np
import pytest

from bixolon_scanner.training.yolo_objectness_export import canonical_outputs


def test_canonical_outputs_converts_probability_and_normalizes_xywh() -> None:
    raw = np.asarray([[[320.0], [160.0], [64.0], [32.0], [0.8]]], dtype=np.float32)

    logits, boxes = canonical_outputs(raw, image_size=640)

    assert logits.shape == (1, 1, 1)
    assert float(logits[0, 0, 0]) == pytest.approx(np.log(4.0), rel=1e-5)
    assert boxes[0, 0].tolist() == pytest.approx([0.5, 0.25, 0.1, 0.05])
