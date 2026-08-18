from __future__ import annotations

import numpy as np

from bixolon_scanner.experiments.bread.classifier_siglip_confirmation import (
    batches,
    metrics,
    normalized_tensor_to_image,
)


def test_normalized_tensor_to_image_reverses_imagenet_normalization() -> None:
    tensor = np.zeros((3, 2, 2), dtype=np.float32)
    image = normalized_tensor_to_image(tensor)

    assert image.size == (2, 2)
    assert np.asarray(image)[0, 0].tolist() == [124, 116, 104]


def test_batches_keeps_final_partial_batch() -> None:
    assert list(batches(range(5), 2)) == [[0, 1], [2, 3], [4]]


def test_metrics_reports_top1_and_top3_failures() -> None:
    scores = np.asarray([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    targets = np.asarray([0, 0])

    assert metrics(scores, targets) == {
        "sample_count": 2,
        "top1_error_count": 1,
        "top3_miss_count": 1,
    }
