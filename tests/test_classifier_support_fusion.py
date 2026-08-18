from __future__ import annotations

import numpy as np
import pytest

from bixolon_scanner.experiments.bread.classifier_support_fusion import class_support_scores


def test_class_support_scores_averages_top_k_cosines_per_class() -> None:
    support = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    evaluation = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    scores = class_support_scores(
        evaluation,
        support,
        labels,
        class_count=2,
        top_k=2,
    )

    assert scores.shape == (2, 2)
    assert scores.argmax(axis=1).tolist() == [0, 1]
    assert scores[0, 0] == pytest.approx(scores[1, 1])


def test_class_support_scores_rejects_missing_class_support() -> None:
    with pytest.raises(ValueError, match="class 1"):
        class_support_scores(
            np.ones((1, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            np.asarray([0, 0], dtype=np.int64),
            class_count=2,
            top_k=1,
        )
