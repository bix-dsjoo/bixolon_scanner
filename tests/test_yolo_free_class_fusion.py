from __future__ import annotations

import numpy as np

from bixolon_scanner.runtime.detector_v2 import ensemble_member_class_features


def test_ensemble_member_class_features_uses_nearest_query_per_member() -> None:
    outputs = [
        (
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[0.25, 0.5, 0.2, 0.4], [0.75, 0.5, 0.2, 0.4]], dtype=np.float32),
        ),
        (
            np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            np.asarray([[0.25, 0.5, 0.2, 0.4], [0.75, 0.5, 0.2, 0.4]], dtype=np.float32),
        ),
    ]

    features = ensemble_member_class_features(
        outputs,
        [[65.0, 30.0, 85.0, 70.0]],
        image_width=100,
        image_height=100,
    )

    assert features.shape == (1, 6)
    assert features[0, [0, 1, 3, 4]].tolist() == [3.0, 4.0, 7.0, 8.0]
    assert features[0, 2] == 1.0
    assert features[0, 5] == 1.0
