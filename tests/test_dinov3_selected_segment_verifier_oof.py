import numpy as np

from bixolon_scanner.experiments.bread.dinov3_selected_segment_verifier_oof import (
    selected_segment_features,
)


def test_selected_segment_features_use_matching_candidate_rank() -> None:
    features = selected_segment_features(
        np.asarray([[[1.0, 2.0], [3.0, 4.0]]]),
        np.asarray([[0.0, 0.0, 10.0, 10.0]]),
        np.asarray([[7, 4]]),
        {
            "class_objectness": np.asarray([[0.1, 0.8]]),
            "class_positive": np.asarray([[0.2, 0.9]]),
            "class_iou": np.asarray([[0.3, 0.7]]),
        },
        proposal_index=0,
        class_index=4,
        selected_box=np.asarray([1.0, 0.0, 11.0, 10.0]),
        patch_confidence=0.9,
        cls_confidence=0.8,
        count_agreement=True,
    )

    np.testing.assert_allclose(features[:5], [3.0, 4.0, 0.8, 0.9, 0.7])
    np.testing.assert_allclose(features[-3:], [0.9, 0.8, 1.0])
