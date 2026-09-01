import numpy as np

from bixolon_scanner.experiments.bread.dinov3_set_localizer_oof import (
    normalized_cxcywh_to_absolute_xyxy,
)


def test_normalized_set_boxes_convert_to_original_image_coordinates() -> None:
    converted = normalized_cxcywh_to_absolute_xyxy(
        np.asarray([[0.5, 0.5, 0.2, 0.4]]), {"width": 100, "height": 50}
    )

    np.testing.assert_allclose(converted, [[40.0, 15.0, 60.0, 35.0]])
