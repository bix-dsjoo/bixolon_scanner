import numpy as np

from bixolon_scanner.experiments.bread.foreground_component_detector import (
    foreground_component_boxes,
)


def test_foreground_component_boxes_find_two_colored_objects() -> None:
    image = np.full((100, 120, 3), 245, dtype=np.uint8)
    image[20:50, 10:40] = (130, 70, 25)
    image[55:90, 70:110] = (170, 100, 30)

    boxes = foreground_component_boxes(
        image,
        color_distance=10.0,
        minimum_area_ratio=0.01,
        maximum_area_ratio=0.5,
        opening_size=1,
        closing_size=1,
        padding_ratio=0.0,
    )

    assert boxes == [[10.0, 20.0, 40.0, 50.0], [70.0, 55.0, 110.0, 90.0]]
