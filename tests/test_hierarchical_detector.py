from bixolon_scanner.experiments.bread.hierarchical_detector import (
    hierarchical_containment_nms,
)
from bixolon_scanner.pipeline.ports import Detection


def _select(*detections: Detection) -> list[Detection]:
    return hierarchical_containment_nms(
        list(detections),
        iou_threshold=0.5,
        containment_threshold=0.9,
        group_minimum=2,
    )


def test_hierarchical_nms_suppresses_contained_fragment():
    outer = Detection(0, 0, 100, 100, 0.9, 1)
    fragment = Detection(10, 10, 30, 30, 0.8, 2)

    assert _select(outer, fragment) == [outer]


def test_hierarchical_nms_keeps_different_class_outer_object_with_one_child():
    child = Detection(20, 20, 80, 80, 0.9, 1)
    outer = Detection(0, 0, 100, 100, 0.8, 2)

    assert _select(child, outer) == [child, outer]


def test_hierarchical_nms_suppresses_same_class_outer_duplicate():
    child = Detection(20, 20, 80, 80, 0.9, 1)
    outer = Detection(0, 0, 100, 100, 0.8, 1)

    assert _select(child, outer) == [child]


def test_hierarchical_nms_suppresses_group_box_around_two_stronger_objects():
    left = Detection(0, 0, 40, 40, 0.95, 1)
    right = Detection(60, 60, 100, 100, 0.9, 2)
    group = Detection(0, 0, 100, 100, 0.8, 3)

    assert _select(left, right, group) == [left, right]
