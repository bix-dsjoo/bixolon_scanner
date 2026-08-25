from bixolon_scanner.experiments.bread.yolo_objectness_oof import selected_detections


def test_selected_detections_applies_one_generic_score_and_containment_rule() -> None:
    prediction = {
        "boxes_xyxy": [[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]],
        "scores": [0.9, 0.8, 0.4],
    }

    selected = selected_detections(
        prediction,
        score_threshold=0.65,
        nms_iou_threshold=0.4,
        nms_containment_threshold=0.8,
        maximum_aspect_ratio=20.0,
    )

    assert [(item.x1, item.y1, item.x2, item.y2) for item in selected] == [(0.0, 0.0, 10.0, 10.0)]
