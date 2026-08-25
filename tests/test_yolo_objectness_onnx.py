from bixolon_scanner.evaluation.yolo_objectness_onnx import filter_prediction


def test_filter_prediction_keeps_highest_generic_candidates() -> None:
    filtered = filter_prediction(
        {
            "boxes_xyxy": [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]],
            "scores": [0.2, 0.9, 0.8],
        },
        minimum_score=0.5,
        maximum_detections=1,
    )

    assert filtered == {
        "boxes_xyxy": [[1, 1, 2, 2]],
        "scores": [0.9],
        "class_ids": [0],
    }
