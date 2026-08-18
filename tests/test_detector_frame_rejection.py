from bixolon_scanner.experiments.bread.detector_frame_rejection import (
    box_area_ratio,
    reject_oversized_predictions,
)


def test_rejects_only_predictions_over_inclusive_area_boundary() -> None:
    predictions = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0, 0, 30, 100], [0, 0, 31, 100]],
            "scores": [0.9, 0.8],
            "class_ids": [1, 2],
        }
    ]
    selected, diagnostics = reject_oversized_predictions(
        predictions, {1: (100, 100)}, maximum_box_area_ratio=0.3
    )

    assert selected[0]["boxes_xyxy"] == [[0, 0, 30, 100]]
    assert selected[0]["scores"] == [0.9]
    assert diagnostics[0]["removed"][0]["prediction_index"] == 1


def test_box_area_ratio_clamps_invalid_box() -> None:
    assert box_area_ratio([10, 10, 0, 0], image_width=100, image_height=100) == 0.0
