from bixolon_scanner.experiments.bread.image_embedding_cache import (
    full_image_predictions,
)


def test_full_image_predictions_cover_original_pixel_extent():
    predictions = full_image_predictions([{"image_id": 7, "width": 640, "height": 480}])

    assert predictions == [
        {
            "image_id": 7,
            "boxes_xyxy": [[0.0, 0.0, 640.0, 480.0]],
            "scores": [1.0],
            "class_ids": [0],
        }
    ]
