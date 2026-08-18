from bixolon_scanner.experiments.bread.tiled_onnx_predictions import (
    map_tile_prediction,
    tile_windows,
)


def test_tile_windows_cover_edges_with_overlap() -> None:
    windows = tile_windows(
        100,
        80,
        rows=2,
        columns=2,
        width_fraction=0.6,
        height_fraction=0.75,
    )

    assert windows == [(0, 0, 60, 60), (40, 0, 100, 60), (0, 20, 60, 80), (40, 20, 100, 80)]


def test_map_tile_prediction_restores_original_coordinates() -> None:
    prediction = {
        "boxes_xyxy": [[1.0, 2.0, 11.0, 12.0]],
        "scores": [0.9],
        "class_ids": [1],
    }

    mapped = map_tile_prediction(prediction, x_offset=40, y_offset=20)

    assert mapped["boxes_xyxy"] == [[41.0, 22.0, 51.0, 32.0]]
