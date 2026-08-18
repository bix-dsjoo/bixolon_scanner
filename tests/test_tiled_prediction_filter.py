from bixolon_scanner.experiments.bread.tiled_prediction_filter import (
    touches_internal_tile_boundary,
)


def test_internal_boundary_filter_keeps_outer_image_edges() -> None:
    top_left = [0, 0, 50, 50]

    assert not touches_internal_tile_boundary(
        [0, 5, 20, 30],
        top_left,
        image_width=100,
        image_height=100,
        margin_ratio=0.05,
    )
    assert touches_internal_tile_boundary(
        [30, 5, 49, 30],
        top_left,
        image_width=100,
        image_height=100,
        margin_ratio=0.05,
    )
