from __future__ import annotations

from bixolon_scanner.experiments.bread.classifier_geometry_routing import (
    GeometryRecipe,
    geometry_signal,
    routed_view_name,
)


def test_geometry_routing_moves_away_from_upper_left_overlap() -> None:
    boxes = [
        (20.0, 20.0, 80.0, 80.0),
        (0.0, 0.0, 50.0, 50.0),
    ]
    signal = geometry_signal(boxes, 0)

    assert signal.maximum_overlap_fraction > 0
    assert signal.repulsion_x > 0
    assert signal.repulsion_y > 0
    assert routed_view_name(signal, GeometryRecipe(0.65, 0.05, 0.0)) == ("scale0.650_x+1_y+1")


def test_geometry_routing_keeps_nonoverlapping_box_centered() -> None:
    signal = geometry_signal([(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)], 0)

    assert routed_view_name(signal, GeometryRecipe(0.65, 0.05, 0.0)) == ("scale0.850_x+0_y+0")
