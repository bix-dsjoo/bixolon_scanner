from __future__ import annotations

from bixolon_scanner.training.rpc_class_aware_nms import _keep_indices


def _detection(box: list[float], score: float) -> dict[str, object]:
    return {"bbox_xyxy": box, "score": score}


def test_class_aware_nms_suppresses_only_same_class_overlap() -> None:
    detections = [
        _detection([0, 0, 100, 100], 0.95),
        _detection([10, 10, 90, 90], 0.80),
        _detection([10, 10, 90, 90], 0.70),
        _detection([200, 200, 300, 300], 0.60),
    ]

    kept = _keep_indices(detections, [4, 4, 8, 4], 0.55)

    assert kept == [0, 2, 3]


def test_class_aware_nms_threshold_is_strict() -> None:
    detections = [
        _detection([0, 0, 100, 100], 0.95),
        _detection([0, 0, 50, 100], 0.80),
    ]

    assert _keep_indices(detections, [1, 1], 0.5) == [0, 1]
