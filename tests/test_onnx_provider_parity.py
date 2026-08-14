from __future__ import annotations

from bixolon_scanner.evaluation.onnx_provider_parity import provider_parity_report


def _prediction(score: float, x_offset: float = 0.0):
    return {
        "image_id": 1,
        "boxes_xyxy": [[10.0 + x_offset, 10.0, 30.0 + x_offset, 30.0]],
        "scores": [score],
        "class_ids": [2],
    }


def test_provider_parity_accepts_numerically_close_final_detections():
    report = provider_parity_report(
        [_prediction(0.8)],
        [_prediction(0.8001, 0.001)],
        score_threshold=0.5,
    )

    assert report["final_detection_mismatch_image_count"] == 0
    assert report["passes"] is True


def test_provider_parity_rejects_threshold_status_change():
    report = provider_parity_report(
        [_prediction(0.51)],
        [_prediction(0.49)],
        score_threshold=0.5,
        maximum_score_error=0.1,
    )

    assert report["final_detection_mismatch_image_count"] == 1
    assert report["passes"] is False
