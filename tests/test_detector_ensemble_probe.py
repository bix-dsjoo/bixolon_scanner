from __future__ import annotations

from bixolon_scanner.experiments.archive.bread_1_0_0.detector_ensemble_probe import (
    ensemble_predictions,
)


def test_ensemble_keeps_agreement_and_high_score_recovery_singleton():
    primary = [{"image_id": 1, "boxes_xyxy": [[0, 0, 10, 10]], "scores": [0.6]}]
    recovery = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0, 0, 10, 10], [20, 20, 30, 30]],
            "scores": [0.8, 0.7],
        }
    ]

    result = ensemble_predictions(
        primary,
        recovery,
        primary_threshold=0.5,
        primary_singleton_threshold=0.9,
        recovery_threshold=0.65,
        agreement_iou=0.5,
    )

    assert len(result[0]["boxes_xyxy"]) == 2
