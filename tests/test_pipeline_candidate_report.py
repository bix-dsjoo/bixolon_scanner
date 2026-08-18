from __future__ import annotations

import numpy as np

from bixolon_scanner.experiments.bread.pipeline_candidate_report import (
    gates,
    object_metrics,
)


def test_object_metrics_keeps_unknown_and_segment_recapture_separate() -> None:
    values = object_metrics(
        mask=np.ones(4, dtype=bool),
        targets=np.asarray([0, 1, 2, 3]),
        predictions=np.asarray([0, 0, 2, 1]),
        top3=np.asarray([[0, 1, 2], [0, 1, 2], [2, 1, 0], [1, 2, 0]]),
        approved=np.asarray([True, True, False, False]),
        unknown=np.asarray([False, False, True, False]),
    )

    assert values["approved_count"] == 2
    assert values["approved_misrecognition_count"] == 1
    assert values["unknown_count"] == 1
    assert values["unknown_top3_candidate_out_count"] == 0
    assert values["segment_recapture_count"] == 1


def test_combined_gates_use_final_segmentation_image_denominator_rates() -> None:
    result = gates(
        {
            "segmentation_rate": 0.90,
            "false_negative_image_rate": 0.001,
            "false_positive_image_rate": 0.0,
        },
        {
            "approved_rate": 0.91,
            "approved_misrecognition_rate": 0.001,
            "unknown_top3_candidate_out_rate": 0.0,
        },
        minimum_segmentation_rate=0.90,
        minimum_approved_rate=0.90,
        maximum_error_rate=0.001,
    )

    assert result["all_met"] is True
