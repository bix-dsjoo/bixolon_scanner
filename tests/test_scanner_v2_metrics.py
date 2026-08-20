from __future__ import annotations

import pytest

from bixolon_scanner.evaluation.scanner_v2 import Counts, _reported_metrics


def test_object_status_rates_use_returned_segmentation_objects() -> None:
    counts = Counts(
        image_count=300,
        segmentation_image_count=294,
        image_recapture_count=6,
        ground_truth_count=1410,
        prediction_count=1375,
        approved_count=1340,
        unknown_count=8,
        segment_recapture_count=27,
        approved_misrecognition_count=1,
        latencies_ms=[80.0, 100.0],
    )

    requested, promotion = _reported_metrics(counts)

    assert requested["segmentation_object_count"] == 1375
    assert requested["approved_rate"] == pytest.approx(1340 / 1375)
    assert requested["unknown_top3_rate"] == pytest.approx(8 / 1375)
    assert requested["segment_recapture_rate"] == pytest.approx(27 / 1375)
    assert sum(
        requested[key] for key in ("approved_rate", "unknown_top3_rate", "segment_recapture_rate")
    ) == pytest.approx(1.0)
    assert promotion["approved_all_gt_rate"] == pytest.approx(1340 / 1410)
