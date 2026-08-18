import numpy as np

from bixolon_scanner.evaluation.neighbor_mask_parity import _outcome_counts


def test_outcome_counts_separates_unmatched_detector_predictions() -> None:
    result = _outcome_counts(
        mask=np.asarray([True, True, True, True]),
        states=np.asarray([0, 1, 2, 0], dtype=np.int8),
        top1=np.asarray([0, 1, 2, 3]),
        top3=np.asarray([[0, 1, 2], [0, 1, 2], [2, 1, 0], [3, 2, 1]]),
        targets=np.asarray([0, 2, 2, -1]),
    )

    assert result == {
        "sample_count": 3,
        "approved_count": 1,
        "approved_error_count": 0,
        "unknown_count": 1,
        "unknown_top3_miss_count": 0,
        "segment_recapture_count": 1,
        "segment_recapture_rate": 1 / 3,
        "unmatched_prediction_count": 1,
        "unmatched_approved_count": 1,
        "unmatched_unknown_count": 0,
        "unmatched_segment_recapture_count": 0,
    }
