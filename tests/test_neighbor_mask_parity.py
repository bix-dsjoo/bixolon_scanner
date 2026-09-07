import numpy as np

from bixolon_scanner.evaluation.neighbor_mask_parity import _outcome_counts
from bixolon_scanner.pipeline.ports import Detection
from bixolon_scanner.runtime.preprocessing import classifier_neighbor_ownership_mask


def test_fractional_detector_roundoff_does_not_change_aligned_crop_mask():
    cpu = [
        Detection(423.847277641, 3085.990623593, 2201.103730917, 3876.966552615, 0.99),
        Detection(442.441658914, 2092.529123068, 2292.560795367, 3317.373054743, 0.99),
    ]
    cuda = [
        Detection(423.847277641, 3085.991219401, 2201.103475571, 3876.966637731, 0.99),
        Detection(442.441467404, 2092.529080510, 2292.560986876, 3317.373097301, 0.99),
    ]
    options = dict(
        image_width=4284,
        image_height=5712,
        output_size=224,
        margin_ratio=0,
        distance_bias=-0.1,
        shared_scale=False,
    )
    first = classifier_neighbor_ownership_mask(cpu, 0, **options)
    second = classifier_neighbor_ownership_mask(cuda, 0, **options)
    assert first.any() and not first.all()
    np.testing.assert_array_equal(first, second)


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
