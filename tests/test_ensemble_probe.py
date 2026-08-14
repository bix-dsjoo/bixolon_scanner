import numpy as np

from bixolon_scanner.experiments.archive.bread_1_0_0.ensemble_probe import (
    normalized_scores,
    rank_metrics,
)


def test_normalized_scores_remove_per_sample_scale_and_offset():
    first = normalized_scores(np.asarray([[1.0, 2.0, 4.0]], dtype=np.float32))
    second = normalized_scores(np.asarray([[12.0, 14.0, 18.0]], dtype=np.float32))

    np.testing.assert_allclose(first, second, atol=1e-6)


def test_rank_metrics_report_top1_and_top3():
    metrics = rank_metrics(
        np.asarray([[3.0, 2.0, 1.0], [2.0, 3.0, 1.0]], dtype=np.float32),
        np.asarray([0, 2]),
    )

    assert metrics == {
        "sample_count": 2,
        "top1_accuracy": 0.5,
        "top1_error_count": 1,
        "top3_accuracy": 1.0,
    }
