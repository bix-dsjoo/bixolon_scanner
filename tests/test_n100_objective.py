import pytest

from bixolon_scanner.evaluation.n100_objective import assess


@pytest.mark.parametrize(
    "correct,wrong,missed,latency,expected",
    [
        (1042, 0, 0, 200, True),
        (1041, 0, 0, 200, False),
        (1042, 1, 0, 200, False),
        (1042, 0, 1, 200, False),
        (1042, 0, 0, 200.001, False),
    ],
)
def test_object_targets_and_latency_boundary(correct, wrong, missed, latency, expected):
    summary = {
        "image_count": 132,
        "ground_truth_count": 1096,
        "correct_approved_count": correct,
        "wrong_approved_count": wrong,
        "missed_count": missed,
        "image_status_counts": {},
        "latency": {k: {"p95_ms": latency} for k in ["all", "full_path"]},
    }
    targets = {
        "image_count": 132,
        "ground_truth_count": 1096,
        "minimum_correct_approved_count": 1042,
        "maximum_wrong_approved_count": 0,
        "maximum_missed_count": 0,
        "maximum_http_p95_ms": 200,
    }
    assert assess(summary, targets)["measurement_targets_met"] is expected
    summary["ground_truth_count"] -= 1
    with pytest.raises(ValueError, match="denominator"):
        assess(summary, targets)
