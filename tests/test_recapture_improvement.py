from copy import deepcopy

import pytest

from bixolon_scanner.evaluation.recapture_improvement import (
    compare_repetitions,
    retained_prediction_parity,
)


def result(recapture=38, p95=280):
    return {
        "image_count": 132,
        "ground_truth_count": 1096,
        "matched_count": 1096,
        "wrong_approved_count": 0,
        "missed_count": 0,
        "correct_approved_count": 1078,
        "extra_count": 37,
        "item_status_counts": {"SEGMENT_RECAPTURE": recapture},
        "image_status_counts": {"SEGMENTATION": 132},
        "latency": {"all": {"count": 132, "p95_ms": p95}},
    }


def test_improvement_keeps_all_gt_in_denominator():
    before, after = result(), result(19, 250)
    assert compare_repetitions([before], [after])["accepted"]
    after["missed_count"] = 1
    after["matched_count"] = 1095
    assert not compare_repetitions([before], [after])["accepted"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("correct_approved_count", 1077),
        ("wrong_approved_count", 1),
        ("image_count", 131),
        ("ground_truth_count", 1095),
        ("extra_count", 38),
    ],
)
def test_rejects_apparent_gains_with_accuracy_or_denominator_regression(field, value):
    after = result(1, 200)
    after[field] = value
    assert not compare_repetitions([result()], [after])["accepted"]


def test_errors_and_slower_p95_cannot_pass():
    after = result(1, 281)
    assert not compare_repetitions([result()], [after])["accepted"]
    after = deepcopy(result(1, 200))
    after["image_status_counts"]["ERROR"] = 1
    assert not compare_repetitions([result()], [after])["accepted"]


def test_records_pair_regression_even_when_repetition_median_improves():
    report = compare_repetitions(
        [result(p95=x) for x in (280, 280, 280)], [result(19, x) for x in (250, 290, 250)]
    )
    assert not report["accepted"]
    assert not report["every_pair_no_slower"]


def test_missing_repetitions_or_latency_fail():
    with pytest.raises(ValueError):
        compare_repetitions([], [])
    with pytest.raises(ValueError):
        compare_repetitions([result()], [result(19, None)])


def test_retained_decisions_reject_silent_recapture_relabeling():
    seg = {
        "bbox": {"x": 1, "y": 1, "width": 20, "height": 20},
        "status": "SEGMENT_RECAPTURE",
        "reason_codes": ["SEGMENT_RECAPTURE_REQUIRED"],
        "prediction": None,
        "top3": [],
        "confidence": 0.0,
    }
    before = [{"image_id": 1, "response": {"segmentations": [seg]}}]
    after = deepcopy(before)
    assert retained_prediction_parity(before, after)["unchanged_retained_decisions"]
    after[0]["response"]["segmentations"][0]["status"] = "UNKNOWN"
    assert not retained_prediction_parity(before, after)["unchanged_retained_decisions"]


def test_retained_decisions_require_same_images_and_boxes():
    with pytest.raises(ValueError):
        retained_prediction_parity([{"image_id": 1}], [])
