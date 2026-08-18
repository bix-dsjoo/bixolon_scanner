import numpy as np
from PIL import Image

from bixolon_scanner.experiments.bread.image_recapture_selector import (
    _flag_metrics,
    assign_quality_policy_folds,
    full_recall_threshold,
    pixel_quality_features,
    reason_conjunction_policy,
)


def test_pixel_quality_features_are_fixed_and_finite():
    image = Image.fromarray(np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3))

    features = pixel_quality_features(image, maximum_side=16)

    assert len(features) == 35
    assert np.isfinite(list(features.values())).all()
    assert features["aspect_ratio"] == 16 / 12


def test_full_recall_threshold_includes_lowest_positive_and_ties():
    scores = np.asarray([0.8, 0.2, 0.2, 0.1])
    targets = np.asarray([True, True, False, False])

    threshold = full_recall_threshold(scores, targets)
    flags = scores >= threshold

    assert threshold == 0.2
    assert flags.tolist() == [True, True, True, False]


def test_flag_metrics_reports_reasons_and_false_recapture_rate():
    records = [
        {"expected_reason_codes": ["DETECTOR_NO_OBJECT"]},
        {"expected_reason_codes": []},
        {"expected_reason_codes": []},
    ]

    metrics = _flag_metrics(
        np.asarray([True, True, False]),
        np.asarray([True, False, False]),
        records,
    )

    assert metrics["recapture_recall"] == 1.0
    assert metrics["false_recapture_count"] == 1
    assert metrics["false_recapture_rate"] == 0.5
    assert metrics["by_reason"]["DETECTOR_NO_OBJECT"]["caught"] == 1


def test_reason_conjunction_policy_combines_opposite_quality_tails():
    records = [
        {"image_id": 1, "expected_image_status": "ANNOTATED", "expected_reason_codes": []},
        {"image_id": 2, "expected_image_status": "ANNOTATED", "expected_reason_codes": []},
        {
            "image_id": 3,
            "expected_image_status": "RECAPTURE",
            "expected_reason_codes": ["LOW"],
        },
        {
            "image_id": 4,
            "expected_image_status": "RECAPTURE",
            "expected_reason_codes": ["HIGH"],
        },
    ]
    features = np.asarray(
        [
            [0.4, 0.4],
            [0.6, 0.6],
            [0.1, 0.2],
            [0.9, 0.8],
        ]
    )

    report = reason_conjunction_policy(
        features,
        ["left", "right"],
        records,
        np.asarray([0, 1, 0, 1]),
    )

    assert report["pooled_oof"]["recapture_recall"] == 1.0
    assert report["pooled_oof"]["false_recapture_count"] == 0


def test_quality_policy_folds_stratify_reasons_without_splitting_groups():
    records = []
    for index in range(9):
        records.append(
            {
                "perceptual_group_id": f"normal-{index}",
                "expected_image_status": "ANNOTATED",
                "expected_reason_codes": [],
            }
        )
    for index in range(6):
        records.append(
            {
                "perceptual_group_id": "duplicate" if index < 2 else f"empty-{index}",
                "expected_image_status": "RECAPTURE",
                "expected_reason_codes": ["DETECTOR_NO_OBJECT"],
            }
        )

    folds = assign_quality_policy_folds(records, fold_count=3)

    duplicate_folds = {
        int(fold)
        for fold, record in zip(folds, records)
        if record["perceptual_group_id"] == "duplicate"
    }
    assert len(duplicate_folds) == 1
    reason_folds = {
        int(fold) for fold, record in zip(folds, records) if record["expected_reason_codes"]
    }
    assert reason_folds == {0, 1, 2}
