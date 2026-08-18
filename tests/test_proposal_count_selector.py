import numpy as np

from bixolon_scanner.experiments.bread.proposal_count_selector import (
    _calibrate_ordinal_thresholds,
    count_constrained_select,
    count_features,
    disagreement_recapture_mask,
    proposal_content_count_features,
)


def test_count_features_are_fixed_width_and_finite():
    record = {"width": 100, "height": 200}
    ranked = {
        "scores": [0.9, 0.4],
        "source_ids": [0, 1],
        "boxes_xyxy": [[0, 0, 10, 10], [20, 0, 30, 10]],
        "class_ids": [1, 2],
    }
    raw = {
        "scores": [0.8, 0.3, 0.1],
        "boxes_xyxy": [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]],
        "class_ids": [0, 0, 0],
    }

    features = count_features(record, ranked, raw, rank_length=4)

    assert features.shape == (235,)
    assert np.isfinite(features).all()

    group_features = count_features(
        record,
        ranked,
        raw,
        rank_length=4,
        include_group_count_signals=True,
    )
    assert group_features.shape == (307,)


def test_count_constrained_selection_returns_at_most_predicted_count():
    prediction = {
        "image_id": 1,
        "boxes_xyxy": [[0, 0, 10, 10], [20, 0, 30, 10]],
        "scores": [0.9, 0.8],
        "class_ids": [1, 2],
    }

    selected = count_constrained_select(
        prediction,
        predicted_count=1,
        score_threshold=0.5,
        nms_iou_threshold=0.5,
    )

    assert selected["scores"] == [0.9]


def test_proposal_content_count_features_are_fixed_width_and_finite():
    prediction = {
        "boxes_xyxy": [[0, 0, 10, 10], [20, 0, 30, 10]],
        "scores": [0.9, 0.8],
        "class_ids": [1, 2],
        "source_ids": [0, 1],
    }
    logits = np.asarray([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=np.float32)

    features = proposal_content_count_features(
        prediction,
        logits,
        logits,
        rank_length=2,
    )

    assert features.shape == (156,)
    assert np.isfinite(features).all()


def test_ordinal_threshold_calibration_recovers_separable_counts():
    probabilities = np.asarray([[0.1, 0.0], [0.9, 0.2], [0.8, 0.9]], dtype=np.float32)

    counts, diagnostics = _calibrate_ordinal_thresholds(
        probabilities,
        np.asarray([1, 2, 3]),
        minimum_count=1,
    )

    assert counts.tolist() == [1, 2, 3]
    assert [row["error_count"] for row in diagnostics] == [0, 0]


def test_ordinal_calibration_ignores_unavailable_overcount():
    counts, diagnostics = _calibrate_ordinal_thresholds(
        np.asarray([[0.9], [0.8]], dtype=np.float32),
        np.asarray([1, 2]),
        minimum_count=1,
        maximum_counts=np.asarray([1, 2]),
    )

    assert counts.tolist() == [1, 2]
    assert diagnostics[0]["error_count"] == 0


def test_disagreement_recapture_mask_uses_one_strong_extra_candidate():
    mask = disagreement_recapture_mask(
        [
            {"scores": [0.9, 0.8, 0.6]},
            {"scores": [0.9, 0.8, 0.7, 0.6]},
        ],
        [
            {"scores": [0.9, 0.8]},
            {"scores": [0.9, 0.8]},
        ],
        minimum_selected_count=2,
        extra_candidate_count=1,
        next_score_threshold=0.5,
    )

    assert mask.tolist() == [True, False]
