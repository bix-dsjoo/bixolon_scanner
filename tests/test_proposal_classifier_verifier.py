import json

import numpy as np
import pytest

from bixolon_scanner.experiments.bread.proposal_classifier_verifier import (
    _load_predictions,
    classifier_metadata_for_view,
    proposal_context_features,
    verifier_features,
)


def test_load_predictions_can_explicitly_filter_a_superset(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "\n".join(json.dumps({"image_id": image_id, "scores": []}) for image_id in (1, 2)),
        encoding="utf-8",
    )
    records = [{"image_id": 1}]

    with pytest.raises(ValueError, match="coverage differs"):
        _load_predictions(path, records)

    assert _load_predictions(path, records, allow_superset=True) == [{"image_id": 1, "scores": []}]


def test_verifier_features_include_classifier_and_geometry_signals():
    record = {"width": 100, "height": 200}
    prediction = {
        "boxes_xyxy": [[10, 20, 40, 80]],
        "scores": [0.7],
        "class_ids": [1],
    }
    logits = np.asarray([[1.0, 3.0, 2.0]], dtype=np.float32)
    ranking_logits = np.asarray([[2.0, 1.0, 3.0]], dtype=np.float32)

    features = verifier_features(record, prediction, logits, ranking_logits)

    assert features.shape == (1, 42)
    assert np.isfinite(features).all()


def test_verifier_features_include_detector_ensemble_support_signals():
    record = {"width": 100, "height": 200}
    prediction = {
        "boxes_xyxy": [[10, 20, 40, 80]],
        "scores": [0.7],
        "class_ids": [1],
        "support_counts": [2],
        "source_masks": [5],
        "member_scores": [[0.8, 0.0, 0.6, 0.0]],
    }
    logits = np.asarray([[1.0, 3.0, 2.0]], dtype=np.float32)

    features = verifier_features(record, prediction, logits, logits)

    assert features.shape == (1, 56)
    assert np.isfinite(features).all()
    assert 2.0 in features[0]


def test_verifier_features_reject_partial_ensemble_fields():
    prediction = {
        "boxes_xyxy": [[10, 20, 40, 80]],
        "scores": [0.7],
        "class_ids": [1],
        "support_counts": [2],
    }
    logits = np.asarray([[1.0, 3.0, 2.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="supplied together"):
        verifier_features({"width": 100, "height": 200}, prediction, logits, logits)


def test_proposal_context_features_describe_overlap_and_containment():
    features = proposal_context_features(
        {
            "boxes_xyxy": [[0, 0, 20, 10], [0, 0, 10, 10]],
            "scores": [0.8, 0.9],
            "support_counts": [1, 2],
        }
    )

    assert features.shape == (2, 18)
    assert np.isfinite(features).all()
    assert features[0, 7] == 1
    assert features[0, 10] == 1


def test_classifier_box_resize_view_disables_neighbor_policy():
    class Metadata:
        neighbor_mask_inference = object()

        def model_copy(self, *, update):
            result = Metadata()
            result.neighbor_mask_inference = update["neighbor_mask_inference"]
            return result

    metadata = Metadata()

    assert classifier_metadata_for_view(metadata, "package") is metadata
    assert classifier_metadata_for_view(metadata, "box_resize").neighbor_mask_inference is None
