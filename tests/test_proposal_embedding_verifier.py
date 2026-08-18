import numpy as np
import pytest

from bixolon_scanner.experiments.bread.proposal_embedding_verifier import (
    embedding_features,
    validate_classifier_source,
)


def test_embedding_features_append_raw_adapted_and_distance_signals():
    classifier = np.asarray([[0.2, 0.4]], dtype=np.float32)
    raw = np.asarray([[1.0, 0.0]], dtype=np.float32)
    adapted = np.asarray([[0.8, 0.6]], dtype=np.float32)

    features = embedding_features(classifier, raw, adapted)

    assert features.shape == (1, 8)
    np.testing.assert_allclose(features[0, -2:], [0.8, np.sqrt(0.4)])


def test_validate_classifier_source_rejects_mixed_sources():
    checkpoint = {"dataset_version": "bread-source-1"}
    metadata = {
        "classifier": {
            "selected_source": "single_objects",
            "mixed_sources": True,
            "source_dataset_version": "bread-source-1",
        }
    }

    with pytest.raises(ValueError, match="mixed"):
        validate_classifier_source(checkpoint, metadata)


def test_validate_classifier_source_accepts_single_locked_source():
    checkpoint = {"dataset_version": "bread-source-1"}
    metadata = {
        "classifier": {
            "selected_source": "single_objects_2",
            "mixed_sources": False,
            "source_dataset_version": "bread-source-1",
        }
    }

    assert validate_classifier_source(checkpoint, metadata) == "single_objects_2"
