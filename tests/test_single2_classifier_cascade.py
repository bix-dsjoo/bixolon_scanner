from __future__ import annotations

import numpy as np

from bixolon_scanner.experiments.bread.single2_classifier_cascade import (
    _candidate_config,
    _prototype_initialization,
)


def _feature_bank(values: list[list[float]], labels: list[int], folds: list[int]):
    return {
        "features": np.asarray(values, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "folds": np.asarray(folds, dtype=np.int64),
    }


def test_prototype_initialization_excludes_validation_fold_and_normalizes() -> None:
    clean = _feature_bank([[1, 0], [0, 5], [3, 0], [0, 7]], [0, 1, 0, 1], [0, 0, 1, 1])
    features = {name: clean for name in ("clean", "appearance", "geometry", "context")}

    prototypes = _prototype_initialization(features, excluded_fold=0, class_count=2)

    np.testing.assert_allclose(prototypes, [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(np.linalg.norm(prototypes, axis=1), [1.0, 1.0])


def test_candidate_epochs_override_does_not_mutate_base_config() -> None:
    config = {
        "training": {
            "finetune": {
                "backbone_learning_rate": 1e-6,
                "epochs": 1,
            }
        }
    }

    candidate = _candidate_config(
        config,
        {"backbone_learning_rate": 1e-5, "epochs": 3},
    )

    assert candidate["training"]["finetune"] == {
        "backbone_learning_rate": 1e-5,
        "epochs": 3,
    }
    assert config["training"]["finetune"]["epochs"] == 1
