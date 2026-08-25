import numpy as np

from bixolon_scanner.experiments.bread.classifier_target_label_propagation import (
    folded_target_prototype_probabilities,
    propagate_labels,
)


def test_label_propagation_uses_source_anchors_and_target_neighbors() -> None:
    supports = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    support_labels = np.asarray([0, 1], dtype=np.int64)
    targets = np.asarray([[0.99, 0.01], [0.98, 0.02], [0.01, 0.99]], dtype=np.float32)
    weight = np.eye(2, dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)

    prior, propagated = propagate_labels(
        targets,
        supports,
        support_labels,
        weight,
        bias,
        class_count=2,
        neighbor_count=1,
        iterations=3,
    )

    assert prior.shape == propagated.shape == (3, 2)
    assert propagated.argmax(axis=1).tolist() == [0, 0, 1]


def test_label_propagation_rejects_non_normalized_message_weights() -> None:
    with np.testing.assert_raises(ValueError):
        propagate_labels(
            np.eye(2, dtype=np.float32),
            np.eye(2, dtype=np.float32),
            np.asarray([0, 1], dtype=np.int64),
            np.eye(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            class_count=2,
            source_weight=0.5,
            target_weight=0.5,
            prior_weight=0.5,
        )


def test_folded_target_prototypes_exclude_the_held_out_fold() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    pseudo_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    folds = np.asarray([0, 1, 0, 1], dtype=np.int64)

    probabilities = folded_target_prototype_probabilities(
        embeddings,
        pseudo_labels,
        folds,
        class_count=2,
    )

    assert probabilities.argmax(axis=1).tolist() == pseudo_labels.tolist()
