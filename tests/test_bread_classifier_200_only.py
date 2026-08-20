from __future__ import annotations

import numpy as np

from bixolon_scanner.experiments.bread.classifier_200_only import (
    fit_ridge_head,
    fit_small_sample_head,
    nested_oof_fit,
    select_finite_oof_policy,
)


def test_ridge_head_fits_balanced_separable_features() -> None:
    features = np.asarray([[2.0, 0.0], [1.5, 0.1], [0.0, 2.0], [0.1, 1.5]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    weight, bias = fit_ridge_head(features, labels, alpha=0.1, class_count=2)
    predictions = (features @ weight + bias).argmax(axis=1)

    assert predictions.tolist() == labels.tolist()


def test_small_sample_heads_export_as_linear_weights() -> None:
    features = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [2.5, 0.2, 0.0],
            [0.0, 3.0, 0.0],
            [0.1, 2.5, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.2, 2.5],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    for candidate in (
        {"kind": "cosine_prototype"},
        {"kind": "shrinkage_lda", "shrinkage": 0.5},
    ):
        weight, bias = fit_small_sample_head(features, labels, candidate=candidate, class_count=3)
        predictions = (features @ weight + bias).argmax(axis=1)
        assert predictions.tolist() == labels.tolist()


def test_nested_oof_never_trains_on_outer_fold() -> None:
    train_features = []
    train_labels = []
    train_folds = []
    validation_features = []
    validation_labels = []
    validation_folds = []
    for fold in range(3):
        for label in range(2):
            center = np.asarray([2.0, 0.0] if label == 0 else [0.0, 2.0])
            for offset in (0.0, 0.1):
                train_features.append(center + np.asarray([offset, -offset]))
                train_labels.append(label)
                train_folds.append(fold)
            validation_features.append(center)
            validation_labels.append(label)
            validation_folds.append(fold)
    cache = {
        "train_features": np.asarray(train_features, dtype=np.float32),
        "train_labels": np.asarray(train_labels, dtype=np.int64),
        "train_folds": np.asarray(train_folds, dtype=np.int64),
        "validation_features": np.asarray(validation_features, dtype=np.float32),
        "validation_labels": np.asarray(validation_labels, dtype=np.int64),
        "validation_folds": np.asarray(validation_folds, dtype=np.int64),
    }

    logits, report, head = nested_oof_fit(
        cache,
        head_candidates=[
            {"kind": "regularized_linear_ridge", "alpha": 0.1},
            {"kind": "regularized_linear_ridge", "alpha": 1.0},
        ],
        class_count=2,
    )

    assert logits.argmax(axis=1).tolist() == validation_labels
    assert head["alpha"] in {0.1, 1.0}
    assert {row["outer_fold"] for row in report} == {0, 1, 2}
    assert all(row["outer_fold"] not in {row["inner_train_fold"]} for row in report)


def test_finite_oof_policy_rejects_every_observed_top1_error() -> None:
    logits = np.asarray(
        [
            [5.0, 1.0, 0.0],
            [2.0, 3.0, 0.0],
            [0.0, 1.0, 5.0],
            [0.2, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([0, 1, 2, 1], dtype=np.int64)

    policy = select_finite_oof_policy(logits, targets)

    assert policy["approved_error_count"] == 0
    assert policy["unknown_candidate_out_count"] == 0
    assert policy["approved_count"] < len(targets)
