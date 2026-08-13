from __future__ import annotations

import numpy as np
import pytest

from bixolon_scanner.training.fewshot_adapter import AdapterSpec
from bixolon_scanner.training.ten_shot_training import (
    HeadTrainingConfig,
    feature_cache_fingerprint,
    train_adapter_head,
    validate_feature_cache,
)


def _spec() -> AdapterSpec:
    return AdapterSpec(
        hidden_size=4,
        bottleneck_size=2,
        num_classes=2,
        cosine_scale=10.0,
        cosine_margin=0.1,
    )


def _features():
    support = np.asarray(
        [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0], [0.1, 0.9, 0, 0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    training = np.repeat(support, 3, axis=0)
    training += np.random.default_rng(7).normal(0, 0.01, training.shape).astype(np.float32)
    training_labels = np.repeat(labels, 3)
    source_indices = np.repeat(np.arange(4), 3)
    return support, labels, training, training_labels, source_indices


def test_feature_cache_fingerprint_is_order_stable_for_backgrounds():
    kwargs = {
        "manifest_sha256": "a" * 64,
        "backbone_sha256": "b" * 64,
        "synthetic_recipe_sha256": "c" * 64,
    }
    assert feature_cache_fingerprint(
        **kwargs, background_sha256=["2", "1"]
    ) == feature_cache_fingerprint(**kwargs, background_sha256=["1", "2"])


def test_feature_cache_rejects_source_outside_exact_support_set():
    support, labels, training, training_labels, sources = _features()
    sources[-1] = 4
    with pytest.raises(ValueError, match="outside"):
        validate_feature_cache(
            training,
            training_labels,
            sources,
            spec=_spec(),
            support_features=support,
            support_labels=labels,
        )


def test_adapter_training_is_deterministic_and_learns_separable_features():
    import torch

    support, labels, training, training_labels, sources = _features()
    config = HeadTrainingConfig(
        epochs=3,
        batch_size=8,
        learning_rate=0.01,
        contrastive_weight=0.05,
        seed=13,
    )
    first, first_history = train_adapter_head(
        training,
        training_labels,
        sources,
        support_features=support,
        support_labels=labels,
        spec=_spec(),
        config=config,
    )
    second, second_history = train_adapter_head(
        training,
        training_labels,
        sources,
        support_features=support,
        support_labels=labels,
        spec=_spec(),
        config=config,
    )
    assert first_history == second_history
    with torch.inference_mode():
        predictions = first(torch.from_numpy(support)).argmax(dim=1).numpy()
    assert np.array_equal(predictions, labels)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])
