from __future__ import annotations

import numpy as np
import pytest

from bixolon_scanner.training.fewshot_adapter import (
    AdapterSpec,
    build_residual_cosine_head,
    compatible_proxy_state_dict,
    supervised_contrastive_loss,
    wrap_inference_classifier,
)


def _spec(**overrides) -> AdapterSpec:
    values = {
        "hidden_size": 4,
        "bottleneck_size": 2,
        "num_classes": 2,
        "cosine_scale": 10.0,
        "cosine_margin": 0.2,
    }
    values.update(overrides)
    return AdapterSpec(**values)


def test_adapter_initializes_from_balanced_support_and_preserves_shapes():
    import torch

    head = build_residual_cosine_head(_spec())
    support = np.asarray(
        [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0], [0.1, 0.9, 0, 0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    head.initialize_class_weights_from_support(support, labels)
    features = torch.from_numpy(support)
    logits = head(features)
    assert logits.shape == (4, 2)
    assert logits.argmax(dim=1).tolist() == labels.tolist()
    adapted = head.adapt(features)
    assert torch.allclose(torch.linalg.vector_norm(adapted, dim=1), torch.ones(4))


def test_adapter_rejects_missing_or_unbalanced_support():
    head = build_residual_cosine_head(_spec())
    with pytest.raises(ValueError, match="class 1 has no support"):
        head.initialize_class_weights_from_support(
            np.eye(4, dtype=np.float32), np.asarray([0, 0, 0, 0], dtype=np.int64)
        )


def test_training_logits_apply_margin_only_to_target():
    import torch

    head = build_residual_cosine_head(_spec())
    support = np.asarray(
        [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1])
    head.initialize_class_weights_from_support(support, labels)
    features = torch.from_numpy(support[:2])
    plain = head(features)
    margin, _ = head.training_logits(features, torch.zeros(2, dtype=torch.long))
    assert torch.allclose(margin[:, 0], plain[:, 0] - 2.0)
    assert torch.allclose(margin[:, 1], plain[:, 1])


def test_supervised_contrastive_loss_is_finite_and_differentiable():
    import torch

    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    loss = supervised_contrastive_loss(features, labels, temperature=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None


def test_adapter_spec_locks_margin_range_and_has_no_runtime_support_cache():
    assert "support_keys" not in build_residual_cosine_head(_spec()).state_dict()
    with pytest.raises(ValueError, match="cosine_margin"):
        _spec(cosine_margin=1.0).validate()


def test_local_two_proxy_head_accepts_patch_features_and_side_initialization():
    import torch

    spec = _spec(use_local_features=True, proxies_per_class=2)
    head = build_residual_cosine_head(spec)
    support = np.asarray(
        [[1, 0, 0, 0], [0.8, 0.2, 0, 0], [0, 1, 0, 0], [0.2, 0.8, 0, 0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    proxy_ids = np.asarray([0, 1, 0, 1], dtype=np.int64)
    head.initialize_class_weights_from_support(support, labels, proxy_ids)
    patches = torch.from_numpy(np.repeat(support[:, None, :], 3, axis=1))
    logits = head(torch.from_numpy(support), patches)
    assert logits.shape == (4, 2)
    assert logits.argmax(dim=1).tolist() == labels.tolist()
    assert head.class_weights.shape == (2, 2, 4)


def test_legacy_single_proxy_checkpoint_weights_are_upgraded_on_load():
    import torch

    head = build_residual_cosine_head(_spec())
    legacy = dict(head.state_dict())
    legacy["class_weights"] = legacy["class_weights"].squeeze(1)
    head.load_state_dict(compatible_proxy_state_dict(legacy))
    assert head.class_weights.shape == (2, 1, 4)
    model_state = {"classifier.class_weights": torch.ones(2, 4)}
    assert compatible_proxy_state_dict(model_state)[
        "classifier.class_weights"
    ].shape == (2, 1, 4)


def test_center_crop_classifier_matches_selected_224_view_and_preserves_batch():
    import torch

    class Identity(torch.nn.Module):
        def forward(self, values):
            return values

    values = torch.arange(2 * 3 * 224 * 224, dtype=torch.float32).reshape(
        2, 3, 224, 224
    )
    wrapped = wrap_inference_classifier(
        Identity(), input_size=224, crop_scale=0.88, num_classes=3
    )
    observed = wrapped(values)
    cropped = values[..., 13:210, 13:210]
    expected = torch.nn.functional.interpolate(
        cropped,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    assert observed.shape == values.shape
    assert torch.equal(observed, expected)


def test_center_crop_classifier_rejects_unsafe_scale():
    import torch

    with pytest.raises(ValueError, match="center crop scale"):
        wrap_inference_classifier(
            torch.nn.Identity(), input_size=224, crop_scale=1.0, num_classes=2
        )


def test_inference_logit_policy_is_deterministic_and_ordered():
    import torch

    wrapped = wrap_inference_classifier(
        torch.nn.Identity(),
        input_size=1,
        crop_scale=None,
        num_classes=3,
        logit_quantum=0.44,
        logit_phase=0.066,
        tie_break_bias_span=0.044,
        logit_divisor=50.0,
    )
    values = torch.tensor([[1.01, 1.02, 1.03]])
    observed = wrapped(values)
    assert observed.argmax(dim=1).item() == 0
    assert torch.allclose(
        observed,
        torch.tensor([[0.01628, 0.01584, 0.0154]]),
        atol=1e-6,
    )
