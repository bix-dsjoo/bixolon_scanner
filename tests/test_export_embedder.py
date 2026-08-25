from pathlib import Path

import pytest
import torch

from bixolon_scanner.training.export_dinov2_embedder import dinov2_variant
from bixolon_scanner.training.export_embedder import (
    positive_gem_features,
    spatial_statistics_features,
)
from bixolon_scanner.training.export_frozen_embedder import export_frozen_dinov3_vit16


def test_spatial_statistics_features_preserve_channel_mean_and_variation() -> None:
    values = torch.tensor(
        [
            [
                [[1.0, 3.0], [5.0, 7.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ]
        ]
    )
    mean, standard_deviation = spatial_statistics_features(values)

    assert torch.allclose(mean, torch.tensor([[4.0, 2.0]]))
    assert torch.allclose(
        standard_deviation,
        torch.tensor([[(5.0**0.5), 1e-6]]),
    )


def test_positive_gem_features_ignore_negative_activations() -> None:
    values = torch.tensor([[[[-2.0, 1.0], [3.0, -4.0]]]])

    pooled = positive_gem_features(values, power=2.0)

    assert torch.allclose(pooled, torch.tensor([[(10.0 / 4.0) ** 0.5]]))


def test_frozen_vit_export_rejects_unknown_variant_before_loading_weights(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not-a-checkpoint")

    with pytest.raises(ValueError, match="unsupported frozen DINOv3"):
        export_frozen_dinov3_vit16(
            weights,
            tmp_path / "model.onnx",
            variant="dinov3_unknown",
        )


def test_dinov2_variant_uses_architecture_shape() -> None:
    assert dinov2_variant(hidden_size=384, layer_count=12) == "dinov2_small"
    assert dinov2_variant(hidden_size=768, layer_count=12) == "dinov2_base"

    with pytest.raises(ValueError, match="unsupported DINOv2"):
        dinov2_variant(hidden_size=512, layer_count=12)
