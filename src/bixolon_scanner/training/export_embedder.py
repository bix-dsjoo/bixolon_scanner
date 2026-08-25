from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts.catalog import sha256_file
from .fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from .models import build_dino_classifier, require_torch


def spatial_statistics_features(values):
    """Return channel-wise mean and standard deviation for a ConvNeXt feature map."""
    mean = values.mean(dim=(-2, -1))
    centered = values - mean[:, :, None, None]
    standard_deviation = centered.square().mean(dim=(-2, -1)).clamp_min(1e-12).sqrt()
    return mean, standard_deviation


def positive_gem_features(values, *, power: float = 3.0):
    """Pool positive spatial activations with a fixed generalized mean."""
    if power <= 0.0:
        raise ValueError("GeM pooling power must be positive")
    return values.clamp_min(0.0).pow(power).mean(dim=(-2, -1)).clamp_min(1e-12).pow(1.0 / power)


def export_embedder(
    checkpoint_path: Path,
    output_path: Path,
    *,
    secondary_checkpoint_path: Path | None = None,
    opset: int = 18,
    include_proxy_scores: bool = False,
    image_size: int | None = None,
    multiscale_pooling: bool = False,
    statistics_pooling: bool = False,
    gem_pooling: bool = False,
) -> dict:
    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("backbone_kind") != "dinov3_convnext_tiny":
        raise ValueError("2.0 embedder export requires the DINOv3 ConvNeXt Tiny backbone")
    revision = checkpoint.get("source_revision")
    if not revision:
        raise ValueError("DINOv3 source revision is missing from the checkpoint")
    if checkpoint.get("stage") == "frozen" and "state_dict" in checkpoint:
        model = build_dino_classifier(
            "dinov3_convnext_tiny",
            int(checkpoint["num_classes"]),
            hub_repository=f"facebookresearch/dinov3:{revision}",
            classifier_head_kind="linear",
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        training_scope = "frozen_backbone"
        export_adapted_features = False
        export_proxy_scores = False
    elif "adapter_spec" in checkpoint and "model_state_dict" in checkpoint:
        model = build_ten_shot_classifier(
            backbone_kind="dinov3_convnext_tiny",
            weights_path=None,
            hub_repository=f"facebookresearch/dinov3:{revision}",
            spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
        )
        model.load_state_dict(
            compatible_proxy_state_dict(checkpoint["model_state_dict"]), strict=True
        )
        challenger = checkpoint.get("challenger") or {}
        training_scope = str(challenger.get("trainable_scope") or "trained_backbone")
        export_adapted_features = True
        export_proxy_scores = include_proxy_scores
    else:
        raise ValueError("unsupported DINOv3 ConvNeXt Tiny checkpoint layout")
    secondary_model = None
    secondary_checkpoint = None
    if secondary_checkpoint_path is not None:
        if not export_adapted_features:
            raise ValueError("shared-prefix dual export requires adapted checkpoints")
        secondary_checkpoint = torch.load(
            secondary_checkpoint_path, map_location="cpu", weights_only=True
        )
        if (
            secondary_checkpoint.get("backbone_kind") != checkpoint.get("backbone_kind")
            or secondary_checkpoint.get("source_revision") != revision
            or secondary_checkpoint.get("adapter_spec") != checkpoint.get("adapter_spec")
            or secondary_checkpoint.get("manifest_sha256") != checkpoint.get("manifest_sha256")
        ):
            raise ValueError("shared-prefix dual checkpoints are not source compatible")
        secondary_model = build_ten_shot_classifier(
            backbone_kind="dinov3_convnext_tiny",
            weights_path=None,
            hub_repository=f"facebookresearch/dinov3:{revision}",
            spec=adapter_spec_from_dict(secondary_checkpoint["adapter_spec"]),
        )
        secondary_model.load_state_dict(
            compatible_proxy_state_dict(secondary_checkpoint["model_state_dict"]), strict=True
        )
        primary_state = model.state_dict()
        secondary_state = secondary_model.state_dict()
        branch_prefixes = (
            "backbone.stages.3.",
            "backbone.norm.",
            "backbone.norms.3.",
            "classifier.",
        )
        if any(
            not torch.equal(value, secondary_state[name])
            for name, value in primary_state.items()
            if not name.startswith(branch_prefixes)
        ):
            raise ValueError("dual embedder checkpoints do not share one frozen prefix")
    if multiscale_pooling and not export_adapted_features:
        raise ValueError("multi-scale pooling requires an adapted ConvNeXt checkpoint")
    if multiscale_pooling and export_proxy_scores:
        raise ValueError("multi-scale pooling and proxy-score fusion are mutually exclusive")
    if statistics_pooling and not export_adapted_features:
        raise ValueError("statistics pooling requires an adapted ConvNeXt checkpoint")
    if statistics_pooling and (multiscale_pooling or export_proxy_scores):
        raise ValueError("statistics pooling cannot use another output extension")
    if gem_pooling and not export_adapted_features:
        raise ValueError("GeM pooling requires an adapted ConvNeXt checkpoint")
    if gem_pooling and (multiscale_pooling or statistics_pooling or export_proxy_scores):
        raise ValueError("GeM pooling cannot use another output extension")
    if secondary_model is not None and (
        multiscale_pooling or statistics_pooling or gem_pooling or export_proxy_scores
    ):
        raise ValueError("shared-prefix dual export cannot use another output extension")
    model.eval()
    if secondary_model is not None:
        secondary_model.eval()

    class EmbedderExport(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier
            self.secondary = secondary_model

        def forward(self, pixel_values):
            if self.secondary is not None:
                values = pixel_values
                for index in range(3):
                    values = self.classifier.backbone.downsample_layers[index](values)
                    values = self.classifier.backbone.stages[index](values)
                values = self.classifier.backbone.downsample_layers[3](values)
                branches = []
                for branch in (self.classifier, self.secondary):
                    branch_values = branch.backbone.stages[3](values)
                    branch_features = branch.backbone.norm(branch_values.mean(dim=(-2, -1)))
                    branches.append(branch.classifier.adapt(branch_features))
                return torch.nn.functional.normalize(torch.cat(branches, dim=-1), dim=-1)
            if multiscale_pooling:
                values = pixel_values
                pooled = []
                for index in range(4):
                    values = self.classifier.backbone.downsample_layers[index](values)
                    values = self.classifier.backbone.stages[index](values)
                    stage = values.mean(dim=(-2, -1))
                    if index == 3:
                        stage = self.classifier.backbone.norm(stage)
                        stage = self.classifier.classifier.adapt(stage)
                    pooled.append(torch.nn.functional.normalize(stage, dim=-1))
                return torch.nn.functional.normalize(torch.cat(pooled, dim=-1), dim=-1)
            if statistics_pooling:
                values = pixel_values
                for index in range(4):
                    values = self.classifier.backbone.downsample_layers[index](values)
                    values = self.classifier.backbone.stages[index](values)
                mean, standard_deviation = spatial_statistics_features(values)
                adapted_mean = self.classifier.classifier.adapt(self.classifier.backbone.norm(mean))
                normalized_deviation = torch.nn.functional.normalize(standard_deviation, dim=-1)
                return torch.nn.functional.normalize(
                    torch.cat((adapted_mean, normalized_deviation), dim=-1), dim=-1
                )
            if gem_pooling:
                values = pixel_values
                for index in range(4):
                    values = self.classifier.backbone.downsample_layers[index](values)
                    values = self.classifier.backbone.stages[index](values)
                mean = values.mean(dim=(-2, -1))
                adapted_mean = self.classifier.classifier.adapt(self.classifier.backbone.norm(mean))
                spatially_normalized = self.classifier.backbone.norm(
                    values.permute(0, 2, 3, 1)
                ).permute(0, 3, 1, 2)
                gem = torch.nn.functional.normalize(
                    positive_gem_features(spatially_normalized), dim=-1
                )
                return torch.nn.functional.normalize(torch.cat((adapted_mean, gem), dim=-1), dim=-1)
            features = self.classifier.extract_features(pixel_values)
            if not export_adapted_features:
                return features
            if isinstance(features, tuple):
                adapted = self.classifier.classifier.adapt(*features)
            else:
                adapted = self.classifier.classifier.adapt(features)
            if not export_proxy_scores:
                return adapted
            proxy_scores = self.classifier.classifier._class_scores(adapted)
            return torch.cat((adapted, proxy_scores), dim=-1)

    wrapper = EmbedderExport(model).eval()
    image_size = int(checkpoint["image_size"]) if image_size is None else image_size
    if image_size < 32 or image_size % 32:
        raise ValueError("ConvNeXt embedder export size must be a positive multiple of 32")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (torch.zeros(1, 3, image_size, image_size, dtype=torch.float32),),
        output_path,
        input_names=["pixel_values"],
        output_names=["embeddings"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(output_path))
    return {
        "schema_version": "2.0",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "secondary_checkpoint": (
            None if secondary_checkpoint_path is None else str(secondary_checkpoint_path)
        ),
        "secondary_checkpoint_sha256": (
            None if secondary_checkpoint_path is None else sha256_file(secondary_checkpoint_path)
        ),
        "onnx_sha256": sha256_file(output_path),
        "backbone_kind": checkpoint["backbone_kind"],
        "source_revision": revision,
        "source_weight_sha256": checkpoint.get("source_weight_sha256"),
        "source_weight_filename": checkpoint.get("source_weight_filename"),
        "image_size": image_size,
        "embedding_dimension": (
            1536
            if secondary_model is not None or statistics_pooling or gem_pooling
            else 96 + 192 + 384 + 768
            if multiscale_pooling
            else 768 + int(checkpoint["num_classes"])
            if export_proxy_scores
            else 768
        ),
        "l2_normalized": export_adapted_features and not export_proxy_scores,
        "embedding_space": (
            "shared_frozen_prefix_dual_trained_adapter"
            if secondary_model is not None
            else "trained_adapter_mean_standard_deviation_pooling"
            if statistics_pooling
            else "trained_adapter_positive_gem_pooling"
            if gem_pooling
            else "trained_adapter_multiscale_pooling"
            if multiscale_pooling
            else "trained_adapter_proxy_fusion"
            if export_proxy_scores
            else "trained_adapter"
            if export_adapted_features
            else "backbone"
        ),
        "training_architecture": (
            f"{checkpoint.get('architecture')}+shared-prefix-dual-last-stage"
            if secondary_model is not None
            else f"{checkpoint.get('architecture')}+mean-standard-deviation-pooling"
            if statistics_pooling
            else f"{checkpoint.get('architecture')}+positive-gem-pooling"
            if gem_pooling
            else f"{checkpoint.get('architecture')}+four-stage-normalized-pooling"
            if multiscale_pooling
            else checkpoint.get("architecture")
        ),
        "training_scope": (
            "shared_frozen_prefix_dual_last_stage"
            if secondary_model is not None
            else training_scope
        ),
        "training_dataset_version": checkpoint.get("dataset_version"),
        "training_manifest_sha256": checkpoint.get("manifest_sha256"),
        "opset": opset,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the frozen 2.0 DINOv3 embedder")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--secondary-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--include-proxy-scores", action="store_true")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--multiscale-pooling", action="store_true")
    parser.add_argument("--statistics-pooling", action="store_true")
    parser.add_argument("--gem-pooling", action="store_true")
    args = parser.parse_args(argv)
    report = export_embedder(
        args.checkpoint,
        args.output,
        secondary_checkpoint_path=args.secondary_checkpoint,
        opset=args.opset,
        include_proxy_scores=args.include_proxy_scores,
        image_size=args.image_size,
        multiscale_pooling=args.multiscale_pooling,
        statistics_pooling=args.statistics_pooling,
        gem_pooling=args.gem_pooling,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
