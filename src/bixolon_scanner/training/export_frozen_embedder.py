from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts.catalog import sha256_file
from .models import DINO_V3_HUB_REPOSITORY, require_torch


def export_frozen_dinov3_backbone(
    weights_path: Path,
    output_path: Path,
    *,
    variant: str = "dinov3_vits16",
    image_size: int = 224,
    fixed_batch_size: int | None = None,
    opset: int = 18,
) -> dict:
    """Export an official frozen DINOv3 global image representation."""
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if image_size < 16 or image_size % 16:
        raise ValueError("DINOv3 input size must be a positive multiple of 16")
    dimensions = {
        "dinov3_vits16": 384,
        "dinov3_vitb16": 768,
        "dinov3_convnext_tiny": 768,
    }
    if variant not in dimensions:
        raise ValueError("unsupported frozen DINOv3 ViT/16 variant")
    if fixed_batch_size is not None and fixed_batch_size < 1:
        raise ValueError("fixed DINOv3 batch size must be positive")

    torch = require_torch()
    revision = DINO_V3_HUB_REPOSITORY.rsplit(":", 1)[-1]
    model = torch.hub.load(
        DINO_V3_HUB_REPOSITORY,
        variant,
        source="github",
        trust_repo=True,
        verbose=False,
        pretrained=False,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    class GlobalEmbedder(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, pixel_values):
            return self.backbone.forward_features(pixel_values, masks=None)["x_norm_clstoken"]

    wrapper = GlobalEmbedder(model).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (
            torch.zeros(
                1 if fixed_batch_size is None else fixed_batch_size,
                3,
                image_size,
                image_size,
                dtype=torch.float32,
            ),
        ),
        output_path,
        input_names=["pixel_values"],
        output_names=["embeddings"],
        dynamic_axes=(
            {"pixel_values": {0: "batch"}, "embeddings": {0: "batch"}}
            if fixed_batch_size is None
            else None
        ),
        opset_version=opset,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(output_path))
    return {
        "schema_version": "2.0",
        "backbone_kind": variant,
        "source_revision": revision,
        "source_weight_filename": weights_path.name,
        "source_weight_sha256": sha256_file(weights_path),
        "onnx_sha256": sha256_file(output_path),
        "image_size": image_size,
        "embedding_dimension": dimensions[variant],
        "fixed_batch_size": fixed_batch_size,
        "l2_normalized": True,
        "embedding_space": "frozen_backbone",
        "training_architecture": (
            f"DINOv3 {variant.removeprefix('dinov3_')} frozen global embedding backbone"
        ),
        "training_scope": "frozen_backbone",
        "training_dataset_version": None,
        "training_manifest_sha256": None,
        "opset": opset,
    }


def export_frozen_dinov3_vit16(
    weights_path: Path,
    output_path: Path,
    *,
    variant: str = "dinov3_vits16",
    image_size: int = 224,
    fixed_batch_size: int | None = None,
    opset: int = 18,
) -> dict:
    """Backward-compatible entry point for the frozen DINOv3 exporter."""
    return export_frozen_dinov3_backbone(
        weights_path,
        output_path,
        variant=variant,
        image_size=image_size,
        fixed_batch_size=fixed_batch_size,
        opset=opset,
    )


def export_frozen_dinov3_vits16(
    weights_path: Path,
    output_path: Path,
    *,
    image_size: int = 224,
    opset: int = 18,
) -> dict:
    """Backward-compatible dynamic-batch ViT-S/16 export."""
    return export_frozen_dinov3_vit16(
        weights_path,
        output_path,
        variant="dinov3_vits16",
        image_size=image_size,
        opset=opset,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the frozen DINOv3 ViT-S/16 embedder")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--variant",
        choices=("dinov3_vits16", "dinov3_vitb16", "dinov3_convnext_tiny"),
        default="dinov3_vits16",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fixed-batch-size", type=int)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)
    report = export_frozen_dinov3_backbone(
        args.weights,
        args.output,
        variant=args.variant,
        image_size=args.image_size,
        fixed_batch_size=args.fixed_batch_size,
        opset=args.opset,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
