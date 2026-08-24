from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts.catalog import sha256_file
from .models import DINO_V3_HUB_REPOSITORY, require_torch


def export_frozen_dinov3_vits16(
    weights_path: Path,
    output_path: Path,
    *,
    image_size: int = 224,
    opset: int = 18,
) -> dict:
    """Export the official frozen DINOv3 ViT-S/16 CLS representation."""
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if image_size < 16 or image_size % 16:
        raise ValueError("DINOv3 ViT-S/16 input size must be a positive multiple of 16")

    torch = require_torch()
    revision = DINO_V3_HUB_REPOSITORY.rsplit(":", 1)[-1]
    model = torch.hub.load(
        DINO_V3_HUB_REPOSITORY,
        "dinov3_vits16",
        source="github",
        trust_repo=True,
        verbose=False,
        pretrained=False,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    class ClsEmbedder(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, pixel_values):
            return self.backbone.forward_features(pixel_values, masks=None)["x_norm_clstoken"]

    wrapper = ClsEmbedder(model).eval()

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
        "backbone_kind": "dinov3_vits16",
        "source_revision": revision,
        "source_weight_filename": weights_path.name,
        "source_weight_sha256": sha256_file(weights_path),
        "onnx_sha256": sha256_file(output_path),
        "image_size": image_size,
        "embedding_dimension": 384,
        "l2_normalized": True,
        "embedding_space": "frozen_backbone",
        "training_architecture": "DINOv3 ViT-S/16 frozen CLS embedding backbone",
        "training_scope": "frozen_backbone",
        "training_dataset_version": None,
        "training_manifest_sha256": None,
        "opset": opset,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the frozen DINOv3 ViT-S/16 embedder")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)
    report = export_frozen_dinov3_vits16(
        args.weights,
        args.output,
        image_size=args.image_size,
        opset=args.opset,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
