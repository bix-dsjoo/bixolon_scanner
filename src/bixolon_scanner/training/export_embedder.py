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


def export_embedder(
    checkpoint_path: Path,
    output_path: Path,
    *,
    opset: int = 18,
    include_proxy_scores: bool = False,
    image_size: int | None = None,
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
    model.eval()

    class EmbedderExport(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, pixel_values):
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
        "onnx_sha256": sha256_file(output_path),
        "backbone_kind": checkpoint["backbone_kind"],
        "source_revision": revision,
        "source_weight_sha256": checkpoint.get("source_weight_sha256"),
        "source_weight_filename": checkpoint.get("source_weight_filename"),
        "image_size": image_size,
        "embedding_dimension": 768 + int(checkpoint["num_classes"]) if export_proxy_scores else 768,
        "l2_normalized": export_adapted_features and not export_proxy_scores,
        "embedding_space": (
            "trained_adapter_proxy_fusion"
            if export_proxy_scores
            else "trained_adapter"
            if export_adapted_features
            else "backbone"
        ),
        "training_architecture": checkpoint.get("architecture"),
        "training_scope": training_scope,
        "training_dataset_version": checkpoint.get("dataset_version"),
        "training_manifest_sha256": checkpoint.get("manifest_sha256"),
        "opset": opset,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the frozen 2.0 DINOv3 embedder")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--include-proxy-scores", action="store_true")
    parser.add_argument("--image-size", type=int)
    args = parser.parse_args(argv)
    report = export_embedder(
        args.checkpoint,
        args.output,
        opset=args.opset,
        include_proxy_scores=args.include_proxy_scores,
        image_size=args.image_size,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
