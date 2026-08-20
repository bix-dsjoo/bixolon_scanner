from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts.catalog import sha256_file
from .models import build_dino_classifier, require_torch


def export_embedder(checkpoint_path: Path, output_path: Path, *, opset: int = 18) -> dict:
    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("backbone_kind") != "dinov3_convnext_tiny":
        raise ValueError("2.0 embedder export requires the DINOv3 ConvNeXt Tiny backbone")
    if checkpoint.get("stage") != "frozen":
        raise ValueError("2.0 embedder export requires a frozen-backbone checkpoint")
    revision = checkpoint.get("source_revision")
    if not revision:
        raise ValueError("DINOv3 source revision is missing from the checkpoint")
    model = build_dino_classifier(
        "dinov3_convnext_tiny",
        int(checkpoint["num_classes"]),
        hub_repository=f"facebookresearch/dinov3:{revision}",
        classifier_head_kind="linear",
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    class EmbedderExport(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, pixel_values):
            return self.classifier.extract_features(pixel_values)

    wrapper = EmbedderExport(model).eval()
    image_size = int(checkpoint["image_size"])
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
        "image_size": image_size,
        "embedding_dimension": 768,
        "l2_normalized": False,
        "opset": opset,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the frozen 2.0 DINOv3 embedder")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)
    report = export_embedder(args.checkpoint, args.output, opset=args.opset)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
