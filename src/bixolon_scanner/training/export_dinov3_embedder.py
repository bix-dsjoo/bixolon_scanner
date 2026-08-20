from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts.catalog import sha256_file
from .models import (
    DINO_V3_CONVNEXT_TINY,
    DINO_V3_VIT_BASE_16,
    build_dino_classifier,
    require_torch,
    set_frozen_backbone,
)

DINO_V3_BACKBONES = (DINO_V3_CONVNEXT_TINY, DINO_V3_VIT_BASE_16)


def export_dinov3_embedder(
    weights_path: Path,
    output_path: Path,
    *,
    backbone_kind: str,
    source_revision: str,
    opset: int = 18,
) -> dict:
    if backbone_kind not in DINO_V3_BACKBONES:
        raise ValueError(f"unsupported DINOv3 backbone: {backbone_kind}")
    if not weights_path.is_file() or not source_revision:
        raise ValueError("DINOv3 export requires locked source weights and revision")
    torch = require_torch()
    model = build_dino_classifier(
        backbone_kind,
        1,
        weights_path=weights_path,
        hub_repository=f"facebookresearch/dinov3:{source_revision}",
    )
    set_frozen_backbone(model)
    model.eval()

    class EmbedderExport(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, pixel_values):
            return self.classifier.extract_features(pixel_values)

    wrapper = EmbedderExport(model).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (torch.zeros(1, 3, 224, 224, dtype=torch.float32),),
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
        "backbone_kind": backbone_kind,
        "source_revision": source_revision,
        "source_weight_filename": weights_path.name,
        "source_weight_sha256": sha256_file(weights_path),
        "onnx_sha256": sha256_file(output_path),
        "image_size": 224,
        "embedding_dimension": 768,
        "l2_normalized": False,
        "opset": opset,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a frozen official DINOv3 embedder")
    parser.add_argument("--backbone", choices=DINO_V3_BACKBONES, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)
    report = export_dinov3_embedder(
        args.weights,
        args.output,
        backbone_kind=args.backbone,
        source_revision=args.source_revision,
        opset=args.opset,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
