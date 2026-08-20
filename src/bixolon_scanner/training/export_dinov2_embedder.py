from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts.catalog import sha256_file
from .models import require_torch


def export_dinov2_embedder(model_dir: Path, output_path: Path, *, opset: int = 18) -> dict:
    torch = require_torch()
    from transformers import AutoModel

    resolved = model_dir.resolve()
    weights = resolved / "model.safetensors"
    if not weights.is_file():
        raise ValueError("DINOv2 model directory must contain model.safetensors")
    model = AutoModel.from_pretrained(
        resolved.as_posix(), local_files_only=True, attn_implementation="eager"
    ).eval()
    if getattr(model.config, "model_type", None) != "dinov2":
        raise ValueError("the selected checkpoint is not a DINOv2 model")

    class EmbedderExport(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, pixel_values):
            return self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]

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
    snapshot_revision = resolved.name if len(resolved.name) == 40 else None
    return {
        "schema_version": "2.0",
        "backbone_kind": "dinov2_base",
        "source_revision": snapshot_revision,
        "source_weight_filename": weights.name,
        "source_weight_sha256": sha256_file(weights),
        "onnx_sha256": sha256_file(output_path),
        "image_size": 224,
        "embedding_dimension": int(model.config.hidden_size),
        "l2_normalized": False,
        "opset": opset,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a frozen DINOv2 embedding backbone")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)
    report = export_dinov2_embedder(args.model_dir, args.output, opset=args.opset)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
