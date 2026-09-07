from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bixolon_scanner.training.models import build_dino_classifier, require_torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a frozen DINOv3 feature embedder")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("dinov3_convnext_tiny", "dinov3_vitb16"),
        required=True,
    )
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--fixed-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.fixed_batch_size < 1:
        raise ValueError("fixed batch size must be positive")

    torch = require_torch()
    model = build_dino_classifier(
        args.variant,
        20,
        weights_path=args.weights,
        feature_l2_normalize=True,
        classifier_head_kind="cosine",
        cosine_scale=16.0,
    ).eval()

    class _Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            return self.model.extract_features(pixel_values)

    wrapper = _Wrapper().eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        torch.zeros(args.fixed_batch_size, 3, args.image_size, args.image_size),
        args.output,
        input_names=["pixel_values"],
        output_names=["embeddings"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    dimension = 768
    report = {
        "schema_version": "1.0",
        "training_scope": "frozen_backbone",
        "training_architecture": f"{args.variant} frozen feature embedder",
        "backbone_kind": args.variant,
        "source_revision": "6876159a11b4df116f30f667f8c9888617df0751",
        "source_weight_filename": args.weights.name,
        "source_weight_sha256": _sha256(args.weights),
        "image_size": args.image_size,
        "fixed_batch_size": args.fixed_batch_size,
        "embedding_dimension": dimension,
        "l2_normalized": True,
        "hardware_specific_optimization": False,
        "onnx_sha256": _sha256(args.output),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
