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
    parser = argparse.ArgumentParser(description="Export the fresh store-2 DINOv3 embedder")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument(
        "--output-kind",
        choices=("features", "cosine-logits", "one-hot"),
        default="features",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch = require_torch()
    model = build_dino_classifier(
        "dinov3_convnext_tiny",
        20,
        weights_path=args.weights,
        feature_l2_normalize=True,
        classifier_head_kind="cosine",
        cosine_scale=16.0,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    class _Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            features = self.model.extract_features(pixel_values)
            if args.output_kind == "features":
                return features
            logits = self.model.classifier(features)
            if args.output_kind == "one-hot":
                return torch.nn.functional.one_hot(logits.argmax(dim=-1), num_classes=20).to(
                    dtype=features.dtype
                )
            return torch.nn.functional.normalize(logits, dim=-1)

    wrapper = _Wrapper().eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        torch.zeros(1, 3, args.image_size, args.image_size),
        args.output,
        input_names=["pixel_values"],
        output_names=["embeddings"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    report = {
        "schema_version": "1.0",
        "architecture": "dinov3_convnext_tiny_finetuned_store2",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "official_weights": str(args.weights),
        "official_weights_sha256": _sha256(args.weights),
        "input_size": args.image_size,
        "output_kind": args.output_kind,
        "embedding_dimension": 768 if args.output_kind == "features" else 20,
        "l2_normalized": args.output_kind != "one-hot",
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
