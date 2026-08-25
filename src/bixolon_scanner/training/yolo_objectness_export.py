from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_outputs(raw_output: np.ndarray, *, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw_output, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("YOLO canonical export expects a batched output")
    if values.shape[1] == 5:
        values = np.transpose(values, (0, 2, 1))
    if values.shape[2] != 5:
        raise ValueError("one-class YOLO output must contain xywh and one score")
    probabilities = np.clip(values[:, :, 4:5], 1e-6, 1.0 - 1e-6)
    logits = np.log(probabilities / (1.0 - probabilities))
    boxes = values[:, :, :4] / np.float32(image_size)
    return logits.astype(np.float32), boxes.astype(np.float32)


def export(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    model = YOLO(args.checkpoint).model.eval().cpu()
    model.model[-1].end2end = False

    class CanonicalObjectness(torch.nn.Module):
        def __init__(self, detector, image_size: int):
            super().__init__()
            self.detector = detector
            self.image_size = float(image_size)

        def forward(self, pixel_values):
            output = self.detector(pixel_values)
            raw = output[0] if isinstance(output, tuple) else output
            values = raw.transpose(1, 2)
            probabilities = values[:, :, 4:5].clamp(1e-6, 1.0 - 1e-6)
            logits = torch.log(probabilities / (1.0 - probabilities))
            boxes = values[:, :, :4] / self.image_size
            return logits, boxes

    wrapper = CanonicalObjectness(model, args.image_size).eval()
    dummy = torch.zeros((1, 3, args.image_size, args.image_size), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    report = {
        "schema_version": "1.0",
        "architecture": "YOLO26n one-to-many one-class objectness",
        "architecture_contract": "one-class-product-independent-objectness",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "onnx": str(args.output.resolve()),
        "onnx_sha256": _sha256(args.output),
        "input_name": "pixel_values",
        "input_size": [args.image_size, args.image_size],
        "resize_mode": "stretch",
        "outputs": {
            "logits": "one-class pre-sigmoid objectness",
            "pred_boxes": "normalized_cxcywh",
        },
        "opset": args.opset,
        "product_class_count": 0,
        "objectness_class_count": 1,
        "product_labels_exported": False,
        "license": "AGPL-3.0-only",
        "source": "https://github.com/ultralytics/ultralytics",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export YOLO objectness to the Runtime contract")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=18)
    export(parser.parse_args(argv))


if __name__ == "__main__":
    main()
