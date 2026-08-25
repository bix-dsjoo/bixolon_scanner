from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..runtime.onnx import prepare_rgb
from ..runtime.onnx_session import OrtRunner
from .onnx_detector import raw_outputs_to_prediction


def filter_prediction(
    prediction: dict[str, Any], *, minimum_score: float, maximum_detections: int
) -> dict[str, Any]:
    selected = [index for index, score in enumerate(prediction["scores"]) if score >= minimum_score]
    selected = sorted(selected, key=lambda index: prediction["scores"][index], reverse=True)[
        :maximum_detections
    ]
    return {
        "boxes_xyxy": [prediction["boxes_xyxy"][index] for index in selected],
        "scores": [prediction["scores"][index] for index in selected],
        "class_ids": [0] * len(selected),
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    image_paths = sorted(path for path in args.image_dir.iterdir() if path.is_file())
    runner = OrtRunner(args.model, args.provider, args.cuda_dll_dir)
    rows = []
    for path in image_paths:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            tensor = prepare_rgb(
                image,
                (args.image_size, args.image_size),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                reducing_gap=1.0,
            )[None]
        logits, boxes = runner.run([args.logits_output, args.boxes_output], args.input_name, tensor)
        prediction = raw_outputs_to_prediction(
            np.asarray(logits)[0],
            np.asarray(boxes)[0],
            image_width=width,
            image_height=height,
        )
        row = filter_prediction(
            prediction,
            minimum_score=args.minimum_score,
            maximum_detections=args.maximum_detections,
        )
        row.update(
            {
                "image_id": int(path.stem),
                "architecture_contract": "one-class-product-independent-objectness",
                "head": "one-to-many",
                "resize_mode": "stretch",
            }
        )
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "model": str(args.model.resolve()),
        "provider": args.provider,
        "image_count": len(rows),
        "prediction_count": sum(len(row["scores"]) for row in rows),
        "minimum_score": args.minimum_score,
        "maximum_detections": args.maximum_detections,
        "output": str(args.output.resolve()),
    }
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export canonical YOLO ONNX predictions")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--logits-output", default="logits")
    parser.add_argument("--boxes-output", default="pred_boxes")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--minimum-score", type=float, default=0.001)
    parser.add_argument("--maximum-detections", type=int, default=300)
    export(parser.parse_args(argv))


if __name__ == "__main__":
    main()
