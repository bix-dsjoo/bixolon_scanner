from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..runtime.onnx import prepare_rgb


def prediction_row(
    *,
    image_path: str | Path,
    boxes_xyxy: Iterable[Iterable[float]],
    scores: Iterable[float],
    end2end: bool,
) -> dict[str, Any]:
    image_id = int(Path(image_path).stem)
    boxes = [[float(value) for value in box] for box in boxes_xyxy]
    confidences = [float(value) for value in scores]
    if len(boxes) != len(confidences):
        raise ValueError("YOLO boxes and scores are not aligned")
    return {
        "image_id": image_id,
        "boxes_xyxy": boxes,
        "scores": confidences,
        "class_ids": [0] * len(boxes),
        "architecture_contract": "one-class-product-independent-objectness",
        "head": "one-to-one" if end2end else "one-to-many",
    }


def raw_stretch_prediction_row(
    *,
    image_path: Path,
    raw_output: np.ndarray,
    input_size: int,
    original_width: int,
    original_height: int,
    minimum_score: float,
    maximum_detections: int,
) -> dict[str, Any]:
    values = np.asarray(raw_output, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("YOLO stretch output must have two dimensions")
    if values.shape[0] == 5:
        values = values.T
    if values.shape[1] != 5:
        raise ValueError("one-class YOLO stretch output must contain xywh and one score")
    selected = np.flatnonzero(values[:, 4] >= minimum_score)
    selected = selected[np.argsort(-values[selected, 4], kind="stable")[:maximum_detections]]
    boxes = []
    scores = []
    for index in selected:
        cx, cy, width, height, score = [float(value) for value in values[index]]
        boxes.append(
            [
                (cx - width * 0.5) * original_width / input_size,
                (cy - height * 0.5) * original_height / input_size,
                (cx + width * 0.5) * original_width / input_size,
                (cy + height * 0.5) * original_height / input_size,
            ]
        )
        scores.append(score)
    return prediction_row(
        image_path=image_path,
        boxes_xyxy=boxes,
        scores=scores,
        end2end=False,
    )


def _export_stretch_predictions(
    checkpoint: Path,
    image_paths: list[Path],
    *,
    image_size: int,
    minimum_score: float,
    maximum_detections: int,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch
    from ultralytics import YOLO

    torch_device = f"cuda:{device}" if device.isdigit() else device
    model = YOLO(checkpoint).model.to(torch_device).eval()
    model.model[-1].end2end = False
    rows = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            sizes = []
            for path in batch_paths:
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    sizes.append(image.size)
                    tensors.append(
                        prepare_rgb(
                            image,
                            (image_size, image_size),
                            (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0),
                            reducing_gap=1.0,
                        )
                    )
            pixels = torch.from_numpy(np.stack(tensors)).to(torch_device)
            output = model(pixels)
            raw = output[0] if isinstance(output, tuple) else output
            for path, size, item in zip(batch_paths, sizes, raw.float().cpu().numpy()):
                rows.append(
                    raw_stretch_prediction_row(
                        image_path=path,
                        raw_output=item,
                        input_size=image_size,
                        original_width=size[0],
                        original_height=size[1],
                        minimum_score=minimum_score,
                        maximum_detections=maximum_detections,
                    )
                )
    return rows


def export_predictions(
    checkpoint: Path,
    image_dir: Path,
    output: Path,
    *,
    image_size: int,
    minimum_score: float,
    maximum_detections: int,
    end2end: bool,
    device: str,
    resize_mode: str,
    batch_size: int,
) -> dict[str, Any]:
    from ultralytics import YOLO

    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    if not image_paths:
        raise ValueError(f"no validation images found under {image_dir}")
    if resize_mode == "stretch":
        if end2end:
            raise ValueError("stretch evaluation currently requires the one-to-many head")
        rows = _export_stretch_predictions(
            checkpoint,
            image_paths,
            image_size=image_size,
            minimum_score=minimum_score,
            maximum_detections=maximum_detections,
            device=device,
            batch_size=batch_size,
        )
    else:
        model = YOLO(checkpoint)
        results = model.predict(
            source=[str(path) for path in image_paths],
            imgsz=image_size,
            conf=minimum_score,
            max_det=maximum_detections,
            end2end=end2end,
            device=device,
            stream=True,
            verbose=False,
        )
        rows = []
        for image_path, result in zip(image_paths, results, strict=True):
            rows.append(
                prediction_row(
                    image_path=image_path,
                    boxes_xyxy=result.boxes.xyxy.detach().cpu().tolist(),
                    scores=result.boxes.conf.detach().cpu().tolist(),
                    end2end=end2end,
                )
            )
    observed_ids = [int(row["image_id"]) for row in rows]
    expected_ids = [int(path.stem) for path in image_paths]
    if observed_ids != expected_ids:
        raise ValueError("YOLO predictions are not aligned with the validation images")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "image_count": len(rows),
        "prediction_count": sum(len(row["scores"]) for row in rows),
        "image_size": image_size,
        "minimum_score": minimum_score,
        "maximum_detections": maximum_detections,
        "end2end": end2end,
        "resize_mode": resize_mode,
        "product_labels_read": False,
        "output": str(output.resolve()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export generic YOLO Detector predictions")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--minimum-score", type=float, default=0.001)
    parser.add_argument("--maximum-detections", type=int, default=300)
    parser.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resize-mode", choices=("letterbox", "stretch"), default="letterbox")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            export_predictions(
                args.checkpoint,
                args.image_dir,
                args.output,
                image_size=args.image_size,
                minimum_score=args.minimum_score,
                maximum_detections=args.maximum_detections,
                end2end=args.end2end,
                device=args.device,
                resize_mode=args.resize_mode,
                batch_size=args.batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
