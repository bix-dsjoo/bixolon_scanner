from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ....training.fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from ....training.models import require_torch

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def _rgb(tensor: np.ndarray) -> np.ndarray:
    values = tensor.transpose(1, 2, 0) * STD + MEAN
    return np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


def _tensor(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32) / 255.0
    return ((values - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def isolate_foreground(
    tensor: np.ndarray,
    *,
    color_distance: float,
    selection: str,
    refit_padding: float | None,
) -> np.ndarray:
    image = _rgb(tensor)
    height, width = image.shape[:2]
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0)
    background = np.median(border, axis=0).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(background.reshape(1, 1, 3), cv2.COLOR_RGB2LAB).astype(
        np.float32
    )[0, 0]
    distance = np.linalg.norm(lab - background_lab, axis=2)
    mask = (distance >= color_distance).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    center = np.asarray((width / 2.0, height / 2.0), dtype=np.float32)
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        if area < max(32, round(height * width * 0.002)):
            continue
        touches_border = (
            x <= 1
            or y <= 1
            or x + component_width >= width - 1
            or y + component_height >= height - 1
        )
        distance_to_center = float(np.linalg.norm(centroids[label] - center))
        centrality = np.exp(-((distance_to_center / (0.38 * min(height, width))) ** 2))
        candidates.append(
            {
                "label": label,
                "area": int(area),
                "touches_border": touches_border,
                "score": float(area * (0.25 + 0.75 * centrality)),
            }
        )
    if not candidates:
        return tensor.astype(np.float32, copy=True)
    if selection == "largest_nonborder":
        nonborder = [row for row in candidates if not row["touches_border"]]
        chosen = max(nonborder or candidates, key=lambda row: row["area"])
    elif selection == "center_weighted":
        chosen = max(candidates, key=lambda row: row["score"])
    elif selection == "largest":
        chosen = max(candidates, key=lambda row: row["area"])
    else:
        raise ValueError(f"unsupported component selection: {selection}")
    selected = labels == chosen["label"]
    selected = cv2.dilate(selected.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    isolated = np.broadcast_to(background, image.shape).copy()
    isolated[selected] = image[selected]
    if refit_padding is not None:
        y_values, x_values = np.nonzero(selected)
        left, right = int(x_values.min()), int(x_values.max()) + 1
        top, bottom = int(y_values.min()), int(y_values.max()) + 1
        padding = round(max(right - left, bottom - top) * refit_padding)
        left, top = max(0, left - padding), max(0, top - padding)
        right, bottom = min(width, right + padding), min(height, bottom + padding)
        crop = isolated[top:bottom, left:right]
        scale = min(width / crop.shape[1], height / crop.shape[0])
        resized = cv2.resize(
            crop,
            (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        isolated = np.broadcast_to(background, image.shape).copy()
        target_top = (height - resized.shape[0]) // 2
        target_left = (width - resized.shape[1]) // 2
        isolated[
            target_top : target_top + resized.shape[0],
            target_left : target_left + resized.shape[1],
        ] = resized
    return _tensor(isolated)


def _center_crop(values, scale: float):
    torch = require_torch()
    if scale == 1.0:
        return values
    height, width = values.shape[-2:]
    crop_height, crop_width = round(height * scale), round(width * scale)
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    return torch.nn.functional.interpolate(
        values[..., top : top + crop_height, left : left + crop_width],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


def _logits(model, tensors: np.ndarray, *, crop_scale: float, device, batch_size: int):
    torch = require_torch()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(
                np.array(tensors[start : start + batch_size], dtype=np.float32, copy=True)
            ).to(device)
            parts.append(model(_center_crop(batch, crop_scale)).float().cpu().numpy())
    return np.concatenate(parts)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_ten_shot_classifier(
        backbone_kind=str(checkpoint["backbone_kind"]),
        weights_path=None,
        hub_repository="facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751",
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    device = torch.device("cpu" if args.cpu else "cuda")
    model = model.to(device).eval()
    tensors = np.load(args.tensors, mmap_mode="r")
    rows = [
        json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line
    ]
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    results = []
    logits_by_variant = {}
    baseline = _logits(
        model,
        tensors,
        crop_scale=args.baseline_crop_scale,
        device=device,
        batch_size=args.batch_size,
    )
    logits_by_variant["baseline"] = baseline
    results.append(
        {
            "name": "baseline",
            "top1_accuracy": float((baseline.argmax(axis=1) == targets).mean()),
        }
    )
    for color_distance in args.color_distances:
        for selection in ("largest_nonborder", "center_weighted", "largest"):
            for refit_padding in (None, 0.03, 0.08, 0.15):
                transformed = np.asarray(
                    [
                        isolate_foreground(
                            tensor,
                            color_distance=color_distance,
                            selection=selection,
                            refit_padding=refit_padding,
                        )
                        for tensor in tensors
                    ],
                    dtype=np.float32,
                )
                for crop_scale in (args.baseline_crop_scale,) if refit_padding is None else (1.0,):
                    name = (
                        f"distance{color_distance:g}:{selection}:"
                        f"padding{refit_padding}:crop{crop_scale}"
                    )
                    logits = _logits(
                        model,
                        transformed,
                        crop_scale=crop_scale,
                        device=device,
                        batch_size=args.batch_size,
                    )
                    logits_by_variant[name] = logits
                    results.append(
                        {
                            "name": name,
                            "top1_accuracy": float((logits.argmax(axis=1) == targets).mean()),
                        }
                    )
                print(json.dumps(results[-1]), flush=True)
    results.sort(key=lambda row: (row["top1_accuracy"], row["name"]), reverse=True)
    best_name = results[0]["name"]
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.logits_output,
        baseline=baseline,
        best=logits_by_variant[best_name],
        targets=targets,
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "label_free_foreground_component_isolation_probe",
        "sample_count": len(targets),
        "selected": results[0],
        "top_candidates": results[:20],
        "passes_top1_gate": results[0]["top1_accuracy"] >= 0.99,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe foreground component isolation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits-output", type=Path, required=True)
    parser.add_argument("--color-distances", type=float, nargs="+", default=(16, 24, 32, 40, 48))
    parser.add_argument("--baseline-crop-scale", type=float, default=0.855)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
