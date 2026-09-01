from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ...training.data import read_manifest
from ...training.dinov3_objectness_detector import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_dinov3_convnext_tiny,
)


def expand_boxes(boxes, *, ratio: float, maximum: float):
    import torch

    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    size = (boxes[:, 2:] - boxes[:, :2]) * (1.0 + 2.0 * ratio)
    expanded = torch.cat((center - size * 0.5, center + size * 0.5), dim=1)
    return expanded.clamp_(0.0, maximum)


def run(args: argparse.Namespace) -> dict:
    import torch
    from torchvision.ops import roi_align

    records = [
        record
        for record in read_manifest(args.manifest)
        if record["record_type"] == "detection"
        and record["split"] == "development"
        and not record.get("exclude_from_detector_training", False)
    ]
    cache_metadata = json.loads((args.image_cache / "index.json").read_text(encoding="utf-8"))
    cache_images = np.load(args.image_cache / cache_metadata["array_filename"], mmap_mode="r")
    cache_index = cache_metadata["index"]
    image_size = int(cache_metadata["image_size"])
    predictions = {}
    for fold in (0, 1, 2):
        path = Path(str(args.predictions_template).format(fold=fold))
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            predictions[int(row["image_id"])] = row

    model = load_dinov3_convnext_tiny(args.weights, device=args.device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=args.device).view(1, 3, 1, 1)
    feature_rows = []
    image_rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset, record in enumerate(records):
            image_id = int(record["image_id"])
            image = np.asarray(cache_images[int(cache_index[str(image_id)])])
            tensor = torch.from_numpy(image.copy()).to(args.device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            boxes = torch.as_tensor(
                predictions[image_id]["boxes_xyxy"], dtype=torch.float32, device=args.device
            ).reshape(-1, 4)
            boxes *= torch.tensor(
                [
                    image_size / float(record["width"]),
                    image_size / float(record["height"]),
                    image_size / float(record["width"]),
                    image_size / float(record["height"]),
                ],
                device=args.device,
            )
            boxes = expand_boxes(boxes, ratio=args.context_ratio, maximum=float(image_size))
            crops = roi_align(
                tensor,
                [boxes],
                output_size=(args.crop_size, args.crop_size),
                spatial_scale=1.0,
                aligned=True,
            )
            outputs = []
            for start in range(0, len(crops), args.batch_size):
                batch = (crops[start : start + args.batch_size] - mean) / std
                values = model(batch)
                outputs.append(torch.nn.functional.normalize(values, dim=1).cpu())
            features = torch.cat(outputs).numpy().astype(np.float16)
            feature_rows.append(features)
            image_rows.append(np.full(len(features), image_id, dtype=np.int64))
            if (offset + 1) % 25 == 0:
                print(json.dumps({"images_processed": offset + 1}), flush=True)

    features = np.concatenate(feature_rows)
    image_ids = np.concatenate(image_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=features, image_id=image_ids)
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_crop_oof_proposal_features",
        "image_count": len(records),
        "proposal_count": len(features),
        "feature_dimension": int(features.shape[1]),
        "crop_size": args.crop_size,
        "context_ratio": args.context_ratio,
        "elapsed_seconds": time.perf_counter() - started,
        "existing_detector_used": False,
        "yolo_family_used": False,
        "rfdetr_used": False,
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cache DINOv3 crop features for OOF proposals")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions-template", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--context-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
