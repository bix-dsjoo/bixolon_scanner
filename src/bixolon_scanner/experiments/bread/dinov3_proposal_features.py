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
from .proposal_ranker import proposal_iou_matrix


def proposal_geometry(boxes: np.ndarray, scores: np.ndarray, width: int, height: int) -> np.ndarray:
    normalized = boxes / np.asarray([width, height, width, height], dtype=np.float32)
    box_width = np.maximum(normalized[:, 2] - normalized[:, 0], 1e-6)
    box_height = np.maximum(normalized[:, 3] - normalized[:, 1], 1e-6)
    center_x = (normalized[:, 0] + normalized[:, 2]) * 0.5
    center_y = (normalized[:, 1] + normalized[:, 3]) * 0.5
    area = box_width * box_height
    aspect = np.log(box_width / box_height)
    edge = np.minimum.reduce(
        [normalized[:, 0], normalized[:, 1], 1.0 - normalized[:, 2], 1.0 - normalized[:, 3]]
    )
    return np.stack(
        [scores, center_x, center_y, box_width, box_height, area, aspect, edge],
        axis=1,
    ).astype(np.float32)


def proposal_targets(boxes: np.ndarray, annotations: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(
        [
            [x, y, x + width, y + height]
            for x, y, width, height in (annotation["bbox_xywh"] for annotation in annotations)
        ],
        dtype=np.float32,
    ).reshape(-1, 4)
    ious = proposal_iou_matrix(boxes, targets)
    best = np.argmax(ious, axis=1)
    return (
        ious[np.arange(len(boxes)), best].astype(np.float32),
        np.asarray(
            [int(annotations[index]["category_id"]) for index in best],
            dtype=np.int64,
        ),
    )


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
    prediction_by_id = {}
    for fold in (0, 1, 2):
        path = Path(str(args.predictions_template).format(fold=fold))
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prediction_by_id[int(row["image_id"])] = row
    if set(prediction_by_id) != {int(record["image_id"]) for record in records}:
        raise ValueError("proposal predictions do not align with development records")

    model = load_dinov3_convnext_tiny(args.weights, device=args.device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=args.device).view(1, 3, 1, 1)
    feature_rows = []
    geometry_rows = []
    iou_rows = []
    class_rows = []
    image_rows = []
    fold_rows = []
    box_rows = []
    score_rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset, record in enumerate(records):
            image_id = int(record["image_id"])
            cached = np.asarray(cache_images[int(cache_index[str(image_id)])])
            tensor = torch.from_numpy(cached.copy()).to(args.device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            tensor = (tensor - mean) / std
            maps = model.get_intermediate_layers(
                tensor,
                n=[1, 2, 3],
                reshape=True,
                norm=True,
            )
            prediction = prediction_by_id[image_id]
            boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
            scores = np.asarray(prediction["scores"], dtype=np.float32)
            scaled = boxes * np.asarray(
                [
                    image_size / float(record["width"]),
                    image_size / float(record["height"]),
                    image_size / float(record["width"]),
                    image_size / float(record["height"]),
                ],
                dtype=np.float32,
            )
            scaled_tensor = torch.from_numpy(scaled).to(args.device)
            pooled = []
            for feature_map in maps:
                values = roi_align(
                    feature_map,
                    [scaled_tensor],
                    output_size=(1, 1),
                    spatial_scale=feature_map.shape[-1] / image_size,
                    aligned=True,
                ).flatten(1)
                pooled.append(torch.nn.functional.normalize(values, dim=1))
            features = torch.cat(pooled, dim=1)
            features = torch.nn.functional.normalize(features, dim=1)
            max_iou, target_class = proposal_targets(boxes, record["annotations"])
            feature_rows.append(features.cpu().numpy().astype(np.float16))
            geometry_rows.append(
                proposal_geometry(boxes, scores, int(record["width"]), int(record["height"]))
            )
            iou_rows.append(max_iou)
            class_rows.append(target_class)
            image_rows.append(np.full(len(boxes), image_id, dtype=np.int64))
            fold_rows.append(np.full(len(boxes), int(record["fold"]), dtype=np.int8))
            box_rows.append(boxes)
            score_rows.append(scores)
            if (offset + 1) % 25 == 0:
                print(json.dumps({"images_processed": offset + 1}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.concatenate(feature_rows),
        geometry=np.concatenate(geometry_rows),
        max_iou=np.concatenate(iou_rows),
        target_class=np.concatenate(class_rows),
        image_id=np.concatenate(image_rows),
        fold=np.concatenate(fold_rows),
        boxes_xyxy=np.concatenate(box_rows),
        detector_score=np.concatenate(score_rows),
    )
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_dense_oof_proposal_features",
        "image_count": len(records),
        "proposal_count": int(sum(len(row) for row in box_rows)),
        "feature_dimension": int(feature_rows[0].shape[1]),
        "geometry_dimension": int(geometry_rows[0].shape[1]),
        "positive_iou_0_5_count": int(np.count_nonzero(np.concatenate(iou_rows) >= 0.5)),
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
    parser = argparse.ArgumentParser(description="Cache DINOv3 dense features for OOF proposals")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions-template", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
