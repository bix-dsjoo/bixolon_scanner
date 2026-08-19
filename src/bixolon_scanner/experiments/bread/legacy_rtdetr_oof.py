from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...contracts.model_package import sha256_file
from ...evaluation.detector import _metrics, detection_error_rows
from ...training.data import read_manifest
from ...training.models import require_torch


def checkpoint_route(
    record: dict[str, Any], old_records_by_id: dict[int, dict[str, Any]]
) -> tuple[str, str]:
    source_id = record.get("source_image_id")
    old = old_records_by_id.get(int(source_id)) if source_id is not None else None
    if old is not None and old["split"] == "development":
        return f"fold{int(old['fold'])}", "legacy_outer_fold"
    if old is not None and old["split"] == "test":
        return "final", "legacy_locked_test_unseen_by_final_training"
    return "final", "new_multi_object_unseen_by_final_training"


def unseen_checkpoint_routes(
    record: dict[str, Any], old_records_by_id: dict[int, dict[str, Any]]
) -> tuple[tuple[str, ...], str]:
    source_id = record.get("source_image_id")
    old = old_records_by_id.get(int(source_id)) if source_id is not None else None
    if old is not None and old["split"] == "development":
        route = f"fold{int(old['fold'])}"
        return (route,), "legacy_outer_fold"
    if old is not None and old["split"] == "test":
        return (
            ("fold0", "fold1", "fold2", "final"),
            "legacy_locked_test_unseen_by_every_legacy_checkpoint",
        )
    return (
        ("fold0", "fold1", "fold2", "final"),
        "new_multi_object_unseen_by_every_legacy_checkpoint",
    )


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def collect_predictions(
    records: list[dict[str, Any]],
    route_sets: list[tuple[str, ...]],
    checkpoints: dict[str, Path],
    *,
    dataset_root: Path,
    batch_size: int,
    cpu: bool,
) -> list[dict[str, Any]]:
    torch = require_torch()
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    device = torch.device("cpu" if cpu else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu to opt into CPU inference")
    predictions_by_id: dict[str, list[dict[str, Any]]] = {
        str(record["image_id"]): [] for record in records
    }
    for route, checkpoint in checkpoints.items():
        selected = [
            record
            for record, selected_routes in zip(records, route_sets)
            if route in selected_routes
        ]
        if not selected:
            continue
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint).to(device).eval()
        for start in range(0, len(selected), batch_size):
            batch_records = selected[start : start + batch_size]
            images = [_open_rgb(dataset_root / row["image_path"]) for row in batch_records]
            try:
                inputs = processor(images=images, return_tensors="pt")
            finally:
                for image in images:
                    image.close()
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ),
            ):
                outputs = model(**inputs)
            sizes = torch.asarray(
                [[row["height"], row["width"]] for row in batch_records], device=device
            )
            processed = processor.post_process_object_detection(
                outputs,
                threshold=0.0,
                target_sizes=sizes,
            )
            for record, result in zip(batch_records, processed):
                scores = result["scores"].float().cpu().numpy()
                predictions_by_id[str(record["image_id"])].append(
                    {
                        "image_id": record["image_id"],
                        "boxes_xyxy": result["boxes"].float().cpu().numpy().tolist(),
                        "scores": scores.tolist(),
                        "class_ids": np.zeros(len(scores), dtype=np.int64).tolist(),
                        "top3_class_ids": np.zeros((len(scores), 3), dtype=np.int64).tolist(),
                        "checkpoint_route": route,
                    }
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    outputs = []
    route_ids = {route: index for index, route in enumerate(checkpoints)}
    for record in records:
        parts = predictions_by_id[str(record["image_id"])]
        if not parts:
            raise ValueError("no unseen checkpoint route was selected for an image")
        outputs.append(
            {
                "image_id": record["image_id"],
                "boxes_xyxy": [box for part in parts for box in part["boxes_xyxy"]],
                "scores": [score for part in parts for score in part["scores"]],
                "class_ids": [value for part in parts for value in part["class_ids"]],
                "top3_class_ids": [value for part in parts for value in part["top3_class_ids"]],
                "source_ids": [
                    route_ids[part["checkpoint_route"]] for part in parts for _ in part["scores"]
                ],
                "checkpoint_routes": [part["checkpoint_route"] for part in parts],
            }
        )
    return outputs


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    folds = set(args.folds)
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") == "ANNOTATED"
    ]
    old_records = [
        row for row in read_manifest(args.legacy_manifest) if row["record_type"] == "detection"
    ]
    old_by_id = {int(row["image_id"]): row for row in old_records}
    if args.all_unseen_ensemble:
        route_rows = [unseen_checkpoint_routes(record, old_by_id) for record in records]
        route_sets = [row[0] for row in route_rows]
    else:
        legacy_route_rows = [checkpoint_route(record, old_by_id) for record in records]
        route_rows = [((row[0],), row[1]) for row in legacy_route_rows]
        route_sets = [row[0] for row in route_rows]
    checkpoints = {
        "fold0": args.fold0_checkpoint,
        "fold1": args.fold1_checkpoint,
        "fold2": args.fold2_checkpoint,
        "final": args.final_checkpoint,
    }
    predictions = collect_predictions(
        records,
        route_sets,
        checkpoints,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        cpu=args.cpu,
    )
    candidates = []
    for score_threshold, nms_threshold in product(args.score_thresholds, args.nms_thresholds):
        metrics = _metrics(
            records,
            predictions,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_threshold,
            match_iou_threshold=0.5,
            max_queries=300,
        )
        candidates.append(
            {
                "score_threshold": score_threshold,
                "nms_iou_threshold": nms_threshold,
                "metrics": metrics,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            -row["metrics"]["false_positive_count"] - row["metrics"]["false_negative_count"],
            row["metrics"]["exact_image_rate"],
            -row["metrics"]["false_negative_count"],
            row["score_threshold"],
        ),
    )
    full_recall = [row for row in candidates if row["metrics"]["false_negative_count"] == 0]
    route_counts = {
        route: sum(route in route_set for route_set in route_sets)
        for route in checkpoints
        if any(route in route_set for route_set in route_sets)
    }
    route_reason_counts = {
        reason: sum(row[1] == reason for row in route_rows) for _, reason in route_rows
    }
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_legacy_rtdetr_unseen_oof",
        "folds": sorted(folds),
        "all_unseen_ensemble": bool(args.all_unseen_ensemble),
        "route_counts": route_counts,
        "route_reason_counts": route_reason_counts,
        "checkpoint_sha256": {
            name: sha256_file(path / "model.safetensors") for name, path in checkpoints.items()
        },
        "candidate_count": len(candidates),
        "zero_error_candidate_count": sum(
            row["metrics"]["false_positive_count"] == 0
            and row["metrics"]["false_negative_count"] == 0
            for row in candidates
        ),
        "selected": selected,
        "full_recall_selected": (
            min(full_recall, key=lambda row: row["metrics"]["false_positive_count"])
            if full_recall
            else None
        ),
        "error_images": detection_error_rows(
            records,
            predictions,
            score_threshold=selected["score_threshold"],
            nms_iou_threshold=selected["nms_iou_threshold"],
            match_iou_threshold=0.5,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unseen legacy RT-DETR proposals")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fold0-checkpoint", type=Path, required=True)
    parser.add_argument("--fold1-checkpoint", type=Path, required=True)
    parser.add_argument("--fold2-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--all-unseen-ensemble", action="store_true")
    parser.add_argument("--score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--nms-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
