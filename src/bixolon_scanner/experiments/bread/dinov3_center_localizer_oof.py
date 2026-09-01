from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ...training.data import read_manifest
from ...training.dinov3_center_localizer import (
    build_dinov3_center_localizer,
    center_localizer_loss,
    center_targets,
    decode_center_predictions,
    flip_detection_records,
)
from .dinov3_set_localizer_oof import (
    _spatial_exact,
    cache_backbone_features,
    normalized_cxcywh_to_absolute_xyxy,
)


def _train_fold(args, records, folds, stride8, stride16, stride32, held_fold):
    import torch

    torch.manual_seed(args.seed + held_fold)
    training_indices = np.flatnonzero(folds != held_fold)
    held_indices = np.flatnonzero(folds == held_fold)
    model = build_dinov3_center_localizer(hidden_dimension=args.hidden_dimension).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    generator = np.random.default_rng(args.seed + held_fold)
    final_loss = None
    for epoch in range(args.epochs):
        model.train()
        order = generator.permutation(training_indices)
        loss_sum = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            features = [
                torch.from_numpy(np.asarray(values[indices]).copy()).to(
                    args.device, dtype=torch.float32
                )
                for values in (stride8, stride16, stride32)
            ]
            flips = generator.random(len(indices)) < 0.5
            if np.any(flips):
                flip_tensor = torch.from_numpy(flips).to(args.device)
                for feature in features:
                    feature[flip_tensor] = feature[flip_tensor].flip(-1)
            batch_records = flip_detection_records([records[index] for index in indices], flips)
            targets = center_targets(
                batch_records,
                height=features[0].shape[-2],
                width=features[0].shape[-1],
                device=args.device,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*features)
            loss, _ = center_localizer_loss(prediction, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        final_loss = loss_sum / max(batches, 1)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                json.dumps(
                    {
                        "held_out_fold": int(held_fold),
                        "epoch": epoch + 1,
                        "mean_loss": final_loss,
                    }
                ),
                flush=True,
            )
    output_boxes = np.zeros((len(held_indices), args.maximum_objects, 4), dtype=np.float32)
    output_scores = np.zeros((len(held_indices), args.maximum_objects), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(held_indices), args.batch_size):
            indices = held_indices[start : start + args.batch_size]
            features = [
                torch.from_numpy(np.asarray(values[indices]).copy()).to(
                    args.device, dtype=torch.float32
                )
                for values in (stride8, stride16, stride32)
            ]
            scores, boxes = decode_center_predictions(
                model(*features), maximum_objects=args.maximum_objects
            )
            output_boxes[start : start + len(indices)] = boxes.cpu().numpy()
            output_scores[start : start + len(indices)] = scores.cpu().numpy()
    return (
        output_boxes,
        output_scores,
        {
            "held_out_fold": int(held_fold),
            "training_image_count": len(training_indices),
            "held_out_image_count": len(held_indices),
            "final_training_loss": final_loss,
        },
    )


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and not row.get("exclude_from_detector_training", False)
    ]
    records.sort(key=lambda row: int(row["image_id"]))
    metadata = cache_backbone_features(args, records)
    image_ids = np.asarray(metadata["image_ids"], dtype=np.int64)
    folds = np.asarray(metadata["folds"], dtype=np.int8)
    stride8 = np.load(args.feature_cache / metadata["stride8"], mmap_mode="r")
    stride16 = np.load(args.feature_cache / metadata["stride16"], mmap_mode="r")
    stride32 = np.load(args.feature_cache / metadata["stride32"], mmap_mode="r")
    output_boxes = np.zeros((len(records), args.maximum_objects, 4), dtype=np.float32)
    output_scores = np.zeros((len(records), args.maximum_objects), dtype=np.float32)
    diagnostics = []
    for held_fold in sorted(np.unique(folds)):
        boxes, scores, diagnostic = _train_fold(
            args, records, folds, stride8, stride16, stride32, int(held_fold)
        )
        selected = folds == held_fold
        output_boxes[selected] = boxes
        output_scores[selected] = scores
        diagnostics.append(diagnostic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        boxes_cxcywh=output_boxes.astype(np.float16),
        objectness=output_scores.astype(np.float16),
        image_id=image_ids,
        fold=folds,
    )
    count_rows = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.patch_count_oof.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    oracle_exact = 0
    predicted_exact = 0
    scene_rows = []
    for index, record in enumerate(records):
        true_count = len(record["annotations"])
        predicted_count = int(count_rows[int(record["image_id"])]["predicted_count"])
        absolute = normalized_cxcywh_to_absolute_xyxy(output_boxes[index], record)
        oracle_safe, oracle_iou = _spatial_exact(absolute[:true_count], record)
        predicted_safe, predicted_iou = _spatial_exact(absolute[:predicted_count], record)
        oracle_exact += int(oracle_safe)
        predicted_exact += int(predicted_safe)
        scene_rows.append(
            {
                "image_id": int(record["image_id"]),
                "fold": int(record["fold"]),
                "true_count": true_count,
                "predicted_count": predicted_count,
                "oracle_count_spatial_exact": oracle_safe,
                "predicted_count_spatial_exact": predicted_safe,
                "oracle_assigned_iou": oracle_iou,
                "predicted_assigned_iou": predicted_iou,
                "ordered_boxes_xyxy": absolute.tolist(),
                "ordered_objectness": output_scores[index].tolist(),
            }
        )
    args.output.with_suffix(".scenes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scene_rows),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_center_heatmap_localizer_true_group_oof",
        "selection_scope": "fixed-epoch heads are trained on the other folds",
        "image_count": len(records),
        "oracle_count_spatial_exact_image_count": oracle_exact,
        "predicted_count_spatial_exact_image_count": predicted_exact,
        "folds": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
        "existing_detector_used": False,
        "yolo_family_used": False,
        "rfdetr_used": False,
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train DINOv3 center heatmap localization OOF")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--patch-count-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--maximum-objects", type=int, default=7)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20260831)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
