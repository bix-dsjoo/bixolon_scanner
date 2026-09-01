from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...training.data import read_manifest
from ...training.dinov3_objectness_detector import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_dinov3_convnext_tiny,
)
from ...training.dinov3_set_localizer import (
    build_dinov3_set_localizer,
    hungarian_set_loss,
    normalized_target_boxes,
)
from .proposal_ranker import proposal_iou_matrix


def cache_backbone_features(args: argparse.Namespace, records: list[dict]) -> dict:
    import torch

    args.feature_cache.mkdir(parents=True, exist_ok=True)
    metadata_path = args.feature_cache / "index.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "stride8" in metadata and (args.feature_cache / metadata["stride8"]).exists():
            return metadata
    image_metadata = json.loads((args.image_cache / "index.json").read_text(encoding="utf-8"))
    images = np.load(args.image_cache / image_metadata["array_filename"], mmap_mode="r")
    stride8 = np.lib.format.open_memmap(
        args.feature_cache / "stride8.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(records), 192, 80, 80),
    )
    stride16 = np.lib.format.open_memmap(
        args.feature_cache / "stride16.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(records), 384, 40, 40),
    )
    stride32 = np.lib.format.open_memmap(
        args.feature_cache / "stride32.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(records), 768, 20, 20),
    )
    backbone = load_dinov3_convnext_tiny(args.weights, device=args.device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=args.device).view(1, 3, 1, 1)
    with torch.inference_mode():
        for start in range(0, len(records), args.feature_batch_size):
            batch_records = records[start : start + args.feature_batch_size]
            batch_rows = [
                int(image_metadata["index"][str(record["image_id"])]) for record in batch_records
            ]
            batch = np.stack([np.asarray(images[row]) for row in batch_rows])
            tensor = torch.from_numpy(batch.copy()).to(args.device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            tensor = (tensor - mean) / std
            maps = backbone.get_intermediate_layers(tensor, n=[1, 2, 3], reshape=True, norm=True)
            stride8[start : start + len(batch)] = maps[0].cpu().numpy().astype(np.float16)
            stride16[start : start + len(batch)] = maps[1].cpu().numpy().astype(np.float16)
            stride32[start : start + len(batch)] = maps[2].cpu().numpy().astype(np.float16)
            print(json.dumps({"features_cached": start + len(batch)}), flush=True)
    stride8.flush()
    stride16.flush()
    stride32.flush()
    metadata = {
        "schema_version": "1.0",
        "image_ids": [int(record["image_id"]) for record in records],
        "folds": [int(record["fold"]) for record in records],
        "stride8": "stride8.npy",
        "stride16": "stride16.npy",
        "stride32": "stride32.npy",
        "backbone": "official_dinov3_convnext_tiny",
        "existing_detector_used": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _train_fold(
    args: argparse.Namespace,
    records: list[dict],
    folds: np.ndarray,
    stride16: np.ndarray,
    stride32: np.ndarray,
    held_fold: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch

    torch.manual_seed(args.seed + held_fold)
    training_indices = np.flatnonzero(folds != held_fold)
    held_indices = np.flatnonzero(folds == held_fold)
    targets = [normalized_target_boxes(record) for record in records]
    model = build_dinov3_set_localizer(
        maximum_objects=args.maximum_objects,
        hidden_dimension=args.hidden_dimension,
        decoder_layers=args.decoder_layers,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    generator = np.random.default_rng(args.seed + held_fold)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = generator.permutation(training_indices)
        loss_sum = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            feature16 = torch.from_numpy(np.asarray(stride16[indices]).copy()).to(
                args.device, dtype=torch.float32
            )
            feature32 = torch.from_numpy(np.asarray(stride32[indices]).copy()).to(
                args.device, dtype=torch.float32
            )
            batch_targets = [
                torch.from_numpy(targets[index].copy()).to(args.device) for index in indices
            ]
            flip = generator.random(len(indices)) < 0.5
            if np.any(flip):
                flip_tensor = torch.from_numpy(flip).to(args.device)
                feature16[flip_tensor] = feature16[flip_tensor].flip(-1)
                feature32[flip_tensor] = feature32[flip_tensor].flip(-1)
                for target, should_flip in zip(batch_targets, flip, strict=True):
                    if should_flip:
                        target[:, 0] = 1.0 - target[:, 0]
            optimizer.zero_grad(set_to_none=True)
            object_logits, predicted_boxes = model(feature16, feature32)
            loss, _ = hungarian_set_loss(object_logits, predicted_boxes, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        mean_loss = loss_sum / max(batches, 1)
        history.append(mean_loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                json.dumps(
                    {
                        "held_out_fold": held_fold,
                        "epoch": epoch + 1,
                        "mean_loss": mean_loss,
                    }
                ),
                flush=True,
            )
    model.eval()
    held_boxes = np.zeros((len(held_indices), args.maximum_objects, 4), dtype=np.float32)
    held_scores = np.zeros((len(held_indices), args.maximum_objects), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(held_indices), args.batch_size):
            indices = held_indices[start : start + args.batch_size]
            feature16 = torch.from_numpy(np.asarray(stride16[indices]).copy()).to(
                args.device, dtype=torch.float32
            )
            feature32 = torch.from_numpy(np.asarray(stride32[indices]).copy()).to(
                args.device, dtype=torch.float32
            )
            logits, boxes = model(feature16, feature32)
            held_boxes[start : start + len(indices)] = boxes.cpu().numpy()
            held_scores[start : start + len(indices)] = logits.sigmoid().cpu().numpy()
    return (
        held_boxes,
        held_scores,
        {
            "held_out_fold": held_fold,
            "training_image_count": len(training_indices),
            "held_out_image_count": len(held_indices),
            "final_training_loss": history[-1],
        },
    )


def normalized_cxcywh_to_absolute_xyxy(boxes: np.ndarray, record: dict) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    center_x = boxes[:, 0] * record["width"]
    center_y = boxes[:, 1] * record["height"]
    width = boxes[:, 2] * record["width"]
    height = boxes[:, 3] * record["height"]
    return np.column_stack(
        (center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2)
    ).astype(np.float32)


def _spatial_exact(boxes: np.ndarray, record: dict) -> tuple[bool, list[float]]:
    targets = np.asarray(
        [
            [x, y, x + width, y + height]
            for x, y, width, height in (
                annotation["bbox_xywh"] for annotation in record["annotations"]
            )
        ],
        dtype=np.float32,
    )
    if len(boxes) != len(targets):
        return False, []
    overlap = proposal_iou_matrix(boxes, targets)
    proposal_indices, target_indices = linear_sum_assignment(-overlap)
    assigned = overlap[proposal_indices, target_indices]
    return bool(np.all(assigned >= 0.5)), assigned.tolist()


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
    if image_ids.tolist() != [int(record["image_id"]) for record in records]:
        raise ValueError("DINOv3 set feature cache does not align with manifest")
    stride16 = np.load(args.feature_cache / metadata["stride16"], mmap_mode="r")
    stride32 = np.load(args.feature_cache / metadata["stride32"], mmap_mode="r")
    output_boxes = np.zeros((len(records), args.maximum_objects, 4), dtype=np.float32)
    output_scores = np.zeros((len(records), args.maximum_objects), dtype=np.float32)
    diagnostics = []
    for held_fold in sorted(np.unique(folds)):
        boxes, scores, diagnostic = _train_fold(
            args, records, folds, stride16, stride32, int(held_fold)
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
    scene_rows = []
    oracle_exact = 0
    predicted_exact = 0
    count_exact = 0
    for index, record in enumerate(records):
        order = np.argsort(-output_scores[index], kind="stable")
        true_count = len(record["annotations"])
        predicted_count = int(count_rows[int(record["image_id"])]["predicted_count"])
        absolute = normalized_cxcywh_to_absolute_xyxy(output_boxes[index], record)
        oracle_boxes = absolute[order[:true_count]]
        predicted_boxes = absolute[order[:predicted_count]]
        oracle_safe, oracle_iou = _spatial_exact(oracle_boxes, record)
        predicted_safe, predicted_iou = _spatial_exact(predicted_boxes, record)
        oracle_exact += int(oracle_safe)
        predicted_exact += int(predicted_safe)
        count_exact += int(predicted_count == true_count)
        scene_rows.append(
            {
                "image_id": int(record["image_id"]),
                "fold": int(record["fold"]),
                "true_count": true_count,
                "predicted_count": predicted_count,
                "count_correct": predicted_count == true_count,
                "oracle_count_spatial_exact": oracle_safe,
                "predicted_count_spatial_exact": predicted_safe,
                "oracle_assigned_iou": oracle_iou,
                "predicted_assigned_iou": predicted_iou,
                "ordered_boxes_xyxy": absolute[order].tolist(),
                "ordered_objectness": output_scores[index, order].tolist(),
            }
        )
    scene_path = args.output.with_suffix(".scenes.jsonl")
    scene_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scene_rows),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_count_conditioned_set_localizer_true_group_oof",
        "selection_scope": "fixed-epoch set heads are trained on the other folds",
        "image_count": len(records),
        "patch_count_exact_image_count": count_exact,
        "oracle_count_spatial_exact_image_count": oracle_exact,
        "predicted_count_spatial_exact_image_count": predicted_exact,
        "feature_cache": args.feature_cache.resolve().as_posix(),
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
    parser = argparse.ArgumentParser(
        description="Train true group-OOF DINOv3 count-conditioned set localization"
    )
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
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20260831)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
