from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...evaluation.detector import _metrics_grid
from ...training.models import require_torch
from ...training.ssdlite_objectness_detector import (
    CachedObjectnessDataset,
    build_ssdlite_objectness,
    collate_detection_batch,
    enable_empty_image_hard_negative_loss,
    export_ssdlite_onnx,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _standardize_records(dataset: CachedObjectnessDataset) -> list[dict[str, Any]]:
    records = []
    for record in dataset.records:
        image_id = str(record["image_id"])
        width = record.get("width")
        height = record.get("height")
        if width is None or height is None:
            width, height = dataset.source_shapes[image_id]
        records.append(
            {
                "image_id": int(record["image_id"]),
                "width": int(width),
                "height": int(height),
                "annotations": [
                    {
                        "bbox_xywh": [
                            float(value)
                            for value in annotation.get("bbox_xywh", annotation.get("bbox"))
                        ],
                        "category_id": int(annotation["category_id"]),
                    }
                    for annotation in record["annotations"]
                ],
            }
        )
    return records


def _evaluate_dataset(model, dataset: CachedObjectnessDataset, *, batch_size: int) -> list[dict]:
    torch = require_torch()
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_detection_batch,
    )
    predictions: list[dict] = []
    model.eval()
    with torch.inference_mode():
        for images, _targets in loader:
            outputs = model([image.cuda(non_blocking=True) for image in images])
            for output, record in zip(
                outputs,
                dataset.records[len(predictions) : len(predictions) + len(outputs)],
                strict=True,
            ):
                image_id = str(record["image_id"])
                width = record.get("width")
                height = record.get("height")
                if width is None or height is None:
                    width, height = dataset.source_shapes[image_id]
                scale_x = float(width) / dataset.image_size
                scale_y = float(height) / dataset.image_size
                boxes = output["boxes"].detach().cpu().numpy().astype(np.float32)
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                predictions.append(
                    {
                        "image_id": int(record["image_id"]),
                        "boxes_xyxy": boxes.tolist(),
                        "scores": output["scores"].detach().cpu().numpy().tolist(),
                        "class_ids": output["labels"].detach().cpu().numpy().tolist(),
                    }
                )
    return predictions


def _class_metrics(
    records: list[dict[str, Any]],
    predictions: list[dict],
    *,
    score_threshold: float,
    nms_threshold: float,
) -> dict[str, int | float]:
    torch = require_torch()
    from torchvision.ops import box_iou, nms

    matched_count = 0
    correct_count = 0
    for record, prediction in zip(records, predictions, strict=True):
        boxes = torch.as_tensor(prediction["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4)
        scores = torch.as_tensor(prediction["scores"], dtype=torch.float32)
        class_ids = torch.as_tensor(prediction["class_ids"], dtype=torch.int64)
        eligible = scores >= score_threshold
        boxes = boxes[eligible]
        scores = scores[eligible]
        class_ids = class_ids[eligible]
        if len(boxes):
            keep = nms(boxes, scores, nms_threshold)
            boxes = boxes[keep]
            class_ids = class_ids[keep]
        annotations = record["annotations"]
        ground_truth = torch.as_tensor(
            [
                [
                    annotation["bbox_xywh"][0],
                    annotation["bbox_xywh"][1],
                    annotation["bbox_xywh"][0] + annotation["bbox_xywh"][2],
                    annotation["bbox_xywh"][1] + annotation["bbox_xywh"][3],
                ]
                for annotation in annotations
            ],
            dtype=torch.float32,
        ).reshape(-1, 4)
        unmatched = set(range(len(annotations)))
        overlaps = box_iou(boxes, ground_truth) if len(boxes) and len(ground_truth) else None
        for detection_index in range(len(boxes)):
            if not unmatched:
                break
            best = max(unmatched, key=lambda index: float(overlaps[detection_index, index]))
            if float(overlaps[detection_index, best]) < 0.5:
                continue
            unmatched.remove(best)
            matched_count += 1
            correct_count += int(class_ids[detection_index]) == int(
                annotations[best]["category_id"]
            )
    return {
        "detector_class_matched_count": matched_count,
        "detector_class_correct_count": correct_count,
        "detector_class_error_count": matched_count - correct_count,
        "detector_class_accuracy": correct_count / matched_count if matched_count else 0.0,
    }


def _evaluate(
    model,
    real_dataset: CachedObjectnessDataset,
    operational_dataset: CachedObjectnessDataset,
    *,
    batch_size: int,
    threshold_steps: int,
    nms_threshold: float,
    class_aware: bool,
) -> dict[str, Any]:
    real_records = _standardize_records(real_dataset)
    operational_records = _standardize_records(operational_dataset)
    real_predictions = _evaluate_dataset(model, real_dataset, batch_size=batch_size)
    operational_predictions = _evaluate_dataset(model, operational_dataset, batch_size=batch_size)
    thresholds = np.linspace(0.01, 0.99, threshold_steps, dtype=np.float64)
    real_grid = _metrics_grid(
        real_records,
        real_predictions,
        score_thresholds=thresholds,
        nms_iou_threshold=nms_threshold,
        match_iou_threshold=0.5,
        max_queries=300,
    )
    operational_grid = _metrics_grid(
        operational_records,
        operational_predictions,
        score_thresholds=thresholds,
        nms_iou_threshold=nms_threshold,
        match_iou_threshold=0.5,
        max_queries=300,
    )
    combined = []
    for real, operational in zip(real_grid, operational_grid, strict=True):
        error_count = (
            int(real["false_positive_count"])
            + int(real["false_negative_count"])
            + int(operational["false_positive_count"])
            + int(operational["false_negative_count"])
        )
        combined.append(
            {
                "score_threshold": float(real["score_threshold"]),
                "error_count": error_count,
                "real": real,
                "operational": operational,
            }
        )
    selected = min(
        combined,
        key=lambda row: (
            row["error_count"],
            int(row["real"]["false_negative_count"])
            + int(row["operational"]["false_negative_count"]),
            int(row["real"]["false_positive_count"])
            + int(row["operational"]["false_positive_count"]),
            -int(row["real"]["exact_image_count"]),
            -int(row["operational"]["exact_image_count"]),
            -float(row["score_threshold"]),
        ),
    )
    if class_aware:
        selected["real"].update(
            _class_metrics(
                real_records,
                real_predictions,
                score_threshold=float(selected["score_threshold"]),
                nms_threshold=nms_threshold,
            )
        )
        selected["operational"].update(
            _class_metrics(
                operational_records,
                operational_predictions,
                score_threshold=float(selected["score_threshold"]),
                nms_threshold=nms_threshold,
            )
        )
        selected["detector_class_error_count"] = int(
            selected["real"]["detector_class_error_count"]
        ) + int(selected["operational"]["detector_class_error_count"])
    else:
        selected["detector_class_error_count"] = None
    return selected


def _train(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    from torch.utils.data import ConcatDataset, DataLoader, Subset

    if not torch.cuda.is_available():
        raise RuntimeError("SSDLite experiment requires CUDA")
    operational_args = (
        args.operational_manifest,
        args.operational_root,
        args.operational_cache,
    )
    if not args.fixed_epochs_no_selection and any(value is None for value in operational_args):
        raise ValueError(
            "operational manifest, root, and cache are required unless "
            "--fixed-epochs-no-selection is enabled"
        )
    if (
        args.fixed_epochs_no_selection
        and args.operational_repeat
        and any(value is None for value in operational_args)
    ):
        raise ValueError(
            "fixed-epoch operational training requires operational manifest, root, and cache"
        )
    _seed_everything(args.seed)
    if (
        args.fixed_epochs_no_selection
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        raise FileExistsError(
            f"fixed-epoch training output must be a new empty directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real_train = CachedObjectnessDataset(
        args.real_manifest,
        args.real_root,
        args.real_cache,
        training=True,
        class_aware=args.class_aware,
        quarter_turn_augmentation=args.quarter_turn_augmentation,
    )
    synthetic_train = CachedObjectnessDataset(
        args.synthetic_manifest,
        args.synthetic_root,
        args.synthetic_cache,
        training=True,
        class_aware=args.class_aware,
        quarter_turn_augmentation=args.quarter_turn_augmentation,
    )
    real_evaluation = None
    operational_evaluation = None
    operational_train = None
    if not args.fixed_epochs_no_selection:
        real_evaluation = CachedObjectnessDataset(
            args.real_manifest, args.real_root, args.real_cache, training=False
        )
        operational_evaluation = CachedObjectnessDataset(
            args.operational_manifest,
            args.operational_root,
            args.operational_cache,
            training=False,
        )
        operational_train = CachedObjectnessDataset(
            args.operational_manifest,
            args.operational_root,
            args.operational_cache,
            training=True,
            class_aware=args.class_aware,
            quarter_turn_augmentation=args.quarter_turn_augmentation,
        )
    elif args.operational_repeat:
        operational_train = CachedObjectnessDataset(
            args.operational_manifest,
            args.operational_root,
            args.operational_cache,
            training=True,
            class_aware=args.class_aware,
            quarter_turn_augmentation=args.quarter_turn_augmentation,
        )
    negative_sources = [real_train]
    negative_records = list(real_train.records)
    if operational_train is not None:
        negative_sources.append(operational_train)
        negative_records.extend(operational_train.records)
    hard_negative_train = Subset(
        ConcatDataset(negative_sources),
        [index for index, record in enumerate(negative_records) if not record["annotations"]],
    )
    train_parts = [real_train for _ in range(args.real_repeat)]
    if operational_train is not None:
        train_parts.extend(operational_train for _ in range(args.operational_repeat))
    train_parts.extend(hard_negative_train for _ in range(args.hard_negative_repeat))
    train_parts.append(synthetic_train)
    train_loader = DataLoader(
        ConcatDataset(train_parts),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_detection_batch,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = build_ssdlite_objectness(
        foreground_class_count=20 if args.class_aware else 1,
        pretrained_backbone=args.pretrained_backbone,
        pretrained_detector_transfer=args.pretrained_detector_transfer,
        device="cuda",
    )
    transferred_parameter_count = None
    if args.initial_checkpoint is not None:
        state = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
        current = model.state_dict()
        transferable = {
            name: value
            for name, value in state.items()
            if name in current and current[name].shape == value.shape
        }
        model.load_state_dict(transferable, strict=len(transferable) == len(current))
        transferred_parameter_count = len(transferable)
    if args.minimum_empty_negatives:
        enable_empty_image_hard_negative_loss(
            model,
            minimum_negatives=args.minimum_empty_negatives,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.02
    )
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best_rank = None
    best_evaluation = None
    best_epoch = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running: dict[str, float] = {}
        batch_count = 0
        epoch_started = time.perf_counter()
        for images, targets in train_loader:
            images = [image.cuda(non_blocking=True) for image in images]
            targets = [
                {
                    key: value.cuda(non_blocking=True)
                    for key, value in target.items()
                    if key != "image_id"
                }
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + float(value.detach())
            batch_count += 1
        scheduler.step()
        entry: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "losses": {name: value / batch_count for name, value in running.items()},
            "duration_seconds": time.perf_counter() - epoch_started,
        }
        if not args.fixed_epochs_no_selection and (
            epoch % args.evaluate_every == 0 or epoch == args.epochs
        ):
            assert real_evaluation is not None
            assert operational_evaluation is not None
            evaluation = _evaluate(
                model,
                real_evaluation,
                operational_evaluation,
                batch_size=args.evaluation_batch_size,
                threshold_steps=args.threshold_steps,
                nms_threshold=args.nms_threshold,
                class_aware=args.class_aware,
            )
            entry["evaluation"] = evaluation
            rank = (
                int(evaluation["error_count"]),
                int(evaluation["detector_class_error_count"]) if args.class_aware else 0,
                int(evaluation["real"]["false_negative_count"])
                + int(evaluation["operational"]["false_negative_count"]),
                int(evaluation["real"]["false_positive_count"])
                + int(evaluation["operational"]["false_positive_count"]),
                -int(evaluation["real"]["exact_image_count"]),
                -int(evaluation["operational"]["exact_image_count"]),
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_evaluation = evaluation
                best_epoch = epoch
                torch.save(model.state_dict(), args.output_dir / "best.pt")
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if (
            not args.fixed_epochs_no_selection
            and best_evaluation is not None
            and best_evaluation["error_count"] == 0
            and (not args.class_aware or best_evaluation["detector_class_error_count"] == 0)
            and epoch >= args.minimum_epochs
        ):
            break
    torch.save(model.state_dict(), args.output_dir / "last.pt")
    selected_checkpoint = args.output_dir / "last.pt"
    if not args.fixed_epochs_no_selection:
        if best_epoch is None:
            raise RuntimeError("training completed without an evaluation checkpoint")
        selected_checkpoint = args.output_dir / "best.pt"
        model.load_state_dict(
            torch.load(selected_checkpoint, map_location="cpu", weights_only=True)
        )
    else:
        best_epoch = args.epochs
    export_ssdlite_onnx(
        model,
        args.output_dir / "detector.onnx",
        detector_class_count=20 if args.class_aware else 1,
    )
    if args.initial_checkpoint is not None:
        initialization = "continued_checkpoint"
    elif args.pretrained_detector_transfer:
        initialization = "torchvision_coco_detector_transfer"
    elif args.pretrained_backbone:
        initialization = "torchvision_imagenet_backbone"
    else:
        initialization = "random"
    pretrained_weights_used = initialization != "random"
    report = {
        "experiment": f"ssdlite320_mobilenet_v3_objectness_{initialization}",
        "license_boundary": {
            "architecture_implementation": "torchvision BSD-3-Clause",
            "pretrained_weights_used": pretrained_weights_used,
            "initialization": initialization,
            "foundation_weight_source": (
                "torchvision SSDLite320 MobileNetV3 COCO default weights"
                if args.pretrained_detector_transfer
                else (
                    "torchvision MobileNetV3-Large ImageNet-1K V2 weights"
                    if args.pretrained_backbone
                    else None
                )
            ),
            "bread_specific_training_data": "project-owned real and generated synthetic data",
        },
        "settings": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "batch_size": args.batch_size,
            "real_repeat": args.real_repeat,
            "operational_repeat": args.operational_repeat,
            "hard_negative_repeat": args.hard_negative_repeat,
            "minimum_empty_negatives": args.minimum_empty_negatives,
            "class_aware": args.class_aware,
            "pretrained_backbone": args.pretrained_backbone,
            "pretrained_detector_transfer": args.pretrained_detector_transfer,
            "quarter_turn_augmentation": args.quarter_turn_augmentation,
            "minimum_epochs": args.minimum_epochs,
            "transferred_parameter_count": transferred_parameter_count,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "nms_threshold": args.nms_threshold,
            "selection_policy": (
                "fixed_last_epoch_without_evaluation"
                if args.fixed_epochs_no_selection
                else "development_evaluation"
            ),
            "development_evaluation_accessed": not args.fixed_epochs_no_selection,
            "initial_checkpoint": (
                None if args.initial_checkpoint is None else str(args.initial_checkpoint)
            ),
        },
        "sample_counts": {
            "real_training": len(real_train),
            "synthetic_training": len(synthetic_train),
            "operational_evaluation": (
                0 if operational_evaluation is None else len(operational_evaluation)
            ),
            "operational_training": (
                len(operational_train)
                if operational_train is not None and args.operational_repeat
                else 0
            ),
            "hard_negative_training": len(hard_negative_train),
        },
        "best_epoch": best_epoch,
        "best_evaluation": best_evaluation,
        "history": history,
        "duration_seconds": time.perf_counter() - started,
        "artifacts": {
            "checkpoint": selected_checkpoint.name,
            "onnx": "detector.onnx",
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO-free SSDLite objectness detector")
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--real-cache", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--operational-manifest", type=Path)
    parser.add_argument("--operational-root", type=Path)
    parser.add_argument("--operational-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument(
        "--fixed-epochs-no-selection",
        action="store_true",
        help=(
            "train exactly --epochs without development "
            "evaluation, early stopping, or checkpoint selection"
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--real-repeat", type=int, default=2)
    parser.add_argument("--operational-repeat", type=int, default=0)
    parser.add_argument("--hard-negative-repeat", type=int, default=0)
    parser.add_argument("--minimum-empty-negatives", type=int, default=0)
    parser.add_argument("--class-aware", action="store_true")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--pretrained-detector-transfer", action="store_true")
    parser.add_argument("--quarter-turn-augmentation", action="store_true")
    parser.add_argument("--minimum-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--evaluate-every", type=int, default=2)
    parser.add_argument("--threshold-steps", type=int, default=197)
    parser.add_argument("--seed", type=int, default=20260901)
    _train(parser.parse_args())


if __name__ == "__main__":
    main()
