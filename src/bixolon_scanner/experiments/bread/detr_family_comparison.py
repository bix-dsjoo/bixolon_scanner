from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ...evaluation.detector import _metrics_grid
from ...training.models import require_torch
from ...training.ssdlite_objectness_detector import CachedObjectnessDataset


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    revision: str


MODEL_SPECS = {
    "lw-detr-tiny": ModelSpec(
        checkpoint="AnnaZhang/lwdetr_tiny_60e_coco",
        revision="4b636b514dcf623f6eafc9e1ab63b8ad5c513925",
    ),
    "dfine-s": ModelSpec(
        checkpoint="ustc-community/dfine-small-coco",
        revision="f79e65b5fbb33ceb9d3ebba042955d7410c608f8",
    ),
    "rtdetrv2-s": ModelSpec(
        checkpoint="PekingU/rtdetr_v2_r18vd",
        revision="5650961749fa93567c0d46fc7f43ea4f9e914107",
    ),
    "rf-detr-nano": ModelSpec(
        checkpoint="Roboflow/rf-detr-nano",
        revision="9f201577d9415f0378084748d27e68586e6b3600",
    ),
}


class HfObjectnessDataset:
    def __init__(self, source: CachedObjectnessDataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        image, target = self.source[index]
        boxes = target["boxes"]
        annotations = []
        for annotation_id, box in enumerate(boxes.tolist(), start=1):
            x1, y1, x2, y2 = [float(value) for value in box]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": int(target["image_id"]),
                    "category_id": 0,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
        return image, {
            "image_id": int(target["image_id"]),
            "annotations": annotations,
        }


def _process_images(processor, images, *, annotations=None):
    """Process cached float images without applying a second 1/255 rescale."""
    options = {
        "images": list(images),
        "return_tensors": "pt",
        "do_rescale": False,
    }
    if annotations is not None:
        options["annotations"] = list(annotations)
    return processor(**options)


def _collate(batch, *, processor):
    images, annotations = zip(*batch, strict=True)
    encoded = _process_images(processor, images, annotations=annotations)
    encoded["raw_annotations"] = list(annotations)
    return encoded


def _move_batch(batch: dict[str, Any], device) -> dict[str, Any]:
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key not in {"labels", "raw_annotations"}
    }
    inputs["labels"] = [
        {key: value.to(device) for key, value in label.items()} for label in batch["labels"]
    ]
    return inputs


def _records_and_predictions(model, processor, loader, *, device, image_size: int):
    torch = require_torch()
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            raw_annotations = batch["raw_annotations"]
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key not in {"labels", "raw_annotations"}
            }
            outputs = model(**inputs)
            target_sizes = torch.full(
                (len(raw_annotations), 2), image_size, dtype=torch.int64, device=device
            )
            results = processor.post_process_object_detection(
                outputs,
                threshold=0.0,
                target_sizes=target_sizes,
            )
            for annotation, result in zip(raw_annotations, results, strict=True):
                records.append(
                    {
                        "image_id": int(annotation["image_id"]),
                        "width": image_size,
                        "height": image_size,
                        "annotations": [
                            {
                                "bbox_xywh": [float(value) for value in row["bbox"]],
                                "category_id": 1,
                            }
                            for row in annotation["annotations"]
                        ],
                    }
                )
                predictions.append(
                    {
                        "image_id": int(annotation["image_id"]),
                        "boxes_xyxy": result["boxes"].float().cpu().numpy().tolist(),
                        "scores": result["scores"].float().cpu().numpy().tolist(),
                        "class_ids": result["labels"].long().cpu().numpy().tolist(),
                    }
                )
    return records, predictions


def _select_threshold(
    real_records,
    real_predictions,
    operational_records,
    operational_predictions,
    *,
    minimum_threshold: float,
    threshold_steps: int,
    nms_threshold: float,
):
    thresholds = np.linspace(minimum_threshold, 0.999, threshold_steps, dtype=np.float64)
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
        false_negative_count = int(real["false_negative_count"]) + int(
            operational["false_negative_count"]
        )
        false_positive_count = int(real["false_positive_count"]) + int(
            operational["false_positive_count"]
        )
        combined.append(
            {
                "score_threshold": float(real["score_threshold"]),
                "error_count": false_negative_count + false_positive_count,
                "false_negative_count": false_negative_count,
                "false_positive_count": false_positive_count,
                "real": real,
                "operational": operational,
            }
        )

    def image_tie_breakers(row):
        return (
            -int(row["real"]["exact_image_count"]),
            -int(row["operational"]["exact_image_count"]),
            -row["score_threshold"],
        )

    return {
        "recall_first": min(
            combined,
            key=lambda row: (
                row["false_negative_count"],
                row["false_positive_count"],
                *image_tie_breakers(row),
            ),
        ),
        "minimum_total_error": min(
            combined,
            key=lambda row: (
                row["error_count"],
                row["false_negative_count"],
                row["false_positive_count"],
                *image_tie_breakers(row),
            ),
        ),
    }


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "sample_count": len(values),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def _cpu_latency(model, processor, dataset, *, warmup_count: int, cpu_threads: int):
    torch = require_torch()
    torch.set_num_threads(cpu_threads)
    model = model.to("cpu").eval()

    def run(index: int) -> None:
        image, _annotation = dataset[index]
        inputs = _process_images(processor, [image])
        with torch.inference_mode():
            model(**inputs)

    for index in range(warmup_count):
        run(index % len(dataset))
    values = []
    for index in range(len(dataset)):
        started = time.perf_counter()
        run(index)
        values.append((time.perf_counter() - started) * 1000.0)
    return _latency_summary(values)


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    from torch.utils.data import ConcatDataset, DataLoader, Subset
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    if not torch.cuda.is_available():
        raise RuntimeError("DETR family comparison training requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"comparison output must be empty: {args.output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    spec = MODEL_SPECS[args.model]
    if args.resume_from is None:
        processor = AutoImageProcessor.from_pretrained(
            spec.checkpoint,
            revision=spec.revision,
            size={"height": args.image_size, "width": args.image_size},
        )
        model = AutoModelForObjectDetection.from_pretrained(
            spec.checkpoint,
            revision=spec.revision,
            num_labels=1,
            id2label={0: "bakery_item"},
            label2id={"bakery_item": 0},
            ignore_mismatched_sizes=True,
        ).to("cuda")
    else:
        processor = AutoImageProcessor.from_pretrained(args.resume_from)
        model = AutoModelForObjectDetection.from_pretrained(args.resume_from).to("cuda")

    real_train_source = CachedObjectnessDataset(
        args.real_manifest,
        args.real_root,
        args.real_cache,
        training=True,
    )
    operational_train_source = CachedObjectnessDataset(
        args.operational_manifest,
        args.operational_root,
        args.operational_cache,
        training=True,
    )
    synthetic_train_source = CachedObjectnessDataset(
        args.synthetic_manifest,
        args.synthetic_root,
        args.synthetic_cache,
        training=True,
    )
    real_evaluation = HfObjectnessDataset(
        CachedObjectnessDataset(
            args.real_manifest,
            args.real_root,
            args.real_cache,
            training=False,
        )
    )
    operational_evaluation = HfObjectnessDataset(
        CachedObjectnessDataset(
            args.operational_manifest,
            args.operational_root,
            args.operational_cache,
            training=False,
        )
    )
    negative_source = ConcatDataset([real_train_source, operational_train_source])
    negative_records = list(real_train_source.records) + list(operational_train_source.records)
    hard_negative_source = Subset(
        negative_source,
        [index for index, record in enumerate(negative_records) if not record["annotations"]],
    )
    train_dataset = ConcatDataset(
        [
            *([HfObjectnessDataset(real_train_source)] * args.real_repeat),
            *([HfObjectnessDataset(operational_train_source)] * args.operational_repeat),
            *([HfObjectnessDataset(hard_negative_source)] * args.hard_negative_repeat),
            HfObjectnessDataset(synthetic_train_source),
        ]
    )
    collate = partial(_collate, processor=processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    evaluation_batch_size = max(1, args.evaluation_batch_size)
    real_loader = DataLoader(
        real_evaluation,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    operational_loader = DataLoader(
        operational_evaluation,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.learning_rate * 0.02,
    )
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        epoch_started = time.perf_counter()
        for batch in train_loader:
            inputs = _move_batch(batch, "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = {
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "duration_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    real_records, real_predictions = _records_and_predictions(
        model,
        processor,
        real_loader,
        device="cuda",
        image_size=args.image_size,
    )
    operational_records, operational_predictions = _records_and_predictions(
        model,
        processor,
        operational_loader,
        device="cuda",
        image_size=args.image_size,
    )
    selections = _select_threshold(
        real_records,
        real_predictions,
        operational_records,
        operational_predictions,
        minimum_threshold=args.minimum_threshold,
        threshold_steps=args.threshold_steps,
        nms_threshold=args.nms_threshold,
    )
    torch.cuda.empty_cache()
    cpu_latency = _cpu_latency(
        model,
        processor,
        real_evaluation,
        warmup_count=args.warmup_count,
        cpu_threads=args.cpu_threads,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "model")
    processor.save_pretrained(args.output_dir / "model")
    report = {
        "experiment": "same-data-one-class-detector-family-comparison",
        "model": args.model,
        "checkpoint": spec.checkpoint,
        "checkpoint_revision": spec.revision,
        "settings": {
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "real_repeat": args.real_repeat,
            "operational_repeat": args.operational_repeat,
            "hard_negative_repeat": args.hard_negative_repeat,
            "nms_threshold": args.nms_threshold,
            "match_iou_threshold": 0.5,
            "minimum_threshold": args.minimum_threshold,
        },
        "sample_counts": {
            "real": len(real_train_source),
            "operational": len(operational_train_source),
            "synthetic": len(synthetic_train_source),
            "hard_negative": len(hard_negative_source),
            "weighted_epoch": len(train_dataset),
        },
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
        "selected_evaluation": selections["recall_first"],
        "minimum_total_error_evaluation": selections["minimum_total_error"],
        "cpu_pytorch_latency": cpu_latency,
        "latency_scope": "processor+PyTorch-model-forward, batch=1; not ONNX Runtime",
        "duration_seconds": time.perf_counter() - started,
        "limitations": [
            "The evaluation records overlap training and threshold selection.",
            "This comparison measures memorization/regression fit, not independent generalization.",
            "CPU latency is PyTorch eager screening and is not a deployable ONNX benchmark.",
        ],
        "resume_from": str(args.resume_from) if args.resume_from is not None else None,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare one-class DETR families with the SSDLite320 training data"
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--real-manifest",
        type=Path,
        default=Path(
            "artifacts/experiments/scanner-0.1.7-safety-20260828/"
            "detector415-source-ground-truth-manifest.jsonl"
        ),
    )
    parser.add_argument("--real-root", type=Path, default=Path("datasets/bread_dataset"))
    parser.add_argument(
        "--real-cache", type=Path, default=Path("artifacts/cache/ssdlite320-real415")
    )
    parser.add_argument(
        "--operational-manifest",
        type=Path,
        default=Path(
            "artifacts/evaluations/scanner-0.1.5/operational-20260827-annotated69-manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--operational-root",
        type=Path,
        default=Path("datasets/bread_dataset/operational_collections/2026-08-27"),
    )
    parser.add_argument(
        "--operational-cache",
        type=Path,
        default=Path("artifacts/cache/ssdlite320-operational69"),
    )
    parser.add_argument(
        "--synthetic-manifest",
        type=Path,
        default=Path(
            "artifacts/synthetic/"
            "_preliminary-single-objects-1500-unbalanced-seed-20260831/manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=Path(
            "artifacts/synthetic/_preliminary-single-objects-1500-unbalanced-seed-20260831"
        ),
    )
    parser.add_argument(
        "--synthetic-cache",
        type=Path,
        default=Path("artifacts/cache/ssdlite320-synthetic1500"),
    )
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--real-repeat", type=int, default=4)
    parser.add_argument("--operational-repeat", type=int, default=4)
    parser.add_argument("--hard-negative-repeat", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument("--minimum-threshold", type=float, default=0.001)
    parser.add_argument("--threshold-steps", type=int, default=999)
    parser.add_argument("--warmup-count", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260901)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
