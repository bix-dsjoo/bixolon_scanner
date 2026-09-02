from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from bixolon_scanner.training.dinov3_objectness_detector import (
    load_dinov3_convnext_tiny,
)
from bixolon_scanner.training.models import require_torch

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _xyxy(annotation: dict) -> tuple[float, float, float, float]:
    x, y, width, height = annotation["bbox_xywh"]
    return float(x), float(y), float(x + width), float(y + height)


def _iou(box: tuple[float, float, float, float], other) -> float:
    left = max(box[0], other[0])
    top = max(box[1], other[1])
    right = min(box[2], other[2])
    bottom = min(box[3], other[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    other_area = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
    return intersection / max(area + other_area - intersection, 1e-6)


def _partial_box(box, rng: np.random.Generator):
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    if rng.random() < 0.5:
        fraction = float(rng.uniform(0.30, 0.72))
        if rng.random() < 0.5:
            return left, top, left + width * fraction, bottom
        return right - width * fraction, top, right, bottom
    fraction = float(rng.uniform(0.30, 0.72))
    if rng.random() < 0.5:
        return left, top, right, top + height * fraction
    return left, bottom - height * fraction, right, bottom


def _background_box(boxes, rng: np.random.Generator, size: int):
    for _ in range(50):
        width = float(rng.uniform(0.10, 0.34) * size)
        height = float(rng.uniform(0.10, 0.34) * size)
        left = float(rng.uniform(0, size - width))
        top = float(rng.uniform(0, size - height))
        candidate = (left, top, left + width, top + height)
        if all(_iou(candidate, box) < 0.05 for box in boxes):
            return candidate
    return None


def _union_box(boxes, rng: np.random.Generator):
    if len(boxes) < 2:
        return None
    order = rng.permutation(len(boxes))
    for first in order:
        for second in order:
            if first == second:
                continue
            a, b = boxes[int(first)], boxes[int(second)]
            candidate = (
                min(a[0], b[0]),
                min(a[1], b[1]),
                max(a[2], b[2]),
                max(a[3], b[3]),
            )
            if max(_iou(candidate, box) for box in boxes) < 0.50:
                return candidate
    return None


def _make_samples(
    records: list[dict[str, Any]],
    *,
    split: str,
    maximum_per_label: int,
    seed: int,
    elongated_repeat: int = 1,
) -> list[tuple[int, tuple[float, float, float, float], int, str]]:
    rng = np.random.default_rng(seed + (0 if split == "train" else 1))
    positive = []
    negative = []
    for record in records:
        session_index = (int(record["image_id"]) - 1) // 3
        record_split = "validation" if session_index % 10 == 0 else "train"
        if record_split != split:
            continue
        image_id = int(record["image_id"])
        boxes = [_xyxy(row) for row in record["annotations"]]
        for box in boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            elongated = max(width / max(height, 1e-6), height / max(width, 1e-6)) >= 2.0
            repeat = elongated_repeat if elongated and split == "train" else 1
            for _ in range(repeat):
                positive.append(
                    (
                        image_id,
                        box,
                        1,
                        "complete_elongated_gt" if elongated else "complete_gt",
                    )
                )
                negative.append(
                    (
                        image_id,
                        _partial_box(box, rng),
                        0,
                        "elongated_fragment" if elongated else "partial_object",
                    )
                )
        union = _union_box(boxes, rng)
        if union is not None:
            negative.append((image_id, union, 0, "multi_object_union"))
        background = _background_box(boxes, rng, 640)
        if background is not None:
            negative.append((image_id, background, 0, "background"))

    per_label = min(maximum_per_label, len(positive), len(negative))

    def select(rows):
        if len(rows) <= per_label:
            return rows
        indexes = rng.choice(len(rows), per_label, replace=False)
        return [rows[int(index)] for index in indexes]

    selected_positive = select(positive)
    selected_negative = select(negative)
    samples = selected_positive + selected_negative
    rng.shuffle(samples)
    return samples


def _containment_keep(candidates: list[tuple[tuple[float, ...], float]]) -> list[bool]:
    keep = [True] * len(candidates)
    for index, (box, score) in enumerate(candidates):
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        for other_index, (other, other_score) in enumerate(candidates):
            if index == other_index or other_score < score:
                continue
            intersection = max(0.0, min(box[2], other[2]) - max(box[0], other[0])) * max(
                0.0, min(box[3], other[3]) - max(box[1], other[1])
            )
            other_area = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
            containment = intersection / max(area, 1e-6)
            area_ratio = area / max(other_area, 1e-6)
            if containment >= 0.70 and area_ratio <= 0.55:
                keep[index] = False
                break
    return keep


def _make_mined_samples(
    records: list[dict[str, Any]],
    predictions: dict[int, dict[str, Any]],
    *,
    split: str,
    maximum_per_label: int,
    detector_threshold: float,
    seed: int,
) -> list[tuple[int, tuple[float, float, float, float], int, str]]:
    rng = np.random.default_rng(seed + (10 if split == "train" else 11))
    positive = []
    negative = []
    for record in records:
        session_index = (int(record["image_id"]) - 1) // 3
        record_split = "validation" if session_index % 10 == 0 else "train"
        if record_split != split:
            continue
        image_id = int(record["image_id"])
        raw = predictions[image_id]
        candidates = [
            (tuple(float(value) for value in box), float(score))
            for box, score in zip(raw["boxes_xyxy"], raw["scores"], strict=True)
            if float(score) >= detector_threshold
        ]
        candidates.sort(key=lambda row: row[1], reverse=True)
        candidates = [
            candidate
            for candidate, keep in zip(candidates, _containment_keep(candidates), strict=True)
            if keep
        ]
        truth = [_xyxy(row) for row in record["annotations"]]
        unmatched = set(range(len(truth)))
        for box, _score in candidates:
            best = max(unmatched, key=lambda index: _iou(box, truth[index])) if unmatched else None
            if best is not None and _iou(box, truth[best]) >= 0.50:
                unmatched.remove(best)
                positive.append((image_id, box, 1, "mined_matched_proposal"))
            else:
                negative.append((image_id, box, 0, "mined_false_proposal"))

    per_label = min(maximum_per_label, len(positive), len(negative))

    def select(rows):
        if len(rows) <= per_label:
            return rows
        indexes = rng.choice(len(rows), per_label, replace=False)
        return [rows[int(index)] for index in indexes]

    selected = select(positive) + select(negative)
    rng.shuffle(selected)
    return selected


class _CropDataset:
    def __init__(self, cache: Path, samples: list[tuple], *, training: bool) -> None:
        metadata = json.loads((cache / "index.json").read_text(encoding="utf-8"))
        self.images = np.load(cache / metadata["array_filename"], mmap_mode="r")
        self.index = {int(key): int(value) for key, value in metadata["index"].items()}
        self.samples = samples
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        torch = require_torch()
        image_id, box, label, _kind = self.samples[index]
        array = np.asarray(self.images[self.index[image_id]])
        image = Image.fromarray(array, mode="RGB")
        left, top, right, bottom = box
        crop = image.crop(
            (
                max(0, math.floor(left)),
                max(0, math.floor(top)),
                min(image.width, math.ceil(right)),
                min(image.height, math.ceil(bottom)),
            )
        )
        if self.training:
            if random.random() < 0.5:
                crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            crop = ImageEnhance.Brightness(crop).enhance(random.uniform(0.82, 1.18))
            crop = ImageEnhance.Contrast(crop).enhance(random.uniform(0.85, 1.15))
            crop = ImageEnhance.Color(crop).enhance(random.uniform(0.88, 1.12))
        crop = crop.resize((192, 192), Image.Resampling.BICUBIC)
        values = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
        values = (values - MEAN) / STD
        return torch.from_numpy(values), torch.tensor(label, dtype=torch.int64)


class _Verifier:
    def __init__(self, weights: Path, *, device: str):
        torch = require_torch()

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = load_dinov3_convnext_tiny(weights, device=device)
                self.classifier = torch.nn.Linear(768, 2)

            def forward(self, pixel_values):
                features = self.backbone(pixel_values)
                return self.classifier(features)

        self.module = Module().to(device)


def _metrics(model, loader) -> dict[str, float | int]:
    torch = require_torch()
    truth = []
    scores = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.cuda(non_blocking=True))
            scores.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
            truth.extend(labels.tolist())
    truth_array = np.asarray(truth, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float32)
    predicted = score_array >= 0.5
    false_negative = int(np.count_nonzero((truth_array == 1) & ~predicted))
    false_positive = int(np.count_nonzero((truth_array == 0) & predicted))
    return {
        "sample_count": len(truth),
        "positive_count": int(np.count_nonzero(truth_array == 1)),
        "negative_count": int(np.count_nonzero(truth_array == 0)),
        "false_negative_count": false_negative,
        "false_positive_count": false_positive,
        "accuracy": float(np.mean(predicted == truth_array)),
        "positive_score_minimum": float(score_array[truth_array == 1].min()),
        "negative_score_maximum": float(score_array[truth_array == 0].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fresh DINOv3 completeness verifier")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-per-label", type=int, default=18000)
    parser.add_argument("--validation-per-label", type=int, default=3000)
    parser.add_argument("--detector-predictions", type=Path)
    parser.add_argument("--detector-threshold", type=float, default=0.20)
    parser.add_argument("--mined-train-per-label", type=int, default=30000)
    parser.add_argument("--mined-validation-per-label", type=int, default=5000)
    parser.add_argument("--elongated-repeat", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.00001)
    parser.add_argument("--head-learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    if args.elongated_repeat < 1:
        parser.error("--elongated-repeat must be positive")

    actual_sha256 = _sha256(args.weights)
    if actual_sha256 != args.weights_sha256.lower():
        raise ValueError("official DINOv3 weight checksum mismatch")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch = require_torch()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    records = _records(args.manifest)
    train_samples = _make_samples(
        records,
        split="train",
        maximum_per_label=args.train_per_label,
        seed=args.seed,
        elongated_repeat=args.elongated_repeat,
    )
    validation_samples = _make_samples(
        records,
        split="validation",
        maximum_per_label=args.validation_per_label,
        seed=args.seed,
        elongated_repeat=1,
    )
    if args.detector_predictions is not None:
        predictions = {int(row["image_id"]): row for row in _records(args.detector_predictions)}
        missing = sorted({int(row["image_id"]) for row in records} - predictions.keys())
        if missing:
            raise ValueError(f"detector predictions are missing image ids: {missing[:10]}")
        train_samples.extend(
            _make_mined_samples(
                records,
                predictions,
                split="train",
                maximum_per_label=args.mined_train_per_label,
                detector_threshold=args.detector_threshold,
                seed=args.seed,
            )
        )
        validation_samples.extend(
            _make_mined_samples(
                records,
                predictions,
                split="validation",
                maximum_per_label=args.mined_validation_per_label,
                detector_threshold=args.detector_threshold,
                seed=args.seed,
            )
        )
    train_dataset = _CropDataset(args.cache, train_samples, training=True)
    validation_dataset = _CropDataset(args.cache, validation_samples, training=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = _Verifier(args.weights, device="cuda").module
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.backbone.stages[-1].parameters():
        parameter.requires_grad = True
    for parameter in model.backbone.norm.parameters():
        parameter.requires_grad = True
    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.learning_rate},
            {"params": model.classifier.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.02,
    )
    scaler = torch.amp.GradScaler("cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    history = []
    best_rank = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.stages[0].eval()
        model.backbone.stages[1].eval()
        model.backbone.stages[2].eval()
        loss_sum = 0.0
        count = 0
        epoch_started = time.perf_counter()
        for images, labels in train_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * len(labels)
            count += len(labels)
        scheduler.step()
        validation = _metrics(model, validation_loader)
        row = {
            "epoch": epoch,
            "loss": loss_sum / count,
            "duration_seconds": time.perf_counter() - epoch_started,
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        rank = (
            int(validation["false_negative_count"]) + int(validation["false_positive_count"]),
            int(validation["false_negative_count"]),
            int(validation["false_positive_count"]),
            epoch,
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            torch.save(model.state_dict(), args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    torch.save(model.state_dict(), args.output_dir / "last.pt")
    report = {
        "schema_version": "1.0",
        "experiment": "store2_source_only_dinov3_completeness_verifier",
        "training_source_policy": "single_objects_4-derived-synthetic-only",
        "evaluation_data_used_for_fitting": False,
        "initial_checkpoint": None,
        "official_weights_sha256": actual_sha256,
        "elongated_repeat": args.elongated_repeat,
        "detector_predictions": (
            {
                "path": str(args.detector_predictions.resolve()),
                "sha256": _sha256(args.detector_predictions),
                "score_threshold": args.detector_threshold,
                "containment_minimum": 0.70,
                "containment_maximum_area_ratio": 0.55,
            }
            if args.detector_predictions is not None
            else None
        ),
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
        "train_kind_counts": {
            kind: sum(row[3] == kind for row in train_samples)
            for kind in sorted({row[3] for row in train_samples})
        },
        "history": history,
        "best_rank": best_rank,
        "duration_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
