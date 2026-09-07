from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from bixolon_scanner.training.class_conditional_quality import (
    ClassConditionalQualitySample,
    build_class_conditional_quality_samples,
)
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


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _QualityDataset:
    def __init__(
        self,
        cache: Path,
        samples: list[ClassConditionalQualitySample],
        *,
        training: bool,
        image_size: int,
        seed: int,
        context_margin: float,
    ) -> None:
        metadata = json.loads((cache / "index.json").read_text(encoding="utf-8"))
        self.images = np.load(cache / metadata["array_filename"], mmap_mode="r")
        self.index = {int(key): int(value) for key, value in metadata["index"].items()}
        self.source_shapes = {
            int(key): tuple(int(value) for value in shape)
            for key, shape in metadata["source_shapes"].items()
        }
        self.samples = samples
        self.training = training
        self.image_size = image_size
        self.seed = seed
        self.context_margin = context_margin

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        torch = require_torch()
        sample = self.samples[index]
        array = np.asarray(self.images[self.index[sample.image_id]])
        image = Image.fromarray(array, mode="RGB")
        source_height, source_width = self.source_shapes[sample.image_id]
        scale_x = image.width / source_width
        scale_y = image.height / source_height
        left, top, right, bottom = sample.box_xyxy
        left, right = left * scale_x, right * scale_x
        top, bottom = top * scale_y, bottom * scale_y
        width = right - left
        height = bottom - top
        generator = random.Random(self.seed + index * 65537)
        if self.training:
            margin = generator.uniform(
                self.context_margin * 0.75,
                self.context_margin * 1.25,
            )
        else:
            margin = self.context_margin
        crop = image.crop(
            (
                max(0, math.floor(left - width * margin)),
                max(0, math.floor(top - height * margin)),
                min(image.width, math.ceil(right + width * margin)),
                min(image.height, math.ceil(bottom + height * margin)),
            )
        )
        if self.training:
            if generator.random() < 0.5:
                crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            turns = generator.randrange(4)
            if turns:
                crop = crop.rotate(90 * turns, expand=True)
            crop = ImageEnhance.Brightness(crop).enhance(generator.uniform(0.78, 1.22))
            crop = ImageEnhance.Contrast(crop).enhance(generator.uniform(0.82, 1.18))
            crop = ImageEnhance.Color(crop).enhance(generator.uniform(0.85, 1.15))
        crop = crop.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        values = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
        values = (values - MEAN) / STD
        return (
            torch.from_numpy(values),
            torch.tensor(sample.class_index, dtype=torch.int64),
            torch.tensor(float(sample.is_complete), dtype=torch.float32),
        )


class _QualityModel:
    def __init__(self, weights: Path, *, device: str):
        torch = require_torch()

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = load_dinov3_convnext_tiny(weights, device=device)
                self.quality_head = torch.nn.Linear(768, 20)

            def forward(self, pixel_values):
                return self.quality_head(self.backbone(pixel_values))

        self.module = Module().to(device)


def _validation_metrics(model, loader) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    torch = require_torch()
    scores = []
    labels = []
    classes = []
    model.eval()
    with torch.inference_mode():
        for images, class_indexes, complete in loader:
            logits = model(images.cuda(non_blocking=True))
            selected = logits.gather(1, class_indexes.cuda(non_blocking=True)[:, None])[:, 0]
            scores.extend(torch.sigmoid(selected.float()).cpu().tolist())
            labels.extend(complete.tolist())
            classes.extend(class_indexes.tolist())
    score_array = np.asarray(scores, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    class_array = np.asarray(classes, dtype=np.int64)
    incomplete_scores = score_array[label_array == 0]
    threshold = float(np.nextafter(incomplete_scores.max(), np.float32(1.0)))
    safe_complete = int(np.count_nonzero(score_array[label_array == 1] >= threshold))
    complete_count = int(np.count_nonzero(label_array == 1))
    metrics = {
        "sample_count": len(labels),
        "complete_count": complete_count,
        "incomplete_count": int(np.count_nonzero(label_array == 0)),
        "zero_unsafe_threshold": threshold,
        "zero_unsafe_complete_accepted_count": safe_complete,
        "zero_unsafe_complete_accepted_rate": safe_complete / complete_count,
        "complete_score_minimum": float(score_array[label_array == 1].min()),
        "incomplete_score_maximum": float(incomplete_scores.max()),
        "accuracy_at_0_5": float(np.mean((score_array >= 0.5) == label_array)),
    }
    return metrics, score_array, label_array, class_array


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a source-only class-conditional complete/partial verifier"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--complete-minimum", type=float, default=0.85)
    parser.add_argument("--incomplete-maximum", type=float, default=0.60)
    parser.add_argument("--train-per-class-label", type=int, default=1000)
    parser.add_argument("--validation-per-class-label", type=int, default=250)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--context-margin", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.00001)
    parser.add_argument("--head-learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()

    if not 0.0 <= args.context_margin <= 1.0:
        parser.error("--context-margin must be between zero and one")
    if _sha256(args.weights) != args.weights_sha256.lower():
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
    train_samples = build_class_conditional_quality_samples(
        records,
        split="train",
        complete_minimum=args.complete_minimum,
        incomplete_maximum=args.incomplete_maximum,
        maximum_per_class_label=args.train_per_class_label,
        seed=args.seed,
    )
    validation_samples = build_class_conditional_quality_samples(
        records,
        split="validation",
        complete_minimum=args.complete_minimum,
        incomplete_maximum=args.incomplete_maximum,
        maximum_per_class_label=args.validation_per_class_label,
        seed=args.seed,
    )
    train_loader = torch.utils.data.DataLoader(
        _QualityDataset(
            args.cache,
            train_samples,
            training=True,
            image_size=args.image_size,
            seed=args.seed,
            context_margin=args.context_margin,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = torch.utils.data.DataLoader(
        _QualityDataset(
            args.cache,
            validation_samples,
            training=False,
            image_size=args.image_size,
            seed=args.seed + 1,
            context_margin=args.context_margin,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = _QualityModel(args.weights, device="cuda").module
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
            {"params": model.quality_head.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.02
    )
    scaler = torch.amp.GradScaler("cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    history = []
    best_rank = None
    best_metrics = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for stage in model.backbone.stages[:-1]:
            stage.eval()
        loss_sum = 0.0
        count = 0
        epoch_started = time.perf_counter()
        for images, class_indexes, complete in train_loader:
            images = images.cuda(non_blocking=True)
            class_indexes = class_indexes.cuda(non_blocking=True)
            complete = complete.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(images).gather(1, class_indexes[:, None])[:, 0]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, complete)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * len(complete)
            count += len(complete)
        scheduler.step()
        metrics, scores, labels, classes = _validation_metrics(model, validation_loader)
        row = {
            "epoch": epoch,
            "loss": loss_sum / count,
            "duration_seconds": time.perf_counter() - epoch_started,
            "validation": metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        rank = (-int(metrics["zero_unsafe_complete_accepted_count"]), epoch)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_metrics = metrics
            torch.save(model.state_dict(), args.output_dir / "best.pt")
            np.savez_compressed(
                args.output_dir / "best-validation-scores.npz",
                scores=scores,
                labels=labels,
                class_indexes=classes,
            )
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    torch.save(model.state_dict(), args.output_dir / "last.pt")
    report = {
        "schema_version": "1.0",
        "experiment": "store2_source_only_class_conditional_quality",
        "training_manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": _sha256(args.manifest),
            "source_datasets": sorted({str(row.get("source_dataset")) for row in records}),
        },
        "evaluation_data_used_for_fitting": False,
        "official_weights_sha256": _sha256(args.weights),
        "settings": vars(args) | {"manifest": str(args.manifest), "cache": str(args.cache)},
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
        "best_rank": best_rank,
        "best_validation": best_metrics,
        "history": history,
        "duration_seconds": time.perf_counter() - started,
    }
    report["settings"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in report["settings"].items()
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
