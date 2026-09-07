from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.runtime.onnx import prepare_rgb
from bixolon_scanner.training.count_verifier import (
    load_source_only_count_records,
    source_revision,
)
from bixolon_scanner.training.dinov3_objectness_detector import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_dinov3_convnext_tiny,
)
from bixolon_scanner.training.models import require_torch


def _dataset(records, image_size: int, *, augment: bool, seed: int):
    torch = require_torch()

    class CountDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            record = records[index]
            with Image.open(record["resolved_path"]) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                if augment:
                    rng = np.random.default_rng(seed + index)
                    if rng.random() < 0.5:
                        image = ImageOps.mirror(image)
                    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.88, 1.12)))
                    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.90, 1.10)))
                tensor = prepare_rgb(
                    image,
                    (image_size, image_size),
                    IMAGENET_MEAN,
                    IMAGENET_STD,
                    reducing_gap=1.0,
                )
            return torch.from_numpy(tensor), int(record["count_label"])

    return CountDataset()


def _model(weights: Path, *, device: str, unfreeze_last_stages: int):
    torch = require_torch()

    class CountModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = load_dinov3_convnext_tiny(weights, device=device)
            self.head = torch.nn.Linear(768, 9)
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for stage in self.backbone.stages[-unfreeze_last_stages:]:
                for parameter in stage.parameters():
                    parameter.requires_grad = True
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True

        def forward(self, pixel_values):
            return self.head(self.backbone(pixel_values))

    return CountModel().to(device)


def _evaluate(model, loader, *, device: str) -> dict:
    torch = require_torch()
    model.eval()
    count = 0
    exact = 0
    within_one = 0
    error_sum = 0
    confidence = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(images)
            probabilities = torch.softmax(logits.float(), dim=1)
            predictions = probabilities.argmax(dim=1)
            errors = (predictions - labels).abs()
            count += len(labels)
            exact += int((errors == 0).sum())
            within_one += int((errors <= 1).sum())
            error_sum += int(errors.sum())
            confidence.extend(probabilities.max(dim=1).values.cpu().tolist())
    return {
        "image_count": count,
        "exact_correct_count": exact,
        "exact_accuracy": exact / count,
        "within_one_accuracy": within_one / count,
        "mean_absolute_error": error_sum / count,
        "mean_confidence": float(np.mean(confidence)),
        "minimum_confidence": float(np.min(confidence)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a source-only DINOv3 count verifier")
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--backbone-learning-rate", type=float, default=0.00001)
    parser.add_argument("--head-learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--unfreeze-last-stages", type=int, default=1)
    # This script defines its paired-view dataset locally, which is not picklable
    # under Windows' spawn multiprocessing mode.  Keep the portable default at
    # zero; callers on fork-based platforms may opt in explicitly.
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    torch = require_torch()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = load_source_only_count_records(
        args.training_manifest,
        args.training_root,
        expected_dataset_version=args.dataset_version,
    )
    validation = load_source_only_count_records(
        args.validation_manifest,
        args.validation_root,
        expected_dataset_version=args.dataset_version,
    )
    if {row["image_sha256"] for row in training} & {row["image_sha256"] for row in validation}:
        raise ValueError("count verifier training and validation images overlap")
    train_loader = torch.utils.data.DataLoader(
        _dataset(training, args.image_size, augment=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=args.device == "cuda",
    )
    validation_loader = torch.utils.data.DataLoader(
        _dataset(validation, args.image_size, augment=False, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=args.device == "cuda",
    )
    model = _model(
        args.weights,
        device=args.device,
        unfreeze_last_stages=args.unfreeze_last_stages,
    )
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("head.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
            {"params": model.head.parameters(), "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    label_counts = np.bincount(
        np.asarray([row["count_label"] for row in training]), minlength=9
    ).astype(np.float32)
    class_weights = label_counts.sum() / np.maximum(label_counts, 1.0)
    class_weights /= class_weights.mean()
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(args.device),
        label_smoothing=0.03,
    )
    history = []
    best_accuracy = -1.0
    best_path = args.output_dir / "best.pt"
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for images, labels in train_loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=args.device == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            sample_count += len(labels)
        scheduler.step()
        validation_metrics = _evaluate(model, validation_loader, device=args.device)
        row = {
            "epoch": epoch,
            "training_loss": total_loss / sample_count,
            "validation": validation_metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_metrics["exact_accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["exact_accuracy"]
            torch.save(model.state_dict(), best_path)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=True))
    selected_metrics = _evaluate(model, validation_loader, device=args.device)
    model = model.to("cpu").eval()
    output_path = args.output_dir / "count-verifier.onnx"
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, args.image_size, args.image_size),),
        output_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(output_path))
    best_epoch = max(history, key=lambda row: row["validation"]["exact_accuracy"])["epoch"]
    report = {
        "schema_version": "1.0",
        "experiment": "store2_source_only_dinov3_convnext_exact_count_finetune",
        "model_role": "exact_count_verifier",
        "comparison_mode": "exact_count",
        "active_runtime_modified": False,
        "dataset_version": args.dataset_version,
        "training_source_policy": "bix_bakery_dataset-only",
        "training_manifest": str(args.training_manifest),
        "training_manifest_sha256": sha256_file(args.training_manifest),
        "training_image_count": len(training),
        "validation_manifest": str(args.validation_manifest),
        "validation_manifest_sha256": sha256_file(args.validation_manifest),
        "validation_image_count": len(validation),
        "source_revision": source_revision(),
        "source_weight_filename": args.weights.name,
        "source_weight_sha256": sha256_file(args.weights),
        "backbone_kind": "dinov3_convnext_tiny",
        "image_size": args.image_size,
        "count_labels": list(range(9)),
        "temperature": 1.0,
        "confidence_threshold": 0.5,
        "selected_epoch": best_epoch,
        "validation": selected_metrics,
        "history": history,
        "settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "backbone_learning_rate": args.backbone_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay,
            "unfreeze_last_stages": args.unfreeze_last_stages,
            "seed": args.seed,
        },
        "training_seconds": time.perf_counter() - started,
        "onnx_sha256": sha256_file(output_path),
        "opset": args.opset,
        "evaluation_images_used": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
