from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from bixolon_scanner.runtime.onnx import prepare_rgb
from bixolon_scanner.training.models import (
    build_dino_classifier,
    require_torch,
    set_frozen_backbone,
)
from bixolon_scanner.training.synthetic_roi import (
    DirectRoiRecipe,
    augment_direct_roi,
    prepare_direct_roi_source,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _AugmentedSupportDataset:
    def __init__(self, manifest: Path, root: Path, *, views: int, image_size: int) -> None:
        self.records = [
            json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line
        ]
        if len(self.records) != 200:
            raise ValueError("classifier training requires exactly 200 support images")
        self.views = views
        self.image_size = image_size
        self.recipe = DirectRoiRecipe(
            output_size=image_size,
            canvas_scale_min=0.78,
            canvas_scale_max=1.0,
            rotation_degrees=180.0,
            perspective_fraction=0.05,
            side_view_probability=0.42,
            side_view_minimum_compression=0.28,
            side_view_maximum_compression=0.58,
            brightness_min=0.65,
            brightness_max=1.25,
            contrast_min=0.75,
            contrast_max=1.25,
            saturation_min=0.75,
            saturation_max=1.2,
            blur_probability=0.2,
            blur_radius_max=0.9,
            jpeg_quality_min=76,
            jpeg_quality_max=96,
            crop_mode="border_connected_composite",
            procedural_gradient=True,
            procedural_shadow=True,
        )
        self.images = []
        self.cutouts = []
        for record in self.records:
            with Image.open(root / record["image_path"]) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                self.images.append(image.copy())
                self.cutouts.append(prepare_direct_roi_source(image, self.recipe))

    def __len__(self) -> int:
        return len(self.records) * self.views

    def __getitem__(self, index: int):
        torch = require_torch()
        source_index = index % len(self.records)
        view_index = index // len(self.records)
        record = self.records[source_index]
        if view_index == 0:
            image = self.images[source_index]
        else:
            image = augment_direct_roi(
                self.images[source_index],
                source_sha256=str(record["image_sha256"]),
                category_id=int(record["category_id"]),
                seed=20260905 + source_index * 1000 + view_index,
                recipe=self.recipe,
                prepared_cutout=self.cutouts[source_index],
            ).image
        values = prepare_rgb(
            image,
            (self.image_size, self.image_size),
            MEAN,
            STD,
            reducing_gap=1.0,
        )
        return torch.from_numpy(values), int(record["category_id"]) - 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-epoch source-only DINOv3 classifier fine-tuning"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--views", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backbone-learning-rate", type=float, default=0.00001)
    parser.add_argument("--head-learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

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
    dataset = _AugmentedSupportDataset(
        args.manifest,
        args.root,
        views=args.views,
        image_size=args.image_size,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = build_dino_classifier(
        "dinov3_convnext_tiny",
        20,
        weights_path=args.weights,
        feature_l2_normalize=True,
        classifier_head_kind="cosine",
        cosine_scale=16.0,
    ).cuda()
    set_frozen_backbone(model, unfreeze_last_stages=1)
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in backbone_ids
    ]
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.backbone_learning_rate * 0.02,
    )
    scaler = torch.amp.GradScaler("cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.stages[0].eval()
        model.backbone.stages[1].eval()
        model.backbone.stages[2].eval()
        loss_sum = 0.0
        correct = 0
        count = 0
        epoch_started = time.perf_counter()
        for images, labels in loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.05)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite classifier loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            count += len(labels)
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": loss_sum / count,
            "training_accuracy": correct / count,
            "backbone_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
            "duration_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    checkpoint = args.output_dir / "last.pt"
    torch.save(model.state_dict(), checkpoint)
    report = {
        "schema_version": "1.0",
        "experiment": "store2_source_only_dinov3_classifier_finetune",
        "selection_policy": "fixed_last_epoch_without_evaluation",
        "development_evaluation_accessed": False,
        "initial_classifier_checkpoint": None,
        "official_backbone_weights_sha256": actual_sha256,
        "fresh_components": ["cosine classifier head"],
        "settings": vars(args) | {"output_dir": str(args.output_dir), "weights": str(args.weights)},
        "augmentation_recipe": dataset.recipe.__dict__,
        "history": history,
        "duration_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint.name,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
