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

from bixolon_scanner.training.dinov3_objectness_detector import build_dinov3_fcos
from bixolon_scanner.training.models import require_torch
from bixolon_scanner.training.ssdlite_objectness_detector import (
    CachedObjectnessDataset,
    collate_detection_batch,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _FcosCachedDataset(CachedObjectnessDataset):
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        if self.class_aware:
            target["labels"] = target["labels"] - 1
        else:
            target["labels"] = target["labels"].new_zeros(target["labels"].shape)
        return image, target


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    from torch.utils.data import ConcatDataset, DataLoader, Subset

    if not torch.cuda.is_available():
        raise RuntimeError("DINOv3 FCOS training requires CUDA")
    weights = args.weights.resolve()
    actual_weights_sha256 = _sha256(weights)
    if actual_weights_sha256 != args.weights_sha256.lower():
        raise ValueError(
            "official DINOv3 weight checksum mismatch: "
            f"expected={args.weights_sha256.lower()} actual={actual_weights_sha256}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)

    real = _FcosCachedDataset(
        args.real_manifest,
        args.real_root,
        args.real_cache,
        training=True,
        class_aware=args.class_aware,
    )
    synthetic = _FcosCachedDataset(
        args.synthetic_manifest,
        args.synthetic_root,
        args.synthetic_cache,
        training=True,
        class_aware=args.class_aware,
    )
    empty = Subset(real, [i for i, row in enumerate(real.records) if not row["annotations"]])
    parts = [real for _ in range(args.real_repeat)]
    parts.extend(empty for _ in range(args.empty_repeat))
    parts.append(synthetic)
    loader = DataLoader(
        ConcatDataset(parts),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_detection_batch,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = build_dinov3_fcos(
        weights,
        image_size=real.image_size,
        score_threshold=0.01,
        nms_threshold=0.4,
        detections_per_image=100,
        topk_candidates=1000,
        fpn_channels=args.fpn_channels,
        unfreeze_last_stages=args.unfreeze_last_stages,
        foreground_class_count=20 if args.class_aware else 1,
        device="cuda",
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    minimum_ratio = 0.02

    def learning_rate_factor(epoch_index: int) -> float:
        progress = min(epoch_index, args.epochs) / args.epochs
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    scaler = torch.amp.GradScaler("cuda")
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        batches = 0
        epoch_started = time.perf_counter()
        for images, targets in loader:
            images = [image.cuda(non_blocking=True) for image in images]
            targets = [
                {key: value.cuda(non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss_parts = model(images, targets)
                loss = sum(loss_parts.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            for name, value in loss_parts.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach().cpu())
            batches += 1
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "losses": {name: total / batches for name, total in sorted(sums.items())},
            "duration_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    checkpoint = args.output_dir / "last.pt"
    torch.save(model.state_dict(), checkpoint)
    report = {
        "schema_version": "1.0",
        "experiment": "store2_source_only_dinov3_convnext_tiny_fcos",
        "selection_policy": "fixed_last_epoch_without_evaluation",
        "development_evaluation_accessed": False,
        "initial_detector_checkpoint": None,
        "backbone": {
            "source": "official DINOv3 pretrained weights",
            "path": str(weights),
            "sha256": actual_weights_sha256,
            "unfreeze_last_stages": args.unfreeze_last_stages,
        },
        "fresh_components": ["FPN", "FCOS classification head", "FCOS box head"],
        "settings": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": real.image_size,
            "fpn_channels": args.fpn_channels,
            "class_aware": args.class_aware,
            "real_repeat": args.real_repeat,
            "empty_repeat": args.empty_repeat,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "sample_counts": {
            "real": len(real),
            "synthetic": len(synthetic),
            "empty": len(empty),
        },
        "history": history,
        "duration_seconds": time.perf_counter() - started,
        "artifacts": {"checkpoint": checkpoint.name},
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a source-only, fixed-epoch DINOv3 FCOS detector"
    )
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--real-cache", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--real-repeat", type=int, default=8)
    parser.add_argument("--empty-repeat", type=int, default=8)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--unfreeze-last-stages", type=int, default=0)
    parser.add_argument("--class-aware", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20260903)
    _train(parser.parse_args())


if __name__ == "__main__":
    main()
