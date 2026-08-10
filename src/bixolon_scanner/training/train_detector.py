from __future__ import annotations

import argparse
from functools import partial
import json
import random
from pathlib import Path

import numpy as np

from .data import DetectionDataset
from .models import require_torch
from .run_record import write_run_record
from .config_file import parse_args_with_config


def collate_detection_batch(batch, *, processor):
    images, annotations = zip(*batch)
    return processor(images=list(images), annotations=list(annotations), return_tensors="pt")


def train(args: argparse.Namespace) -> None:
    torch = require_torch()
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.pretrained_name, size={"height": args.image_size, "width": args.image_size})
    model = RTDetrV2ForObjectDetection.from_pretrained(
        args.pretrained_name,
        num_labels=1,
        id2label={0: "bread"},
        label2id={"bread": 0},
        ignore_mismatched_sizes=True,
    ).to(device)
    training_mode = "final_train" if args.final_training else "train"
    train_dataset = DetectionDataset(
        args.manifest, args.dataset_root, mode=training_mode, fold=args.fold, cache_dir=args.cache_dir
    )
    validation_dataset = None if args.final_training else DetectionDataset(
        args.manifest, args.dataset_root, mode="validation", fold=args.fold, cache_dir=args.cache_dir
    )

    collate = partial(collate_detection_batch, processor=processor)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate
    )
    validation_loader = None if validation_dataset is None else DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    stale_epochs = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_record(
        args.output_dir,
        task="detector_training",
        args=args,
        device=str(device),
        dataset_sizes={
            "train": len(train_dataset),
            "validation": 0 if validation_dataset is None else len(validation_dataset),
        },
    )

    def move_batch(batch):
        inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        inputs["labels"] = [
            {key: value.to(device) for key, value in label.items()} for label in batch["labels"]
        ]
        return inputs

    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            inputs = move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**inputs).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation_loss = None
        if validation_loader is not None:
            model.eval()
            validation_losses = []
            with torch.inference_mode():
                for batch in validation_loader:
                    validation_losses.append(float(model(**move_batch(batch)).loss.detach().cpu()))
            validation_loss = float(np.mean(validation_losses))
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
        }
        history.append(record)
        print(json.dumps(record))
        if validation_loss is None:
            continue
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            stale_epochs = 0
            model.save_pretrained(args.output_dir / "best")
            processor.save_pretrained(args.output_dir / "best")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if args.final_training:
        model.save_pretrained(args.output_dir / "best")
        processor.save_pretrained(args.output_dir / "best")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the class-agnostic RT-DETRv2 detector")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--final-training", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--pretrained-name", default="PekingU/rtdetr_v2_r18vd")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--cpu", action="store_true")
    train(parse_args_with_config(parser, section="detector"))


if __name__ == "__main__":
    main()
