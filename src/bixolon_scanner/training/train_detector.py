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
    progress_path = args.output_dir / "training_progress.pt"
    last_dir = args.output_dir / "last"
    resume_training = bool(
        getattr(args, "resume", False)
        and progress_path.is_file()
        and (last_dir / "config.json").is_file()
    )
    processor_source = last_dir if resume_training else args.pretrained_name
    processor = AutoImageProcessor.from_pretrained(
        processor_source, size={"height": args.image_size, "width": args.image_size}
    )
    if resume_training:
        model = RTDetrV2ForObjectDetection.from_pretrained(last_dir).to(device)
    else:
        model = RTDetrV2ForObjectDetection.from_pretrained(
            args.pretrained_name,
            num_labels=1,
            id2label={0: "product"},
            label2id={"product": 0},
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
    start_epoch = 0
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
    if resume_training:
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if (
            int(progress.get("total_epochs", -1)) != int(args.epochs)
            or int(progress.get("fold", -1)) != int(args.fold)
            or bool(progress.get("final_training")) != bool(args.final_training)
        ):
            raise ValueError("detector progress identity mismatch")
        optimizer.load_state_dict(progress["optimizer"])
        scheduler.load_state_dict(progress["scheduler"])
        scaler.load_state_dict(progress["scaler"])
        history = list(progress["history"])
        start_epoch = int(progress["completed_epochs"])
        best_loss = float(progress["best_loss"])
        stale_epochs = int(progress["stale_epochs"])
        random.setstate(progress["rng"]["python"])
        np.random.set_state(progress["rng"]["numpy"])
        torch.set_rng_state(progress["rng"]["torch"])
        if device.type == "cuda" and progress["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(progress["rng"]["cuda"])
    for epoch in range(start_epoch, args.epochs):
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
            best_loss = float(record["train_loss"])
        elif validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            stale_epochs = 0
            model.save_pretrained(args.output_dir / "best")
            processor.save_pretrained(args.output_dir / "best")
        else:
            stale_epochs += 1
        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)
        progress = {
            "total_epochs": int(args.epochs),
            "fold": int(args.fold),
            "final_training": bool(args.final_training),
            "completed_epochs": epoch + 1,
            "best_loss": best_loss,
            "stale_epochs": stale_epochs,
            "history": history,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            },
        }
        temporary = progress_path.with_suffix(".tmp")
        torch.save(progress, temporary)
        temporary.replace(progress_path)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        if validation_loss is not None and stale_epochs >= args.patience:
            break
    if args.final_training:
        model.save_pretrained(args.output_dir / "best")
        processor.save_pretrained(args.output_dir / "best")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    progress_path.unlink(missing_ok=True)


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
    parser.add_argument("--resume", action="store_true")
    train(parse_args_with_config(parser, section="detector"))


if __name__ == "__main__":
    main()
