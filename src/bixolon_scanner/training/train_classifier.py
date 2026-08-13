from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

from ..package import sha256_file
from .calibration import fit_temperature, select_approval_threshold, softmax, topk_accuracy
from .config_file import parse_args_with_config
from .data import ClassifierDataset
from .models import (
    DINO_V3_HUB_REPOSITORY,
    build_dino_classifier,
    require_torch,
    set_frozen_backbone,
)
from .run_record import write_run_record


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model, loader, device):
    torch = require_torch()
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            output = model(images).float().cpu().numpy()
            logits.append(output)
            targets.append(labels.numpy())
    return np.concatenate(logits), np.concatenate(targets)


def _calibration_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    temperature = fit_temperature(logits, targets)
    probabilities = softmax(logits, temperature)
    # Per-fold validation is too small to certify a 0.5% risk upper bound.
    # It is used only to compare frozen and partial stages; final thresholding
    # is performed once over all OOF predictions in evaluate.py.
    threshold = select_approval_threshold(probabilities, targets, confidence_level=None)
    return {
        "accuracy": float((probabilities.argmax(axis=1) == targets).mean()),
        "top3_accuracy": topk_accuracy(probabilities, targets),
        "temperature": temperature,
        "approval_threshold": threshold.threshold,
        "approved_precision": threshold.approved_precision,
        "approval_coverage": threshold.coverage,
        "empirical_risk_constraint_satisfied": threshold.risk_control_satisfied,
    }


def train(args: argparse.Namespace) -> None:
    torch = require_torch()
    from torch.utils.data import DataLoader

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    training_mode = "final_train" if args.final_training else "train"
    train_dataset = ClassifierDataset(
        args.manifest,
        args.dataset_root,
        mode=training_mode,
        fold=args.fold,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
    )
    validation_dataset = (
        None
        if args.final_training
        else ClassifierDataset(
            args.manifest,
            args.dataset_root,
            mode="validation",
            fold=args.fold,
            image_size=args.image_size,
            cache_dir=args.cache_dir,
        )
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        None
        if validation_dataset is None
        else DataLoader(
            validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )
    )
    if args.backbone_kind == "dinov3_convnext_tiny" and args.weights is None:
        raise ValueError("--weights is required for the licensed DINOv3 checkpoint")
    model = build_dino_classifier(
        args.backbone_kind,
        args.num_classes,
        pretrained_name=args.pretrained_name,
        weights_path=args.weights,
        hub_repository=args.hub_repository,
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_record(
        args.output_dir,
        task="classifier_training",
        args=args,
        device=str(device),
        dataset_sizes={
            "train": len(train_dataset),
            "validation": len(validation_dataset) if validation_dataset is not None else 0,
        },
    )

    def run_stage(name: str, epochs: int, learning_rate: float, unfreeze_blocks: int):
        set_frozen_backbone(model, unfreeze_last_blocks=unfreeze_blocks)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=0.05,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        history = []
        for epoch in range(epochs):
            model.train()
            losses = []
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    logits = model(images)
                    loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.05)
                losses.append(float(loss.detach().cpu()))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            epoch_record = {
                "stage": name,
                "epoch": epoch + 1,
                "training_loss": float(np.mean(losses)),
            }
            history.append(epoch_record)
            print(json.dumps(epoch_record), flush=True)
        if validation_loader is not None:
            logits, targets = _evaluate(model, validation_loader, device)
            metrics = _calibration_metrics(logits, targets)
        else:
            logits = targets = None
            metrics = {"final_training_loss": history[-1]["training_loss"]}
        checkpoint_path = args.output_dir / f"{name}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "backbone_kind": args.backbone_kind,
                "pretrained_name": args.pretrained_name,
                "backbone_architecture": (
                    "dinov3_convnext_tiny"
                    if args.backbone_kind == "dinov3_convnext_tiny"
                    else args.pretrained_name
                ),
                "source_revision": args.hub_repository.split(":", 1)[-1]
                if args.backbone_kind == "dinov3_convnext_tiny"
                else None,
                "source_weight_filename": args.weights.name if args.weights else None,
                "source_weight_sha256": sha256_file(args.weights) if args.weights else None,
                "num_classes": args.num_classes,
                "image_size": args.image_size,
                "stage": name,
                "metrics": metrics,
            },
            checkpoint_path,
        )
        if logits is not None and targets is not None:
            np.savez_compressed(
                args.output_dir / f"{name}_validation.npz", logits=logits, targets=targets
            )
        (args.output_dir / f"{name}_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        return metrics, checkpoint_path

    frozen_metrics, frozen_path = run_stage("frozen", args.frozen_epochs, args.frozen_lr, 0)
    frozen_state = torch.load(frozen_path, map_location=device, weights_only=False)["state_dict"]
    fine_metrics, fine_path = run_stage("partial", args.finetune_epochs, args.finetune_lr, 2)
    use_fine = args.final_training or (
        fine_metrics["accuracy"] >= frozen_metrics["accuracy"]
        and fine_metrics["approval_coverage"] > frozen_metrics["approval_coverage"]
    )
    selected_path = fine_path if use_fine else frozen_path
    selected_stage = "partial" if use_fine else "frozen"
    if not use_fine:
        model.load_state_dict(frozen_state)
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    selected["selection"] = {
        "frozen": frozen_metrics,
        "partial": fine_metrics,
        "selected": selected["stage"],
    }
    torch.save(selected, args.output_dir / "best.pt")
    if not args.final_training:
        shutil.copyfile(
            args.output_dir / f"{selected_stage}_validation.npz",
            args.output_dir / "best_validation.npz",
        )
    (args.output_dir / "selection.json").write_text(
        json.dumps(selected["selection"], indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DINOv3/DINOv2 bread classifier")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--final-training", action="store_true")
    parser.add_argument(
        "--backbone-kind",
        choices=("dinov3_convnext_tiny", "dinov2"),
        default="dinov3_convnext_tiny",
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--hub-repository", default=DINO_V3_HUB_REPOSITORY)
    parser.add_argument("--pretrained-name", default="facebook/dinov2-small")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frozen-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--frozen-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--cpu", action="store_true")
    train(parse_args_with_config(parser, section="classifier"))


if __name__ == "__main__":
    main()
