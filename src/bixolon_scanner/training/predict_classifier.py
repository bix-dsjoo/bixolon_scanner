from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import ClassifierDataset
from .models import build_dino_classifier, require_torch


def predict(args: argparse.Namespace) -> None:
    torch = require_torch()
    from torch.utils.data import DataLoader

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    repository = (
        f"facebookresearch/dinov3:{checkpoint['source_revision']}"
        if checkpoint.get("source_revision")
        else "facebookresearch/dinov3"
    )
    model = build_dino_classifier(
        checkpoint.get("backbone_kind", "dinov2"),
        checkpoint["num_classes"],
        pretrained_name=checkpoint["pretrained_name"],
        hub_repository=repository,
        classifier_head_kind=checkpoint.get("classifier_head_kind", "linear"),
        cosine_scale=float(checkpoint.get("cosine_scale", 16.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device).eval()
    dataset = ClassifierDataset(
        args.manifest,
        args.dataset_root,
        mode=args.mode,
        fold=args.fold,
        image_size=checkpoint["image_size"],
        cache_dir=args.cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits.append(model(images.to(device, non_blocking=True)).float().cpu().numpy())
            targets.append(labels.numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        logits=np.concatenate(logits),
        targets=np.concatenate(targets),
        checkpoint_stage=np.asarray(checkpoint["stage"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create classifier logits for validation or final test"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("validation", "test"), default="test")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    predict(parser.parse_args())


if __name__ == "__main__":
    main()
