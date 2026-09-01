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

from ...configuration import load_json_config
from ...training.data import DetectionDataset, read_manifest
from ...training.dinov3_objectness_detector import (
    build_dinov3_fcos,
    collate_detection_batch,
    image_to_tensor,
    target_to_tensors,
)
from .rfdetr_box_evaluation import proposal_coverage_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = load_json_config(path)
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported DINOv3 detector experiment schema")
    if config.get("architecture") not in {
        "dinov3_convnext_tiny_fpn_fcos_objectness",
        "dinov3_convnext_tiny_last_stage_fpn_fcos_objectness",
    }:
        raise ValueError("experiment must use the pinned DINOv3 objectness architecture")
    exclusions = config.get("prohibited_inputs", {})
    if exclusions != {
        "existing_detector_weights": True,
        "existing_detector_predictions": True,
        "yolo_family": True,
        "rfdetr": True,
    }:
        raise ValueError("DINOv3 detector exclusions must be explicit and complete")
    if config["dataset"]["folds"] != [0, 1, 2]:
        raise ValueError("DINOv3 detector requires the locked three group-aware folds")
    if config["dataset"]["group_fold_overlap_allowed"] is not False:
        raise ValueError("DINOv3 detector cannot allow group-fold leakage")
    if config["training"]["run_test"] is not False:
        raise ValueError("development folds cannot be reported as a held-out test")
    return config


def validate_inputs(
    config: dict[str, Any], repository_root: Path, weights_path: Path, fold: int
) -> tuple[Path, Path]:
    if fold not in config["dataset"]["folds"]:
        raise ValueError(f"fold {fold} is not part of the DINOv3 detector experiment")
    manifest = repository_root / config["dataset"]["manifest"]
    dataset_root = repository_root / config["dataset"]["root"]
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if _sha256(manifest) != config["dataset"]["manifest_sha256"]:
        raise ValueError("DINOv3 detector manifest checksum mismatch")
    if _sha256(weights_path) != config["backbone"]["weights_sha256"]:
        raise ValueError("DINOv3 detector backbone checksum mismatch")
    return manifest, dataset_root


def checkpoint_rank(metrics: dict[str, Any], epoch: int) -> tuple[int, int, int, int]:
    """Rank recall-first: misses dominate every proposal-workload consideration."""
    return (
        -int(metrics["false_negative_count"]),
        -int(metrics["missed_image_count"]),
        -int(metrics["surplus_proposal_count"]),
        epoch,
    )


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _TorchDetectionDataset:
    def __init__(self, source: DetectionDataset):
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        image, target = self.source[index]
        return image_to_tensor(image), target_to_tensors(target)


def _validation_records(manifest: Path, fold: int) -> list[dict[str, Any]]:
    return [
        record
        for record in read_manifest(manifest)
        if record["record_type"] == "detection"
        and record["split"] == "development"
        and int(record["fold"]) == fold
        and not record.get("exclude_from_detector_training", False)
    ]


def _predict(model, loader, records: list[dict[str, Any]], device: str):
    import torch

    model.eval()
    rows: list[dict[str, Any]] = []
    elapsed = []
    record_by_id = {int(record["image_id"]): record for record in records}
    with torch.inference_mode():
        for images, targets in loader:
            inputs = [image.to(device, non_blocking=True) for image in images]
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - started) * 1000.0 / len(inputs))
            for image, target, output in zip(images, targets, outputs, strict=True):
                image_id = int(target["image_id"])
                record = record_by_id[image_id]
                boxes = output["boxes"].detach().cpu().clone()
                boxes[:, (0, 2)] *= float(record["width"]) / float(image.shape[-1])
                boxes[:, (1, 3)] *= float(record["height"]) / float(image.shape[-2])
                rows.append(
                    {
                        "image_id": str(image_id),
                        "image_path": record["image_path"],
                        "boxes_xyxy": boxes.tolist(),
                        "scores": output["scores"].detach().cpu().tolist(),
                        "class_ids": [0] * int(output["boxes"].shape[0]),
                    }
                )
    return rows, {
        "sample_count": len(elapsed),
        "p50_ms": float(np.percentile(elapsed, 50)),
        "p95_ms": float(np.percentile(elapsed, 95)),
        "p99_ms": float(np.percentile(elapsed, 99)),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    config = load_config(args.config)
    repository_root = args.repository_root.resolve()
    manifest, dataset_root = validate_inputs(
        config, repository_root, args.weights.resolve(), args.fold
    )
    device = str(config["model"]["device"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DINOv3 detector training requires CUDA")
    seed = int(config["training"]["seed"]) + args.fold
    _seed_everything(seed)

    output_dir = repository_root / config["output"]["fold_template"].format(fold=args.fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        None
        if args.disable_image_cache
        else (
            args.image_cache.resolve()
            if args.image_cache is not None
            else repository_root / config["dataset"]["image_cache"]
        )
    )
    if cache_dir is not None and not (cache_dir / "index.json").is_file():
        raise FileNotFoundError(
            f"DINOv3 detector image cache is missing: {cache_dir}; "
            "run bixolon tools cache-detector with image-size 640"
        )
    train_source = DetectionDataset(
        manifest,
        dataset_root,
        mode="train",
        fold=args.fold,
        cache_dir=cache_dir,
    )
    validation_source = DetectionDataset(
        manifest,
        dataset_root,
        mode="validation",
        fold=args.fold,
        cache_dir=cache_dir,
    )
    train_dataset = _TorchDetectionDataset(train_source)
    validation_dataset = _TorchDetectionDataset(validation_source)
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "collate_fn": collate_detection_batch,
        "pin_memory": device == "cuda",
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **loader_options,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    proposal_score_floor = (
        args.proposal_score_floor
        if args.proposal_score_floor is not None
        else float(config["model"]["proposal_score_floor"])
    )
    inference_nms_threshold = (
        args.inference_nms_threshold
        if args.inference_nms_threshold is not None
        else float(config["model"]["inference_nms_threshold"])
    )
    detections_per_image = (
        args.detections_per_image
        if args.detections_per_image is not None
        else int(config["model"]["detections_per_image"])
    )
    image_size = args.image_size or int(config["model"]["image_size"])
    model = build_dinov3_fcos(
        args.weights.resolve(),
        image_size=image_size,
        score_threshold=proposal_score_floor,
        nms_threshold=inference_nms_threshold,
        detections_per_image=detections_per_image,
        topk_candidates=int(config["model"]["topk_candidates"]),
        fpn_channels=int(config["model"]["fpn_channels"]),
        unfreeze_last_stages=int(config["backbone"]["unfreeze_last_stages"]),
        device=device,
    )
    records = _validation_records(manifest, args.fold)
    if args.evaluate_checkpoint is not None:
        checkpoint = torch.load(
            args.evaluate_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        if checkpoint.get("architecture") != config["architecture"]:
            raise ValueError("DINOv3 detector checkpoint architecture mismatch")
        if int(checkpoint.get("fold")) != args.fold:
            raise ValueError("DINOv3 detector checkpoint fold mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        predictions, latency = _predict(model, validation_loader, records, device)
        metrics = proposal_coverage_metrics(records, predictions)
        diagnostic = {
            "schema_version": "1.0",
            "experiment": config["experiment"],
            "mode": "proposal_superset_diagnostic",
            "fold": args.fold,
            "checkpoint_sha256": _sha256(args.evaluate_checkpoint.resolve()),
            "proposal_score_floor": proposal_score_floor,
            "inference_nms_threshold": inference_nms_threshold,
            "detections_per_image": detections_per_image,
            "image_size": image_size,
            "image_cache_used": cache_dir is not None,
            "proposal_metrics": metrics,
            "latency": latency,
            "existing_detector_used": False,
            "yolo_family_used": False,
            "rfdetr_used": False,
        }
        suffix = (
            f"superset-s{proposal_score_floor:g}-nms{inference_nms_threshold:g}"
            f"-k{detections_per_image}-size{image_size}"
            f"-{'cached' if cache_dir is not None else 'original'}"
        )
        _write_jsonl(output_dir / f"predictions-{suffix}.jsonl", predictions)
        (output_dir / f"report-{suffix}.json").write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
        return diagnostic
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    backbone_parameter_ids = {
        id(parameter) for parameter in model.backbone.dinov3.parameters() if parameter.requires_grad
    }
    backbone_parameters = [
        parameter for parameter in trainable if id(parameter) in backbone_parameter_ids
    ]
    head_parameters = [
        parameter for parameter in trainable if id(parameter) not in backbone_parameter_ids
    ]
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": head_parameters,
            "lr": float(config["training"]["learning_rate"]),
        }
    ]
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": float(config["training"]["backbone_learning_rate"]),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(args.epochs or config["training"]["epochs"])
    minimum_ratio = float(config["training"]["minimum_learning_rate_ratio"])

    def learning_rate_factor(epoch_index: int) -> float:
        progress = min(epoch_index, epochs) / epochs
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    history: list[dict[str, Any]] = []
    best_rank: tuple[int, int, int, int] | None = None
    best_path = output_dir / "checkpoint_best_recall_first.pth"
    evaluation_interval = int(config["training"]["evaluation_interval"])

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, targets in train_loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {
                    key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
                    for key, value in target.items()
                }
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss_parts = model(images, targets)
                loss = sum(loss_parts.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if epoch % evaluation_interval == 0 or epoch == epochs:
            predictions, latency = _predict(model, validation_loader, records, device)
            proposal_metrics = proposal_coverage_metrics(records, predictions)
            row["proposal_metrics"] = proposal_metrics
            row["latency"] = latency
            prediction_path = output_dir / f"predictions-epoch{epoch:03d}.jsonl"
            _write_jsonl(prediction_path, predictions)
            rank = checkpoint_rank(proposal_metrics, epoch)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                torch.save(
                    {
                        "schema_version": "1.0",
                        "architecture": config["architecture"],
                        "fold": args.fold,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "proposal_metrics": proposal_metrics,
                        "config": config,
                    },
                    best_path,
                )
                _write_jsonl(output_dir / "predictions-best.jsonl", predictions)
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    report = {
        "schema_version": "1.0",
        "experiment": config["experiment"],
        "fold": args.fold,
        "selection_scope": "group-held-out development fold; not independent test",
        "existing_detector_used": False,
        "yolo_family_used": False,
        "rfdetr_used": False,
        "backbone": {
            "kind": config["backbone"]["kind"],
            "source_revision": config["backbone"]["source_revision"],
            "weights_sha256": _sha256(args.weights.resolve()),
            "unfreeze_last_stages": config["backbone"]["unfreeze_last_stages"],
        },
        "data": {
            "manifest_sha256": _sha256(manifest),
            "training_image_count": len(train_dataset),
            "validation_image_count": len(validation_dataset),
            "group_fold_overlap_count": 0,
        },
        "training": {
            "epochs": epochs,
            "seed": seed,
            "best_epoch": int(best["epoch"]),
            "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        },
        "best_proposal_metrics": best["proposal_metrics"],
        "checkpoint_sha256": _sha256(best_path),
        "active_runtime_modified": False,
        "limitations": [
            "threshold and checkpoint were selected on development folds",
            "the accepted-route zero-error claim requires all-fold OOF plus separate calibration",
            "ONNX export and runtime parity are deferred until the detector clears proposal recall",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a recall-first DINOv3 class-agnostic detector"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/bread/dinov3_objectness_fcos_0.1.json"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--proposal-score-floor", type=float)
    parser.add_argument("--inference-nms-threshold", type=float)
    parser.add_argument("--detections-per-image", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--image-cache", type=Path)
    parser.add_argument("--disable-image-cache", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.proposal_score_floor is not None and not 0.0 <= args.proposal_score_floor <= 1.0:
        raise ValueError("proposal score floor must be between zero and one")
    if args.inference_nms_threshold is not None and not 0.0 <= args.inference_nms_threshold <= 1.0:
        raise ValueError("inference NMS threshold must be between zero and one")
    if args.detections_per_image is not None and args.detections_per_image < 1:
        raise ValueError("detections per image must be positive")
    if args.image_size is not None and args.image_size < 32:
        raise ValueError("image size must be at least 32")
    if args.image_cache is not None and args.disable_image_cache:
        raise ValueError("image cache override conflicts with disabled image cache")
    run(args)


if __name__ == "__main__":
    main()
