from __future__ import annotations

import argparse
from functools import partial
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from .data import DetectionDataset
from .evaluate_detector import _metrics_grid
from .models import require_torch
from .run_record import write_run_record
from .config_file import parse_args_with_config


def collate_detection_batch(batch, *, processor):
    images, annotations = zip(*batch)
    return processor(images=list(images), annotations=list(annotations), return_tensors="pt")


def detector_optimizer_recipe(args: argparse.Namespace) -> dict[str, Any]:
    """Return the exact optimizer/initialization recipe bound to a checkpoint."""
    base_lr = float(args.learning_rate)
    head_lr_multiplier = float(getattr(args, "head_lr_multiplier", 1.0))
    prior_probability = float(getattr(args, "class_head_prior_probability", 0.5))
    warmup_epochs = int(getattr(args, "warmup_epochs", 0))
    weight_decay = float(getattr(args, "weight_decay", 1e-4))
    min_score_threshold = float(getattr(args, "min_score_threshold", 0.05))
    max_score_threshold = float(getattr(args, "max_score_threshold", 0.95))
    threshold_steps = int(getattr(args, "threshold_steps", 91))
    nms_iou_threshold = float(getattr(args, "nms_iou_threshold", 0.7))
    match_iou_threshold = float(getattr(args, "match_iou_threshold", 0.5))
    target_recall = float(getattr(args, "target_recall", 0.99))
    max_queries = int(getattr(args, "max_queries", 300))
    if base_lr <= 0:
        raise ValueError("detector learning_rate must be positive")
    if head_lr_multiplier <= 0:
        raise ValueError("detector head_lr_multiplier must be positive")
    if not 0.0 < prior_probability < 1.0:
        raise ValueError("detector class_head_prior_probability must be between zero and one")
    if not 0 <= warmup_epochs < int(args.epochs):
        raise ValueError("detector warmup_epochs must be non-negative and less than epochs")
    if weight_decay < 0:
        raise ValueError("detector weight_decay must be non-negative")
    if not 0.0 <= min_score_threshold <= max_score_threshold <= 1.0:
        raise ValueError("detector score-threshold range must be within [0, 1]")
    if threshold_steps < 1:
        raise ValueError("detector threshold_steps must be positive")
    if not 0.0 <= nms_iou_threshold <= 1.0:
        raise ValueError("detector nms_iou_threshold must be within [0, 1]")
    if not 0.0 <= match_iou_threshold <= 1.0:
        raise ValueError("detector match_iou_threshold must be within [0, 1]")
    if not 0.0 <= target_recall <= 1.0:
        raise ValueError("detector target_recall must be within [0, 1]")
    if max_queries < 1:
        raise ValueError("detector max_queries must be positive")
    return {
        "schema_version": 4,
        "optimizer": "AdamW",
        "base_learning_rate": base_lr,
        "randomly_reinitialized_learning_rate": base_lr * head_lr_multiplier,
        "head_lr_multiplier": head_lr_multiplier,
        "weight_decay": weight_decay,
        "class_head_prior_probability": prior_probability,
        "class_head_bias": math.log(prior_probability / (1.0 - prior_probability)),
        "scheduler": (
            "cosine" if warmup_epochs == 0 else "linear_warmup_then_cosine"
        ),
        "warmup_epochs": warmup_epochs,
        "total_epochs": int(args.epochs),
        "randomly_reinitialized_modules": [
            "model.enc_score_head",
            "class_embed.*",
            "model.denoising_class_embed",
        ],
        "checkpoint_selection": {
            "policy": "target_recall_then_count_precision_threshold",
            "min_score_threshold": min_score_threshold,
            "max_score_threshold": max_score_threshold,
            "threshold_steps": threshold_steps,
            "nms_iou_threshold": nms_iou_threshold,
            "match_iou_threshold": match_iou_threshold,
            "target_recall": target_recall,
            "max_queries": max_queries,
        },
    }


def detector_classification_heads(model) -> list[tuple[str, Any]]:
    """Locate the one-class encoder and decoder score heads in RT-DETRv2."""
    heads: list[tuple[str, Any]] = []
    encoder_head = getattr(getattr(model, "model", None), "enc_score_head", None)
    if encoder_head is not None:
        heads.append(("model.enc_score_head", encoder_head))
    decoder_heads = getattr(model, "class_embed", None)
    if decoder_heads is not None:
        heads.extend((f"class_embed.{index}", head) for index, head in enumerate(decoder_heads))
    if not heads:
        raise ValueError("RT-DETRv2 classification heads were not found")
    identities = [id(head) for _, head in heads]
    if len(identities) != len(set(identities)):
        raise ValueError("RT-DETRv2 classification heads contain duplicate modules")
    return heads


def initialize_detector_classification_head_biases(model, prior_probability: float) -> float:
    """Set encoder/decoder score-head biases to a low foreground prior."""
    if not 0.0 < float(prior_probability) < 1.0:
        raise ValueError("class-head prior probability must be between zero and one")
    bias = math.log(float(prior_probability) / (1.0 - float(prior_probability)))
    heads = detector_classification_heads(model)
    for name, head in heads:
        parameter = getattr(head, "bias", None)
        if parameter is None:
            raise ValueError(f"RT-DETRv2 classification head has no bias: {name}")
        parameter.data.fill_(bias)
    return bias


def detector_randomly_reinitialized_modules(model) -> list[tuple[str, Any]]:
    """Return every module replaced by the 1-class RT-DETRv2 load."""
    modules = detector_classification_heads(model)
    denoising_embedding = getattr(
        getattr(model, "model", None), "denoising_class_embed", None
    )
    if denoising_embedding is None:
        raise ValueError("RT-DETRv2 denoising class embedding was not found")
    modules.append(("model.denoising_class_embed", denoising_embedding))
    identities = [id(module) for _, module in modules]
    if len(identities) != len(set(identities)):
        raise ValueError("RT-DETRv2 randomly reinitialized modules contain duplicates")
    return modules


def detector_optimizer_parameter_groups(model, recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """Split model parameters into disjoint base and confidence-head LR groups."""
    head_parameters = []
    head_ids: set[int] = set()
    for _, module in detector_randomly_reinitialized_modules(model):
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in head_ids:
                raise ValueError("duplicate parameter found across RT-DETRv2 reinitialized modules")
            head_ids.add(id(parameter))
            head_parameters.append(parameter)
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    base_ids = {id(parameter) for parameter in base_parameters}
    if base_ids & head_ids:
        raise RuntimeError("detector optimizer parameter groups overlap")
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if base_ids | head_ids != trainable_ids:
        raise RuntimeError("detector optimizer parameter groups do not cover the model")
    if not base_parameters or not head_parameters:
        raise ValueError("detector optimizer requires non-empty base and class-head groups")
    return [
        {
            "params": base_parameters,
            "lr": float(recipe["base_learning_rate"]),
            "group_name": "base",
        },
        {
            "params": head_parameters,
            "lr": float(recipe["randomly_reinitialized_learning_rate"]),
            "group_name": "randomly_reinitialized_modules",
        },
    ]


def build_detector_scheduler(torch, optimizer, recipe: dict[str, Any]):
    warmup_epochs = int(recipe["warmup_epochs"])
    total_epochs = int(recipe["total_epochs"])
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / float(warmup_epochs),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def detector_metric_quality_key(
    metrics: dict[str, Any], target_recall: float
) -> tuple[float, ...]:
    """Build the exact lexicographic checkpoint quality key."""
    if float(metrics["recall"]) >= float(target_recall):
        return (
            1.0,
            float(metrics["count_accuracy"]),
            float(metrics["precision"]),
            float(metrics["score_threshold"]),
        )
    return (
        0.0,
        float(metrics["recall"]),
        float(metrics["count_accuracy"]),
        float(metrics["precision"]),
    )


def select_detector_metric_candidate(
    candidates: list[dict[str, Any]], target_recall: float
) -> tuple[dict[str, Any], tuple[float, ...]]:
    if not candidates:
        raise ValueError("detector validation produced no metric candidates")
    selected = max(
        candidates,
        key=lambda item: detector_metric_quality_key(item, target_recall),
    )
    return selected, detector_metric_quality_key(selected, target_recall)


def evaluate_detector_validation_predictions(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    recipe: dict[str, Any],
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("detector validation records/predictions are not aligned")
    expected_ids = [int(record["image_id"]) for record in records]
    prediction_ids = [int(prediction["image_id"]) for prediction in predictions]
    if prediction_ids != expected_ids:
        raise ValueError("detector validation prediction image order is not aligned")
    options = recipe["checkpoint_selection"]
    thresholds = np.linspace(
        float(options["min_score_threshold"]),
        float(options["max_score_threshold"]),
        int(options["threshold_steps"]),
    )
    candidates = _metrics_grid(
        records,
        predictions,
        score_thresholds=thresholds,
        nms_iou_threshold=float(options["nms_iou_threshold"]),
        match_iou_threshold=float(options["match_iou_threshold"]),
        max_queries=int(options["max_queries"]),
    )
    selected, quality_key = select_detector_metric_candidate(
        candidates, float(options["target_recall"])
    )
    return {
        "detector_metrics": selected,
        "selected_score_threshold": float(selected["score_threshold"]),
        "target_recall_satisfied": float(selected["recall"])
        >= float(options["target_recall"]),
        "detector_quality_key": list(quality_key),
    }


def postprocess_detector_validation_batch(processor, outputs, records, *, torch, device):
    target_sizes = torch.asarray(
        [[int(record["height"]), int(record["width"])] for record in records],
        device=device,
    )
    processed = processor.post_process_object_detection(
        outputs, threshold=0.0, target_sizes=target_sizes
    )
    if len(processed) != len(records):
        raise ValueError("detector validation postprocessing changed batch cardinality")
    return [
        {
            "image_id": int(record["image_id"]),
            "boxes_xyxy": result["boxes"].float().cpu().numpy().tolist(),
            "scores": result["scores"].float().cpu().numpy().tolist(),
        }
        for record, result in zip(records, processed)
    ]


def validate_detector_progress_identity(
    progress: dict[str, Any], args: argparse.Namespace, recipe: dict[str, Any]
) -> None:
    mismatch = (
        int(progress.get("total_epochs", -1)) != int(args.epochs)
        or int(progress.get("fold", -1)) != int(args.fold)
        or bool(progress.get("final_training")) != bool(args.final_training)
        or progress.get("optimizer_recipe") != recipe
    )
    if not bool(args.final_training) and int(progress.get("completed_epochs", 0)) > 0:
        required = {
            "detector_metrics",
            "selected_score_threshold",
            "target_recall_satisfied",
            "detector_quality_key",
        }
        history = progress.get("history")
        mismatch = mismatch or not isinstance(progress.get("best_detector_quality_key"), list)
        mismatch = mismatch or progress.get("last_detector_metrics") is None
        mismatch = mismatch or progress.get("last_selected_score_threshold") is None
        mismatch = mismatch or not isinstance(
            progress.get("last_target_recall_satisfied"), bool
        )
        mismatch = mismatch or not isinstance(history, list) or any(
            not required <= set(record) for record in history
        )
    if mismatch:
        raise ValueError("detector progress identity mismatch")


def validate_detector_run_identity(run_path: Path, recipe: dict[str, Any]) -> None:
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("detector run provenance is missing or invalid") from error
    if run.get("optimizer_recipe") != recipe:
        raise ValueError("detector run optimizer recipe mismatch")


def write_detector_run_record(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    device: str,
    dataset_sizes: dict[str, int],
    recipe: dict[str, Any],
) -> None:
    write_run_record(
        output_dir,
        task="detector_training",
        args=args,
        device=device,
        dataset_sizes=dataset_sizes,
    )
    run_path = output_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["optimizer_recipe"] = recipe
    temporary = run_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(run_path)


def train(args: argparse.Namespace) -> None:
    torch = require_torch()
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    recipe = detector_optimizer_recipe(args)
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
        initialize_detector_classification_head_biases(
            model, float(recipe["class_head_prior_probability"])
        )
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
    optimizer = torch.optim.AdamW(
        detector_optimizer_parameter_groups(model, recipe),
        lr=float(recipe["base_learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    scheduler = build_detector_scheduler(torch, optimizer, recipe)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    best_detector_quality_key: list[float] | None = None
    stale_epochs = 0
    start_epoch = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def move_batch(batch):
        inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        inputs["labels"] = [
            {key: value.to(device) for key, value in label.items()} for label in batch["labels"]
        ]
        return inputs

    history = []
    if resume_training:
        validate_detector_run_identity(args.output_dir / "run.json", recipe)
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        validate_detector_progress_identity(progress, args, recipe)
        optimizer.load_state_dict(progress["optimizer"])
        scheduler.load_state_dict(progress["scheduler"])
        scaler.load_state_dict(progress["scaler"])
        history = list(progress["history"])
        start_epoch = int(progress["completed_epochs"])
        best_loss = float(progress["best_loss"])
        best_detector_quality_key = progress.get("best_detector_quality_key")
        stale_epochs = int(progress["stale_epochs"])
        random.setstate(progress["rng"]["python"])
        np.random.set_state(progress["rng"]["numpy"])
        torch.set_rng_state(progress["rng"]["torch"])
        if device.type == "cuda" and progress["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(progress["rng"]["cuda"])
    if not resume_training:
        write_detector_run_record(
            args.output_dir,
            args=args,
            device=str(device),
            dataset_sizes={
                "train": len(train_dataset),
                "validation": 0 if validation_dataset is None else len(validation_dataset),
            },
            recipe=recipe,
        )
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
        validation_quality = None
        if validation_loader is not None:
            model.eval()
            validation_losses = []
            validation_predictions: list[dict[str, Any]] = []
            validation_offset = 0
            with torch.inference_mode():
                for batch in validation_loader:
                    outputs = model(**move_batch(batch))
                    validation_losses.append(float(outputs.loss.detach().cpu()))
                    batch_size = len(batch["labels"])
                    batch_records = validation_dataset.records[
                        validation_offset : validation_offset + batch_size
                    ]
                    validation_predictions.extend(
                        postprocess_detector_validation_batch(
                            processor,
                            outputs,
                            batch_records,
                            torch=torch,
                            device=device,
                        )
                    )
                    validation_offset += batch_size
            validation_loss = float(np.mean(validation_losses))
            validation_quality = evaluate_detector_validation_predictions(
                validation_dataset.records, validation_predictions, recipe
            )
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
        }
        if validation_quality is not None:
            record.update(validation_quality)
        history.append(record)
        print(json.dumps(record))
        if validation_loss is None:
            best_loss = float(record["train_loss"])
        else:
            best_loss = min(best_loss, validation_loss)
            quality_key = list(record["detector_quality_key"])
            if best_detector_quality_key is None or quality_key > best_detector_quality_key:
                best_detector_quality_key = quality_key
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
            "best_detector_quality_key": best_detector_quality_key,
            "last_detector_metrics": record.get("detector_metrics"),
            "last_selected_score_threshold": record.get("selected_score_threshold"),
            "last_target_recall_satisfied": record.get("target_recall_satisfied"),
            "stale_epochs": stale_epochs,
            "history": history,
            "optimizer_recipe": recipe,
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
    parser.add_argument("--head-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--class-head-prior-probability", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-score-threshold", type=float, default=0.05)
    parser.add_argument("--max-score-threshold", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=91)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.7)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    train(parse_args_with_config(parser, section="detector"))


if __name__ == "__main__":
    main()
