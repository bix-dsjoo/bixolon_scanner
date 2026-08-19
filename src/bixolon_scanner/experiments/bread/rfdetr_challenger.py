from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported RF-DETR challenger config schema")
    folds = [int(value) for value in config["dataset"]["folds"]]
    if folds != [0, 1, 2]:
        raise ValueError("RF-DETR challenger requires the locked three group-aware folds")
    if config["dataset"]["group_fold_overlap_allowed"] is not False:
        raise ValueError("RF-DETR challenger cannot allow group-fold overlap")
    if config["dataset"]["class_mode"] != "class_aware_20":
        raise ValueError("RF-DETR challenger must keep the locked class-aware detector task")
    if int(config["model"]["num_classes"]) != 20:
        raise ValueError("RF-DETR challenger requires exactly 20 detector classes")
    if config["training"]["run_test"] is not False:
        raise ValueError("RF-DETR development folds cannot be reported as a held-out test")
    return config


def _resolve_repository_path(repository_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def validate_inputs(config: dict[str, Any], repository_root: Path, fold: int) -> dict[str, Path]:
    if fold not in {int(value) for value in config["dataset"]["folds"]}:
        raise ValueError(f"fold {fold} is not part of the locked RF-DETR experiment")
    installed_version = importlib.metadata.version(config["implementation"]["package"])
    if installed_version != config["implementation"]["version"]:
        raise ValueError(
            f"rfdetr version mismatch: expected {config['implementation']['version']}, "
            f"found {installed_version}"
        )
    checkpoint = _resolve_repository_path(repository_root, config["pretraining"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RF-DETR pretrained checkpoint is missing: {checkpoint}")
    if _sha256(checkpoint) != config["pretraining"]["sha256"]:
        raise ValueError("RF-DETR pretrained checkpoint checksum mismatch")
    dataset = _resolve_repository_path(
        repository_root, config["dataset"]["fold_dataset_template"].format(fold=fold)
    )
    provenance = dataset / "provenance.json"
    if not provenance.is_file():
        raise FileNotFoundError(f"RF-DETR fold provenance is missing: {provenance}")
    dataset_report = json.loads(provenance.read_text(encoding="utf-8"))
    if int(dataset_report["validation_fold"]) != fold:
        raise ValueError("RF-DETR fold dataset provenance mismatch")
    if int(dataset_report["group_fold_overlap_count"]) != 0:
        raise ValueError("RF-DETR fold dataset contains group leakage")
    if dataset_report["source_manifest_sha256"] != config["dataset"]["historical_manifest_sha256"]:
        raise ValueError("RF-DETR fold dataset manifest checksum mismatch")
    output = _resolve_repository_path(
        repository_root, config["dataset"]["fold_output_template"].format(fold=fold)
    )
    return {"checkpoint": checkpoint, "dataset": dataset, "output": output}


def training_kwargs(
    config: dict[str, Any], paths: dict[str, Path], *, epochs: int | None = None
) -> dict[str, Any]:
    training = config["training"]
    return {
        "dataset_dir": str(paths["dataset"]),
        "output_dir": str(paths["output"]),
        "epochs": int(epochs if epochs is not None else training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "grad_accum_steps": int(training["grad_accum_steps"]),
        "lr": float(training["learning_rate"]),
        "lr_encoder": float(training["encoder_learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "warmup_epochs": float(training["warmup_epochs"]),
        "lr_drop": int(training["learning_rate_drop_epoch"]),
        "square_resize_div_64": bool(training["square_resize_div_64"]),
        "multi_scale": bool(training["multi_scale"]),
        "expanded_scales": bool(training["expanded_scales"]),
        "use_ema": bool(training["use_ema"]),
        "early_stopping": bool(training["early_stopping"]),
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "eval_interval": int(training["evaluation_interval"]),
        "num_workers": int(training["num_workers"]),
        "seed": int(training["seed"]),
        "run_test": bool(training["run_test"]),
        "save_dataset_grids": bool(training["save_dataset_grids"]),
        "tensorboard": bool(training["tensorboard"]),
        "wandb": bool(training["wandb"]),
        "resolution": int(config["model"]["resolution"]),
        "device": str(config["model"]["device"]),
    }


def run(config_path: Path, repository_root: Path, fold: int, epochs: int | None) -> None:
    import torch
    from rfdetr import RFDETRLarge

    config = load_config(config_path)
    paths = validate_inputs(config, repository_root, fold)
    if not torch.cuda.is_available():
        raise RuntimeError("RF-DETR challenger training requires CUDA")
    paths["output"].mkdir(parents=True, exist_ok=True)
    notes = {
        "experiment": config["experiment"],
        "candidate": config["candidate"],
        "fold": fold,
        "manifest_sha256": config["dataset"]["historical_manifest_sha256"],
        "pretrained_sha256": config["pretraining"]["sha256"],
        "group_aware": True,
        "held_out_test_set": False,
    }
    model = RFDETRLarge(
        pretrain_weights=str(paths["checkpoint"]),
        num_classes=int(config["model"]["num_classes"]),
        device=str(config["model"]["device"]),
    )
    kwargs = training_kwargs(config, paths, epochs=epochs)
    model.train(**kwargs, notes=notes)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train the group-aware bread RF-DETR challenger")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/bread/rfdetr_large_bread_1.1.0.json"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    run(args.config, args.repository_root.resolve(), args.fold, args.epochs)


if __name__ == "__main__":
    main()
