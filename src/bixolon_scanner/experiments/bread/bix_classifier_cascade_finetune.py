from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...configuration import load_json_config
from ...training.classifier_allowlist import sha256_file
from ...training.models import build_dino_classifier, require_torch, set_frozen_backbone
from .bix_classifier_cascade import (
    _cascade_metrics,
    _classification_metrics,
    _manifest_rows,
    _masked_tensor,
    _normalized_logits,
    build_source_records,
    route_predictions,
    validate_source_directory,
)
from .classifier_200_only import fit_small_sample_head, predict_ridge


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _entries(
    *,
    records: list[dict[str, Any]],
    dataset_root: Path,
    generated_dir: Path,
    banks: tuple[str, ...],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if "clean" in banks:
        entries.extend(
            {
                "path": str((dataset_root / str(record["image_path"])).resolve()),
                "label": int(record["target"]),
                "fold": int(record["fold"]),
                "provenance": {},
                "bank": "clean",
            }
            for record in records
        )
    for bank in banks:
        if bank == "clean":
            continue
        for row in _manifest_rows(generated_dir / "manifests" / f"{bank}.jsonl"):
            path = Path(str(row["derived_path"]))
            entries.append(
                {
                    "path": str(path.resolve()),
                    "label": int(row["target"]),
                    "fold": int(row["fold"]),
                    "provenance": dict(row["provenance"]),
                    "bank": bank,
                    "view": int(row["view_index"]),
                }
            )
    return entries


class PairedResolutionDataset:
    def __init__(
        self,
        entries: list[dict[str, Any]],
        *,
        margin_ratio: float = 0.05,
        distance_bias: float = 0.0,
    ) -> None:
        self.entries = entries
        self.margin_ratio = margin_ratio
        self.distance_bias = distance_bias

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        torch = require_torch()
        entry = self.entries[index]
        with Image.open(entry["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensor_192 = _masked_tensor(
                image,
                entry["provenance"],
                192,
                margin_ratio=self.margin_ratio,
                distance_bias=self.distance_bias,
            )
            tensor_224 = _masked_tensor(
                image,
                entry["provenance"],
                224,
                margin_ratio=self.margin_ratio,
                distance_bias=self.distance_bias,
            )
        return (
            torch.from_numpy(tensor_192),
            torch.from_numpy(tensor_224),
            int(entry["label"]),
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    *,
    config: dict[str, Any],
    entries: list[dict[str, Any]],
    weights: Path,
    output_path: Path,
    seed: int,
    cpu: bool,
    head_initialization: np.ndarray | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    torch = require_torch()
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    _seed_everything(seed)
    settings = config["training"]["finetune"]
    model = build_dino_classifier(
        "dinov3_convnext_tiny",
        int(config["dataset"]["class_count"]),
        weights_path=weights,
        feature_l2_normalize=True,
        classifier_head_kind="cosine",
        cosine_scale=16.0,
    ).to(device)
    if head_initialization is not None:
        expected = tuple(model.classifier.weight.shape)
        if head_initialization.shape != expected:
            raise ValueError(f"head initialization shape {head_initialization.shape} != {expected}")
        with torch.no_grad():
            model.classifier.weight.copy_(torch.from_numpy(head_initialization).to(device))
    set_frozen_backbone(model, unfreeze_last_stages=int(settings["unfreeze_last_stages"]))
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
            {
                "params": backbone_parameters,
                "lr": float(settings["backbone_learning_rate"]),
            },
            {"params": head_parameters, "lr": float(settings["head_learning_rate"])},
        ],
        weight_decay=float(settings["weight_decay"]),
    )
    loader = torch.utils.data.DataLoader(
        PairedResolutionDataset(
            entries,
            margin_ratio=float(config.get("preprocessing", {}).get("crop_margin_ratio", 0.05)),
            distance_bias=float(config.get("preprocessing", {}).get("neighbor_distance_bias", 0.0)),
        ),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=int(settings["workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(settings["workers"]) > 0,
        generator=torch.Generator().manual_seed(seed),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        for stage in model.backbone.stages[:-1]:
            stage.eval()
        loss_sum = 0.0
        classification_sum = 0.0
        consistency_sum = 0.0
        correct_192 = 0
        correct_224 = 0
        sample_count = 0
        started = time.perf_counter()
        for tensor_192, tensor_224, labels in loader:
            tensor_192 = tensor_192.to(device, non_blocking=True)
            tensor_224 = tensor_224.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                feature_192 = model.extract_features(tensor_192)
                feature_224 = model.extract_features(tensor_224)
                logits_192 = model.classifier(feature_192)
                logits_224 = model.classifier(feature_224)
                loss_192 = torch.nn.functional.cross_entropy(
                    logits_192,
                    labels,
                    label_smoothing=float(settings["label_smoothing"]),
                )
                loss_224 = torch.nn.functional.cross_entropy(
                    logits_224,
                    labels,
                    label_smoothing=float(settings["label_smoothing"]),
                )
                classification = (loss_192 + loss_224) * 0.5
                consistency = (
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        feature_192.float(), feature_224.float(), dim=1
                    )
                ).mean()
                loss = (
                    classification + float(settings["resolution_consistency_weight"]) * consistency
                )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite paired-resolution training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_count = len(labels)
            loss_sum += float(loss.detach().cpu()) * batch_count
            classification_sum += float(classification.detach().cpu()) * batch_count
            consistency_sum += float(consistency.detach().cpu()) * batch_count
            correct_192 += int((logits_192.argmax(1) == labels).sum().detach().cpu())
            correct_224 += int((logits_224.argmax(1) == labels).sum().detach().cpu())
            sample_count += batch_count
        row = {
            "epoch": epoch,
            "sample_count": sample_count,
            "loss": loss_sum / sample_count,
            "classification_loss": classification_sum / sample_count,
            "resolution_consistency_loss": consistency_sum / sample_count,
            "training_accuracy_192": correct_192 / sample_count,
            "training_accuracy_224": correct_224 / sample_count,
            "duration_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return model, history


def _predict(
    model,
    entries: list[dict[str, Any]],
    *,
    batch_size: int,
    cpu: bool,
    workers: int = 2,
    margin_ratio: float = 0.05,
    distance_bias: float = 0.0,
):
    torch = require_torch()
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    loader = torch.utils.data.DataLoader(
        PairedResolutionDataset(
            entries,
            margin_ratio=margin_ratio,
            distance_bias=distance_bias,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    logits_192: list[np.ndarray] = []
    logits_224: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for tensor_192, tensor_224, target in loader:
            tensor_192 = tensor_192.to(device, non_blocking=True)
            tensor_224 = tensor_224.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits_192.append(model(tensor_192).float().cpu().numpy())
                logits_224.append(model(tensor_224).float().cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(logits_192), np.concatenate(logits_224), np.concatenate(labels)


def run_finetune(
    *,
    config_path: Path,
    source_dir: Path,
    convnext_weights: Path,
    cascade_dir: Path,
    output_dir: Path,
    cpu: bool,
) -> dict[str, Any]:
    config = load_json_config(config_path)
    validate_source_directory(config, source_dir)
    dataset_root = source_dir.parent.resolve()
    records = build_source_records(source_dir, dataset_root)
    source_manifest = cascade_dir / "source-manifest.jsonl"
    if not source_manifest.is_file():
        raise ValueError("cascade source manifest is missing")
    original_report = json.loads((cascade_dir / "cascade-report.json").read_text(encoding="utf-8"))
    feature_root = cascade_dir / "cascade-features"
    vit_clean_npz = np.load(feature_root / "vit-160-clean.npz")
    vit_probe_npz = np.load(feature_root / "vit-160-probe.npz")
    vit_reject_npz = np.load(feature_root / "vit-160-reject.npz")
    vit_clean = {key: vit_clean_npz[key] for key in vit_clean_npz.files}
    vit_probe = {key: vit_probe_npz[key] for key in vit_probe_npz.files}
    vit_reject = {key: vit_reject_npz[key] for key in vit_reject_npz.files}
    train_entries = _entries(
        records=records,
        dataset_root=dataset_root,
        generated_dir=cascade_dir,
        banks=("clean", "appearance", "geometry", "context"),
    )
    probe_entries = _entries(
        records=records,
        dataset_root=dataset_root,
        generated_dir=cascade_dir,
        banks=("probe",),
    )
    reject_entries = _entries(
        records=records,
        dataset_root=dataset_root,
        generated_dir=cascade_dir,
        banks=("reject",),
    )
    probe_folds = np.asarray([entry["fold"] for entry in probe_entries])
    probe_views = np.asarray([entry["view"] for entry in probe_entries])
    reject_folds = np.asarray([entry["fold"] for entry in reject_entries])
    count = len(probe_entries)
    oof_192 = np.empty((count, 20), dtype=np.float32)
    oof_224 = np.empty((count, 20), dtype=np.float32)
    oof_vit = np.empty((count, 20), dtype=np.float32)
    oof_scores = np.empty((count, 20), dtype=np.float32)
    oof_approved = np.zeros(count, dtype=bool)
    oof_fallback = np.zeros(count, dtype=bool)
    oof_verifier = np.zeros(count, dtype=bool)
    reject_approved = np.zeros(len(reject_entries), dtype=bool)
    fold_reports: list[dict[str, Any]] = []
    seed = int(config["experiment"]["seed"])
    batch_size = int(config["training"]["finetune"]["batch_size"])

    for outer_fold in (0, 1, 2):
        fold_train = [entry for entry in train_entries if entry["fold"] != outer_fold]
        checkpoint = output_dir / "checkpoints" / f"fold-{outer_fold}.pt"
        model, history = train_model(
            config=config,
            entries=fold_train,
            weights=convnext_weights,
            output_path=checkpoint,
            seed=seed + outer_fold,
            cpu=cpu,
        )
        fold_probe = [entry for entry in probe_entries if entry["fold"] == outer_fold]
        logits_192, logits_224, labels = _predict(model, fold_probe, batch_size=batch_size, cpu=cpu)
        verifier_head = original_report["folds"][outer_fold]["selected"]["verifier_head"]
        verifier_train = vit_clean["folds"] != outer_fold
        verifier_weight, verifier_bias = fit_small_sample_head(
            vit_clean["features"][verifier_train],
            vit_clean["labels"][verifier_train],
            candidate=verifier_head,
            class_count=20,
        )
        fold_probe_mask = vit_probe["folds"] == outer_fold
        logits_vit = predict_ridge(
            vit_probe["features"][fold_probe_mask], verifier_weight, verifier_bias
        )
        policy = dict(original_report["folds"][outer_fold]["selected"]["policy"])
        result = route_predictions(logits_192, logits_224, logits_vit, **policy)
        target_mask = probe_folds == outer_fold
        oof_192[target_mask] = logits_192
        oof_224[target_mask] = logits_224
        oof_vit[target_mask] = logits_vit
        oof_scores[target_mask] = result["scores"]
        oof_approved[target_mask] = result["approved"]
        oof_fallback[target_mask] = result["fallback"]
        oof_verifier[target_mask] = result["verifier"]

        fold_reject = [entry for entry in reject_entries if entry["fold"] == outer_fold]
        reject_192, reject_224, _ = _predict(model, fold_reject, batch_size=batch_size, cpu=cpu)
        fold_reject_mask = vit_reject["folds"] == outer_fold
        reject_vit = predict_ridge(
            vit_reject["features"][fold_reject_mask], verifier_weight, verifier_bias
        )
        reject_result = route_predictions(reject_192, reject_224, reject_vit, **policy)
        reject_approved[reject_folds == outer_fold] = reject_result["approved"]
        fold_reports.append(
            {
                "outer_fold": outer_fold,
                "training_source_count": len(
                    {entry["path"] for entry in fold_train if entry["bank"] == "clean"}
                ),
                "training_roi_count": len(fold_train),
                "history": history,
                "policy_locked_from_frozen_inner_selection": policy,
                "verifier_head_locked_from_frozen_inner_selection": verifier_head,
                "metrics": _cascade_metrics(result, labels),
                "synthetic_reject_approved_count": int(np.count_nonzero(reject_result["approved"])),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        del model
        torch = require_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    labels = np.asarray([entry["label"] for entry in probe_entries])
    combined = {
        "scores": oof_scores,
        "predictions": oof_scores.argmax(axis=1),
        "approved": oof_approved,
        "fallback": oof_fallback,
        "verifier": oof_verifier,
    }
    by_probe_kind = {}
    for view, name in enumerate(("appearance", "geometry", "context")):
        mask = probe_views == view
        by_probe_kind[name] = _cascade_metrics(
            {key: value[mask] for key, value in combined.items()}, labels[mask]
        )

    final_checkpoint = output_dir / "classifier-package" / "convnext-shared-final.pt"
    final_model, final_history = train_model(
        config=config,
        entries=train_entries,
        weights=convnext_weights,
        output_path=final_checkpoint,
        seed=seed + 100,
        cpu=cpu,
    )
    del final_model
    selected_verifiers = [
        json.dumps(row["verifier_head_locked_from_frozen_inner_selection"], sort_keys=True)
        for row in fold_reports
    ]
    verifier_counts = Counter(selected_verifiers)
    final_verifier = json.loads(
        min(
            selected_verifiers,
            key=lambda value: (-verifier_counts[value], selected_verifiers.index(value)),
        )
    )
    verifier_weight, verifier_bias = fit_small_sample_head(
        vit_clean["features"],
        vit_clean["labels"],
        candidate=final_verifier,
        class_count=20,
    )
    verifier_path = output_dir / "classifier-package" / "vit160-verifier-head.npz"
    np.savez_compressed(verifier_path, weight=verifier_weight, bias=verifier_bias)
    final_policy = {
        key: float(
            np.median(
                [row["policy_locked_from_frozen_inner_selection"][key] for row in fold_reports]
            )
        )
        for key in ("primary_margin", "fallback_margin", "verifier_margin")
    }
    package = {
        "schema_version": "1.0",
        "status": "EXPERIMENTAL_NOT_RUNTIME_ACTIVATED",
        "architecture": "shared fine-tuned DINOv3 ConvNeXt-Tiny 192/224 plus frozen DINOv3 ViT-B/16 160",
        "training_image_root": str(source_dir.resolve()),
        "external_images_used": False,
        "convnext_checkpoint": final_checkpoint.name,
        "convnext_checkpoint_sha256": sha256_file(final_checkpoint),
        "vit160_head": verifier_path.name,
        "vit160_head_sha256": sha256_file(verifier_path),
        "routing": final_policy,
        "final_training_history": final_history,
        "limitation": "Not an ONNX Runtime/Catalog bundle; source-derived view OOF only.",
    }
    _write_json(output_dir / "classifier-package" / "metadata.json", package)
    report = {
        "schema_version": "1.0",
        "experiment": "bix_single_objects_paired_192_224_last_stage_finetune",
        "source_manifest_sha256": sha256_file(source_manifest),
        "external_images_used": False,
        "source_count": len(records),
        "stored_training_roi_count": len(train_entries),
        "paired_resolution_presentations_per_full_epoch": len(train_entries) * 2,
        "evaluation_contract": "source-view grouped OOF; all derived images inherit source fold",
        "primary_192": _classification_metrics(oof_192, labels),
        "detail_224": _classification_metrics(oof_224, labels),
        "convnext_equal_fusion": _classification_metrics(
            _normalized_logits(oof_192) + _normalized_logits(oof_224), labels
        ),
        "verifier_160": _classification_metrics(oof_vit, labels),
        "selective_cascade": _cascade_metrics(combined, labels),
        "selective_cascade_by_probe_kind": by_probe_kind,
        "synthetic_quality_reject": {
            "sample_count": len(reject_entries),
            "approved_count": int(np.count_nonzero(reject_approved)),
            "not_approved_count": int(np.count_nonzero(~reject_approved)),
            "not_approved_rate": float(np.mean(~reject_approved)),
            "warning": "Synthetic quality variants are diagnostic proxies, not real recapture ground truth.",
        },
        "folds": fold_reports,
        "final_experimental_package": package,
        "interpretation_guardrail": "No independent physical item, capture session, store, detector ROI, or external test was available.",
    }
    _write_json(output_dir / "finetune-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune the shared 192/224 ConvNeXt classifier and evaluate the ViT160 cascade"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--convnext-weights", type=Path, required=True)
    parser.add_argument("--cascade-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_finetune(
        config_path=args.config,
        source_dir=args.source_dir,
        convnext_weights=args.convnext_weights,
        cascade_dir=args.cascade_dir,
        output_dir=args.output_dir,
        cpu=args.cpu,
    )
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "finetune-report.json"),
                "selective_cascade": report["selective_cascade"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
