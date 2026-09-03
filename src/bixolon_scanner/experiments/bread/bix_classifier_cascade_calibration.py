from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...configuration import load_json_config
from ...training.classifier_allowlist import sha256_file
from ...training.models import build_dino_classifier, require_torch
from .bix_classifier_cascade import (
    _cascade_metrics,
    _classification_metrics,
    _normalized_logits,
    build_source_records,
    route_predictions,
    validate_source_directory,
)
from .bix_classifier_cascade_finetune import _entries, _predict, train_model
from .classifier_200_only import fit_small_sample_head, predict_ridge


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_convnext(weights: Path, checkpoint: Path, *, cpu: bool):
    torch = require_torch()
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model = build_dino_classifier(
        "dinov3_convnext_tiny",
        20,
        weights_path=weights,
        feature_l2_normalize=True,
        classifier_head_kind="cosine",
        cosine_scale=16.0,
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    return model.eval()


def _select_policy(
    *,
    config: dict[str, Any],
    logits_192: np.ndarray,
    logits_224: np.ndarray,
    logits_vit: np.ndarray,
    labels: np.ndarray,
    reject_192: np.ndarray,
    reject_224: np.ndarray,
    reject_vit: np.ndarray,
) -> dict[str, Any]:
    candidates = []
    index = 0
    grid = config["finetune_calibration"]
    for primary in grid["primary_margin_candidates"]:
        for fallback in grid["fallback_margin_candidates"]:
            for verifier in grid["verifier_margin_candidates"]:
                policy = {
                    "primary_margin": float(primary),
                    "fallback_margin": float(fallback),
                    "verifier_margin": float(verifier),
                }
                result = route_predictions(logits_192, logits_224, logits_vit, **policy)
                metrics = _cascade_metrics(result, labels)
                reject_result = route_predictions(reject_192, reject_224, reject_vit, **policy)
                reject_approved = int(np.count_nonzero(reject_result["approved"]))
                key = (
                    metrics["wrong_approved_count"],
                    metrics["top3_miss_count"],
                    -metrics["correct_approved_count"],
                    reject_approved,
                    metrics["fallback_count"],
                    metrics["verifier_count"],
                    index,
                )
                candidates.append(
                    {
                        "policy": policy,
                        "metrics": metrics,
                        "reject_approved_count": reject_approved,
                        "selection_key": list(key),
                    }
                )
                index += 1
    return min(candidates, key=lambda row: tuple(row["selection_key"]))


def run_calibration(
    *,
    config_path: Path,
    source_dir: Path,
    convnext_weights: Path,
    cascade_dir: Path,
    finetune_dir: Path,
    output_dir: Path,
    cpu: bool,
) -> dict[str, Any]:
    config = load_json_config(config_path)
    validate_source_directory(config, source_dir)
    dataset_root = source_dir.parent.resolve()
    records = build_source_records(source_dir, dataset_root)
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
    feature_root = cascade_dir / "cascade-features"
    vit_clean_file = np.load(feature_root / "vit-160-clean.npz")
    vit_probe_file = np.load(feature_root / "vit-160-probe.npz")
    vit_reject_file = np.load(feature_root / "vit-160-reject.npz")
    vit_clean = {key: vit_clean_file[key] for key in vit_clean_file.files}
    vit_probe = {key: vit_probe_file[key] for key in vit_probe_file.files}
    vit_reject = {key: vit_reject_file[key] for key in vit_reject_file.files}
    frozen_report = json.loads((cascade_dir / "cascade-report.json").read_text(encoding="utf-8"))
    probe_folds = np.asarray([entry["fold"] for entry in probe_entries])
    probe_views = np.asarray([entry["view"] for entry in probe_entries])
    reject_folds = np.asarray([entry["fold"] for entry in reject_entries])
    labels = np.asarray([entry["label"] for entry in probe_entries])
    oof_192 = np.empty((len(probe_entries), 20), dtype=np.float32)
    oof_224 = np.empty_like(oof_192)
    oof_vit = np.empty_like(oof_192)
    oof_scores = np.empty_like(oof_192)
    oof_approved = np.zeros(len(probe_entries), dtype=bool)
    oof_fallback = np.zeros(len(probe_entries), dtype=bool)
    oof_verifier = np.zeros(len(probe_entries), dtype=bool)
    reject_approved = np.zeros(len(reject_entries), dtype=bool)
    fold_reports = []
    batch_size = int(config["training"]["finetune"]["batch_size"])
    seed = int(config["experiment"]["seed"])

    for outer in (0, 1, 2):
        inner_train = (outer + 1) % 3
        inner_validation = (outer + 2) % 3
        inner_entries = [entry for entry in train_entries if entry["fold"] == inner_train]
        calibration_checkpoint = output_dir / "calibration-checkpoints" / f"outer-{outer}.pt"
        if calibration_checkpoint.is_file():
            calibration_model = _load_convnext(convnext_weights, calibration_checkpoint, cpu=cpu)
            history = [{"reused_checkpoint": True}]
        else:
            calibration_model, history = train_model(
                config=config,
                entries=inner_entries,
                weights=convnext_weights,
                output_path=calibration_checkpoint,
                seed=seed + 200 + outer,
                cpu=cpu,
            )
        validation_entries = [entry for entry in probe_entries if entry["fold"] == inner_validation]
        validation_192, validation_224, validation_labels = _predict(
            calibration_model,
            validation_entries,
            batch_size=batch_size,
            cpu=cpu,
        )
        verifier_head = frozen_report["folds"][outer]["selected"]["verifier_head"]
        verifier_train = vit_clean["folds"] == inner_train
        verifier_weight, verifier_bias = fit_small_sample_head(
            vit_clean["features"][verifier_train],
            vit_clean["labels"][verifier_train],
            candidate=verifier_head,
            class_count=20,
        )
        validation_vit = predict_ridge(
            vit_probe["features"][vit_probe["folds"] == inner_validation],
            verifier_weight,
            verifier_bias,
        )
        inner_reject_entries = [
            entry for entry in reject_entries if entry["fold"] == inner_validation
        ]
        reject_192, reject_224, _ = _predict(
            calibration_model,
            inner_reject_entries,
            batch_size=batch_size,
            cpu=cpu,
        )
        reject_vit = predict_ridge(
            vit_reject["features"][vit_reject["folds"] == inner_validation],
            verifier_weight,
            verifier_bias,
        )
        selected = _select_policy(
            config=config,
            logits_192=validation_192,
            logits_224=validation_224,
            logits_vit=validation_vit,
            labels=validation_labels,
            reject_192=reject_192,
            reject_224=reject_224,
            reject_vit=reject_vit,
        )
        del calibration_model
        torch = require_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        outer_checkpoint = finetune_dir / "checkpoints" / f"fold-{outer}.pt"
        outer_model = _load_convnext(convnext_weights, outer_checkpoint, cpu=cpu)
        outer_entries = [entry for entry in probe_entries if entry["fold"] == outer]
        outer_192, outer_224, outer_labels = _predict(
            outer_model, outer_entries, batch_size=batch_size, cpu=cpu
        )
        outer_verifier_train = vit_clean["folds"] != outer
        outer_verifier_weight, outer_verifier_bias = fit_small_sample_head(
            vit_clean["features"][outer_verifier_train],
            vit_clean["labels"][outer_verifier_train],
            candidate=verifier_head,
            class_count=20,
        )
        outer_vit = predict_ridge(
            vit_probe["features"][vit_probe["folds"] == outer],
            outer_verifier_weight,
            outer_verifier_bias,
        )
        outer_result = route_predictions(outer_192, outer_224, outer_vit, **selected["policy"])
        outer_mask = probe_folds == outer
        oof_192[outer_mask] = outer_192
        oof_224[outer_mask] = outer_224
        oof_vit[outer_mask] = outer_vit
        oof_scores[outer_mask] = outer_result["scores"]
        oof_approved[outer_mask] = outer_result["approved"]
        oof_fallback[outer_mask] = outer_result["fallback"]
        oof_verifier[outer_mask] = outer_result["verifier"]

        outer_reject_entries = [entry for entry in reject_entries if entry["fold"] == outer]
        outer_reject_192, outer_reject_224, _ = _predict(
            outer_model,
            outer_reject_entries,
            batch_size=batch_size,
            cpu=cpu,
        )
        outer_reject_vit = predict_ridge(
            vit_reject["features"][vit_reject["folds"] == outer],
            outer_verifier_weight,
            outer_verifier_bias,
        )
        outer_reject_result = route_predictions(
            outer_reject_192,
            outer_reject_224,
            outer_reject_vit,
            **selected["policy"],
        )
        reject_approved[reject_folds == outer] = outer_reject_result["approved"]
        fold_reports.append(
            {
                "outer_fold": outer,
                "inner_train_fold": inner_train,
                "inner_validation_fold": inner_validation,
                "calibration_training_history": history,
                "selected": selected,
                "outer_metrics": _cascade_metrics(outer_result, outer_labels),
                "outer_reject_approved_count": int(
                    np.count_nonzero(outer_reject_result["approved"])
                ),
                "calibration_checkpoint_sha256": sha256_file(calibration_checkpoint),
                "outer_checkpoint_sha256": sha256_file(outer_checkpoint),
            }
        )
        del outer_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined = {
        "scores": oof_scores,
        "predictions": oof_scores.argmax(axis=1),
        "approved": oof_approved,
        "fallback": oof_fallback,
        "verifier": oof_verifier,
    }
    by_kind = {}
    for view, name in enumerate(("appearance", "geometry", "context")):
        mask = probe_views == view
        by_kind[name] = _cascade_metrics(
            {key: value[mask] for key, value in combined.items()}, labels[mask]
        )
    report = {
        "schema_version": "1.0",
        "experiment": "nested_calibration_of_finetuned_bix_classifier_cascade",
        "external_images_used": False,
        "calibration_contract": "For each outer fold, train on one disjoint inner source-view fold and select policy on the other; apply only to the untouched outer fold.",
        "primary_192": _classification_metrics(oof_192, labels),
        "detail_224": _classification_metrics(oof_224, labels),
        "convnext_equal_fusion": _classification_metrics(
            _normalized_logits(oof_192) + _normalized_logits(oof_224), labels
        ),
        "verifier_160": _classification_metrics(oof_vit, labels),
        "selective_cascade": _cascade_metrics(combined, labels),
        "selective_cascade_by_probe_kind": by_kind,
        "synthetic_quality_reject": {
            "sample_count": len(reject_entries),
            "approved_count": int(np.count_nonzero(reject_approved)),
            "not_approved_count": int(np.count_nonzero(~reject_approved)),
            "not_approved_rate": float(np.mean(~reject_approved)),
        },
        "folds": fold_reports,
        "interpretation_guardrail": "No external image or independent physical item/capture/store test was available.",
    }
    _write_json(output_dir / "calibrated-report.json", report)
    final_policy = {
        key: float(np.median([row["selected"]["policy"][key] for row in fold_reports]))
        for key in ("primary_margin", "fallback_margin", "verifier_margin")
    }
    final_checkpoint = finetune_dir / "classifier-package" / "convnext-shared-final.pt"
    verifier_head = finetune_dir / "classifier-package" / "vit160-verifier-head.npz"
    package = {
        "schema_version": "1.0",
        "status": "EXPERIMENTAL_NOT_RUNTIME_ACTIVATED",
        "architecture": "shared fine-tuned DINOv3 ConvNeXt-Tiny 192/224 plus frozen DINOv3 ViT-B/16 160",
        "training_image_root": str(source_dir.resolve()),
        "external_images_used": False,
        "convnext_checkpoint": str(final_checkpoint.resolve()),
        "convnext_checkpoint_sha256": sha256_file(final_checkpoint),
        "vit160_head": str(verifier_head.resolve()),
        "vit160_head_sha256": sha256_file(verifier_head),
        "calibrated_routing": final_policy,
        "calibration_method": report["calibration_contract"],
        "activation_blockers": [
            "Correct approval rate is below 99% on source-derived OOF probe.",
            "Synthetic quality rejection is not representative of real recapture performance.",
            "No independent physical-item, capture-session, store, or detector-ROI test exists.",
            "ONNX export and Runtime/Catalog parity have not been run.",
        ],
    }
    _write_json(output_dir / "classifier-package" / "metadata.json", package)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested calibration for the fine-tuned BIX cascade"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--convnext-weights", type=Path, required=True)
    parser.add_argument("--cascade-dir", type=Path, required=True)
    parser.add_argument("--finetune-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    report = run_calibration(
        config_path=args.config,
        source_dir=args.source_dir,
        convnext_weights=args.convnext_weights,
        cascade_dir=args.cascade_dir,
        finetune_dir=args.finetune_dir,
        output_dir=args.output_dir,
        cpu=args.cpu,
    )
    print(json.dumps(report["selective_cascade"], ensure_ascii=False))


if __name__ == "__main__":
    main()
