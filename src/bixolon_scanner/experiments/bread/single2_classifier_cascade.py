from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...configuration import load_json_config
from ...training.classifier_allowlist import sha256_file
from ...training.models import require_torch
from .bix_classifier_cascade import (
    _build_vit,
    _cascade_metrics,
    _classification_metrics,
    _normalized_logits,
    route_predictions,
)
from .bix_classifier_cascade_calibration import _load_convnext, _select_policy
from .bix_classifier_cascade_finetune import _entries, _predict, train_model
from .classifier_200_only import _extract_features, fit_small_sample_head, predict_ridge
from .single2_augmentation_ablation import (
    _load_source_records,
    load_feature_cache,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prototype_initialization(
    frozen_features: dict[str, dict[str, np.ndarray]],
    *,
    excluded_fold: int | None,
    class_count: int,
) -> np.ndarray:
    values = np.concatenate(
        [
            frozen_features[bank]["features"]
            for bank in ("clean", "appearance", "geometry", "context")
        ]
    )
    labels = np.concatenate(
        [frozen_features[bank]["labels"] for bank in ("clean", "appearance", "geometry", "context")]
    )
    folds = np.concatenate(
        [frozen_features[bank]["folds"] for bank in ("clean", "appearance", "geometry", "context")]
    )
    selected = np.ones(len(labels), dtype=bool)
    if excluded_fold is not None:
        selected &= folds != excluded_fold
    prototypes = []
    for class_index in range(class_count):
        class_values = values[selected & (labels == class_index)]
        if not len(class_values):
            raise ValueError(f"prototype class {class_index} has no training feature")
        prototype = class_values.mean(axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
        prototypes.append(prototype)
    return np.stack(prototypes).astype(np.float32)


def _entry_tensors(
    entries: list[dict[str, Any]],
    *,
    size: int,
    margin_ratio: float,
    distance_bias: float,
) -> list[np.ndarray]:
    from .bix_classifier_cascade import _masked_tensor

    tensors = []
    for entry in entries:
        with Image.open(entry["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensors.append(
                _masked_tensor(
                    image,
                    entry["provenance"],
                    size,
                    margin_ratio=margin_ratio,
                    distance_bias=distance_bias,
                )
            )
    return tensors


def _prepare_vit_features(
    *,
    entries_by_bank: dict[str, list[dict[str, Any]]],
    vit_weights: Path,
    output_dir: Path,
    batch_size: int,
    margin_ratio: float,
    distance_bias: float,
    cpu: bool,
    reuse: bool,
) -> dict[str, dict[str, np.ndarray]]:
    feature_dir = output_dir / "vit-features"
    paths = {bank: feature_dir / f"{bank}.npz" for bank in entries_by_bank}
    if reuse and all(path.is_file() for path in paths.values()):
        return {
            bank: {key: archive[key] for key in archive.files}
            for bank, path in paths.items()
            for archive in [np.load(path)]
        }
    torch = require_torch()
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    vit = _build_vit(vit_weights).to(device).eval()
    output: dict[str, dict[str, np.ndarray]] = {}
    feature_dir.mkdir(parents=True, exist_ok=True)
    for bank, entries in entries_by_bank.items():
        tensors = _entry_tensors(
            entries,
            size=160,
            margin_ratio=margin_ratio,
            distance_bias=distance_bias,
        )
        values = {
            "features": _extract_features(vit, tensors, device=device, batch_size=batch_size),
            "labels": np.asarray([entry["label"] for entry in entries], dtype=np.int64),
            "folds": np.asarray([entry["fold"] for entry in entries], dtype=np.int64),
            "views": np.asarray([entry.get("view", -1) for entry in entries], dtype=np.int64),
        }
        np.savez_compressed(paths[bank], **values)
        output[bank] = values
    del vit
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _vit_oof(
    *,
    features: dict[str, dict[str, np.ndarray]],
    head: dict[str, Any],
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clean = features["clean"]
    probe = features["probe"]
    reject = features["reject"]
    probe_logits = np.empty((len(probe["labels"]), class_count), dtype=np.float32)
    reject_logits = np.empty((len(reject["labels"]), class_count), dtype=np.float32)
    for outer in (0, 1, 2):
        weight, bias = fit_small_sample_head(
            clean["features"][clean["folds"] != outer],
            clean["labels"][clean["folds"] != outer],
            candidate=head,
            class_count=class_count,
        )
        probe_mask = probe["folds"] == outer
        reject_mask = reject["folds"] == outer
        probe_logits[probe_mask] = predict_ridge(probe["features"][probe_mask], weight, bias)
        reject_logits[reject_mask] = predict_ridge(reject["features"][reject_mask], weight, bias)
    return (
        probe_logits,
        reject_logits,
        probe["labels"],
        probe["folds"],
        reject["folds"],
    )


def _candidate_config(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["training"]["finetune"]["backbone_learning_rate"] = float(
        candidate["backbone_learning_rate"]
    )
    value["training"]["finetune"]["epochs"] = int(
        candidate.get("epochs", value["training"]["finetune"]["epochs"])
    )
    return value


def _train_candidate_oof(
    *,
    config: dict[str, Any],
    candidate: dict[str, Any],
    candidate_index: int,
    train_entries: list[dict[str, Any]],
    probe_entries: list[dict[str, Any]],
    reject_entries: list[dict[str, Any]],
    frozen_features: dict[str, dict[str, np.ndarray]],
    convnext_weights: Path,
    output_dir: Path,
    cpu: bool,
) -> dict[str, Any]:
    class_count = int(config["dataset"]["class_count"])
    probe_folds = np.asarray([entry["fold"] for entry in probe_entries])
    reject_folds = np.asarray([entry["fold"] for entry in reject_entries])
    count = len(probe_entries)
    reject_count = len(reject_entries)
    oof_192 = np.empty((count, class_count), dtype=np.float32)
    oof_224 = np.empty_like(oof_192)
    reject_192 = np.empty((reject_count, class_count), dtype=np.float32)
    reject_224 = np.empty_like(reject_192)
    histories = []
    run_config = _candidate_config(config, candidate)
    preprocessing = config["preprocessing"]
    seed = int(config["experiment"]["seed"])
    for outer in (0, 1, 2):
        fold_train = [entry for entry in train_entries if int(entry["fold"]) != outer]
        initialization = None
        if candidate["head_initialization"] == "frozen_feature_prototype":
            initialization = _prototype_initialization(
                frozen_features,
                excluded_fold=outer,
                class_count=class_count,
            )
        checkpoint = output_dir / "checkpoints" / str(candidate["name"]) / f"fold-{outer}.pt"
        if checkpoint.is_file():
            model = _load_convnext(convnext_weights, checkpoint, cpu=cpu)
            history = [{"reused_checkpoint": True}]
        else:
            model, history = train_model(
                config=run_config,
                entries=fold_train,
                weights=convnext_weights,
                output_path=checkpoint,
                seed=seed + candidate_index * 10 + outer,
                cpu=cpu,
                head_initialization=initialization,
            )
        probe_mask = probe_folds == outer
        fold_probe = [entry for entry in probe_entries if int(entry["fold"]) == outer]
        logits_192, logits_224, _ = _predict(
            model,
            fold_probe,
            batch_size=int(config["training"]["finetune"]["batch_size"]),
            cpu=cpu,
            margin_ratio=float(preprocessing["crop_margin_ratio"]),
            distance_bias=float(preprocessing["neighbor_distance_bias"]),
        )
        oof_192[probe_mask] = logits_192
        oof_224[probe_mask] = logits_224
        reject_mask = reject_folds == outer
        fold_reject = [entry for entry in reject_entries if int(entry["fold"]) == outer]
        logits_192, logits_224, _ = _predict(
            model,
            fold_reject,
            batch_size=int(config["training"]["finetune"]["batch_size"]),
            cpu=cpu,
            margin_ratio=float(preprocessing["crop_margin_ratio"]),
            distance_bias=float(preprocessing["neighbor_distance_bias"]),
        )
        reject_192[reject_mask] = logits_192
        reject_224[reject_mask] = logits_224
        histories.append(
            {
                "outer_fold": outer,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_history": history,
            }
        )
        del model
        torch = require_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    labels = np.asarray([entry["label"] for entry in probe_entries])
    return {
        "candidate": candidate,
        "primary_192": _classification_metrics(oof_192, labels),
        "detail_224": _classification_metrics(oof_224, labels),
        "equal_fusion": _classification_metrics(
            _normalized_logits(oof_192) + _normalized_logits(oof_224), labels
        ),
        "histories": histories,
        "oof_192": oof_192,
        "oof_224": oof_224,
        "reject_192": reject_192,
        "reject_224": reject_224,
    }


def _calibrate_crossfit(
    *,
    config: dict[str, Any],
    candidate_result: dict[str, Any],
    vit_probe: np.ndarray,
    vit_reject: np.ndarray,
    labels: np.ndarray,
    probe_folds: np.ndarray,
    reject_folds: np.ndarray,
    probe_views: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float]]:
    count = len(labels)
    scores = np.empty((count, int(config["dataset"]["class_count"])), dtype=np.float32)
    approved = np.zeros(count, dtype=bool)
    fallback = np.zeros(count, dtype=bool)
    verifier = np.zeros(count, dtype=bool)
    reject_approved = np.zeros(len(reject_folds), dtype=bool)
    folds = []
    for outer in (0, 1, 2):
        calibration = probe_folds != outer
        calibration_reject = reject_folds != outer
        selected = _select_policy(
            config=config,
            logits_192=candidate_result["oof_192"][calibration],
            logits_224=candidate_result["oof_224"][calibration],
            logits_vit=vit_probe[calibration],
            labels=labels[calibration],
            reject_192=candidate_result["reject_192"][calibration_reject],
            reject_224=candidate_result["reject_224"][calibration_reject],
            reject_vit=vit_reject[calibration_reject],
        )
        test = probe_folds == outer
        result = route_predictions(
            candidate_result["oof_192"][test],
            candidate_result["oof_224"][test],
            vit_probe[test],
            **selected["policy"],
        )
        scores[test] = result["scores"]
        approved[test] = result["approved"]
        fallback[test] = result["fallback"]
        verifier[test] = result["verifier"]
        reject_test = reject_folds == outer
        reject_result = route_predictions(
            candidate_result["reject_192"][reject_test],
            candidate_result["reject_224"][reject_test],
            vit_reject[reject_test],
            **selected["policy"],
        )
        reject_approved[reject_test] = reject_result["approved"]
        folds.append({"outer_fold": outer, "selected": selected})
    combined = {
        "scores": scores,
        "predictions": scores.argmax(axis=1),
        "approved": approved,
        "fallback": fallback,
        "verifier": verifier,
    }
    by_kind = {}
    for view, name in enumerate(("appearance", "geometry", "context")):
        mask = probe_views == view
        by_kind[name] = _cascade_metrics(
            {key: value[mask] for key, value in combined.items()}, labels[mask]
        )
    report = {
        "metrics": _cascade_metrics(combined, labels),
        "by_probe_kind": by_kind,
        "synthetic_reject": {
            "sample_count": len(reject_approved),
            "approved_count": int(np.count_nonzero(reject_approved)),
            "not_approved_count": int(np.count_nonzero(~reject_approved)),
        },
        "folds": folds,
    }
    policy = {
        key: float(np.median([row["selected"]["policy"][key] for row in folds]))
        for key in ("primary_margin", "fallback_margin", "verifier_margin")
    }
    return report, policy


def run(
    *,
    config_path: Path,
    dataset_root: Path,
    source_manifest: Path,
    generated_dir: Path,
    convnext_weights: Path,
    vit_weights: Path,
    output_dir: Path,
    cpu: bool,
    reuse_vit: bool,
) -> dict[str, Any]:
    config = load_json_config(config_path)
    source_dir = (dataset_root / "single_objects_2").resolve()
    if source_dir != Path(str(config["dataset"]["allowed_root"])).resolve():
        raise ValueError("single_objects_2 source directory differs from the experiment lock")
    if sha256_file(source_manifest) != config["dataset"]["source_manifest_sha256"]:
        raise ValueError("single_objects_2 source manifest checksum mismatch")
    records = _load_source_records(source_manifest, dataset_root.resolve())
    train_entries = _entries(
        records=records,
        dataset_root=dataset_root.resolve(),
        generated_dir=generated_dir,
        banks=("clean", "appearance", "geometry", "context"),
    )
    probe_entries = _entries(
        records=records,
        dataset_root=dataset_root.resolve(),
        generated_dir=generated_dir,
        banks=("probe",),
    )
    reject_entries = _entries(
        records=records,
        dataset_root=dataset_root.resolve(),
        generated_dir=generated_dir,
        banks=("reject",),
    )
    frozen_features, _ = load_feature_cache(generated_dir)
    preprocessing = config["preprocessing"]
    vit_features = _prepare_vit_features(
        entries_by_bank={
            "clean": _entries(
                records=records,
                dataset_root=dataset_root.resolve(),
                generated_dir=generated_dir,
                banks=("clean",),
            ),
            "probe": probe_entries,
            "reject": reject_entries,
        },
        vit_weights=vit_weights,
        output_dir=output_dir,
        batch_size=int(config["training"]["feature_batch_size"]),
        margin_ratio=float(preprocessing["crop_margin_ratio"]),
        distance_bias=float(preprocessing["neighbor_distance_bias"]),
        cpu=cpu,
        reuse=reuse_vit,
    )
    vit_probe, vit_reject, labels, probe_folds, reject_folds = _vit_oof(
        features=vit_features,
        head=dict(config["training"]["verifier_head"]),
        class_count=int(config["dataset"]["class_count"]),
    )
    candidate_results = []
    for index, candidate in enumerate(config["training"]["candidates"]):
        print(json.dumps({"candidate_started": candidate["name"]}), flush=True)
        candidate_results.append(
            _train_candidate_oof(
                config=config,
                candidate=dict(candidate),
                candidate_index=index,
                train_entries=train_entries,
                probe_entries=probe_entries,
                reject_entries=reject_entries,
                frozen_features=frozen_features,
                convnext_weights=convnext_weights,
                output_dir=output_dir,
                cpu=cpu,
            )
        )
    selected_index = min(
        range(len(candidate_results)),
        key=lambda index: (
            candidate_results[index]["detail_224"]["top3_miss_count"],
            candidate_results[index]["detail_224"]["top1_error_count"],
            index,
        ),
    )
    selected = candidate_results[selected_index]
    probe_views = np.asarray([entry["view"] for entry in probe_entries])
    cascade, policy = _calibrate_crossfit(
        config=config,
        candidate_result=selected,
        vit_probe=vit_probe,
        vit_reject=vit_reject,
        labels=labels,
        probe_folds=probe_folds,
        reject_folds=reject_folds,
        probe_views=probe_views,
    )
    final_config = _candidate_config(config, selected["candidate"])
    final_initialization = None
    if selected["candidate"]["head_initialization"] == "frozen_feature_prototype":
        final_initialization = _prototype_initialization(
            frozen_features,
            excluded_fold=None,
            class_count=int(config["dataset"]["class_count"]),
        )
    package_dir = output_dir / "classifier-package"
    final_checkpoint = package_dir / "convnext-shared-final.pt"
    _, final_history = train_model(
        config=final_config,
        entries=train_entries,
        weights=convnext_weights,
        output_path=final_checkpoint,
        seed=int(config["experiment"]["seed"]) + 100,
        cpu=cpu,
        head_initialization=final_initialization,
    )
    clean = vit_features["clean"]
    vit_weight, vit_bias = fit_small_sample_head(
        clean["features"],
        clean["labels"],
        candidate=dict(config["training"]["verifier_head"]),
        class_count=int(config["dataset"]["class_count"]),
    )
    vit_head = package_dir / "vit160-verifier-head.npz"
    np.savez_compressed(vit_head, weight=vit_weight, bias=vit_bias)
    np.savez_compressed(
        output_dir / "selected-oof-logits.npz",
        logits_192=selected["oof_192"],
        logits_224=selected["oof_224"],
        logits_vit=vit_probe,
        labels=labels,
        folds=probe_folds,
    )
    public_candidates = []
    for row in candidate_results:
        public_candidates.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"oof_192", "oof_224", "reject_192", "reject_224"}
            }
        )
    report = {
        "schema_version": "1.0",
        "experiment": config["experiment"],
        "data_contract": {
            "training_source": str(source_dir),
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_count": len(records),
            "positive_roi_count": len(train_entries),
            "probe_count": len(probe_entries),
            "reject_count": len(reject_entries),
            "multi_object_used_for_model_or_threshold_selection": False,
            "physical_item_limit": "One physical-item lineage per class; held-out folds isolate source views only.",
        },
        "preprocessing": preprocessing,
        "candidates": public_candidates,
        "selected_candidate": selected["candidate"],
        "verifier_160_oof": _classification_metrics(vit_probe, labels),
        "selective_cascade_oof": cascade,
        "final_policy": policy,
        "final_training_history": final_history,
        "package": {
            "convnext_checkpoint": str(final_checkpoint.resolve()),
            "convnext_checkpoint_sha256": sha256_file(final_checkpoint),
            "vit_head": str(vit_head.resolve()),
            "vit_head_sha256": sha256_file(vit_head),
        },
    }
    _write_json(output_dir / "training-report.json", report)
    _write_json(
        package_dir / "metadata.json",
        {
            "schema_version": "1.0",
            "status": "EXPERIMENTAL_NOT_RUNTIME_ACTIVATED",
            "architecture": "fine-tuned DINOv3 ConvNeXt-Tiny 192/224 plus frozen DINOv3 ViT-B/16 160",
            "training_image_root": str(source_dir),
            "source_manifest_sha256": sha256_file(source_manifest),
            "preprocessing": preprocessing,
            "selected_training_candidate": selected["candidate"],
            "convnext_checkpoint": str(final_checkpoint.resolve()),
            "convnext_checkpoint_sha256": sha256_file(final_checkpoint),
            "vit160_head": str(vit_head.resolve()),
            "vit160_head_sha256": sha256_file(vit_head),
            "calibrated_routing": policy,
            "runtime_activated": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the single_objects_2 classifier cascade")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--convnext-weights", type=Path, required=True)
    parser.add_argument("--vit-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--reuse-vit", action="store_true")
    args = parser.parse_args()
    report = run(
        config_path=args.config,
        dataset_root=args.dataset_root,
        source_manifest=args.source_manifest,
        generated_dir=args.generated_dir,
        convnext_weights=args.convnext_weights,
        vit_weights=args.vit_weights,
        output_dir=args.output_dir,
        cpu=args.cpu,
        reuse_vit=args.reuse_vit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
