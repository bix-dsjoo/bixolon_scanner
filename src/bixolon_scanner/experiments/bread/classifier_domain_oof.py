from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...training.fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from ...training.models import require_torch
from .classifier_geometry_mask import apply_background_mask, neighbor_ownership_mask


def unique_class_predictions(scores: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    """Assign at most one ROI to each class using prediction scores only."""
    from scipy.optimize import linear_sum_assignment

    values = np.asarray(scores)
    groups = np.asarray(image_ids)
    if values.ndim != 2 or len(values) != len(groups):
        raise ValueError("scores and image ids are not aligned")
    predictions = np.empty(len(values), dtype=np.int64)
    for image_id in np.unique(groups):
        indices = np.flatnonzero(groups == image_id)
        if len(indices) > values.shape[1]:
            raise ValueError("unique-class assignment has more ROIs than classes")
        rows, columns = linear_sum_assignment(-values[indices])
        predictions[indices[rows]] = columns
    return predictions


def classification_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    image_ids: np.ndarray,
    *,
    unique_classes: bool,
) -> dict[str, Any]:
    values = np.asarray(scores)
    labels = np.asarray(targets, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(labels):
        raise ValueError("scores and targets are not aligned")
    predictions = (
        unique_class_predictions(values, image_ids) if unique_classes else np.argmax(values, axis=1)
    )
    order = np.argsort(-values, axis=1, kind="stable")
    return {
        "sample_count": len(labels),
        "top1_error_count": int(np.count_nonzero(predictions != labels)),
        "top1_accuracy": float(np.mean(predictions == labels)),
        "top3_miss_count": int(np.count_nonzero(~np.any(order[:, :3] == labels[:, None], axis=1))),
        "top3_accuracy": float(np.mean(np.any(order[:, :3] == labels[:, None], axis=1))),
    }


def assert_group_fold_isolation(rows: Sequence[dict[str, Any]]) -> None:
    group_folds: dict[str, set[int]] = {}
    for row in rows:
        group_folds.setdefault(str(row["group_id"]), set()).add(int(row["fold"]))
    overlap = sorted(group for group, folds in group_folds.items() if len(folds) != 1)
    if overlap:
        raise ValueError(f"group-aware fold overlap: {overlap[:3]}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def extract(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_ten_shot_classifier(
        backbone_kind=str(checkpoint["backbone_kind"]),
        weights_path=None,
        hub_repository=args.hub_repository,
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    device = torch.device("cpu" if args.cpu else "cuda")
    model = model.to(device).eval()

    tensors = np.load(args.evaluation_tensors, mmap_mode="r")
    rows = _read_jsonl(args.evaluation_records)
    assert_group_fold_isolation(rows)
    predictions = {int(row["image_id"]): row for row in _read_jsonl(args.predictions)}
    manifest = {int(row["image_id"]): row for row in _read_jsonl(args.manifest)}
    if len(tensors) != len(rows):
        raise ValueError("evaluation tensors and records are not aligned")

    raw_features = []
    adapted_features = []
    base_logits = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = np.array(tensors[start : start + args.batch_size], dtype=np.float32, copy=True)
            batch_rows = rows[start : start + len(batch)]
            masks = np.stack(
                [
                    neighbor_ownership_mask(
                        image_width=int(manifest[int(row["image_id"])]["width"]),
                        image_height=int(manifest[int(row["image_id"])]["height"]),
                        boxes=row.get(
                            "mask_boxes_xyxy",
                            predictions[int(row["image_id"])]["boxes_xyxy"],
                        ),
                        target_index=int(row.get("mask_target_index", row["detection_index"])),
                        output_size=batch.shape[-1],
                        margin_ratio=args.margin_ratio,
                        distance_bias=args.distance_bias,
                        shared_scale=False,
                    )
                    for row in batch_rows
                ]
            )
            pixels = torch.from_numpy(apply_background_mask(batch, masks)).to(device)
            raw = model.extract_features(pixels)
            adapted = model.classifier.adapt(raw)
            raw_features.append(raw.float().cpu().numpy())
            adapted_features.append(adapted.float().cpu().numpy())
            base_logits.append(model.classifier(raw).float().cpu().numpy())

    payload = {
        "raw": np.concatenate(raw_features).astype(np.float32),
        "adapted": np.concatenate(adapted_features).astype(np.float32),
        "base_logits": np.concatenate(base_logits).astype(np.float32),
        "targets": np.asarray([int(row["target"]) for row in rows], dtype=np.int64),
        "folds": np.asarray([int(row["fold"]) for row in rows], dtype=np.int64),
        "image_ids": np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_same_domain_feature_extraction",
        "sample_count": len(rows),
        "feature_dimensions": {
            "raw": payload["raw"].shape[1],
            "adapted": payload["adapted"].shape[1],
        },
        "fold_counts": {
            str(fold): int(np.count_nonzero(payload["folds"] == fold)) for fold in (0, 1, 2)
        },
        "group_fold_overlap_count": 0,
        "geometry_only_mask": True,
        "evaluation_targets_used_for_feature_preprocessing": False,
        "output": str(args.output),
    }
    print(json.dumps(report, indent=2))
    return report


def _fit_scores(
    features: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    *,
    validation_fold: int,
    c_value: float,
) -> np.ndarray:
    from sklearn.svm import LinearSVC

    training = folds != validation_fold
    validation = folds == validation_fold
    model = LinearSVC(C=c_value, dual="auto", max_iter=20_000)
    model.fit(features[training], targets[training])
    return model.decision_function(features[validation])


def _fit_lda_scores(
    features: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    *,
    validation_fold: int,
    shrinkage: float,
) -> np.ndarray:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    training = folds != validation_fold
    validation = folds == validation_fold
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
    model.fit(features[training], targets[training])
    return model.decision_function(features[validation])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cache = np.load(args.features)
    targets = np.asarray(cache["targets"], dtype=np.int64)
    folds = np.asarray(cache["folds"], dtype=np.int64)
    image_ids = np.asarray(cache["image_ids"], dtype=np.int64)
    if set(np.unique(folds)) != {0, 1, 2}:
        raise ValueError("same-domain OOF requires folds 0, 1, and 2")

    candidates = []
    candidate_scores: dict[tuple[str, float], np.ndarray] = {}
    for family in args.feature_families:
        features = np.asarray(cache[family], dtype=np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-12)
        for c_value in args.c_values:
            scores = np.zeros((len(targets), int(targets.max()) + 1), dtype=np.float64)
            by_fold = {}
            for fold in (0, 1, 2):
                validation = folds == fold
                scores[validation] = _fit_scores(
                    features,
                    targets,
                    folds,
                    validation_fold=fold,
                    c_value=c_value,
                )
                by_fold[str(fold)] = classification_metrics(
                    scores[validation],
                    targets[validation],
                    image_ids[validation],
                    unique_classes=True,
                )
            independent = classification_metrics(scores, targets, image_ids, unique_classes=False)
            unique = classification_metrics(scores, targets, image_ids, unique_classes=True)
            candidate_scores[(family, float(c_value))] = scores
            candidates.append(
                {
                    "feature_family": family,
                    "c": float(c_value),
                    "independent": independent,
                    "unique_class_assignment": unique,
                    "by_fold_unique_class_assignment": by_fold,
                }
            )

    selected = min(
        candidates,
        key=lambda row: (
            row["unique_class_assignment"]["top1_error_count"],
            row["unique_class_assignment"]["top3_miss_count"],
            row["independent"]["top1_error_count"],
            row["c"],
            row["feature_family"],
        ),
    )
    selected_scores = candidate_scores[(selected["feature_family"], selected["c"])]
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.logits_output,
        scores=selected_scores.astype(np.float32),
        targets=targets,
        folds=folds,
        image_ids=image_ids,
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_same_domain_grouped_oof_probe",
        "selection_scope": "three_fold_grouped_development_oof_not_locked_test",
        "training_source": "accepted multi_object_scenes ROIs from the other two folds",
        "sample_count": len(targets),
        "candidate_count": len(candidates),
        "selected": selected,
        "candidates": candidates,
        "locked_test_accessed": False,
        "promotion_ready": False,
        "promotion_blocker": "detector, selective-risk policy, final model, and locked test pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def evaluate_lda(args: argparse.Namespace) -> dict[str, Any]:
    cache = np.load(args.features)
    targets = np.asarray(cache["targets"], dtype=np.int64)
    folds = np.asarray(cache["folds"], dtype=np.int64)
    image_ids = np.asarray(cache["image_ids"], dtype=np.int64)
    if set(np.unique(folds)) != {0, 1, 2}:
        raise ValueError("same-domain OOF requires folds 0, 1, and 2")

    candidates = []
    candidate_scores: dict[tuple[str, float], np.ndarray] = {}
    for family in args.feature_families:
        features = np.asarray(cache[family], dtype=np.float64)
        features /= np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-12)
        for shrinkage in args.shrinkages:
            scores = np.zeros((len(targets), int(targets.max()) + 1), dtype=np.float64)
            by_fold = {}
            for fold in (0, 1, 2):
                validation = folds == fold
                scores[validation] = _fit_lda_scores(
                    features,
                    targets,
                    folds,
                    validation_fold=fold,
                    shrinkage=shrinkage,
                )
                by_fold[str(fold)] = classification_metrics(
                    scores[validation],
                    targets[validation],
                    image_ids[validation],
                    unique_classes=False,
                )
            independent = classification_metrics(scores, targets, image_ids, unique_classes=False)
            candidate_scores[(family, float(shrinkage))] = scores
            candidates.append(
                {
                    "feature_family": family,
                    "shrinkage": float(shrinkage),
                    "independent": independent,
                    "by_fold_independent": by_fold,
                }
            )

    selected = min(
        candidates,
        key=lambda row: (
            row["independent"]["top1_error_count"],
            row["independent"]["top3_miss_count"],
            row["shrinkage"],
            row["feature_family"],
        ),
    )
    selected_scores = candidate_scores[(selected["feature_family"], selected["shrinkage"])]
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.logits_output,
        scores=selected_scores.astype(np.float32),
        targets=targets,
        folds=folds,
        image_ids=image_ids,
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_same_domain_grouped_oof_lda_probe",
        "selection_scope": "three_fold_grouped_development_oof_not_locked_test",
        "training_source": "repaired detector ROIs from the other two folds",
        "sample_count": len(targets),
        "candidate_count": len(candidates),
        "selected": selected,
        "candidates": candidates,
        "unique_class_assignment_used": False,
        "locked_test_accessed": False,
        "promotion_ready": False,
        "promotion_blocker": "selective-risk policy, final model, parity, latency, and locked test pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def selective_margin_policy(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    maximum_approved_errors: int,
    maximum_unknown_top3_misses: int,
) -> dict[str, Any]:
    """Find the minimum-rejection class-conditional margin policy on development rows."""
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    predictions = np.argmax(values, axis=1)
    order = np.argsort(-values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    margins = sorted_values[:, 0] - sorted_values[:, 1]
    errors = np.flatnonzero(predictions != labels)
    top3_misses = ~np.any(order[:, :3] == labels[:, None], axis=1)
    candidates = []
    for leave_count in range(min(maximum_approved_errors, len(errors)) + 1):
        for approved_error_indices in itertools.combinations(errors.tolist(), leave_count):
            approved_errors = set(approved_error_indices)
            thresholds: list[float | None] = [None] * values.shape[1]
            for index in errors:
                if int(index) in approved_errors:
                    continue
                class_id = int(predictions[index])
                threshold = float(margins[index])
                current = thresholds[class_id]
                thresholds[class_id] = threshold if current is None else max(current, threshold)
            rejected = np.zeros(len(labels), dtype=bool)
            for class_id, threshold in enumerate(thresholds):
                if threshold is not None:
                    rejected |= (predictions == class_id) & (margins <= threshold)
            approved_wrong = int(np.count_nonzero((predictions != labels) & ~rejected))
            unknown_top3_misses = int(np.count_nonzero(top3_misses & rejected))
            if (
                approved_wrong <= maximum_approved_errors
                and unknown_top3_misses <= maximum_unknown_top3_misses
            ):
                candidates.append(
                    {
                        "thresholds": thresholds,
                        "rejected": rejected,
                        "approved_count": int(np.count_nonzero(~rejected)),
                        "approved_error_count": approved_wrong,
                        "unknown_count": int(np.count_nonzero(rejected)),
                        "unknown_top3_miss_count": unknown_top3_misses,
                    }
                )
    if not candidates:
        raise ValueError("no class-conditional margin policy satisfies the supplied error limits")
    return max(
        candidates,
        key=lambda row: (
            row["approved_count"],
            -row["approved_error_count"],
            -row["unknown_top3_miss_count"],
        ),
    )


def _apply_margin_thresholds(scores: np.ndarray, thresholds: Sequence[float | None]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-values, axis=1, kind="stable")
    predictions = order[:, 0]
    sorted_values = np.take_along_axis(values, order, axis=1)
    margins = sorted_values[:, 0] - sorted_values[:, 1]
    rejected = np.zeros(len(values), dtype=bool)
    for class_id, threshold in enumerate(thresholds):
        if threshold is not None:
            rejected |= (predictions == class_id) & (margins <= threshold)
    return rejected


def _policy_metrics(
    scores: np.ndarray, targets: np.ndarray, rejected: np.ndarray
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    predictions = np.argmax(values, axis=1)
    order = np.argsort(-values, axis=1, kind="stable")
    top3_misses = ~np.any(order[:, :3] == labels[:, None], axis=1)
    sample_count = len(labels)
    approved_count = int(np.count_nonzero(~rejected))
    approved_error_count = int(np.count_nonzero((predictions != labels) & ~rejected))
    unknown_count = int(np.count_nonzero(rejected))
    unknown_top3_miss_count = int(np.count_nonzero(top3_misses & rejected))
    return {
        "sample_count": sample_count,
        "approved_count": approved_count,
        "approved_rate": approved_count / sample_count,
        "approved_error_count": approved_error_count,
        "approved_error_rate_all_gt": approved_error_count / sample_count,
        "unknown_count": unknown_count,
        "unknown_rate": unknown_count / sample_count,
        "unknown_top3_miss_count": unknown_top3_miss_count,
        "unknown_top3_miss_rate_all_gt": unknown_top3_miss_count / sample_count,
    }


def evaluate_policy(args: argparse.Namespace) -> dict[str, Any]:
    cache = np.load(args.logits)
    scores = np.asarray(cache["scores"], dtype=np.float64)
    targets = np.asarray(cache["targets"], dtype=np.int64)
    folds = np.asarray(cache["folds"], dtype=np.int64)
    sample_count = len(targets)
    maximum_approved_errors = math.floor(args.maximum_approved_error_rate * sample_count)
    maximum_unknown_top3_misses = math.floor(args.maximum_unknown_top3_miss_rate * sample_count)
    policy = selective_margin_policy(
        scores,
        targets,
        maximum_approved_errors=maximum_approved_errors,
        maximum_unknown_top3_misses=maximum_unknown_top3_misses,
    )
    pooled_metrics = _policy_metrics(scores, targets, policy["rejected"])

    nested_rejected = np.zeros(sample_count, dtype=bool)
    nested_thresholds = {}
    nested_infeasible_folds = []
    for fold in (0, 1, 2):
        training = folds != fold
        held_out = folds == fold
        training_count = int(np.count_nonzero(training))
        try:
            fold_policy = selective_margin_policy(
                scores[training],
                targets[training],
                maximum_approved_errors=math.floor(
                    args.maximum_approved_error_rate * training_count
                ),
                maximum_unknown_top3_misses=math.floor(
                    args.maximum_unknown_top3_miss_rate * training_count
                ),
            )
        except ValueError:
            nested_infeasible_folds.append(fold)
            continue
        nested_rejected[held_out] = _apply_margin_thresholds(
            scores[held_out], fold_policy["thresholds"]
        )
        nested_thresholds[str(fold)] = fold_policy["thresholds"]
    nested_metrics = (
        None if nested_infeasible_folds else _policy_metrics(scores, targets, nested_rejected)
    )

    target_minimum = math.ceil(args.approved_rate_target * sample_count)
    operational_minimum = math.ceil(args.approved_rate_minimum * sample_count)
    gates = {
        "approved_99_target_met": pooled_metrics["approved_count"] >= target_minimum,
        "approved_operational_minimum_met": pooled_metrics["approved_count"] >= operational_minimum,
        "approved_error_gate_met": pooled_metrics["approved_error_count"]
        <= maximum_approved_errors,
        "unknown_top3_gate_met": pooled_metrics["unknown_top3_miss_count"]
        <= maximum_unknown_top3_misses,
    }
    serializable_policy = {key: value for key, value in policy.items() if key != "rejected"}
    output = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_same_domain_selective_margin_policy",
        "selection_scope": "pooled_grouped_development_oof_not_locked_test",
        "all_gt_denominator": sample_count,
        "policy": serializable_policy,
        "pooled_development_metrics": pooled_metrics,
        "nested_cross_calibration_diagnostic": {
            "metrics": nested_metrics,
            "thresholds_by_held_out_fold": nested_thresholds,
            "feasible": not nested_infeasible_folds,
            "infeasible_training_folds": nested_infeasible_folds,
            "selection_only_not_deployable": True,
        },
        "gates": gates,
        "all_classifier_gates_met": all(gates.values()),
        "locked_test_accessed": False,
        "promotion_ready": False,
        "promotion_blocker": "final model, detector integration, parity, latency, and locked test pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe same-domain grouped OOF classification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--checkpoint", type=Path, required=True)
    extract_parser.add_argument("--evaluation-tensors", type=Path, required=True)
    extract_parser.add_argument("--evaluation-records", type=Path, required=True)
    extract_parser.add_argument("--predictions", type=Path, required=True)
    extract_parser.add_argument("--manifest", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--margin-ratio", type=float, default=0.05)
    extract_parser.add_argument("--distance-bias", type=float, default=0.0)
    extract_parser.add_argument("--batch-size", type=int, default=96)
    extract_parser.add_argument(
        "--hub-repository",
        default="facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751",
    )
    extract_parser.add_argument("--cpu", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--features", type=Path, required=True)
    evaluate_parser.add_argument("--feature-families", nargs="+", default=("raw", "adapted"))
    evaluate_parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    )
    evaluate_parser.add_argument("--logits-output", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)

    lda_parser = subparsers.add_parser("evaluate-lda")
    lda_parser.add_argument("--features", type=Path, required=True)
    lda_parser.add_argument("--feature-families", nargs="+", default=("raw", "adapted"))
    lda_parser.add_argument(
        "--shrinkages",
        type=float,
        nargs="+",
        default=(0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1),
    )
    lda_parser.add_argument("--logits-output", type=Path, required=True)
    lda_parser.add_argument("--output", type=Path, required=True)

    policy_parser = subparsers.add_parser("policy")
    policy_parser.add_argument("--logits", type=Path, required=True)
    policy_parser.add_argument("--approved-rate-target", type=float, default=0.99)
    policy_parser.add_argument("--approved-rate-minimum", type=float, default=0.90)
    policy_parser.add_argument("--maximum-approved-error-rate", type=float, default=0.001)
    policy_parser.add_argument("--maximum-unknown-top3-miss-rate", type=float, default=0.001)
    policy_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "extract":
        extract(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "evaluate-lda":
        evaluate_lda(args)
    else:
        evaluate_policy(args)


if __name__ == "__main__":
    main()
