from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ...training.data import read_manifest
from .dinov3_class_conditional_ranker_oof import (
    class_conditional_targets,
    pair_features,
)
from .proposal_ranker import proposal_iou_matrix


def _fractional_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return 1.0 - ranks / max(len(values) - 1, 1)


def relative_pair_features(
    base_features: np.ndarray,
    boxes: np.ndarray,
    image_ids: np.ndarray,
    candidates: np.ndarray,
    class_objectness: np.ndarray,
    class_iou: np.ndarray,
) -> np.ndarray:
    """Add image×SKU ranking and spatial-consensus features to every pair."""
    base = np.asarray(base_features, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    image_ids = np.asarray(image_ids, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    objectness = np.asarray(class_objectness, dtype=np.float32)
    predicted_iou = np.asarray(class_iou, dtype=np.float32)
    if base.shape[:2] != candidates.shape or objectness.shape != candidates.shape:
        raise ValueError("relative pair inputs are not aligned")
    if predicted_iou.shape != candidates.shape or len(boxes) != len(candidates):
        raise ValueError("relative class scores do not align with proposal boxes")

    relative_dimension = base.shape[2] * 2 + 29
    relative = np.zeros((*candidates.shape, relative_dimension), dtype=np.float32)
    flat_candidates = candidates.reshape(-1)
    flat_objectness = objectness.reshape(-1)
    flat_iou = predicted_iou.reshape(-1)
    flat_product = np.sqrt(np.clip(flat_objectness * flat_iou, 0.0, 1.0))
    pair_image_ids = np.repeat(image_ids[:, None], candidates.shape[1], axis=1).reshape(-1)
    proposal_indices = np.repeat(
        np.arange(len(boxes), dtype=np.int64)[:, None], candidates.shape[1], axis=1
    ).reshape(-1)
    flat_base = base.reshape(-1, base.shape[2])
    flat_relative = relative.reshape(-1, relative_dimension)
    class_count = int(candidates.max()) + 1
    group_ids = pair_image_ids * class_count + flat_candidates
    for group_id in np.unique(group_ids):
        pair_indices = np.flatnonzero(group_ids == group_id)
        group_base = flat_base[pair_indices]
        mean = group_base.mean(axis=0)
        std = group_base.std(axis=0).clip(min=1e-5)
        standardized = (group_base - mean) / std
        ranks = np.column_stack(
            [_fractional_rank(group_base[:, index]) for index in range(group_base.shape[1])]
        )
        group_boxes = boxes[proposal_indices[pair_indices]]
        overlap = proposal_iou_matrix(group_boxes, group_boxes)
        np.fill_diagonal(overlap, 0.0)
        score_columns = np.column_stack(
            (
                flat_objectness[pair_indices],
                flat_iou[pair_indices],
                flat_product[pair_indices],
            )
        )
        score_ranks = np.column_stack(
            [_fractional_rank(score_columns[:, index]) for index in range(3)]
        )
        consensus = []
        for threshold in (0.1, 0.3, 0.5, 0.7):
            neighbors = overlap >= threshold
            consensus.append(neighbors.mean(axis=1))
            for index in range(3):
                weights = np.exp(
                    np.clip(
                        (score_columns[:, index] - score_columns[:, index].max()) / 0.1,
                        -30.0,
                        0.0,
                    )
                )
                consensus.append((neighbors * weights[None]).sum(axis=1) / weights.sum())
        anchor_overlap = np.column_stack(
            [overlap[:, int(np.argmax(score_columns[:, index]))] for index in range(3)]
        )
        centers = (group_boxes[:, :2] + group_boxes[:, 2:]) * 0.5
        sizes = np.maximum(group_boxes[:, 2:] - group_boxes[:, :2], 1e-6)
        geometry_deviation = []
        product_weights = np.exp(
            np.clip((score_columns[:, 2] - score_columns[:, 2].max()) / 0.1, -30.0, 0.0)
        )
        product_weights /= product_weights.sum()
        center = np.sum(centers * product_weights[:, None], axis=0)
        log_size = np.sum(np.log(sizes) * product_weights[:, None], axis=0)
        geometry_deviation.extend(((centers - center) / sizes).T)
        geometry_deviation.extend((np.log(sizes) - log_size).T)
        extra = np.column_stack(
            (
                score_columns,
                score_ranks,
                *consensus,
                anchor_overlap,
                *geometry_deviation,
            )
        )
        flat_relative[pair_indices] = np.column_stack((standardized, ranks, extra))
    return np.concatenate((base, relative), axis=2)


def best_pair_labels(group_ids: np.ndarray, target_iou: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(group_ids), dtype=bool)
    for group_id in np.unique(group_ids):
        indices = np.flatnonzero(group_ids == group_id)
        best = indices[int(np.argmax(target_iou[indices]))]
        labels[best] = target_iou[best] >= 0.5
    return labels


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    records = {
        int(row["image_id"]): row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and not row.get("exclude_from_detector_training", False)
    }
    with np.load(args.dense_features, allow_pickle=False) as payload:
        geometry = payload["geometry"].astype(np.float32)
        image_ids = payload["image_id"].copy()
        folds = payload["fold"].copy()
        boxes = payload["boxes_xyxy"].astype(np.float32)
    with np.load(args.catalog_logits, allow_pickle=False) as payload:
        catalog = {name: payload[name].copy() for name in payload.files}
    with np.load(args.proposal_ranker_scores, allow_pickle=False) as payload:
        proposal_ranker = {name: payload[name].copy() for name in payload.files}
    with np.load(args.class_conditional_scores, allow_pickle=False) as payload:
        conditional = {name: payload[name].copy() for name in payload.files}
    for name, payload in (
        ("Catalog", catalog),
        ("proposal ranker", proposal_ranker),
        ("class conditional", conditional),
    ):
        if not np.array_equal(payload["image_id"], image_ids):
            raise ValueError(f"{name} cache does not align with dense features")

    candidates = conditional["candidate_classes"].astype(np.int64)
    target_iou = np.zeros(candidates.shape, dtype=np.float32)
    for image_id in np.unique(image_ids):
        selected = image_ids == image_id
        target_iou[selected] = class_conditional_targets(
            boxes[selected], candidates[selected], records[int(image_id)]["annotations"]
        )
    proposal_scores = np.column_stack(
        [
            proposal_ranker[name]
            for name in (
                "dense_objectness",
                "compact_objectness",
                "compact_assignment",
                "compact_iou",
            )
        ]
    ).astype(np.float32)
    base_features = pair_features(
        geometry,
        proposal_scores,
        catalog["logits"],
        catalog["retrieval_logits"],
        candidates,
        catalog["approval_scores"],
        catalog["approval_blocked"],
        catalog["segment_recapture"],
    )
    features = relative_pair_features(
        base_features,
        boxes,
        image_ids,
        candidates,
        conditional["class_objectness"],
        conditional["class_iou"],
    )
    flat_features = features.reshape(-1, features.shape[-1])
    flat_target = target_iou.reshape(-1)
    pair_folds = np.repeat(folds[:, None], candidates.shape[1], axis=1).reshape(-1)
    pair_image_ids = np.repeat(image_ids[:, None], candidates.shape[1], axis=1).reshape(-1)
    group_ids = pair_image_ids * catalog["logits"].shape[1] + candidates.reshape(-1)
    best_labels = best_pair_labels(group_ids, flat_target)
    positive_labels = flat_target >= 0.5
    best_score = np.zeros(len(flat_target), dtype=np.float32)
    positive_score = np.zeros(len(flat_target), dtype=np.float32)
    iou_score = np.zeros(len(flat_target), dtype=np.float32)
    diagnostics = []
    for held_fold in sorted(np.unique(folds)):
        training = pair_folds != held_fold
        held = pair_folds == held_fold
        outputs = []
        for coordinate, labels in enumerate((best_labels, positive_labels)):
            positive_weight = (training.sum() - labels[training].sum()) / max(
                int(labels[training].sum()), 1
            )
            model = HistGradientBoostingClassifier(
                learning_rate=args.learning_rate,
                max_iter=args.maximum_iterations,
                max_leaf_nodes=args.maximum_leaf_nodes,
                min_samples_leaf=args.minimum_leaf,
                l2_regularization=args.l2_regularization,
                random_state=args.seed + 10 * int(held_fold) + coordinate,
            )
            model.fit(
                flat_features[training],
                labels[training],
                sample_weight=np.where(labels[training], positive_weight, 1.0),
            )
            outputs.append(model.predict_proba(flat_features[held])[:, 1])
        best_score[held], positive_score[held] = outputs
        regressor = HistGradientBoostingRegressor(
            learning_rate=args.learning_rate,
            max_iter=args.maximum_iterations,
            max_leaf_nodes=args.maximum_leaf_nodes,
            min_samples_leaf=args.minimum_leaf,
            l2_regularization=args.l2_regularization,
            random_state=args.seed + 100 + int(held_fold),
        )
        regressor.fit(
            flat_features[training],
            flat_target[training],
            sample_weight=1.0 + 8.0 * flat_target[training],
        )
        iou_score[held] = np.clip(regressor.predict(flat_features[held]), 0.0, 1.0)
        diagnostics.append(
            {
                "held_out_fold": int(held_fold),
                "training_pair_count": int(training.sum()),
                "held_out_pair_count": int(held.sum()),
            }
        )
        print(json.dumps(diagnostics[-1]), flush=True)

    output_shape = candidates.shape
    scores = {
        "class_objectness": best_score.reshape(output_shape),
        "class_positive": positive_score.reshape(output_shape),
        "class_iou": iou_score.reshape(output_shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        candidate_classes=candidates.astype(np.int8),
        **{name: value.astype(np.float16) for name, value in scores.items()},
        image_id=image_ids,
        fold=folds,
    )
    eligible = 0
    selected_success = {name: 0 for name in scores}
    for group_id in np.unique(group_ids):
        indices = np.flatnonzero(group_ids == group_id)
        if flat_target[indices].max() < 0.5:
            continue
        eligible += 1
        for name, values in scores.items():
            flat_values = values.reshape(-1)
            selected_success[name] += int(
                flat_target[indices[int(np.argmax(flat_values[indices]))]] >= 0.5
            )
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_relative_class_ranker_true_group_oof",
        "selection_scope": "each held fold is ranked by models trained on other folds",
        "feature_dimension": int(features.shape[-1]),
        "eligible_image_class_group_count": eligible,
        "selection": {
            name: {"count": count, "rate": count / eligible}
            for name, count in selected_success.items()
        },
        "folds": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
        "existing_detector_used": False,
        "yolo_family_used": False,
        "rfdetr_used": False,
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit an image-SKU-relative DINOv3 proposal ranker"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dense-features", type=Path, required=True)
    parser.add_argument("--catalog-logits", type=Path, required=True)
    parser.add_argument("--proposal-ranker-scores", type=Path, required=True)
    parser.add_argument("--class-conditional-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--maximum-iterations", type=int, default=220)
    parser.add_argument("--maximum-leaf-nodes", type=int, default=31)
    parser.add_argument("--minimum-leaf", type=int, default=30)
    parser.add_argument("--l2-regularization", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260831)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
