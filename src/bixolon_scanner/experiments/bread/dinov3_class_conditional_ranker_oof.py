from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from ...training.data import read_manifest
from .proposal_ranker import proposal_iou_matrix


def candidate_classes(logits: np.ndarray, *, top_k: int) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 2 or not 1 <= top_k <= values.shape[1]:
        raise ValueError("invalid Catalog logits or candidate count")
    return np.argsort(-values, axis=1, kind="stable")[:, :top_k]


def class_conditional_targets(
    boxes: np.ndarray,
    candidates: np.ndarray,
    annotations: list[dict],
) -> np.ndarray:
    targets = np.zeros(candidates.shape, dtype=np.float32)
    class_boxes = {
        int(row["category_id"]) - 1: np.asarray(
            [
                row["bbox_xywh"][0],
                row["bbox_xywh"][1],
                row["bbox_xywh"][0] + row["bbox_xywh"][2],
                row["bbox_xywh"][1] + row["bbox_xywh"][3],
            ],
            dtype=np.float32,
        )
        for row in annotations
    }
    boxes = np.asarray(boxes, dtype=np.float32)
    for class_index, target_box in class_boxes.items():
        matching = candidates == class_index
        if not np.any(matching):
            continue
        iou = proposal_iou_matrix(boxes, target_box[None])[:, 0]
        targets[matching] = np.broadcast_to(iou[:, None], candidates.shape)[matching]
    return targets


def pair_features(
    geometry: np.ndarray,
    proposal_scores: np.ndarray,
    logits: np.ndarray,
    retrieval_logits: np.ndarray,
    candidates: np.ndarray,
    approval_scores: np.ndarray,
    approval_blocked: np.ndarray,
    segment_recapture: np.ndarray,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    retrieval = np.asarray(retrieval_logits, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.shape[0] != logits.shape[0] or logits.shape != retrieval.shape:
        raise ValueError("class-conditional inputs are not aligned")
    rows = np.arange(len(logits))[:, None]
    adapter_norm = np.linalg.norm(logits, axis=1).clip(min=1e-12)
    retrieval_norm = np.linalg.norm(retrieval, axis=1).clip(min=1e-12)
    adapter_order = np.argsort(-logits, axis=1, kind="stable")
    retrieval_order = np.argsort(-retrieval, axis=1, kind="stable")
    adapter_ranks = np.empty_like(adapter_order)
    retrieval_ranks = np.empty_like(retrieval_order)
    np.put_along_axis(
        adapter_ranks,
        adapter_order,
        np.arange(logits.shape[1], dtype=np.int64)[None],
        axis=1,
    )
    np.put_along_axis(
        retrieval_ranks,
        retrieval_order,
        np.arange(retrieval.shape[1], dtype=np.int64)[None],
        axis=1,
    )
    adapter_sorted = np.take_along_axis(logits, adapter_order, axis=1)
    retrieval_sorted = np.take_along_axis(retrieval, retrieval_order, axis=1)
    adapter_value = logits[rows, candidates]
    retrieval_value = retrieval[rows, candidates]
    adapter_rank = adapter_ranks[rows, candidates]
    retrieval_rank = retrieval_ranks[rows, candidates]
    top_k = candidates.shape[1]
    repeated = np.repeat(
        np.column_stack(
            (
                np.asarray(geometry, dtype=np.float32),
                np.asarray(proposal_scores, dtype=np.float32),
                adapter_sorted[:, :3],
                retrieval_sorted[:, :3],
                (adapter_sorted[:, 0] - adapter_sorted[:, 1]) / adapter_norm,
                (retrieval_sorted[:, 0] - retrieval_sorted[:, 1]) / retrieval_norm,
                adapter_norm,
                retrieval_norm,
                np.asarray(approval_scores, dtype=np.float32),
                np.asarray(approval_blocked, dtype=np.float32),
                np.asarray(segment_recapture, dtype=np.float32),
            )
        )[:, None, :],
        top_k,
        axis=1,
    )
    conditional = np.stack(
        (
            adapter_value,
            retrieval_value,
            adapter_value / adapter_norm[:, None],
            retrieval_value / retrieval_norm[:, None],
            1.0 / (adapter_rank + 1.0),
            1.0 / (retrieval_rank + 1.0),
            adapter_sorted[:, 0, None] - adapter_value,
            retrieval_sorted[:, 0, None] - retrieval_value,
            adapter_order[:, 0, None] == candidates,
            retrieval_order[:, 0, None] == candidates,
        ),
        axis=2,
    ).astype(np.float32)
    return np.concatenate((repeated, conditional), axis=2)


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
    with np.load(args.ranker_scores, allow_pickle=False) as payload:
        ranker = {name: payload[name].copy() for name in payload.files}
    for name, payload in (("Catalog", catalog), ("proposal ranker", ranker)):
        if not np.array_equal(payload["image_id"], image_ids):
            raise ValueError(f"{name} cache does not align with dense features")

    candidates = candidate_classes(catalog["logits"], top_k=args.top_k)
    target_iou = np.zeros(candidates.shape, dtype=np.float32)
    for image_id in np.unique(image_ids):
        selected = image_ids == image_id
        target_iou[selected] = class_conditional_targets(
            boxes[selected], candidates[selected], records[int(image_id)]["annotations"]
        )
    proposal_scores = np.column_stack(
        [
            ranker[name]
            for name in (
                "dense_objectness",
                "compact_objectness",
                "compact_assignment",
                "compact_iou",
            )
        ]
    ).astype(np.float32)
    features = pair_features(
        geometry,
        proposal_scores,
        catalog["logits"],
        catalog["retrieval_logits"],
        candidates,
        catalog["approval_scores"],
        catalog["approval_blocked"],
        catalog["segment_recapture"],
    )
    pair_folds = np.repeat(folds[:, None], args.top_k, axis=1).reshape(-1)
    flat_features = features.reshape(-1, features.shape[-1])
    flat_target = target_iou.reshape(-1)
    positive = flat_target >= 0.5
    objectness = np.zeros(len(flat_target), dtype=np.float32)
    predicted_iou = np.zeros(len(flat_target), dtype=np.float32)
    diagnostics = []
    for held_fold in sorted(np.unique(folds)):
        training = pair_folds != held_fold
        held = pair_folds == held_fold
        positive_weight = float(training.sum() - positive[training].sum()) / max(
            int(positive[training].sum()), 1
        )
        sample_weight = np.where(positive[training], positive_weight, 1.0)
        classifier = HistGradientBoostingClassifier(
            learning_rate=args.learning_rate,
            max_iter=args.maximum_iterations,
            max_leaf_nodes=args.maximum_leaf_nodes,
            min_samples_leaf=args.minimum_leaf,
            l2_regularization=args.l2_regularization,
            random_state=args.seed + int(held_fold),
        )
        classifier.fit(
            flat_features[training],
            positive[training],
            sample_weight=sample_weight,
        )
        objectness[held] = classifier.predict_proba(flat_features[held])[:, 1]
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
        predicted_iou[held] = np.clip(regressor.predict(flat_features[held]), 0.0, 1.0)
        diagnostics.append(
            {
                "held_out_fold": int(held_fold),
                "training_pair_count": int(training.sum()),
                "held_out_pair_count": int(held.sum()),
                "training_positive_count": int(positive[training].sum()),
                "held_out_positive_count": int(positive[held].sum()),
                "held_out_roc_auc": float(roc_auc_score(positive[held], objectness[held])),
            }
        )
        print(json.dumps(diagnostics[-1]), flush=True)

    objectness = objectness.reshape(candidates.shape)
    predicted_iou = predicted_iou.reshape(candidates.shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        candidate_classes=candidates.astype(np.int8),
        class_objectness=objectness.astype(np.float16),
        class_iou=predicted_iou.astype(np.float16),
        image_id=image_ids,
        fold=folds,
    )

    covered_ground_truth_count = 0
    candidate_covered_ground_truth_count = 0
    for image_id, record in records.items():
        selected = image_ids == image_id
        image_boxes = boxes[selected]
        image_candidates = candidates[selected]
        for annotation in record["annotations"]:
            class_index = int(annotation["category_id"]) - 1
            x, y, width, height = annotation["bbox_xywh"]
            target_box = np.asarray([[x, y, x + width, y + height]], dtype=np.float32)
            iou = proposal_iou_matrix(image_boxes, target_box)[:, 0]
            covered_ground_truth_count += int(np.any(iou >= 0.5))
            candidate_covered_ground_truth_count += int(
                np.any((iou[:, None] >= 0.5) & (image_candidates == class_index))
            )
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_class_conditional_proposal_ranker_true_group_oof",
        "selection_scope": "each class-proposal score is predicted by models trained on other folds",
        "proposal_count": len(image_ids),
        "candidate_pair_count": int(candidates.size),
        "candidate_top_k": args.top_k,
        "feature_dimension": int(features.shape[-1]),
        "positive_pair_count": int(positive.sum()),
        "stage1_covered_ground_truth_count": covered_ground_truth_count,
        "class_candidate_covered_ground_truth_count": candidate_covered_ground_truth_count,
        "class_candidate_recall_over_stage1_covered": (
            candidate_covered_ground_truth_count / covered_ground_truth_count
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "folds": diagnostics,
        "existing_detector_used": False,
        "yolo_family_used": False,
        "rfdetr_used": False,
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit class-conditional DINOv3 Catalog proposal rankers"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dense-features", type=Path, required=True)
    parser.add_argument("--catalog-logits", type=Path, required=True)
    parser.add_argument("--ranker-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--maximum-iterations", type=int, default=200)
    parser.add_argument("--maximum-leaf-nodes", type=int, default=31)
    parser.add_argument("--minimum-leaf", type=int, default=20)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260831)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
