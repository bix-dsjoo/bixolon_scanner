from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import SGDClassifier

from .proposal_ranker import proposal_intersection_matrix, proposal_iou_matrix


def classifier_score_features(
    logits: np.ndarray,
    retrieval_logits: np.ndarray,
    approval_scores: np.ndarray,
    approval_blocked: np.ndarray,
    segment_recapture: np.ndarray,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    retrieval = np.asarray(retrieval_logits, dtype=np.float32)
    if logits.shape != retrieval.shape or logits.ndim != 2:
        raise ValueError("Catalog score arrays must be aligned matrices")
    order = np.argsort(-logits, axis=1, kind="stable")
    retrieval_order = np.argsort(-retrieval, axis=1, kind="stable")
    sorted_logits = np.take_along_axis(logits, order, axis=1)
    sorted_retrieval = np.take_along_axis(retrieval, retrieval_order, axis=1)
    top_k = min(5, logits.shape[1])
    logit_norm = np.linalg.norm(logits, axis=1).clip(min=1e-12)
    retrieval_norm = np.linalg.norm(retrieval, axis=1).clip(min=1e-12)
    return np.column_stack(
        (
            sorted_logits[:, :top_k],
            sorted_retrieval[:, :top_k],
            (sorted_logits[:, 0] - sorted_logits[:, 1]) / logit_norm,
            (sorted_retrieval[:, 0] - sorted_retrieval[:, 1]) / retrieval_norm,
            logit_norm,
            retrieval_norm,
            order[:, 0] == retrieval_order[:, 0],
            approval_scores,
            approval_blocked,
            segment_recapture,
        )
    ).astype(np.float32)


def proposal_context_features(
    boxes: np.ndarray,
    detector_scores: np.ndarray,
    predicted_classes: np.ndarray,
) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(detector_scores, dtype=np.float32)
    classes = np.asarray(predicted_classes, dtype=np.int64)
    overlap = proposal_iou_matrix(boxes, boxes)
    intersection = proposal_intersection_matrix(boxes, boxes)
    areas = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    np.fill_diagonal(overlap, 0.0)
    np.fill_diagonal(intersection, 0.0)
    higher = scores[None, :] > scores[:, None]
    same_class = classes[:, None] == classes[None, :]
    containment = np.divide(
        intersection,
        areas[None, :],
        out=np.zeros_like(intersection),
        where=areas[None, :] > 0.0,
    )
    return np.column_stack(
        (
            overlap.max(axis=1),
            np.where(higher, overlap, 0.0).max(axis=1),
            (overlap >= 0.3).sum(axis=1),
            (overlap >= 0.5).sum(axis=1),
            (overlap >= 0.7).sum(axis=1),
            np.where(same_class, overlap, 0.0).max(axis=1),
            (same_class & (overlap >= 0.3)).sum(axis=1),
            (same_class & (overlap >= 0.5)).sum(axis=1),
            containment.max(axis=1),
            (containment >= 0.8).sum(axis=1),
        )
    ).astype(np.float32)


def _aligned(cache: dict[str, np.ndarray], reference_ids: np.ndarray, name: str) -> None:
    if not np.array_equal(cache["image_id"], reference_ids):
        raise ValueError(f"{name} does not align with dense proposal features")


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    with np.load(args.dense_features, allow_pickle=False) as payload:
        dense = payload["features"].copy()
        geometry = payload["geometry"].astype(np.float32)
        max_iou = payload["max_iou"].astype(np.float32)
        image_ids = payload["image_id"].copy()
        folds = payload["fold"].copy()
        boxes = payload["boxes_xyxy"].astype(np.float32)
        detector_scores = payload["detector_score"].astype(np.float32)
    with np.load(args.catalog_logits, allow_pickle=False) as payload:
        catalog = {name: payload[name].copy() for name in payload.files}
    _aligned(catalog, image_ids, "Catalog logits")

    logits = catalog["logits"].astype(np.float32)
    retrieval = catalog["retrieval_logits"].astype(np.float32)
    predicted_classes = np.argmax(logits, axis=1)
    classifier_features = classifier_score_features(
        logits,
        retrieval,
        catalog["approval_scores"],
        catalog["approval_blocked"],
        catalog["segment_recapture"],
    )
    context = np.empty((len(image_ids), 10), dtype=np.float32)
    for image_id in np.unique(image_ids):
        selected = image_ids == image_id
        context[selected] = proposal_context_features(
            boxes[selected], detector_scores[selected], predicted_classes[selected]
        )
    compact = np.column_stack((geometry, classifier_features, context, logits, retrieval)).astype(
        np.float32
    )
    positive = max_iou >= 0.5
    dense_objectness = np.zeros(len(image_ids), dtype=np.float32)
    compact_objectness = np.zeros(len(image_ids), dtype=np.float32)
    compact_assignment = np.zeros(len(image_ids), dtype=np.float32)
    compact_iou = np.zeros(len(image_ids), dtype=np.float32)
    assignment = np.zeros(len(image_ids), dtype=np.bool_)
    for image_id in np.unique(image_ids):
        image_rows = np.flatnonzero(image_ids == image_id)
        for target_class in np.unique(catalog["target_index"][image_rows]):
            candidates = image_rows[catalog["target_index"][image_rows] == target_class]
            best = candidates[int(np.argmax(max_iou[candidates]))]
            if max_iou[best] >= 0.5:
                assignment[best] = True
    diagnostics = []
    for held_fold in sorted(np.unique(folds)):
        training = folds != held_fold
        held = folds == held_fold
        dense_model = SGDClassifier(
            loss="log_loss",
            alpha=args.dense_alpha,
            class_weight="balanced",
            average=True,
            max_iter=500,
            tol=1e-4,
            random_state=args.seed + int(held_fold),
        )
        dense_model.fit(
            dense[training].astype(np.float32) * np.float32(args.dense_scale),
            positive[training],
        )
        dense_objectness[held] = dense_model.predict_proba(
            dense[held].astype(np.float32) * np.float32(args.dense_scale)
        )[:, 1]

        objectness_model = ExtraTreesClassifier(
            n_estimators=args.estimators,
            min_samples_leaf=args.minimum_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed + 100 + int(held_fold),
        )
        objectness_model.fit(compact[training], positive[training])
        compact_objectness[held] = objectness_model.predict_proba(compact[held])[:, 1]

        assignment_model = ExtraTreesClassifier(
            n_estimators=args.estimators,
            min_samples_leaf=args.minimum_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed + 150 + int(held_fold),
        )
        assignment_model.fit(compact[training], assignment[training])
        compact_assignment[held] = assignment_model.predict_proba(compact[held])[:, 1]

        iou_model = ExtraTreesRegressor(
            n_estimators=args.estimators,
            min_samples_leaf=args.minimum_leaf,
            max_features=args.max_features,
            n_jobs=-1,
            random_state=args.seed + 200 + int(held_fold),
        )
        iou_model.fit(compact[training], max_iou[training])
        compact_iou[held] = np.clip(iou_model.predict(compact[held]), 0.0, 1.0)
        diagnostics.append(
            {
                "held_out_fold": int(held_fold),
                "training_proposal_count": int(training.sum()),
                "held_out_proposal_count": int(held.sum()),
                "training_positive_count": int(positive[training].sum()),
                "held_out_positive_count": int(positive[held].sum()),
                "training_assignment_count": int(assignment[training].sum()),
                "held_out_assignment_count": int(assignment[held].sum()),
            }
        )
        print(json.dumps(diagnostics[-1]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        dense_objectness=dense_objectness.astype(np.float16),
        compact_objectness=compact_objectness.astype(np.float16),
        compact_assignment=compact_assignment.astype(np.float16),
        compact_iou=compact_iou.astype(np.float16),
        image_id=image_ids,
        fold=folds,
    )
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_proposal_ranker_true_group_oof",
        "selection_scope": "each proposal score is predicted by models trained on other folds",
        "proposal_count": len(image_ids),
        "positive_iou_0_5_count": int(positive.sum()),
        "assignment_positive_count": int(assignment.sum()),
        "compact_feature_dimension": int(compact.shape[1]),
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
    parser = argparse.ArgumentParser(description="Cross-fit DINOv3 proposal ranking scores")
    parser.add_argument("--dense-features", type=Path, required=True)
    parser.add_argument("--catalog-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--estimators", type=int, default=256)
    parser.add_argument("--minimum-leaf", type=int, default=2)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--dense-alpha", type=float, default=0.0001)
    parser.add_argument("--dense-scale", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=20260831)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
