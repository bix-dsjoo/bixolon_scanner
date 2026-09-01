from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ...training.data import read_manifest
from .dinov3_class_conditional_ranker_oof import (
    class_conditional_targets,
    pair_features,
)


def box_regression_targets(
    boxes: np.ndarray,
    candidates: np.ndarray,
    annotations: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Return FCOS-style class-conditional offsets and valid target mask."""
    boxes = np.asarray(boxes, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.shape[0] != len(boxes):
        raise ValueError("candidate classes do not align with proposal boxes")
    box_width = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6)
    box_height = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6)
    box_center_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
    box_center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
    offsets = np.zeros((*candidates.shape, 4), dtype=np.float32)
    valid = np.zeros(candidates.shape, dtype=bool)
    for annotation in annotations:
        class_index = int(annotation["category_id"]) - 1
        matching = candidates == class_index
        if not np.any(matching):
            continue
        x, y, width, height = (float(value) for value in annotation["bbox_xywh"])
        target = np.column_stack(
            (
                ((x + width * 0.5) - box_center_x) / box_width,
                ((y + height * 0.5) - box_center_y) / box_height,
                np.log(np.maximum(width, 1e-6) / box_width),
                np.log(np.maximum(height, 1e-6) / box_height),
            )
        )
        offsets[matching] = np.broadcast_to(target[:, None, :], offsets.shape)[matching]
        valid[matching] = True
    return offsets, valid


def apply_box_offsets(boxes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    if offsets.shape[:-1] != (len(boxes),) or offsets.shape[-1] != 4:
        raise ValueError("box offsets must align with proposal boxes")
    width = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6)
    height = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6)
    center_x = (boxes[:, 0] + boxes[:, 2]) * 0.5 + offsets[:, 0] * width
    center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5 + offsets[:, 1] * height
    refined_width = width * np.exp(np.clip(offsets[:, 2], -1.5, 1.5))
    refined_height = height * np.exp(np.clip(offsets[:, 3], -1.5, 1.5))
    return np.column_stack(
        (
            center_x - refined_width * 0.5,
            center_y - refined_height * 0.5,
            center_x + refined_width * 0.5,
            center_y + refined_height * 0.5,
        )
    ).astype(np.float32)


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
    source_iou = np.zeros(candidates.shape, dtype=np.float32)
    target_offsets = np.zeros((*candidates.shape, 4), dtype=np.float32)
    valid = np.zeros(candidates.shape, dtype=bool)
    for image_id in np.unique(image_ids):
        selected = image_ids == image_id
        annotations = records[int(image_id)]["annotations"]
        source_iou[selected] = class_conditional_targets(
            boxes[selected], candidates[selected], annotations
        )
        target_offsets[selected], valid[selected] = box_regression_targets(
            boxes[selected], candidates[selected], annotations
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
    flat_features = features.reshape(-1, features.shape[-1])
    flat_offsets = target_offsets.reshape(-1, 4)
    flat_iou = source_iou.reshape(-1)
    flat_valid = valid.reshape(-1)
    pair_folds = np.repeat(folds[:, None], candidates.shape[1], axis=1).reshape(-1)
    predicted_offsets = np.zeros_like(flat_offsets)
    diagnostics = []
    for held_fold in sorted(np.unique(folds)):
        training = (pair_folds != held_fold) & flat_valid & (flat_iou >= args.minimum_training_iou)
        held = pair_folds == held_fold
        for coordinate in range(4):
            regressor = HistGradientBoostingRegressor(
                learning_rate=args.learning_rate,
                max_iter=args.maximum_iterations,
                max_leaf_nodes=args.maximum_leaf_nodes,
                min_samples_leaf=args.minimum_leaf,
                l2_regularization=args.l2_regularization,
                loss="absolute_error",
                random_state=args.seed + 10 * int(held_fold) + coordinate,
            )
            regressor.fit(
                flat_features[training],
                flat_offsets[training, coordinate],
                sample_weight=0.25 + flat_iou[training],
            )
            predicted_offsets[held, coordinate] = regressor.predict(flat_features[held])
        diagnostics.append(
            {
                "held_out_fold": int(held_fold),
                "training_pair_count": int(training.sum()),
                "held_out_pair_count": int(held.sum()),
            }
        )
        print(json.dumps(diagnostics[-1]), flush=True)

    predicted_offsets = np.clip(
        predicted_offsets.reshape(*candidates.shape, 4), -args.maximum_offset, args.maximum_offset
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        candidate_classes=candidates.astype(np.int8),
        box_offsets=predicted_offsets.astype(np.float16),
        image_id=image_ids,
        fold=folds,
    )
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_class_conditional_box_refiner_true_group_oof",
        "selection_scope": "each held fold is refined by models trained on other folds",
        "proposal_count": len(image_ids),
        "candidate_pair_count": int(candidates.size),
        "training_pair_count": int(
            np.count_nonzero(flat_valid & (flat_iou >= args.minimum_training_iou))
        ),
        "feature_dimension": int(features.shape[-1]),
        "minimum_training_iou": args.minimum_training_iou,
        "maximum_offset": args.maximum_offset,
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
        description="Cross-fit class-conditional DINOv3 proposal box refinement"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dense-features", type=Path, required=True)
    parser.add_argument("--catalog-logits", type=Path, required=True)
    parser.add_argument("--proposal-ranker-scores", type=Path, required=True)
    parser.add_argument("--class-conditional-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-training-iou", type=float, default=0.1)
    parser.add_argument("--maximum-offset", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--maximum-iterations", type=int, default=240)
    parser.add_argument("--maximum-leaf-nodes", type=int, default=31)
    parser.add_argument("--minimum-leaf", type=int, default=20)
    parser.add_argument("--l2-regularization", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260831)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
