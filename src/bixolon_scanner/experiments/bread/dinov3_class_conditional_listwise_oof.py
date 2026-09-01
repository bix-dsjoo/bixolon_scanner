from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ...training.data import read_manifest
from .dinov3_class_conditional_ranker_oof import (
    class_conditional_targets,
    pair_features,
)


def best_positive_pair_indices(
    group_ids: np.ndarray,
    target_iou: np.ndarray,
    *,
    minimum_iou: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(group_ids, dtype=np.int64)
    targets = np.asarray(target_iou, dtype=np.float32)
    if groups.shape != targets.shape or groups.ndim != 1:
        raise ValueError("group ids and targets must be aligned vectors")
    best_indices = []
    best_groups = []
    for group_id in np.unique(groups):
        indices = np.flatnonzero(groups == group_id)
        best = indices[int(np.argmax(targets[indices]))]
        if targets[best] >= minimum_iou:
            best_indices.append(best)
            best_groups.append(int(group_id))
    return np.asarray(best_indices, dtype=np.int64), np.asarray(best_groups, dtype=np.int64)


def fixed_random_projection(
    features: np.ndarray,
    *,
    output_dimension: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or output_dimension <= 0:
        raise ValueError("random projection requires a feature matrix and positive dimension")
    generator = np.random.default_rng(seed)
    matrix = generator.normal(
        0.0,
        1.0 / np.sqrt(output_dimension),
        size=(values.shape[1], output_dimension),
    ).astype(np.float32)
    projected = values @ matrix
    return projected / np.linalg.norm(projected, axis=1, keepdims=True).clip(min=1e-12)


def _build_model(input_dimension: int):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(input_dimension, 96),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(96, 48),
        torch.nn.GELU(),
        torch.nn.Linear(48, 1),
    )


def _listwise_loss(scores, group_ids, best_indices, best_groups, group_count: int):
    import torch

    group_max = torch.full(
        (group_count,),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    group_max.scatter_reduce_(0, group_ids, scores, reduce="amax", include_self=True)
    stabilized = torch.exp(scores - group_max[group_ids])
    group_sum = torch.zeros((group_count,), dtype=scores.dtype, device=scores.device)
    group_sum.scatter_add_(0, group_ids, stabilized)
    log_denominator = group_max + torch.log(group_sum.clamp(min=1e-12))
    return (-scores[best_indices] + log_denominator[best_groups]).mean()


def run(args: argparse.Namespace) -> dict:
    import torch

    if args.projection_dimension < 0:
        raise ValueError("projection dimension cannot be negative")
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
        dense_content = (
            payload["features"].astype(np.float32) if args.projection_dimension > 0 else None
        )
        image_ids = payload["image_id"].copy()
        folds = payload["fold"].copy()
        boxes = payload["boxes_xyxy"].astype(np.float32)
    crop_content = None
    if args.projection_dimension > 0:
        if args.crop_features is None:
            raise ValueError("crop features are required when content projection is enabled")
        with np.load(args.crop_features, allow_pickle=False) as payload:
            crop_content = payload["features"].astype(np.float32)
            if not np.array_equal(payload["image_id"], image_ids):
                raise ValueError("crop feature cache does not align with dense features")
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
    if args.projection_dimension > 0:
        if dense_content is None or crop_content is None:
            raise ValueError("content projection inputs were not loaded")
        proposal_content = np.column_stack(
            (
                fixed_random_projection(
                    dense_content,
                    output_dimension=args.projection_dimension,
                    seed=args.seed,
                ),
                fixed_random_projection(
                    crop_content,
                    output_dimension=args.projection_dimension,
                    seed=args.seed + 1,
                ),
            )
        ).astype(np.float32)
        features = np.concatenate(
            (
                features,
                np.repeat(proposal_content[:, None, :], candidates.shape[1], axis=1),
            ),
            axis=2,
        )
    flat_features = features.reshape(-1, features.shape[-1])
    flat_targets = target_iou.reshape(-1)
    pair_folds = np.repeat(folds[:, None], candidates.shape[1], axis=1).reshape(-1)
    pair_image_ids = np.repeat(image_ids[:, None], candidates.shape[1], axis=1).reshape(-1)
    flat_candidates = candidates.reshape(-1)
    global_group_ids = pair_image_ids * catalog["logits"].shape[1] + flat_candidates
    output_scores = np.zeros(len(flat_features), dtype=np.float32)
    diagnostics = []
    device = torch.device(args.device)
    for held_fold in sorted(np.unique(folds)):
        torch.manual_seed(args.seed + int(held_fold))
        training = pair_folds != held_fold
        held = pair_folds == held_fold
        training_features = flat_features[training]
        mean = training_features.mean(axis=0, keepdims=True)
        std = training_features.std(axis=0, keepdims=True).clip(min=1e-5)
        train_tensor = torch.from_numpy((training_features - mean) / std).to(device)
        training_global_groups = global_group_ids[training]
        unique_groups, local_group_ids = np.unique(training_global_groups, return_inverse=True)
        del unique_groups
        best_indices, best_groups = best_positive_pair_indices(
            local_group_ids,
            flat_targets[training],
        )
        group_tensor = torch.from_numpy(local_group_ids).to(device)
        best_index_tensor = torch.from_numpy(best_indices).to(device)
        best_group_tensor = torch.from_numpy(best_groups).to(device)
        model = _build_model(flat_features.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        model.train()
        final_loss = None
        for epoch in range(args.epochs):
            optimizer.zero_grad(set_to_none=True)
            scores = model(train_tensor).squeeze(1)
            loss = _listwise_loss(
                scores,
                group_tensor,
                best_index_tensor,
                best_group_tensor,
                int(local_group_ids.max()) + 1,
            )
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if (epoch + 1) % 20 == 0:
                print(
                    json.dumps(
                        {
                            "held_out_fold": int(held_fold),
                            "epoch": epoch + 1,
                            "listwise_loss": final_loss,
                        }
                    ),
                    flush=True,
                )
        model.eval()
        with torch.inference_mode():
            for start in range(0, int(held.sum()), args.inference_batch_size):
                batch = flat_features[held][start : start + args.inference_batch_size]
                tensor = torch.from_numpy((batch - mean) / std).to(device)
                indices = np.flatnonzero(held)[start : start + args.inference_batch_size]
                output_scores[indices] = model(tensor).squeeze(1).cpu().numpy()
        diagnostics.append(
            {
                "held_out_fold": int(held_fold),
                "training_pair_count": int(training.sum()),
                "held_out_pair_count": int(held.sum()),
                "training_positive_group_count": len(best_indices),
                "final_training_listwise_loss": final_loss,
            }
        )

    output_scores = output_scores.reshape(candidates.shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        candidate_classes=candidates.astype(np.int8),
        class_rank_score=output_scores.astype(np.float16),
        image_id=image_ids,
        fold=folds,
    )

    eligible_groups = 0
    listwise_successes = 0
    regression_successes = 0
    for group_id in np.unique(global_group_ids):
        indices = np.flatnonzero(global_group_ids == group_id)
        if flat_targets[indices].max() < 0.5:
            continue
        eligible_groups += 1
        listwise_successes += int(
            flat_targets[indices[int(np.argmax(output_scores.reshape(-1)[indices]))]] >= 0.5
        )
        regression = conditional["class_iou"].reshape(-1)
        regression_successes += int(
            flat_targets[indices[int(np.argmax(regression[indices]))]] >= 0.5
        )
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_class_conditional_listwise_true_group_oof",
        "selection_scope": "each held fold is ranked by a model trained on other folds",
        "eligible_image_class_group_count": eligible_groups,
        "pair_feature_dimension": int(features.shape[-1]),
        "content_projection_dimension_per_source": args.projection_dimension,
        "listwise_iou_0_5_selection_count": listwise_successes,
        "listwise_iou_0_5_selection_rate": listwise_successes / eligible_groups,
        "regression_iou_0_5_selection_count": regression_successes,
        "regression_iou_0_5_selection_rate": regression_successes / eligible_groups,
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
        description="Cross-fit a listwise DINOv3 class-conditional proposal ranker"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dense-features", type=Path, required=True)
    parser.add_argument("--crop-features", type=Path)
    parser.add_argument("--catalog-logits", type=Path, required=True)
    parser.add_argument("--proposal-ranker-scores", type=Path, required=True)
    parser.add_argument("--class-conditional-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--inference-batch-size", type=int, default=65536)
    parser.add_argument(
        "--projection-dimension",
        type=int,
        default=0,
        help="Opt-in fixed projection per content source; zero keeps the stronger scalar baseline",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
