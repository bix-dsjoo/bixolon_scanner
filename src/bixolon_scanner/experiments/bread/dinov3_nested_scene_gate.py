from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from ...training.data import read_manifest
from .proposal_ranker import proposal_iou_matrix


def zero_error_threshold(scores: np.ndarray, safe: np.ndarray) -> tuple[float, int]:
    """Return a threshold strictly above every observed unsafe score."""
    scores = np.asarray(scores, dtype=np.float64)
    safe = np.asarray(safe, dtype=bool)
    if scores.shape != safe.shape or scores.ndim != 1:
        raise ValueError("risk scores and safety labels must be aligned vectors")
    unsafe_scores = scores[~safe]
    threshold = (
        float(np.nextafter(unsafe_scores.max(), np.inf)) if len(unsafe_scores) else float(-np.inf)
    )
    accepted = scores >= threshold
    return threshold, int(accepted.sum())


def _summary(values: list[float] | np.ndarray, *, default: float = 0.0) -> list[float]:
    array = np.asarray(values, dtype=np.float32)
    if not len(array):
        return [default] * 7
    return [
        float(array.min()),
        float(np.quantile(array, 0.25)),
        float(np.median(array)),
        float(array.mean()),
        float(np.quantile(array, 0.75)),
        float(array.max()),
        float(array.std()),
    ]


def scene_features(
    scene: dict,
    record: dict,
    patch_count: dict,
    cls_count: dict,
) -> np.ndarray:
    """Build inference-only scene features; no GT assignment field is read."""
    selectable = bool(scene.get("selectable", False))
    predicted_count = int(patch_count["predicted_count"])
    base = [
        float(selectable),
        float(predicted_count),
        float(patch_count["confidence"]),
        float(cls_count["confidence"]),
        float(patch_count["predicted_count"] == cls_count["predicted_count"]),
        float(abs(patch_count["predicted_count"] - cls_count["predicted_count"])),
        float(scene.get("minimum_ranker_safety", 0.0)),
        float(scene.get("minimum_identity_margin", -1.0)),
        float(scene.get("minimum_identity_stability", 0.0)),
        float(scene.get("maximum_residual_ratio", 10.0)),
        float(scene.get("all_heads_agree", False)),
    ]
    ranker = _summary(scene.get("ranker_safety", []))
    margin = _summary(scene.get("identity_margin", []), default=-1.0)
    stability = _summary(scene.get("identity_stability", []))
    agreement = _summary([float(value) for value in scene.get("head_agreement", [])])
    segment_extra_trees = _summary(scene.get("segment_safety_extra_trees", []))
    segment_hist_gradient = _summary(scene.get("segment_safety_hist_gradient", []))
    boxes = np.asarray(scene.get("selected_boxes_xyxy", []), dtype=np.float32).reshape(-1, 4)
    if len(boxes):
        scale = np.asarray(
            [record["width"], record["height"], record["width"], record["height"]],
            dtype=np.float32,
        )
        normalized = boxes / scale
        sizes = np.maximum(normalized[:, 2:] - normalized[:, :2], 0.0)
        areas = sizes[:, 0] * sizes[:, 1]
        aspects = np.log(np.maximum(sizes[:, 0], 1e-6) / np.maximum(sizes[:, 1], 1e-6))
        edges = np.minimum.reduce(
            [
                normalized[:, 0],
                normalized[:, 1],
                1.0 - normalized[:, 2],
                1.0 - normalized[:, 3],
            ]
        )
        overlap = proposal_iou_matrix(boxes, boxes)
        pair_overlap = overlap[np.triu_indices(len(boxes), 1)]
        geometry = [*_summary(areas), *_summary(aspects), *_summary(edges), *_summary(pair_overlap)]
    else:
        geometry = [0.0] * 28
    return np.asarray(
        [
            *base,
            *ranker,
            *margin,
            *stability,
            *agreement,
            *segment_extra_trees,
            *segment_hist_gradient,
            *geometry,
        ],
        dtype=np.float32,
    )


def _model_factories(seed: int) -> dict[str, Callable[[], object]]:
    return {
        "extra_trees_leaf_2": lambda: ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.75,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees_leaf_5": lambda: ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            max_features=0.75,
            class_weight="balanced",
            random_state=seed + 1,
            n_jobs=-1,
        ),
        "hist_gradient": lambda: HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=0.5,
            class_weight="balanced",
            random_state=seed + 2,
        ),
    }


def _positive_score(model: object, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if not len(positive):
        return np.zeros(len(features), dtype=np.float32)
    return probabilities[:, int(positive[0])].astype(np.float32)


def run(args: argparse.Namespace) -> dict:
    records = {
        int(row["image_id"]): row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and not row.get("exclude_from_detector_training", False)
    }
    scenes = [
        json.loads(line) for line in args.scenes.read_text(encoding="utf-8").splitlines() if line
    ]
    patch_counts = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.patch_count_oof.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    cls_counts = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.cls_count_oof.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    image_ids = np.asarray([int(scene["image_id"]) for scene in scenes], dtype=np.int64)
    folds = np.asarray([int(scene["fold"]) for scene in scenes], dtype=np.int8)
    features = np.stack(
        [
            scene_features(scene, records[image_id], patch_counts[image_id], cls_counts[image_id])
            for image_id, scene in zip(image_ids, scenes, strict=True)
        ]
    )
    safe = np.asarray(
        [
            bool(scene.get("selectable", False))
            and bool(scene.get("spatial_exact", False))
            and bool(scene.get("top3_all_correct", False))
            for scene in scenes
        ],
        dtype=bool,
    )
    factories = _model_factories(args.seed)
    nested_scores = np.zeros(len(scenes), dtype=np.float32)
    accepted = np.zeros(len(scenes), dtype=bool)
    diagnostics = []
    for outer_fold in sorted(np.unique(folds)):
        outer_training = folds != outer_fold
        outer_held = folds == outer_fold
        inner_folds = sorted(np.unique(folds[outer_training]))
        candidates = []
        for model_name, factory in factories.items():
            inner_scores = np.zeros(int(outer_training.sum()), dtype=np.float32)
            inner_features = features[outer_training]
            inner_labels = safe[outer_training]
            inner_fold_values = folds[outer_training]
            for inner_fold in inner_folds:
                fit = inner_fold_values != inner_fold
                validate = inner_fold_values == inner_fold
                model = factory()
                model.fit(inner_features[fit], inner_labels[fit])
                inner_scores[validate] = _positive_score(model, inner_features[validate])
            threshold, calibration_accepted = zero_error_threshold(inner_scores, inner_labels)
            candidates.append((calibration_accepted, model_name, threshold))
        calibration_accepted, model_name, threshold = max(candidates)
        model = factories[model_name]()
        model.fit(features[outer_training], safe[outer_training])
        scores = _positive_score(model, features[outer_held])
        nested_scores[outer_held] = scores
        accepted[outer_held] = scores >= threshold
        diagnostics.append(
            {
                "held_out_fold": int(outer_fold),
                "selected_model": model_name,
                "calibration_threshold": threshold,
                "calibration_accepted_count": calibration_accepted,
                "calibration_error_count": 0,
                "held_out_accepted_count": int(accepted[outer_held].sum()),
                "held_out_error_count": int(
                    np.count_nonzero(accepted[outer_held] & ~safe[outer_held])
                ),
            }
        )

    output_rows = []
    for scene, score, is_accepted in zip(scenes, nested_scores, accepted, strict=True):
        output_rows.append(
            {
                **scene,
                "nested_scene_safety_score": float(score),
                "nested_accepted": bool(is_accepted),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
        newline="\n",
    )
    accepted_safe = accepted & safe
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_nested_scene_safety_gate",
        "selection_scope": (
            "for each held fold, model family and zero-error threshold are selected only "
            "from cross-predictions of the other two folds"
        ),
        "image_count": len(scenes),
        "upstream_safe_scene_count": int(safe.sum()),
        "feature_dimension": int(features.shape[1]),
        "accepted": {
            "image_count": int(accepted.sum()),
            "coverage_over_all_images": float(accepted.mean()),
            "safe_image_count": int(accepted_safe.sum()),
            "product_miss_or_false_positive_image_count": int(np.count_nonzero(accepted & ~safe)),
            "top3_candidate_out_image_count": int(
                sum(
                    is_accepted and not scene.get("top3_all_correct", False)
                    for scene, is_accepted in zip(scenes, accepted, strict=True)
                )
            ),
            "image_ids": image_ids[accepted].tolist(),
        },
        "folds": diagnostics,
        "all_recapture_solution": not bool(accepted.sum()),
        "gate_selection_uses_held_fold_labels": False,
        "upstream_policy_development_selected": True,
        "independent_test": False,
        "activation_allowed": False,
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a nested DINOv3 scene safety gate")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--patch-count-oof", type=Path, required=True)
    parser.add_argument("--cls-count-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
