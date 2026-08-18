from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class GeometrySignal:
    maximum_overlap_fraction: float
    repulsion_x: float
    repulsion_y: float


@dataclass(frozen=True)
class GeometryRecipe:
    crop_scale: float
    overlap_trigger: float
    axis_threshold: float
    fallback_scale: float = 0.85

    @property
    def name(self) -> str:
        return (
            f"scale={self.crop_scale:.3f}:overlap={self.overlap_trigger:.3f}:"
            f"axis={self.axis_threshold:.3f}:fallback={self.fallback_scale:.3f}"
        )


def geometry_signal(boxes: Sequence[Sequence[float]], target_index: int) -> GeometrySignal:
    if not 0 <= target_index < len(boxes):
        raise ValueError("target detection index is outside the box list")
    target = np.asarray(boxes[target_index], dtype=np.float64)
    target_width = target[2] - target[0]
    target_height = target[3] - target[1]
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target detection box is empty")
    target_area = target_width * target_height
    target_center_x = (target[0] + target[2]) / 2.0
    target_center_y = (target[1] + target[3]) / 2.0
    maximum_overlap = 0.0
    repulsion_x = 0.0
    repulsion_y = 0.0
    for index, values in enumerate(boxes):
        if index == target_index:
            continue
        other = np.asarray(values, dtype=np.float64)
        intersection_width = max(0.0, min(target[2], other[2]) - max(target[0], other[0]))
        intersection_height = max(0.0, min(target[3], other[3]) - max(target[1], other[1]))
        overlap = intersection_width * intersection_height / target_area
        if overlap <= 0.0:
            continue
        maximum_overlap = max(maximum_overlap, overlap)
        other_center_x = (other[0] + other[2]) / 2.0
        other_center_y = (other[1] + other[3]) / 2.0
        repulsion_x += overlap * (target_center_x - other_center_x) / target_width
        repulsion_y += overlap * (target_center_y - other_center_y) / target_height
    return GeometrySignal(maximum_overlap, repulsion_x, repulsion_y)


def routed_view_name(signal: GeometrySignal, recipe: GeometryRecipe) -> str:
    if signal.maximum_overlap_fraction < recipe.overlap_trigger:
        return f"scale{recipe.fallback_scale:.3f}_x+0_y+0"

    def position(value: float) -> int:
        if abs(value) < recipe.axis_threshold:
            return 0
        return 1 if value > 0 else -1

    x_position = position(signal.repulsion_x)
    y_position = position(signal.repulsion_y)
    return f"scale{recipe.crop_scale:.3f}_x{x_position:+d}_y{y_position:+d}"


def _metrics(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-logits, axis=1, kind="stable")
    count = int(np.count_nonzero(mask))
    return {
        "sample_count": count,
        "top1_error_count": int(np.count_nonzero(mask & (order[:, 0] != targets))),
        "top3_miss_count": int(
            np.count_nonzero(mask & (~np.any(order[:, :3] == targets[:, None], axis=1)))
        ),
    }


def route_logits(
    spatial_logits: dict[str, np.ndarray],
    signals: Sequence[GeometrySignal],
    recipe: GeometryRecipe,
) -> tuple[np.ndarray, list[str]]:
    view_names = [routed_view_name(signal, recipe) for signal in signals]
    missing = sorted(set(view_names) - set(spatial_logits))
    if missing:
        raise ValueError(f"spatial logits are missing routed views: {missing}")
    routed = np.stack([spatial_logits[name][index] for index, name in enumerate(view_names)])
    return routed.astype(np.float32), view_names


def _select_recipe(
    *,
    spatial_logits: dict[str, np.ndarray],
    targets: np.ndarray,
    signals: Sequence[GeometrySignal],
    recipes: Sequence[GeometryRecipe],
    calibration: np.ndarray,
) -> tuple[GeometryRecipe, np.ndarray, list[str], dict[str, Any]]:
    selected = None
    for recipe in recipes:
        logits, view_names = route_logits(spatial_logits, signals, recipe)
        metrics = _metrics(logits, targets, calibration)
        shifted_count = sum(not name.endswith("_x+0_y+0") for name in view_names)
        key = (
            -metrics["top3_miss_count"],
            -metrics["top1_error_count"],
            -shifted_count,
            recipe.name,
        )
        if selected is None or key > selected[0]:
            selected = (key, recipe, logits, view_names, metrics)
    if selected is None:
        raise ValueError("no geometry routing candidates")
    return selected[1], selected[2], selected[3], selected[4]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    payload = np.load(args.spatial_logits)
    targets = payload["targets"].astype(np.int64)
    spatial_logits = {
        name: payload[name].astype(np.float32) for name in payload.files if name != "targets"
    }
    rows = [
        json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line
    ]
    predictions = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    if len(rows) != len(targets):
        raise ValueError("classifier records and spatial logits are not aligned")
    signals = [
        geometry_signal(
            predictions[int(row["image_id"])]["boxes_xyxy"], int(row["detection_index"])
        )
        for row in rows
    ]
    recipes = [
        GeometryRecipe(crop_scale, overlap_trigger, axis_threshold, args.fallback_scale)
        for crop_scale, overlap_trigger, axis_threshold in itertools.product(
            args.crop_scales,
            args.overlap_triggers,
            args.axis_thresholds,
        )
    ]
    full = np.ones(len(targets), dtype=bool)
    selected_recipe, routed, view_names, selected_metrics = _select_recipe(
        spatial_logits=spatial_logits,
        targets=targets,
        signals=signals,
        recipes=recipes,
        calibration=full,
    )
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    oof_logits = np.zeros_like(routed)
    fold_reports = []
    for held_out_fold in sorted(set(folds.tolist())):
        calibration = folds != held_out_fold
        held_out = folds == held_out_fold
        fold_recipe, fold_logits, fold_views, calibration_metrics = _select_recipe(
            spatial_logits=spatial_logits,
            targets=targets,
            signals=signals,
            recipes=recipes,
            calibration=calibration,
        )
        oof_logits[held_out] = fold_logits[held_out]
        fold_reports.append(
            {
                "held_out_fold": held_out_fold,
                "recipe": fold_recipe.name,
                "calibration_metrics": calibration_metrics,
                "held_out_metrics": _metrics(fold_logits, targets, held_out),
                "held_out_shifted_count": int(
                    sum(
                        bool(held_out[index]) and not name.endswith("_x+0_y+0")
                        for index, name in enumerate(fold_views)
                    )
                ),
            }
        )
    args.output_logits.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_logits, routed=routed, oof=oof_logits, targets=targets)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_detector_geometry_crop_routing",
        "selection_scope": "global_geometry_recipes_without_image_or_answer_branches",
        "candidate_count": len(recipes),
        "selected": {
            "recipe": selected_recipe.name,
            "metrics": selected_metrics,
            "shifted_count": sum(not name.endswith("_x+0_y+0") for name in view_names),
            "routed_view_counts": {
                name: view_names.count(name) for name in sorted(set(view_names))
            },
        },
        "grouped_oof": {
            "metrics": _metrics(oof_logits, targets, full),
            "folds": fold_reports,
        },
        "geometry_only_inputs": [
            "detector boxes",
            "detection index",
            "global crop recipe",
        ],
        "forbidden_inputs_absent": [
            "image id branches",
            "evaluation target",
            "class-specific routing",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route classifier crops away from overlapping boxes"
    )
    parser.add_argument("--spatial-logits", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-scales", type=float, nargs="+", default=(0.65, 0.75, 0.85))
    parser.add_argument(
        "--overlap-triggers", type=float, nargs="+", default=(0.05, 0.10, 0.15, 0.20, 0.25)
    )
    parser.add_argument("--axis-thresholds", type=float, nargs="+", default=(0.0, 0.03, 0.06, 0.10))
    parser.add_argument("--fallback-scale", type=float, default=0.85)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
