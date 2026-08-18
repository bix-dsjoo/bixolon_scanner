from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ...evaluation.detector import _iou


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fused_score(
    members: list[dict[str, Any]],
    *,
    model_weights: np.ndarray,
    mode: str,
) -> float:
    weighted_scores = [
        float(member["score"]) * float(model_weights[int(member["source_id"])])
        for member in members
    ]
    if mode == "maximum":
        return max(float(member["score"]) for member in members)
    if mode == "mean_all":
        return float(sum(weighted_scores) / model_weights.sum())
    if mode == "mean_present":
        present = {int(member["source_id"]) for member in members}
        return float(sum(weighted_scores) / model_weights[list(present)].sum())
    if mode == "support_adjusted_maximum":
        present_weight = sum(
            float(model_weights[source_id])
            for source_id in {int(member["source_id"]) for member in members}
        )
        return float(
            max(float(member["score"]) for member in members) * present_weight / model_weights.sum()
        )
    raise ValueError(f"unsupported ensemble score mode: {mode}")


def _fused_box(members: list[dict[str, Any]], model_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(
        [
            max(float(member["score"]), 1e-9) * float(model_weights[int(member["source_id"])])
            for member in members
        ],
        dtype=np.float64,
    )
    boxes = np.asarray([member["box"] for member in members], dtype=np.float64)
    return np.average(boxes, axis=0, weights=weights).astype(np.float32)


def fuse_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    model_weights: list[float],
    score_thresholds: list[float],
    pre_nms_iou_threshold: float = 1.0,
    max_candidates_per_model: int = 300,
    cluster_iou_threshold: float,
    score_mode: str,
    class_agnostic_output: bool = False,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one detector prediction row is required")
    if len(model_weights) != len(rows) or len(score_thresholds) != len(rows):
        raise ValueError("model weights and score thresholds must match model count")
    if any(weight <= 0.0 for weight in model_weights):
        raise ValueError("model weights must be positive")
    if not 0.0 <= cluster_iou_threshold <= 1.0:
        raise ValueError("cluster IoU threshold must be in [0, 1]")
    if not 0.0 <= pre_nms_iou_threshold <= 1.0:
        raise ValueError("pre-NMS IoU threshold must be in [0, 1]")
    if max_candidates_per_model < 1:
        raise ValueError("maximum candidates per model must be positive")
    image_id = rows[0]["image_id"]
    if any(row["image_id"] != image_id for row in rows[1:]):
        raise ValueError("detector ensemble image ids differ")

    candidates = []
    for source_id, (row, threshold) in enumerate(zip(rows, score_thresholds)):
        boxes = row["boxes_xyxy"]
        scores = row["scores"]
        classes = row["class_ids"]
        if not (len(boxes) == len(scores) == len(classes)):
            raise ValueError("detector prediction arrays are not aligned")
        source_candidates = [
            {
                "box": np.asarray(box, dtype=np.float32),
                "score": float(score),
                "class_id": int(class_id),
                "source_id": source_id,
            }
            for box, score, class_id in zip(boxes, scores, classes)
            if float(score) >= threshold
        ]
        source_candidates.sort(key=lambda item: (-item["score"], *item["box"].tolist()))
        selected = []
        for candidate in source_candidates:
            if all(
                _iou(candidate["box"], current["box"]) <= pre_nms_iou_threshold
                for current in selected
            ):
                selected.append(candidate)
                if len(selected) == max_candidates_per_model:
                    break
        candidates.extend(selected)
    weights = np.asarray(model_weights, dtype=np.float64)
    candidates.sort(
        key=lambda item: (
            -item["score"] * float(weights[item["source_id"]]),
            item["source_id"],
            *item["box"].tolist(),
        )
    )

    clusters: list[list[dict[str, Any]]] = []
    fused_boxes: list[np.ndarray] = []
    for candidate in candidates:
        best_index = None
        best_iou = -1.0
        for index, (cluster, fused_box) in enumerate(zip(clusters, fused_boxes)):
            if any(member["source_id"] == candidate["source_id"] for member in cluster):
                continue
            overlap = _iou(candidate["box"], fused_box)
            if overlap >= cluster_iou_threshold and overlap > best_iou:
                best_index = index
                best_iou = overlap
        if best_index is None:
            clusters.append([candidate])
            fused_boxes.append(candidate["box"].copy())
        else:
            clusters[best_index].append(candidate)
            fused_boxes[best_index] = _fused_box(clusters[best_index], weights)

    outputs = []
    for cluster, box in zip(clusters, fused_boxes):
        class_scores: Counter[int] = Counter()
        for member in cluster:
            class_scores[int(member["class_id"])] += float(member["score"]) * float(
                weights[int(member["source_id"])]
            )
        primary = max(
            cluster,
            key=lambda member: (
                member["score"] * float(weights[int(member["source_id"])]),
                -int(member["source_id"]),
            ),
        )
        source_ids = sorted({int(member["source_id"]) for member in cluster})
        outputs.append(
            {
                "box": box.tolist(),
                "score": _fused_score(cluster, model_weights=weights, mode=score_mode),
                "class_id": (
                    0
                    if class_agnostic_output
                    else int(
                        max(class_scores, key=lambda class_id: (class_scores[class_id], -class_id))
                    )
                ),
                "source_id": int(primary["source_id"]),
                "source_mask": sum(1 << source_id for source_id in source_ids),
                "support_count": len(source_ids),
                "member_scores": [
                    max(
                        (
                            float(member["score"])
                            for member in cluster
                            if int(member["source_id"]) == source_id
                        ),
                        default=0.0,
                    )
                    for source_id in range(len(rows))
                ],
            }
        )
    outputs.sort(
        key=lambda item: (
            -item["score"],
            -item["support_count"],
            item["source_mask"],
            *item["box"],
        )
    )
    return {
        "image_id": image_id,
        "boxes_xyxy": [item["box"] for item in outputs],
        "scores": [item["score"] for item in outputs],
        "class_ids": [item["class_id"] for item in outputs],
        "source_ids": [item["source_id"] for item in outputs],
        "source_masks": [item["source_mask"] for item in outputs],
        "support_counts": [item["support_count"] for item in outputs],
        "member_scores": [item["member_scores"] for item in outputs],
    }


def fuse_prediction_sets(
    prediction_sets: list[list[dict[str, Any]]],
    *,
    model_weights: list[float],
    score_thresholds: list[float],
    pre_nms_iou_threshold: float = 1.0,
    max_candidates_per_model: int = 300,
    cluster_iou_threshold: float,
    score_mode: str,
    class_agnostic_output: bool = False,
) -> list[dict[str, Any]]:
    if not prediction_sets:
        raise ValueError("at least one detector prediction set is required")
    image_count = len(prediction_sets[0])
    if any(len(rows) != image_count for rows in prediction_sets[1:]):
        raise ValueError("detector ensemble image counts differ")
    return [
        fuse_prediction_rows(
            [prediction_set[index] for prediction_set in prediction_sets],
            model_weights=model_weights,
            score_thresholds=score_thresholds,
            pre_nms_iou_threshold=pre_nms_iou_threshold,
            max_candidates_per_model=max_candidates_per_model,
            cluster_iou_threshold=cluster_iou_threshold,
            score_mode=score_mode,
            class_agnostic_output=class_agnostic_output,
        )
        for index in range(image_count)
    ]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    prediction_sets = [_read_predictions(path) for path in args.predictions]
    model_count = len(prediction_sets)
    model_weights = args.model_weights or [1.0] * model_count
    score_thresholds = args.score_thresholds
    if len(score_thresholds) == 1:
        score_thresholds = score_thresholds * model_count
    fused = fuse_prediction_sets(
        prediction_sets,
        model_weights=model_weights,
        score_thresholds=score_thresholds,
        pre_nms_iou_threshold=args.pre_nms_iou_threshold,
        max_candidates_per_model=args.max_candidates_per_model,
        cluster_iou_threshold=args.cluster_iou_threshold,
        score_mode=args.score_mode,
        class_agnostic_output=args.class_agnostic_output,
    )
    support_histogram: Counter[int] = Counter(
        support for row in fused for support in row["support_counts"]
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_deployable_detector_model_ensemble",
        "selection_scope": "prediction-only fusion without labels or image identifiers",
        "prediction_files": [path.name for path in args.predictions],
        "prediction_sha256": {path.name: _sha256(path) for path in args.predictions},
        "model_weights": model_weights,
        "score_thresholds": score_thresholds,
        "pre_nms_iou_threshold": args.pre_nms_iou_threshold,
        "max_candidates_per_model": args.max_candidates_per_model,
        "cluster_iou_threshold": args.cluster_iou_threshold,
        "score_mode": args.score_mode,
        "class_agnostic_output": args.class_agnostic_output,
        "image_count": len(fused),
        "candidate_count": sum(len(row["scores"]) for row in fused),
        "support_histogram": {
            str(support): count for support, count in sorted(support_histogram.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in fused),
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse deployable detector model outputs without label-dependent routing"
    )
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--model-weights", type=float, nargs="+")
    parser.add_argument("--score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--pre-nms-iou-threshold", type=float, default=1.0)
    parser.add_argument("--max-candidates-per-model", type=int, default=300)
    parser.add_argument("--cluster-iou-threshold", type=float, required=True)
    parser.add_argument(
        "--score-mode",
        choices=["maximum", "mean_all", "mean_present", "support_adjusted_maximum"],
        required=True,
    )
    parser.add_argument(
        "--class-agnostic-output",
        action="store_true",
        help="Collapse fused localization candidates to one class before downstream NMS.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
