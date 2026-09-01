from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...training.data import read_manifest
from .proposal_ranker import proposal_iou_matrix


@dataclass(frozen=True)
class SelectionPolicy:
    score_kind: str
    suppression_mode: str
    nms_iou: float
    center_distance: float
    aggregation_iou: float
    class_temperature: float
    candidate_limit: int


def greedy_top_k_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    count: int,
    iou_threshold: float,
    classes: np.ndarray | None = None,
    suppression_mode: str = "iou",
    center_distance_threshold: float = 0.5,
) -> np.ndarray:
    if count < 0:
        raise ValueError("selected count must be non-negative")
    boxes = np.asarray(boxes, dtype=np.float32)
    order = np.argsort(-np.asarray(scores), kind="stable")
    overlap = proposal_iou_matrix(boxes, boxes)
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    areas = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    if suppression_mode not in {"iou", "center", "class_unique", "center_unique"}:
        raise ValueError(f"unsupported suppression mode: {suppression_mode}")
    unique_class = suppression_mode in {"class_unique", "center_unique"}
    center_aware = suppression_mode in {"center", "center_unique"}
    if unique_class and classes is None:
        raise ValueError("class-unique suppression requires proposal classes")
    selected: list[int] = []
    for index in order:
        duplicate = False
        for kept in selected:
            if unique_class and int(classes[int(index)]) == int(classes[kept]):
                duplicate = True
                break
            spatial_duplicate = overlap[int(index), kept] > iou_threshold
            if spatial_duplicate and center_aware:
                scale = np.sqrt(max(min(areas[int(index)], areas[kept]), 1e-12))
                distance = np.linalg.norm(centers[int(index)] - centers[kept]) / scale
                spatial_duplicate = distance <= center_distance_threshold
            if spatial_duplicate:
                duplicate = True
                break
        if not duplicate:
            selected.append(int(index))
            if len(selected) == count:
                break
    return np.asarray(selected, dtype=np.int64)


def class_consensus_top_k(
    boxes: np.ndarray,
    scores: np.ndarray,
    logits: np.ndarray,
    *,
    count: int,
    iou_threshold: float,
    temperature: float,
    candidate_limit: int,
) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("class consensus temperature must be positive")
    if candidate_limit < count:
        raise ValueError("class consensus candidate limit is below the requested count")
    scores = np.asarray(scores, dtype=np.float32)
    candidate_indices = np.argsort(-scores, kind="stable")[:candidate_limit]
    candidate_logits = np.asarray(logits, dtype=np.float32)[candidate_indices]
    shifted = candidate_logits / np.float32(temperature)
    shifted -= shifted.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    evidence = np.sum(probabilities * scores[candidate_indices, None], axis=0)
    overlap = proposal_iou_matrix(np.asarray(boxes, dtype=np.float32), boxes)
    selected: list[int] = []
    for class_index in np.argsort(-evidence, kind="stable"):
        class_scores = scores[candidate_indices] * probabilities[:, int(class_index)]
        for local_index in np.argsort(-class_scores, kind="stable"):
            proposal_index = int(candidate_indices[int(local_index)])
            if all(overlap[proposal_index, kept] <= iou_threshold for kept in selected):
                selected.append(proposal_index)
                break
        if len(selected) == count:
            break
    return np.asarray(selected, dtype=np.int64)


def aggregate_catalog_heads(
    boxes: np.ndarray,
    selected: np.ndarray,
    logits: np.ndarray,
    retrieval_logits: np.ndarray,
    weights: np.ndarray,
    *,
    minimum_iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(selected):
        class_count = int(np.asarray(logits).shape[1])
        return (
            np.empty((0, class_count), dtype=np.float32),
            np.empty((0, class_count), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    overlap = proposal_iou_matrix(np.asarray(boxes)[selected], boxes)
    ownership = np.argmax(overlap, axis=0)
    maximum = overlap[ownership, np.arange(overlap.shape[1])]
    normalized_logits = np.asarray(logits, dtype=np.float32)
    normalized_logits /= np.linalg.norm(normalized_logits, axis=1, keepdims=True).clip(min=1e-12)
    normalized_retrieval = np.asarray(retrieval_logits, dtype=np.float32)
    normalized_retrieval /= np.linalg.norm(normalized_retrieval, axis=1, keepdims=True).clip(
        min=1e-12
    )
    proposal_top1 = np.argmax(normalized_logits, axis=1)
    aggregated_logits = []
    aggregated_retrieval = []
    stability = []
    for selected_position, selected_index in enumerate(selected):
        members = (ownership == selected_position) & (maximum >= minimum_iou)
        members[int(selected_index)] = True
        member_weights = np.asarray(weights[members], dtype=np.float32).clip(min=1e-6)
        member_weights /= member_weights.sum()
        adapter = np.sum(normalized_logits[members] * member_weights[:, None], axis=0)
        retrieval = np.sum(normalized_retrieval[members] * member_weights[:, None], axis=0)
        top1 = int(np.argmax(adapter))
        aggregated_logits.append(adapter)
        aggregated_retrieval.append(retrieval)
        stability.append(float(member_weights[proposal_top1[members] == top1].sum()))
    return (
        np.stack(aggregated_logits),
        np.stack(aggregated_retrieval),
        np.asarray(stability, dtype=np.float32),
    )


def _jsonl_by_id(path: Path) -> dict[int, dict]:
    return {
        int(row["image_id"]): row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    }


def _score(scores: dict[str, np.ndarray], kind: str) -> np.ndarray:
    dense = scores["dense_objectness"]
    compact = scores["compact_objectness"]
    assignment = scores["compact_assignment"]
    iou = scores["compact_iou"]
    if kind == "dense":
        return dense
    if kind == "compact_objectness":
        return compact
    if kind == "compact_iou":
        return iou
    if kind == "compact_assignment":
        return assignment
    if kind == "mean":
        return (dense + compact + iou) / 3.0
    if kind == "compact_mean":
        return (compact + iou) / 2.0
    if kind == "assignment_iou":
        return np.sqrt(np.clip(assignment * iou, 0.0, 1.0))
    if kind == "product":
        return np.cbrt(np.clip(dense * compact * iou, 0.0, 1.0))
    if kind == "iou_objectness":
        return iou * (0.5 + 0.25 * dense + 0.25 * compact)
    raise ValueError(f"unsupported score kind: {kind}")


def _scene(
    record: dict,
    boxes: np.ndarray,
    ranker: dict[str, np.ndarray],
    logits: np.ndarray,
    retrieval: np.ndarray,
    *,
    count: int,
    policy: SelectionPolicy,
) -> dict:
    selection_score = _score(ranker, policy.score_kind)
    if policy.suppression_mode == "class_consensus":
        selected = class_consensus_top_k(
            boxes,
            selection_score,
            logits,
            count=count,
            iou_threshold=policy.nms_iou,
            temperature=policy.class_temperature,
            candidate_limit=policy.candidate_limit,
        )
    else:
        selected = greedy_top_k_nms(
            boxes,
            selection_score,
            count=count,
            iou_threshold=policy.nms_iou,
            classes=np.argmax(logits, axis=1),
            suppression_mode=policy.suppression_mode,
            center_distance_threshold=policy.center_distance,
        )
    if len(selected) != count:
        return {"selectable": False}
    aggregated, aggregated_retrieval, stability = aggregate_catalog_heads(
        boxes,
        selected,
        logits,
        retrieval,
        np.maximum(selection_score, 1e-6),
        minimum_iou=policy.aggregation_iou,
    )
    adapter_order = np.argsort(-aggregated, axis=1, kind="stable")
    retrieval_order = np.argsort(-aggregated_retrieval, axis=1, kind="stable")
    sorted_adapter = np.take_along_axis(aggregated, adapter_order, axis=1)
    identity_margin = (sorted_adapter[:, 0] - sorted_adapter[:, 1]) / np.linalg.norm(
        aggregated, axis=1
    ).clip(min=1e-12)

    targets = np.asarray(
        [
            [x, y, x + width, y + height]
            for x, y, width, height in (
                annotation["bbox_xywh"] for annotation in record["annotations"]
            )
        ],
        dtype=np.float32,
    )
    target_classes = np.asarray(
        [int(annotation["category_id"]) - 1 for annotation in record["annotations"]],
        dtype=np.int64,
    )
    overlap = proposal_iou_matrix(boxes[selected], targets)
    proposal_index, target_index = linear_sum_assignment(-overlap)
    assigned_iou = overlap[proposal_index, target_index]
    assigned_class = np.full(len(selected), -1, dtype=np.int64)
    assigned_class[proposal_index] = target_classes[target_index]
    spatial_exact = (
        len(selected) == len(targets)
        and len(proposal_index) == len(targets)
        and bool(np.all(assigned_iou >= 0.5))
    )
    top1_correct = adapter_order[:, 0] == assigned_class
    top3_correct = np.any(adapter_order[:, :3] == assigned_class[:, None], axis=1)
    head_agreement = adapter_order[:, 0] == retrieval_order[:, 0]
    selected_overlap = proposal_iou_matrix(boxes[selected], boxes)
    covered = selected_overlap.max(axis=0) > policy.nms_iou
    residual = selection_score[~covered]
    minimum_selected_score = float(selection_score[selected].min())
    residual_ratio = (
        0.0 if not len(residual) else float(residual.max() / max(minimum_selected_score, 1e-6))
    )
    ranker_safety = np.minimum.reduce(
        [
            ranker["dense_objectness"][selected],
            ranker["compact_objectness"][selected],
            ranker["compact_assignment"][selected],
            ranker["compact_iou"][selected],
        ]
    )
    return {
        "selectable": True,
        "selected_indices": selected.tolist(),
        "spatial_exact": spatial_exact,
        "top1_all_correct": bool(np.all(top1_correct)),
        "top3_all_correct": bool(spatial_exact and np.all(top3_correct)),
        "minimum_ranker_safety": float(ranker_safety.min()),
        "minimum_identity_margin": float(identity_margin.min()),
        "minimum_identity_stability": float(stability.min()),
        "all_heads_agree": bool(np.all(head_agreement)),
        "maximum_residual_ratio": residual_ratio,
        "assigned_iou": assigned_iou.tolist(),
        "assigned_class": assigned_class.tolist(),
        "top1": adapter_order[:, 0].tolist(),
        "top3": adapter_order[:, :3].tolist(),
        "identity_margin": identity_margin.tolist(),
        "identity_stability": stability.tolist(),
        "head_agreement": head_agreement.tolist(),
        "top1_correct": top1_correct.tolist(),
        "top3_correct": top3_correct.tolist(),
    }


def _accepts(scene: dict, gate: dict) -> bool:
    return bool(
        scene["selectable"]
        and scene["minimum_ranker_safety"] >= gate["minimum_ranker_safety"]
        and scene["minimum_identity_margin"] >= gate["minimum_identity_margin"]
        and scene["minimum_identity_stability"] >= gate["minimum_identity_stability"]
        and scene["maximum_residual_ratio"] <= gate["maximum_residual_ratio"]
        and (not gate["require_head_agreement"] or scene["all_heads_agree"])
    )


def _best_zero_error_gate(scenes: list[dict]) -> tuple[dict, list[dict]]:
    candidates = []
    for ranker, margin, stability, residual, agreement in product(
        [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [-1.0, 0.0, 0.005, 0.01, 0.02, 0.04, 0.08],
        [0.0, 0.5, 0.6, 0.7, 0.8, 0.9],
        [1.01, 0.98, 0.95, 0.9, 0.8, 0.7],
        [False, True],
    ):
        gate = {
            "minimum_ranker_safety": ranker,
            "minimum_identity_margin": margin,
            "minimum_identity_stability": stability,
            "maximum_residual_ratio": residual,
            "require_head_agreement": agreement,
        }
        accepted = [scene for scene in scenes if _accepts(scene, gate)]
        error_count = sum(
            not scene["spatial_exact"] or not scene["top3_all_correct"] for scene in accepted
        )
        candidates.append(
            {
                "gate": gate,
                "accepted_count": len(accepted),
                "error_count": error_count,
                "approved_safe_count": sum(scene["top1_all_correct"] for scene in accepted),
            }
        )
    zero_error = [row for row in candidates if row["error_count"] == 0 and row["accepted_count"]]
    selected = max(
        zero_error or candidates,
        key=lambda row: (
            row["accepted_count"] if row["error_count"] == 0 else -row["error_count"],
            row["approved_safe_count"],
            row["gate"]["minimum_ranker_safety"],
        ),
    )
    return selected, candidates


def run(args: argparse.Namespace) -> dict:
    records = {
        int(row["image_id"]): row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and not row.get("exclude_from_detector_training", False)
    }
    patch_counts = _jsonl_by_id(args.patch_count_oof)
    cls_counts = _jsonl_by_id(args.cls_count_oof)
    with np.load(args.dense_features, allow_pickle=False) as payload:
        image_ids = payload["image_id"].copy()
        boxes = payload["boxes_xyxy"].astype(np.float32)
    with np.load(args.catalog_logits, allow_pickle=False) as payload:
        catalog = {name: payload[name].copy() for name in payload.files}
    with np.load(args.ranker_scores, allow_pickle=False) as payload:
        ranker = {name: payload[name].astype(np.float32) for name in payload.files}
    for name, payload in (("Catalog", catalog), ("ranker", ranker)):
        if not np.array_equal(payload["image_id"], image_ids):
            raise ValueError(f"{name} cache does not align with dense features")

    count_safe_ids = []
    for image_id in sorted(records):
        patch = patch_counts[image_id]
        cls = cls_counts[image_id]
        if (
            patch["confidence"] >= args.patch_count_confidence
            and cls["confidence"] >= args.cls_count_confidence
            and patch["predicted_count"] == cls["predicted_count"]
        ):
            count_safe_ids.append(image_id)

    policy_results = []
    best_scenes: list[dict] | None = None
    best_policy: SelectionPolicy | None = None
    best_gate: dict | None = None
    best_key = None
    for (
        score_kind,
        suppression_mode,
        nms_iou,
        center_distance,
        aggregation_iou,
        class_temperature,
        candidate_limit,
    ) in product(
        args.score_kinds,
        args.suppression_modes,
        args.nms_ious,
        args.center_distances,
        args.aggregation_ious,
        args.class_temperatures,
        args.candidate_limits,
    ):
        if (
            suppression_mode in {"iou", "class_unique", "class_consensus"}
            and center_distance != args.center_distances[0]
        ):
            continue
        if suppression_mode != "class_consensus" and (
            class_temperature != args.class_temperatures[0]
            or candidate_limit != args.candidate_limits[0]
        ):
            continue
        policy = SelectionPolicy(
            score_kind,
            suppression_mode,
            nms_iou,
            center_distance,
            aggregation_iou,
            class_temperature,
            candidate_limit,
        )
        scenes = []
        for image_id in count_safe_ids:
            selected = image_ids == image_id
            scene = _scene(
                records[image_id],
                boxes[selected],
                {name: values[selected] for name, values in ranker.items() if name != "fold"},
                catalog["logits"][selected],
                catalog["retrieval_logits"][selected],
                count=int(patch_counts[image_id]["predicted_count"]),
                policy=policy,
            )
            scene.update(
                {
                    "image_id": image_id,
                    "fold": int(records[image_id]["fold"]),
                    "expected_count": len(records[image_id]["annotations"]),
                    "predicted_count": int(patch_counts[image_id]["predicted_count"]),
                    "patch_count_confidence": float(patch_counts[image_id]["confidence"]),
                    "cls_count_confidence": float(cls_counts[image_id]["confidence"]),
                }
            )
            scenes.append(scene)
        selected_gate, _ = _best_zero_error_gate(scenes)
        accepted = [scene for scene in scenes if _accepts(scene, selected_gate["gate"])]
        row = {
            "policy": {
                "score_kind": score_kind,
                "suppression_mode": suppression_mode,
                "nms_iou": nms_iou,
                "center_distance": center_distance,
                "aggregation_iou": aggregation_iou,
                "class_temperature": class_temperature,
                "candidate_limit": candidate_limit,
            },
            "gate": selected_gate["gate"],
            "count_safe_image_count": len(scenes),
            "pre_gate_spatial_exact_count": sum(
                scene.get("spatial_exact", False) for scene in scenes
            ),
            "pre_gate_top3_safe_count": sum(
                scene.get("top3_all_correct", False) for scene in scenes
            ),
            "accepted_image_count": len(accepted),
            "accepted_spatial_error_count": sum(not scene["spatial_exact"] for scene in accepted),
            "accepted_top3_candidate_out_image_count": sum(
                not scene["top3_all_correct"] for scene in accepted
            ),
            "accepted_top1_all_correct_image_count": sum(
                scene["top1_all_correct"] for scene in accepted
            ),
        }
        policy_results.append(row)
        key = (
            -row["accepted_spatial_error_count"] - row["accepted_top3_candidate_out_image_count"],
            row["accepted_image_count"],
            row["accepted_top1_all_correct_image_count"],
            row["pre_gate_top3_safe_count"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_scenes = scenes
            best_policy = policy
            best_gate = selected_gate["gate"]

    if best_scenes is None or best_policy is None or best_gate is None:
        raise ValueError("no E2E policy candidate was evaluated")
    accepted = [scene for scene in best_scenes if _accepts(scene, best_gate)]

    approved_threshold = None
    approved_segment_count = 0
    approved_error_count = 0
    for threshold in [0.2, 0.16, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.0]:
        approved = [
            (correct, margin)
            for scene in accepted
            for correct, margin, agree in zip(
                scene["top1_correct"],
                scene["identity_margin"],
                scene["head_agreement"],
                strict=True,
            )
            if agree and margin >= threshold
        ]
        errors = sum(not correct for correct, _ in approved)
        if approved and errors == 0:
            approved_threshold = threshold
            approved_segment_count = len(approved)
            approved_error_count = errors
            break

    unknown_segment_count = 0
    unknown_top3_candidate_out_count = 0
    for scene in accepted:
        for correct, top3_correct, margin, agree in zip(
            scene["top1_correct"],
            scene["top3_correct"],
            scene["identity_margin"],
            scene["head_agreement"],
            strict=True,
        ):
            approved = approved_threshold is not None and agree and margin >= approved_threshold
            if not approved:
                unknown_segment_count += 1
                unknown_top3_candidate_out_count += int(not top3_correct)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(
                {**scene, "accepted": _accepts(scene, best_gate)},
                ensure_ascii=False,
            )
            + "\n"
            for scene in best_scenes
        ),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_selective_e2e_development_oof",
        "selection_scope": (
            "proposal generation/ranking/count are group-aware OOF; policy and safety gate "
            "are selected on the same development OOF and require independent confirmation"
        ),
        "prohibited_detector_inputs": {
            "existing_detector_used": False,
            "yolo_family_used": False,
            "rfdetr_used": False,
        },
        "image_count": len(records),
        "count_gate": {
            "patch_confidence_minimum": args.patch_count_confidence,
            "cls_confidence_minimum": args.cls_count_confidence,
            "head_agreement_required": True,
            "passed_image_count": len(count_safe_ids),
        },
        "selected_policy": {
            "score_kind": best_policy.score_kind,
            "suppression_mode": best_policy.suppression_mode,
            "nms_iou": best_policy.nms_iou,
            "center_distance": best_policy.center_distance,
            "aggregation_iou": best_policy.aggregation_iou,
            "class_temperature": best_policy.class_temperature,
            "candidate_limit": best_policy.candidate_limit,
        },
        "selected_gate": best_gate,
        "accepted": {
            "image_count": len(accepted),
            "coverage_over_all_images": len(accepted) / len(records),
            "segmentation_count": sum(len(scene["selected_indices"]) for scene in accepted),
            "product_miss_image_count": sum(not scene["spatial_exact"] for scene in accepted),
            "false_positive_or_duplicate_image_count": sum(
                not scene["spatial_exact"] for scene in accepted
            ),
            "unknown_top3_candidate_out_image_count": sum(
                not scene["top3_all_correct"] for scene in accepted
            ),
            "top1_all_correct_image_count": sum(scene["top1_all_correct"] for scene in accepted),
            "image_ids": [scene["image_id"] for scene in accepted],
        },
        "approval": {
            "minimum_normalized_margin": approved_threshold,
            "retrieval_head_agreement_required": True,
            "approved_segment_count": approved_segment_count,
            "approved_wrong_count": approved_error_count,
        },
        "unknown": {
            "segment_count": unknown_segment_count,
            "top3_candidate_out_count": unknown_top3_candidate_out_count,
        },
        "fallback": {
            "status": "IMAGE_RECAPTURE",
            "image_count": len(records) - len(accepted),
            "all_recapture_solution": len(accepted) == 0,
        },
        "policy_candidate_count": len(policy_results),
        "independent_test": False,
        "activation_allowed": False,
        "scene_diagnostics_path": args.output.resolve().as_posix(),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate selective DINOv3 E2E OOF safety")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dense-features", type=Path, required=True)
    parser.add_argument("--catalog-logits", type=Path, required=True)
    parser.add_argument("--ranker-scores", type=Path, required=True)
    parser.add_argument("--patch-count-oof", type=Path, required=True)
    parser.add_argument("--cls-count-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-count-confidence", type=float, default=0.97)
    parser.add_argument("--cls-count-confidence", type=float, default=0.8)
    parser.add_argument(
        "--score-kinds",
        nargs="+",
        default=[
            "dense",
            "compact_objectness",
            "compact_iou",
            "compact_assignment",
            "mean",
            "compact_mean",
            "assignment_iou",
            "product",
            "iou_objectness",
        ],
    )
    parser.add_argument(
        "--suppression-modes",
        nargs="+",
        default=["iou", "center", "class_unique", "center_unique", "class_consensus"],
    )
    parser.add_argument("--nms-ious", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--aggregation-ious", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--center-distances", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--class-temperatures", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[50, 100, 200])
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
