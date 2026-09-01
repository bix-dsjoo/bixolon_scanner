from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...training.data import read_manifest
from .dinov3_class_conditional_box_refiner_oof import apply_box_offsets
from .dinov3_selective_e2e import aggregate_catalog_heads
from .proposal_ranker import proposal_iou_matrix


@dataclass(frozen=True)
class ClassSelection:
    proposal_indices: np.ndarray
    class_indices: np.ndarray
    scores: np.ndarray
    residual_ratio: float


def fuse_class_conditional_boxes(
    boxes: np.ndarray,
    candidate_classes: np.ndarray,
    pair_scores: np.ndarray,
    selection: ClassSelection,
    *,
    top_n: int,
    cluster_iou: float,
    temperature: float,
    refined_pair_boxes: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse the local proposal cluster supporting each selected SKU.

    The class selector keeps an anchor proposal for identity aggregation.  Localization
    is instead estimated from proposals that support the same SKU and overlap the
    anchor.  This is deliberately class-conditional and uses no detector-family
    prediction or ground-truth information at inference time.
    """
    values = np.asarray(boxes, dtype=np.float32)
    classes = np.asarray(candidate_classes, dtype=np.int64)
    scores = np.asarray(pair_scores, dtype=np.float32)
    if classes.shape != scores.shape or classes.shape[0] != len(values):
        raise ValueError("class candidate scores do not align with proposal boxes")
    if refined_pair_boxes is not None and np.asarray(refined_pair_boxes).shape != (
        *classes.shape,
        4,
    ):
        raise ValueError("refined pair boxes do not align with class candidates")
    if top_n <= 0 or temperature <= 0.0:
        raise ValueError("fusion top_n and temperature must be positive")
    if not 0.0 <= cluster_iou <= 1.0:
        raise ValueError("fusion cluster_iou must be between zero and one")

    fused = []
    for anchor_index, class_index in zip(
        selection.proposal_indices, selection.class_indices, strict=True
    ):
        proposal_indices, ranks = np.nonzero(classes == class_index)
        class_scores = scores[proposal_indices, ranks]
        order = np.argsort(-class_scores, kind="stable")
        proposal_indices = proposal_indices[order]
        ranks = ranks[order]
        class_scores = class_scores[order]
        overlap = proposal_iou_matrix(values[proposal_indices], values[int(anchor_index)][None, :])[
            :, 0
        ]
        local = overlap >= cluster_iou
        proposal_indices = proposal_indices[local][:top_n]
        ranks = ranks[local][:top_n]
        class_scores = class_scores[local][:top_n]
        if len(proposal_indices) == 0:
            proposal_indices = np.asarray([anchor_index], dtype=np.int64)
            selected_positions = np.flatnonzero(classes[int(anchor_index)] == class_index)
            ranks = np.asarray([selected_positions[0]], dtype=np.int64)
            class_scores = np.asarray(
                [scores[int(anchor_index), selected_positions[0]]], dtype=np.float32
            )
        logits = (class_scores - float(class_scores.max())) / temperature
        weights = np.exp(np.clip(logits, -30.0, 0.0))
        weights /= weights.sum()
        localization_boxes = (
            values[proposal_indices]
            if refined_pair_boxes is None
            else np.asarray(refined_pair_boxes)[proposal_indices, ranks]
        )
        fused.append(np.sum(localization_boxes * weights[:, None], axis=0))
    return np.asarray(fused, dtype=np.float32)


def class_conditional_score(
    objectness: np.ndarray,
    predicted_iou: np.ndarray,
    *,
    kind: str,
) -> np.ndarray:
    if kind == "objectness":
        return np.asarray(objectness, dtype=np.float32)
    if kind == "iou":
        return np.asarray(predicted_iou, dtype=np.float32)
    if kind == "product":
        return np.sqrt(
            np.clip(
                np.asarray(objectness, dtype=np.float32)
                * np.asarray(predicted_iou, dtype=np.float32),
                0.0,
                1.0,
            )
        )
    raise ValueError(f"unsupported class-conditional score: {kind}")


def select_class_conditional_top_k(
    boxes: np.ndarray,
    candidate_classes: np.ndarray,
    pair_scores: np.ndarray,
    *,
    count: int,
    nms_iou: float,
) -> ClassSelection:
    boxes = np.asarray(boxes, dtype=np.float32)
    classes = np.asarray(candidate_classes, dtype=np.int64)
    scores = np.asarray(pair_scores, dtype=np.float32)
    if classes.shape != scores.shape or classes.shape[0] != len(boxes):
        raise ValueError("class candidate scores do not align with proposal boxes")
    overlap = proposal_iou_matrix(boxes, boxes)
    class_candidates: dict[int, list[tuple[float, int]]] = {}
    for proposal_index in range(len(boxes)):
        for rank in range(classes.shape[1]):
            class_index = int(classes[proposal_index, rank])
            class_candidates.setdefault(class_index, []).append(
                (float(scores[proposal_index, rank]), proposal_index)
            )
    for values in class_candidates.values():
        values.sort(key=lambda item: item[0], reverse=True)
    class_order = sorted(
        class_candidates,
        key=lambda class_index: class_candidates[class_index][0][0],
        reverse=True,
    )
    selected_proposals: list[int] = []
    selected_classes: list[int] = []
    selected_scores: list[float] = []
    for class_index in class_order:
        for score, proposal_index in class_candidates[class_index]:
            if all(overlap[proposal_index, kept] <= nms_iou for kept in selected_proposals):
                selected_proposals.append(proposal_index)
                selected_classes.append(class_index)
                selected_scores.append(score)
                break
        if len(selected_proposals) == count:
            break
    selected_class_set = set(selected_classes)
    residual_scores = [
        values[0][0]
        for class_index, values in class_candidates.items()
        if class_index not in selected_class_set
    ]
    minimum_selected = min(selected_scores, default=0.0)
    residual_ratio = (
        0.0 if not residual_scores else max(residual_scores) / max(minimum_selected, 1e-6)
    )
    return ClassSelection(
        proposal_indices=np.asarray(selected_proposals, dtype=np.int64),
        class_indices=np.asarray(selected_classes, dtype=np.int64),
        scores=np.asarray(selected_scores, dtype=np.float32),
        residual_ratio=float(residual_ratio),
    )


def _jsonl_by_id(path: Path) -> dict[int, dict]:
    return {
        int(row["image_id"]): row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    }


def _scene(
    record: dict,
    boxes: np.ndarray,
    candidates: np.ndarray,
    pair_scores: np.ndarray,
    catalog_logits: np.ndarray,
    retrieval_logits: np.ndarray,
    *,
    count: int,
    nms_iou: float,
    aggregation_iou: float,
    fusion_top_n: int,
    fusion_cluster_iou: float,
    fusion_temperature: float,
    box_offsets: np.ndarray | None,
) -> dict:
    selection = select_class_conditional_top_k(
        boxes,
        candidates,
        pair_scores,
        count=count,
        nms_iou=nms_iou,
    )
    if len(selection.proposal_indices) != count:
        return {"selectable": False}
    refined_pair_boxes = None
    if box_offsets is not None:
        refined_pair_boxes = np.stack(
            [apply_box_offsets(boxes, box_offsets[:, rank]) for rank in range(candidates.shape[1])],
            axis=1,
        )
        refined_pair_boxes[..., 0::2] = np.clip(
            refined_pair_boxes[..., 0::2], 0.0, float(record["width"])
        )
        refined_pair_boxes[..., 1::2] = np.clip(
            refined_pair_boxes[..., 1::2], 0.0, float(record["height"])
        )
    selected_boxes = fuse_class_conditional_boxes(
        boxes,
        candidates,
        pair_scores,
        selection,
        top_n=fusion_top_n,
        cluster_iou=fusion_cluster_iou,
        temperature=fusion_temperature,
        refined_pair_boxes=refined_pair_boxes,
    )
    aggregated, aggregated_retrieval, stability = aggregate_catalog_heads(
        boxes,
        selection.proposal_indices,
        catalog_logits,
        retrieval_logits,
        np.maximum(pair_scores.max(axis=1), 1e-6),
        minimum_iou=aggregation_iou,
    )
    adapter_top1 = np.argmax(aggregated, axis=1)
    retrieval_top1 = np.argmax(aggregated_retrieval, axis=1)
    conditional_top3 = []
    identity_margins = []
    for proposal_index, selected_class in zip(
        selection.proposal_indices, selection.class_indices, strict=True
    ):
        order = np.argsort(-pair_scores[proposal_index], kind="stable")
        ranked = [
            int(candidates[proposal_index, index])
            for index in order
            if int(candidates[proposal_index, index]) != int(selected_class)
        ]
        top3 = [int(selected_class), *ranked[:2]]
        conditional_top3.append(top3)
        selected_positions = np.flatnonzero(candidates[proposal_index] == selected_class)
        selected_score = float(pair_scores[proposal_index, selected_positions[0]])
        alternative_scores = pair_scores[proposal_index][
            candidates[proposal_index] != selected_class
        ]
        alternative = float(alternative_scores.max()) if len(alternative_scores) else 0.0
        identity_margins.append(selected_score - alternative)
    conditional_top3 = np.asarray(conditional_top3, dtype=np.int64)

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
    overlap = proposal_iou_matrix(selected_boxes, targets)
    proposal_indices, target_indices = linear_sum_assignment(-overlap)
    assigned_iou = overlap[proposal_indices, target_indices]
    assigned_class = np.full(len(selection.proposal_indices), -1, dtype=np.int64)
    assigned_class[proposal_indices] = target_classes[target_indices]
    spatial_exact = (
        len(selection.proposal_indices) == len(targets)
        and len(proposal_indices) == len(targets)
        and bool(np.all(assigned_iou >= 0.5))
    )
    top1_correct = selection.class_indices == assigned_class
    top3_correct = np.any(conditional_top3 == assigned_class[:, None], axis=1)
    head_agreement = (selection.class_indices == adapter_top1) & (
        selection.class_indices == retrieval_top1
    )
    return {
        "selectable": True,
        "selected_indices": selection.proposal_indices.tolist(),
        "selected_boxes_xyxy": selected_boxes.tolist(),
        "selected_classes": selection.class_indices.tolist(),
        "spatial_exact": spatial_exact,
        "top1_all_correct": bool(spatial_exact and np.all(top1_correct)),
        "top3_all_correct": bool(spatial_exact and np.all(top3_correct)),
        "minimum_ranker_safety": float(selection.scores.min()),
        "ranker_safety": selection.scores.tolist(),
        "minimum_identity_margin": float(min(identity_margins)),
        "minimum_identity_stability": float(stability.min()),
        "all_heads_agree": bool(np.all(head_agreement)),
        "maximum_residual_ratio": selection.residual_ratio,
        "assigned_iou": assigned_iou.tolist(),
        "assigned_class": assigned_class.tolist(),
        "top1": selection.class_indices.tolist(),
        "top3": conditional_top3.tolist(),
        "identity_margin": identity_margins,
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


def _best_gate(scenes: list[dict]) -> dict:
    best = None
    for ranker, margin, stability, residual, agreement in product(
        [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [-1.0, -0.1, -0.05, 0.0, 0.01, 0.02, 0.04, 0.08],
        [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [1.5, 1.25, 1.1, 1.01, 0.98, 0.95, 0.9, 0.8],
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
        errors = sum(
            not scene["spatial_exact"] or not scene["top3_all_correct"] for scene in accepted
        )
        candidate = {
            "gate": gate,
            "accepted_count": len(accepted),
            "error_count": errors,
            "approved_safe_count": sum(scene["top1_all_correct"] for scene in accepted),
        }
        key = (
            errors == 0 and len(accepted) > 0,
            len(accepted) if errors == 0 else -errors,
            candidate["approved_safe_count"],
        )
        if best is None or key > best[0]:
            best = (key, candidate)
    if best is None:
        raise ValueError("no class-conditional safety gate was evaluated")
    return best[1]


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
    with np.load(args.class_conditional_scores, allow_pickle=False) as payload:
        conditional = {name: payload[name].copy() for name in payload.files}
    for name, payload in (("Catalog", catalog), ("class-conditional", conditional)):
        if not np.array_equal(payload["image_id"], image_ids):
            raise ValueError(f"{name} cache does not align with dense features")

    count_safe_ids = []
    for image_id in sorted(records):
        patch = patch_counts[image_id]
        cls = cls_counts[image_id]
        if patch["confidence"] < args.patch_count_confidence:
            continue
        if args.require_count_head_agreement and (
            patch["predicted_count"] != cls["predicted_count"]
        ):
            continue
        if cls["confidence"] < args.cls_count_confidence:
            continue
        count_safe_ids.append(image_id)

    policy_rows = []
    best = None
    score_arrays = {
        kind: class_conditional_score(
            conditional["class_objectness"],
            conditional["class_iou"],
            kind=kind,
        )
        for kind in args.score_kinds
    }
    if args.listwise_scores is not None:
        with np.load(args.listwise_scores, allow_pickle=False) as payload:
            if not np.array_equal(payload["image_id"], image_ids):
                raise ValueError("listwise score cache does not align with dense features")
            if not np.array_equal(payload["candidate_classes"], conditional["candidate_classes"]):
                raise ValueError("listwise candidates do not align with class-conditional scores")
            score_arrays["listwise"] = payload["class_rank_score"].astype(np.float32)
    box_offsets = None
    if args.box_refiner is not None:
        with np.load(args.box_refiner, allow_pickle=False) as payload:
            if not np.array_equal(payload["image_id"], image_ids):
                raise ValueError("box refiner cache does not align with dense features")
            if not np.array_equal(payload["candidate_classes"], conditional["candidate_classes"]):
                raise ValueError("box refiner candidates do not align with conditional scores")
            box_offsets = payload["box_offsets"].astype(np.float32)
    for (
        score_kind,
        nms_iou,
        aggregation_iou,
        fusion_top_n,
        fusion_cluster_iou,
        fusion_temperature,
    ) in product(
        score_arrays,
        args.nms_ious,
        args.aggregation_ious,
        args.fusion_top_ns,
        args.fusion_cluster_ious,
        args.fusion_temperatures,
    ):
        pair_scores = score_arrays[score_kind]
        scenes = []
        for image_id in count_safe_ids:
            selected = image_ids == image_id
            scene = _scene(
                records[image_id],
                boxes[selected],
                conditional["candidate_classes"][selected],
                pair_scores[selected],
                catalog["logits"][selected],
                catalog["retrieval_logits"][selected],
                count=int(patch_counts[image_id]["predicted_count"]),
                nms_iou=nms_iou,
                aggregation_iou=aggregation_iou,
                fusion_top_n=fusion_top_n,
                fusion_cluster_iou=fusion_cluster_iou,
                fusion_temperature=fusion_temperature,
                box_offsets=None if box_offsets is None else box_offsets[selected],
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
        gate_result = _best_gate(scenes)
        accepted = [scene for scene in scenes if _accepts(scene, gate_result["gate"])]
        row = {
            "score_kind": score_kind,
            "nms_iou": nms_iou,
            "aggregation_iou": aggregation_iou,
            "fusion_top_n": fusion_top_n,
            "fusion_cluster_iou": fusion_cluster_iou,
            "fusion_temperature": fusion_temperature,
            "gate": gate_result["gate"],
            "scenes": scenes,
            "accepted_count": len(accepted),
            "accepted_error_count": sum(
                not scene["spatial_exact"] or not scene["top3_all_correct"] for scene in accepted
            ),
            "pre_gate_spatial_exact_count": sum(
                scene.get("spatial_exact", False) for scene in scenes
            ),
            "pre_gate_top3_safe_count": sum(
                scene.get("top3_all_correct", False) for scene in scenes
            ),
        }
        policy_rows.append(row)
        # First choose the localization/identity policy, then evaluate its reject gate.
        # Letting a fitted gate choose the upstream policy hides localization failures
        # and compounds development-set selection leakage.
        key = (
            row["pre_gate_top3_safe_count"],
            row["pre_gate_spatial_exact_count"],
            row["accepted_error_count"] == 0,
            row["accepted_count"] if row["accepted_error_count"] == 0 else 0,
        )
        if best is None or key > best[0]:
            best = (key, row)
    if best is None:
        raise ValueError("no class-conditional E2E policy was evaluated")
    selected = best[1]
    accepted = [scene for scene in selected["scenes"] if _accepts(scene, selected["gate"])]

    approval_threshold = None
    for threshold in [0.2, 0.16, 0.12, 0.08, 0.04, 0.02, 0.01, 0.0, -0.05, -0.1]:
        approved = [
            correct
            for scene in accepted
            for correct, margin, agree in zip(
                scene["top1_correct"],
                scene["identity_margin"],
                scene["head_agreement"],
                strict=True,
            )
            if agree and margin >= threshold
        ]
        if approved and all(approved):
            approval_threshold = threshold
            break
    approved_count = 0
    approved_errors = 0
    unknown_count = 0
    unknown_candidate_out = 0
    for scene in accepted:
        for top1_correct, top3_correct, margin, agree in zip(
            scene["top1_correct"],
            scene["top3_correct"],
            scene["identity_margin"],
            scene["head_agreement"],
            strict=True,
        ):
            approved = approval_threshold is not None and agree and margin >= approval_threshold
            if approved:
                approved_count += 1
                approved_errors += int(not top1_correct)
            else:
                unknown_count += 1
                unknown_candidate_out += int(not top3_correct)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(
                {**scene, "accepted": _accepts(scene, selected["gate"])},
                ensure_ascii=False,
            )
            + "\n"
            for scene in selected["scenes"]
        ),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_class_conditional_selective_e2e_development_oof",
        "selection_scope": (
            "proposal/count/class-conditional scores are group-aware OOF; policy and gate "
            "are development-selected and require nested or independent confirmation"
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
            "head_agreement_required": args.require_count_head_agreement,
            "passed_image_count": len(count_safe_ids),
            "observed_count_error_count": sum(
                patch_counts[image_id]["predicted_count"] != len(records[image_id]["annotations"])
                for image_id in count_safe_ids
            ),
        },
        "selected_policy": {
            "score_kind": selected["score_kind"],
            "nms_iou": selected["nms_iou"],
            "aggregation_iou": selected["aggregation_iou"],
            "fusion_top_n": selected["fusion_top_n"],
            "fusion_cluster_iou": selected["fusion_cluster_iou"],
            "fusion_temperature": selected["fusion_temperature"],
            "gate": selected["gate"],
            "pre_gate_spatial_exact_count": selected["pre_gate_spatial_exact_count"],
            "pre_gate_top3_safe_count": selected["pre_gate_top3_safe_count"],
        },
        "accepted": {
            "image_count": len(accepted),
            "coverage_over_all_images": len(accepted) / len(records),
            "segmentation_count": sum(len(scene["selected_indices"]) for scene in accepted),
            "product_miss_image_count": sum(not scene["spatial_exact"] for scene in accepted),
            "false_positive_or_duplicate_image_count": sum(
                not scene["spatial_exact"] for scene in accepted
            ),
            "top3_candidate_out_image_count": sum(
                not scene["top3_all_correct"] for scene in accepted
            ),
            "image_ids": [scene["image_id"] for scene in accepted],
        },
        "approval": {
            "minimum_class_conditional_margin": approval_threshold,
            "catalog_head_agreement_required": True,
            "approved_segment_count": approved_count,
            "approved_wrong_count": approved_errors,
        },
        "unknown": {
            "segment_count": unknown_count,
            "top3_candidate_out_count": unknown_candidate_out,
        },
        "fallback": {
            "status": "IMAGE_RECAPTURE",
            "image_count": len(records) - len(accepted),
            "all_recapture_solution": len(accepted) == 0,
        },
        "policy_candidate_count": len(policy_rows),
        "top_policy_candidates": [
            {
                key: row[key]
                for key in (
                    "score_kind",
                    "nms_iou",
                    "aggregation_iou",
                    "fusion_top_n",
                    "fusion_cluster_iou",
                    "fusion_temperature",
                    "pre_gate_spatial_exact_count",
                    "pre_gate_top3_safe_count",
                    "accepted_count",
                    "accepted_error_count",
                )
            }
            for row in sorted(
                policy_rows,
                key=lambda item: (
                    item["pre_gate_top3_safe_count"],
                    item["pre_gate_spatial_exact_count"],
                ),
                reverse=True,
            )[:20]
        ],
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
    parser = argparse.ArgumentParser(
        description="Evaluate class-conditional DINOv3 selective E2E safety"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dense-features", type=Path, required=True)
    parser.add_argument("--catalog-logits", type=Path, required=True)
    parser.add_argument("--class-conditional-scores", type=Path, required=True)
    parser.add_argument("--listwise-scores", type=Path)
    parser.add_argument("--box-refiner", type=Path)
    parser.add_argument("--patch-count-oof", type=Path, required=True)
    parser.add_argument("--cls-count-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-count-confidence", type=float, default=0.97)
    parser.add_argument("--cls-count-confidence", type=float, default=0.0)
    parser.add_argument(
        "--require-count-head-agreement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--score-kinds", nargs="+", default=["objectness", "iou", "product"])
    parser.add_argument("--nms-ious", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--aggregation-ious", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--fusion-top-ns", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--fusion-cluster-ious", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7]
    )
    parser.add_argument("--fusion-temperatures", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
