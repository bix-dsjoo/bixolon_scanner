from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..contracts.model_package import sha256_file
from ..experiments.bread.classifier_geometry_mask import (
    apply_background_mask,
    neighbor_ownership_mask,
)
from ..experiments.bread.detector_ambiguity_gate import ambiguity_recapture_mask
from ..experiments.bread.detector_model_ensemble import fuse_prediction_sets
from ..experiments.bread.detector_proposal_class_selector import (
    candidate_mask_context,
    filtered_proposal_indices,
    select_class_verified_prediction,
)
from ..experiments.bread.hierarchical_detector import filter_predictions
from ..experiments.bread.proposal_count_selector import count_constrained_select
from ..pipeline.ports import Detection
from ..runtime.onnx import OrtRunner, prepare_rgb
from .detected_roi_dataset import crop_tensor, match_detections
from .onnx_detector import load_records, raw_outputs_to_prediction


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def locked_image_manifest_sha256(dataset_root: Path, annotation_file: Path) -> tuple[str, int]:
    payload = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
    image_base = annotation_file.parent
    lines = []
    for image in sorted(payload["images"], key=lambda row: int(row["id"])):
        relative = str(image["file_name"])
        image_path = (image_base / relative).resolve()
        image_path.relative_to(dataset_root.resolve())
        lines.append(f"{int(image['id']):04d} {sha256_file(image_path).lower()} {relative}\n")
    digest = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    return digest, len(lines)


def verify_locked_inputs(
    candidate_path: Path, dataset_lock_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    candidate = _read_json(candidate_path)
    dataset_lock = _read_json(dataset_lock_path)
    if candidate["lifecycle"] != "locked_final_candidate":
        raise ValueError("candidate must be locked before independent evaluation")
    locked_test = candidate["locked_test"]
    if locked_test["evaluation_attempted"]:
        raise ValueError("locked test evaluation was already attempted")
    if dataset_lock["lifecycle"] != "locked_test":
        raise ValueError("independent dataset is not locked")
    if locked_test["dataset_version"] != dataset_lock["dataset_version"]:
        raise ValueError("candidate and independent dataset versions differ")

    for model in candidate["detector"]["models"]:
        path = Path(model["path"])
        if sha256_file(path).lower() != str(model["sha256"]).lower():
            raise ValueError(f"detector checksum mismatch: {path}")
    classifier_path = Path(candidate["classifier"]["path"])
    if sha256_file(classifier_path).lower() != str(candidate["classifier"]["sha256"]).lower():
        raise ValueError("classifier checksum mismatch")

    dataset_root = Path(dataset_lock["dataset_root"]).resolve()
    annotation_file = (dataset_root / dataset_lock["annotation_file"]).resolve()
    if sha256_file(annotation_file).lower() != str(dataset_lock["annotation_sha256"]).lower():
        raise ValueError("locked annotation checksum mismatch")
    image_digest, image_count = locked_image_manifest_sha256(dataset_root, annotation_file)
    if image_digest != str(dataset_lock["image_manifest_sha256"]).lower():
        raise ValueError("locked image manifest checksum mismatch")
    if image_count != int(dataset_lock["image_count"]):
        raise ValueError("locked image count mismatch")
    return candidate, dataset_lock, dataset_root, annotation_file


def _run_detectors(
    records: list[dict[str, Any]],
    runners: list[OrtRunner],
) -> tuple[list[list[dict[str, Any]]], dict[int, tuple[int, int]], list[float]]:
    prediction_sets: list[list[dict[str, Any]]] = [[] for _ in runners]
    dimensions: dict[int, tuple[int, int]] = {}
    image_times_ms = []
    for record in records:
        started = time.perf_counter()
        with Image.open(record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        try:
            width, height = image.size
            dimensions[int(record["image_id"])] = (width, height)
            tensor = prepare_rgb(
                image,
                (640, 640),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                reducing_gap=1.0,
            )[None]
        finally:
            image.close()
        for rows, runner in zip(prediction_sets, runners):
            logits, boxes = runner.run(["logits", "pred_boxes"], "pixel_values", tensor)
            prediction = raw_outputs_to_prediction(
                np.asarray(logits)[0],
                np.asarray(boxes)[0],
                image_width=width,
                image_height=height,
            )
            prediction["image_id"] = int(record["image_id"])
            rows.append(prediction)
        image_times_ms.append((time.perf_counter() - started) * 1000.0)
    return prediction_sets, dimensions, image_times_ms


def ambiguity_image_ids(
    raw_predictions: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> set[int]:
    combined = np.zeros(len(raw_predictions), dtype=bool)
    for rule in rules:
        available = [
            count_constrained_select(
                prediction,
                predicted_count=600,
                score_threshold=float(rule["availability_score_threshold"]),
                nms_iou_threshold=float(rule["availability_nms_iou_threshold"]),
                containment_threshold=float(rule["availability_containment_threshold"]),
                group_minimum=int(rule["availability_group_minimum"]),
            )
            for prediction in raw_predictions
        ]
        combined |= ambiguity_recapture_mask(
            available,
            base_predictions,
            minimum_selected_count=int(rule["minimum_selected_count"]),
            extra_candidate_count=int(rule["extra_candidate_count"]),
            extra_count_mode=str(rule["extra_count_mode"]),
            next_score_threshold=float(rule["next_score_threshold_inclusive"]),
        )
    return {
        int(row["image_id"]) for row, ambiguous in zip(raw_predictions, combined) if bool(ambiguous)
    }


def _score_masked_tensors(
    tensors: np.ndarray,
    rows: list[dict[str, Any]],
    dimensions: dict[int, tuple[int, int]],
    runner: OrtRunner,
    *,
    batch_size: int,
    margin_ratio: float,
    distance_bias: float,
) -> np.ndarray:
    parts = []
    for start in range(0, len(rows), batch_size):
        batch = np.asarray(tensors[start : start + batch_size], dtype=np.float32).copy()
        batch_rows = rows[start : start + len(batch)]
        masks = np.stack(
            [
                neighbor_ownership_mask(
                    image_width=dimensions[int(row["image_id"])][0],
                    image_height=dimensions[int(row["image_id"])][1],
                    boxes=row["mask_boxes_xyxy"],
                    target_index=int(row["mask_target_index"]),
                    output_size=batch.shape[-1],
                    margin_ratio=margin_ratio,
                    distance_bias=distance_bias,
                    shared_scale=False,
                )
                for row in batch_rows
            ]
        )
        masked = apply_background_mask(batch, masks).astype(np.float32, copy=False)
        (scores,) = runner.run(["logits"], "pixel_values", masked)
        parts.append(np.asarray(scores, dtype=np.float32))
    if not parts:
        return np.empty((0, 20), dtype=np.float32)
    return np.concatenate(parts)


def _proposal_tensors(
    records_by_id: dict[int, dict[str, Any]],
    base_by_id: dict[int, dict[str, Any]],
    raw_by_id: dict[int, dict[str, Any]],
    ambiguity_ids: set[int],
    *,
    minimum_score: float,
    minimum_support: int,
    duplicate_iou: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    tensors = []
    rows = []
    for image_id in sorted(ambiguity_ids):
        record = records_by_id[image_id]
        base_boxes = np.asarray(base_by_id[image_id]["boxes_xyxy"], dtype=np.float32)
        indices = filtered_proposal_indices(
            raw_by_id[image_id],
            minimum_score=minimum_score,
            minimum_support=minimum_support,
        )
        with Image.open(record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        try:
            for proposal_index in indices:
                raw = raw_by_id[image_id]
                box = raw["boxes_xyxy"][proposal_index]
                detection = Detection(
                    *box,
                    float(raw["scores"][proposal_index]),
                    int(raw["class_ids"][proposal_index]),
                )
                mask_boxes, mask_target = candidate_mask_context(
                    base_boxes,
                    np.asarray(box, dtype=np.float32),
                    duplicate_iou=duplicate_iou,
                )
                tensors.append(
                    crop_tensor(
                        image,
                        detection,
                        crop_margin_ratio=0.05,
                        input_size=224,
                    )
                )
                rows.append(
                    {
                        "image_id": image_id,
                        "proposal_index": proposal_index,
                        "mask_boxes_xyxy": mask_boxes,
                        "mask_target_index": mask_target,
                    }
                )
        finally:
            image.close()
    if not tensors:
        return np.empty((0, 3, 224, 224), dtype=np.float32), rows
    return np.stack(tensors).astype(np.float32), rows


def _proposal_entries(
    raw_prediction: dict[str, Any], rows: list[dict[str, Any]], scores: np.ndarray
) -> list[dict[str, Any]]:
    entries = []
    for row, class_scores in zip(rows, scores):
        order = np.argsort(-class_scores, kind="stable")
        index = int(row["proposal_index"])
        entries.append(
            {
                "proposal_index": index,
                "box": np.asarray(raw_prediction["boxes_xyxy"][index], dtype=np.float32),
                "detector_score": float(raw_prediction["scores"][index]),
                "support_count": int(raw_prediction["support_counts"][index]),
                "predicted_class": int(order[0]),
                "class_margin": float(class_scores[order[0]] - class_scores[order[1]]),
            }
        )
    return entries


def _resolve_predictions(
    base_predictions: list[dict[str, Any]],
    raw_predictions: list[dict[str, Any]],
    ambiguity_ids: set[int],
    proposal_rows: list[dict[str, Any]],
    proposal_scores: np.ndarray,
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_by_id = {int(row["image_id"]): row for row in base_predictions}
    raw_by_id = {int(row["image_id"]): row for row in raw_predictions}
    positions: dict[int, list[int]] = {}
    for index, row in enumerate(proposal_rows):
        positions.setdefault(int(row["image_id"]), []).append(index)
    outputs = []
    diagnostics = []
    for base in base_predictions:
        image_id = int(base["image_id"])
        if image_id not in ambiguity_ids:
            outputs.append(base)
            continue
        selected_positions = positions.get(image_id, [])
        rows = [proposal_rows[index] for index in selected_positions]
        entries = _proposal_entries(raw_by_id[image_id], rows, proposal_scores[selected_positions])
        prediction, diagnostic = select_class_verified_prediction(
            base_by_id[image_id],
            raw_by_id[image_id],
            entries,
            minimum_support=int(policy["candidate_minimum_support"]),
            base_match_iou=float(policy["base_match_iou"]),
            group_relation_iou=float(policy["group_relation_iou"]),
            group_area_ratio=float(policy["group_area_ratio"]),
            group_margin_ratio=float(policy["group_margin_ratio"]),
            group_novel_margin=float(policy["group_novel_margin"]),
            group_minimum_score=float(policy["group_minimum_score"]),
            independent_maximum_iou=float(policy["independent_maximum_iou"]),
            independent_margin=float(policy["independent_margin"]),
            independent_minimum_score=float(policy["independent_minimum_score"]),
        )
        outputs.append(prediction)
        diagnostics.append(diagnostic)
    return outputs, diagnostics


def _classification_tensors(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    tensors = []
    rows = []
    for record, prediction in zip(records, predictions):
        image_id = int(record["image_id"])
        boxes = prediction["boxes_xyxy"]
        with Image.open(record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        try:
            for detection_index, (box, score, class_id) in enumerate(
                zip(boxes, prediction["scores"], prediction["class_ids"])
            ):
                tensors.append(
                    crop_tensor(
                        image,
                        Detection(*box, float(score), int(class_id)),
                        crop_margin_ratio=0.05,
                        input_size=224,
                    )
                )
                rows.append(
                    {
                        "image_id": image_id,
                        "detection_index": detection_index,
                        "mask_boxes_xyxy": boxes,
                        "mask_target_index": detection_index,
                    }
                )
        finally:
            image.close()
    if not tensors:
        return np.empty((0, 3, 224, 224), dtype=np.float32), rows
    return np.stack(tensors).astype(np.float32), rows


def build_final_metrics(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    logits: np.ndarray,
    *,
    approval_thresholds: list[float | None],
    approval_default_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    scores_by_key = {
        (int(row["image_id"]), int(row["detection_index"])): values
        for row, values in zip(score_rows, logits)
    }
    total_gt = sum(len(record["annotations"]) for record in records)
    approved_count = approved_wrong = unknown_count = candidate_out = 0
    fp_images = fn_images = fp_count = fn_count = matched_count = 0
    object_decisions = []
    error_images = []
    for record, prediction in zip(records, predictions):
        image_id = int(record["image_id"])
        detections = [
            Detection(*box, float(score), int(class_id))
            for box, score, class_id in zip(
                prediction["boxes_xyxy"], prediction["scores"], prediction["class_ids"]
            )
        ]
        matches = match_detections(detections, record["annotations"], match_iou_threshold=0.5)
        image_fp = len(detections) - len(matches)
        image_fn = len(record["annotations"]) - len(matches)
        fp_count += image_fp
        fn_count += image_fn
        fp_images += image_fp > 0
        fn_images += image_fn > 0
        matched_count += len(matches)
        if image_fp or image_fn:
            error_images.append(
                {
                    "image_id": image_id,
                    "prediction_count": len(detections),
                    "ground_truth_count": len(record["annotations"]),
                    "matched_count": len(matches),
                    "false_positive_count": image_fp,
                    "false_negative_count": image_fn,
                }
            )
        for detection_index, (annotation_index, overlap) in matches.items():
            values = scores_by_key[(image_id, detection_index)]
            order = np.argsort(-values, kind="stable")
            predicted_class = int(order[0])
            target = int(record["annotations"][annotation_index]["category_id"]) - 1
            margin = float(values[order[0]] - values[order[1]])
            configured = approval_thresholds[predicted_class]
            threshold = approval_default_threshold if configured is None else float(configured)
            approved = margin >= threshold
            top3 = [int(value) for value in order[:3]]
            if approved:
                approved_count += 1
                approved_wrong += predicted_class != target
                status = "APPROVED"
            else:
                unknown_count += 1
                candidate_out += target not in top3
                status = "UNKNOWN"
            object_decisions.append(
                {
                    "image_id": image_id,
                    "detection_index": detection_index,
                    "annotation_index": annotation_index,
                    "match_iou": overlap,
                    "target": target,
                    "prediction": predicted_class,
                    "top3": top3,
                    "logit_margin": margin,
                    "approval_threshold": threshold,
                    "status": status,
                }
            )

    image_count = len(records)
    segmentation_count = image_count
    metrics = {
        "counts": {
            "image_count": image_count,
            "segmentation_image_count": segmentation_count,
            "image_recapture_count": 0,
            "judgeable_ground_truth_object_count": total_gt,
            "detector_prediction_count": sum(len(row["scores"]) for row in predictions),
            "detector_matched_count": matched_count,
            "detector_false_positive_count": fp_count,
            "detector_false_negative_count": fn_count,
            "approved_count": approved_count,
            "approved_misrecognition_count": approved_wrong,
            "unknown_count": unknown_count,
            "unknown_top3_candidate_out_count": candidate_out,
        },
        "rates": {
            "segmentation_image_rate": _rate(segmentation_count, image_count),
            "end_to_end_approved_object_rate": _rate(approved_count, total_gt),
            "segmentation_image_false_negative_rate": _rate(fn_images, segmentation_count),
            "segmentation_image_false_positive_rate": _rate(fp_images, segmentation_count),
            "approved_object_misrecognition_rate": _rate(approved_wrong, total_gt),
            "unknown_top3_candidate_out_rate": _rate(candidate_out, total_gt),
            "unknown_rate_diagnostic_only": _rate(unknown_count, total_gt),
        },
    }
    limits = {
        "segmentation_image_rate": 0.90,
        "end_to_end_approved_object_rate": 0.90,
        "segmentation_image_false_negative_rate": 0.001,
        "segmentation_image_false_positive_rate": 0.001,
        "approved_object_misrecognition_rate": 0.001,
        "unknown_top3_candidate_out_rate": 0.001,
        "final_end_to_end_approved_goal": 0.99,
    }
    rates = metrics["rates"]
    gates = {
        "segmentation_image_rate": rates["segmentation_image_rate"]
        >= limits["segmentation_image_rate"],
        "end_to_end_approved_object_rate": rates["end_to_end_approved_object_rate"]
        >= limits["end_to_end_approved_object_rate"],
        "segmentation_image_false_negative_rate": rates["segmentation_image_false_negative_rate"]
        <= limits["segmentation_image_false_negative_rate"],
        "segmentation_image_false_positive_rate": rates["segmentation_image_false_positive_rate"]
        <= limits["segmentation_image_false_positive_rate"],
        "approved_object_misrecognition_rate": rates["approved_object_misrecognition_rate"]
        <= limits["approved_object_misrecognition_rate"],
        "unknown_top3_candidate_out_rate": rates["unknown_top3_candidate_out_rate"]
        <= limits["unknown_top3_candidate_out_rate"],
    }
    metrics["limits"] = limits
    metrics["operational_gates"] = {**gates, "all_met": all(gates.values())}
    metrics["final_end_to_end_approved_goal_met"] = (
        rates["end_to_end_approved_object_rate"] >= limits["final_end_to_end_approved_goal"]
    )
    return metrics, object_decisions, error_images


def _percentiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": len(values),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise ValueError("locked evaluation output directory must not already exist")
    candidate, dataset_lock, dataset_root, _ = verify_locked_inputs(
        args.candidate, args.dataset_lock
    )
    detector_runners = [
        OrtRunner(Path(model["path"]), args.provider, args.cuda_dll_dir)
        for model in candidate["detector"]["models"]
    ]
    classifier_runner = OrtRunner(
        Path(candidate["classifier"]["path"]), args.provider, args.cuda_dll_dir
    )
    # Provider and graph preflight happens before reading any evaluation metrics.
    for runner in detector_runners:
        runner.run(
            ["logits", "pred_boxes"],
            "pixel_values",
            np.zeros((1, 3, 640, 640), dtype=np.float32),
        )
    classifier_runner.run(
        ["logits"],
        "pixel_values",
        np.zeros((1, 3, 224, 224), dtype=np.float32),
    )

    records = load_records(dataset_root, Path(dataset_lock["annotation_file"]).name)
    if len(records) != int(dataset_lock["image_count"]):
        raise ValueError("loaded locked record count differs from the lock")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    prediction_sets, dimensions, detector_times = _run_detectors(records, detector_runners)
    detector = candidate["detector"]
    fusion = detector["fusion"]
    raw = fuse_prediction_sets(
        prediction_sets,
        model_weights=[float(value) for value in fusion["model_weights"]],
        score_thresholds=[float(value) for value in fusion["score_thresholds"]],
        pre_nms_iou_threshold=float(fusion["pre_nms_iou_threshold"]),
        max_candidates_per_model=int(fusion["maximum_candidates_per_model"]),
        cluster_iou_threshold=float(fusion["cluster_iou_threshold"]),
        score_mode=str(fusion["score_mode"]),
    )
    base_policy = detector["base_selection"]
    base = filter_predictions(
        raw,
        score_threshold=float(base_policy["score_threshold"]),
        iou_threshold=float(base_policy["nms_iou_threshold"]),
        containment_threshold=float(base_policy["containment_threshold"]),
        group_minimum=int(base_policy["group_minimum"]),
    )
    ambiguity_ids = ambiguity_image_ids(raw, base, detector["ambiguity_union"])
    records_by_id = {int(row["image_id"]): row for row in records}
    base_by_id = {int(row["image_id"]): row for row in base}
    raw_by_id = {int(row["image_id"]): row for row in raw}
    selector = detector["class_verified_selector"]
    proposal_tensors, proposal_rows = _proposal_tensors(
        records_by_id,
        base_by_id,
        raw_by_id,
        ambiguity_ids,
        minimum_score=float(selector["candidate_minimum_score"]),
        minimum_support=int(selector["candidate_minimum_support"]),
        duplicate_iou=float(selector["candidate_duplicate_iou"]),
    )
    classifier = candidate["classifier"]
    proposal_scores = _score_masked_tensors(
        proposal_tensors,
        proposal_rows,
        dimensions,
        classifier_runner,
        batch_size=args.classifier_batch_size,
        margin_ratio=float(classifier["neighbor_mask"]["margin_ratio"]),
        distance_bias=float(classifier["neighbor_mask"]["distance_bias"]),
    )
    selected, selector_diagnostics = _resolve_predictions(
        base, raw, ambiguity_ids, proposal_rows, proposal_scores, selector
    )
    classifier_tensors, classifier_rows = _classification_tensors(records, selected)
    classifier_scores = _score_masked_tensors(
        classifier_tensors,
        classifier_rows,
        dimensions,
        classifier_runner,
        batch_size=args.classifier_batch_size,
        margin_ratio=float(classifier["neighbor_mask"]["margin_ratio"]),
        distance_bias=float(classifier["neighbor_mask"]["distance_bias"]),
    )
    metrics, object_decisions, error_images = build_final_metrics(
        records,
        selected,
        classifier_rows,
        classifier_scores,
        approval_thresholds=classifier["approval_thresholds"],
        approval_default_threshold=float(classifier["approval_default_threshold"]),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    for index, rows in enumerate(prediction_sets):
        _write_jsonl(args.output_dir / f"detector-model-{index}.jsonl", rows)
    _write_jsonl(args.output_dir / "detector-fused.jsonl", raw)
    _write_jsonl(args.output_dir / "detector-selected.jsonl", selected)
    _write_jsonl(args.output_dir / "object-decisions.jsonl", object_decisions)
    np.savez_compressed(
        args.output_dir / "classifier-scores.npz",
        scores=classifier_scores,
        image_ids=np.asarray([row["image_id"] for row in classifier_rows], dtype=np.int64),
        detection_indices=np.asarray(
            [row["detection_index"] for row in classifier_rows], dtype=np.int64
        ),
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_locked_final_candidate",
        "candidate_id": candidate["candidate_id"],
        "candidate_manifest": str(args.candidate),
        "candidate_manifest_sha256_at_evaluation": sha256_file(args.candidate),
        "dataset_version": dataset_lock["dataset_version"],
        "dataset_lock": str(args.dataset_lock),
        "provider": args.provider,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "detector_session_execution": "sequential_deterministic_evaluation",
        },
        "policy_frozen_before_evaluation": True,
        "target_labels_used_for_prediction_or_threshold_selection": False,
        "ambiguity_image_count": len(ambiguity_ids),
        "ambiguity_image_ids": sorted(ambiguity_ids),
        "proposal_classifier_sample_count": len(proposal_rows),
        "selector_diagnostics": selector_diagnostics,
        "metrics": metrics,
        "detector_error_images": error_images,
        "timing_diagnostic_not_release_benchmark": {
            "detector_four_model_sequential_per_image": _percentiles(detector_times),
            "total_evaluation_ms": elapsed_ms,
        },
        "independent_evaluation_passed": bool(
            metrics["operational_gates"]["all_met"]
            and metrics["final_end_to_end_approved_goal_met"]
        ),
        "policy_changes_after_this_evaluation_allowed": False,
    }
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one locked Bread 1.1 candidate once")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--classifier-batch-size", type=int, default=96)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
