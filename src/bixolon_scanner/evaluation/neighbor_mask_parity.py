from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.model_package import load_model_package
from ..pipeline.ports import Detection
from ..runtime.onnx import OnnxClassifier


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _weighted_reference(
    payload: Any,
    weights: np.ndarray,
    temperature: float,
    ranking_tie_break_bias_span: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.stack([payload["normalized_bias0.000"], payload["normalized_bias0.250"]]).astype(
        np.float32
    )
    logits = np.sum(values * weights[:, None, None], axis=0)
    orders = np.argsort(-values, axis=2, kind="stable")
    ranks = np.empty_like(orders)
    np.put_along_axis(
        ranks,
        orders,
        np.arange(values.shape[2], dtype=orders.dtype)[None, None, :],
        axis=2,
    )
    ranking = np.sum((1.0 / (ranks + 1.0)) * weights[:, None, None], axis=0)
    view_probabilities = np.stack([_softmax(view, temperature) for view in values])
    ranking += np.sum(view_probabilities * weights[:, None, None], axis=0) * 1e-3
    ranking_probabilities = _softmax(ranking, temperature)
    safety = np.sum(ranking_probabilities * np.log(ranking_probabilities.clip(1e-12)), axis=1)
    ranking += np.linspace(
        0.0,
        -ranking_tie_break_bias_span,
        ranking.shape[1],
        dtype=np.float32,
    )[None, :]
    probabilities = _softmax(logits, temperature)
    ordered = np.sort(probabilities, axis=1)
    approval = ordered[:, -1] - ordered[:, -2]
    return logits, ranking, approval, safety


def _states(
    approval: np.ndarray,
    safety: np.ndarray,
    *,
    approval_threshold: float,
    safety_threshold: float,
) -> np.ndarray:
    approved = approval >= approval_threshold
    recapture = (~approved) & (safety < safety_threshold)
    return np.where(approved, 0, np.where(recapture, 2, 1)).astype(np.int8)


def _outcome_counts(
    *,
    mask: np.ndarray,
    states: np.ndarray,
    top1: np.ndarray,
    top3: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    matched = mask & (targets >= 0)
    unmatched = mask & (targets < 0)
    sample_count = int(np.count_nonzero(matched))
    approved = matched & (states == 0)
    unknown = matched & (states == 1)
    recapture = matched & (states == 2)
    return {
        "sample_count": sample_count,
        "approved_count": int(np.count_nonzero(approved)),
        "approved_error_count": int(np.count_nonzero(approved & (top1 != targets))),
        "unknown_count": int(np.count_nonzero(unknown)),
        "unknown_top3_miss_count": int(
            np.count_nonzero(unknown & (~np.any(top3 == targets[:, None], axis=1)))
        ),
        "segment_recapture_count": int(np.count_nonzero(recapture)),
        "segment_recapture_rate": (
            float(np.count_nonzero(recapture) / sample_count) if sample_count else 0.0
        ),
        "unmatched_prediction_count": int(np.count_nonzero(unmatched)),
        "unmatched_approved_count": int(np.count_nonzero(unmatched & (states == 0))),
        "unmatched_unknown_count": int(np.count_nonzero(unmatched & (states == 1))),
        "unmatched_segment_recapture_count": int(np.count_nonzero(unmatched & (states == 2))),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    package = load_model_package(args.package)
    metadata = package.metadata.classifier
    policy = metadata.neighbor_mask_inference
    if policy is None:
        raise ValueError("package does not define neighbor-mask classifier inference")
    classifier = OnnxClassifier(
        package.classifier_path,
        metadata,
        args.provider,
        cuda_dll_dir=args.cuda_dll_dir,
    )
    classifier.warmup()
    tensors = np.load(args.evaluation_tensors, mmap_mode="r")
    rows = [
        json.loads(line)
        for line in args.evaluation_records.read_text(encoding="utf-8").splitlines()
        if line
    ]
    predictions = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    indices_by_image: defaultdict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_image[int(row["image_id"])].append(index)
    logits = np.zeros((len(rows), len(metadata.labels)), dtype=np.float32)
    ranking = np.zeros_like(logits)
    approval = np.zeros(len(rows), dtype=np.float32)
    safety = np.zeros(len(rows), dtype=np.float32)
    for image_id, indices in indices_by_image.items():
        prediction = predictions[image_id]
        detections = [
            Detection(*box, float(score), int(class_id))
            for box, score, class_id in zip(
                prediction["boxes_xyxy"],
                prediction["scores"],
                prediction["class_ids"],
            )
        ]
        detection_indices = [int(rows[index]["detection_index"]) for index in indices]
        if detection_indices != list(range(len(detections))):
            raise ValueError(f"classifier rows for image {image_id} are not detection-ordered")
        result = classifier._neighbor_mask_classify(
            np.asarray(tensors[indices], dtype=np.float32),
            detections,
            image_width=int(args.image_width_by_id.get(image_id, 0)),
            image_height=int(args.image_height_by_id.get(image_id, 0)),
        )
        logits[indices] = result.logits
        ranking[indices] = result.ranking_logits
        approval[indices] = result.approval_scores
        safety[indices] = result.top3_safety_scores
    states = _states(
        approval,
        safety,
        approval_threshold=metadata.approval_threshold,
        safety_threshold=policy.top3_safety_threshold,
    )
    top1 = np.argmax(logits, axis=1)
    top3 = np.argsort(-ranking, axis=1, kind="stable")[:, :3]
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    excluded = set(args.evaluation_exclude_image_ids)
    evaluation_mask = np.asarray([int(row["image_id"]) not in excluded for row in rows], dtype=bool)
    difficulty = np.asarray(
        [args.difficulty_by_id[int(row["image_id"])] for row in rows], dtype=object
    )
    report = {
        "schema_version": "1.0",
        "evaluation": (
            "bread_neighbor_mask_pytorch_onnx_parity"
            if args.reference_logits is not None
            else "bread_neighbor_mask_onnx_outcomes"
        ),
        "provider": args.provider,
        "sample_count": len(rows),
        "image_count": len(indices_by_image),
        "reference_logits": str(args.reference_logits) if args.reference_logits else None,
        "state_counts": {
            "approved": int(np.count_nonzero(states == 0)),
            "unknown": int(np.count_nonzero(states == 1)),
            "segment_recapture": int(np.count_nonzero(states == 2)),
        },
        "outcomes_after_image_gates": _outcome_counts(
            mask=evaluation_mask,
            states=states,
            top1=top1,
            top3=top3,
            targets=targets,
        ),
        "outcomes_by_difficulty": {
            level: _outcome_counts(
                mask=evaluation_mask & (difficulty == level),
                states=states,
                top1=top1,
                top3=top3,
                targets=targets,
            )
            for level in ("EASY", "MEDIUM", "HARD", "SCAN_LOG")
        },
        "evaluation_excluded_image_ids": sorted(excluded),
    }
    if args.reference_logits is not None:
        reference_payload = np.load(args.reference_logits)
        weights = np.asarray([view.weight for view in policy.views], dtype=np.float32)
        expected_logits, expected_ranking, expected_approval, expected_safety = _weighted_reference(
            reference_payload,
            weights,
            metadata.temperature,
            policy.ranking_tie_break_bias_span,
        )
        expected_states = _states(
            expected_approval,
            expected_safety,
            approval_threshold=metadata.approval_threshold,
            safety_threshold=policy.top3_safety_threshold,
        )
        expected_top1 = np.argmax(expected_logits, axis=1)
        expected_top3 = np.argsort(-expected_ranking, axis=1, kind="stable")[:, :3]
        report.update(
            {
                "maximum_absolute_logit_difference": float(
                    np.max(np.abs(logits - expected_logits))
                ),
                "maximum_absolute_ranking_difference": float(
                    np.max(np.abs(ranking - expected_ranking))
                ),
                "maximum_absolute_approval_difference": float(
                    np.max(np.abs(approval - expected_approval))
                ),
                "maximum_absolute_safety_difference": float(
                    np.max(np.abs(safety - expected_safety))
                ),
                "top1_equal": bool(np.array_equal(top1, expected_top1)),
                "top3_equal": bool(np.array_equal(top3, expected_top3)),
                "state_equal": bool(np.array_equal(states, expected_states)),
            }
        )
    if args.cross_provider_reference_arrays is not None:
        cross_reference = np.load(args.cross_provider_reference_arrays)
        report["cross_provider"] = {
            "reference_arrays": str(args.cross_provider_reference_arrays),
            "maximum_absolute_logit_difference": float(
                np.max(np.abs(logits - cross_reference["logits"]))
            ),
            "maximum_absolute_ranking_difference": float(
                np.max(np.abs(ranking - cross_reference["ranking_logits"]))
            ),
            "top1_equal": bool(np.array_equal(top1, cross_reference["top1"])),
            "top3_equal": bool(np.array_equal(top3, cross_reference["top3"])),
            "state_equal": bool(np.array_equal(states, cross_reference["states"])),
        }
    args.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_arrays,
        logits=logits,
        ranking_logits=ranking,
        approval_scores=approval,
        top3_safety_scores=safety,
        states=states,
        top1=top1,
        top3=top3,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate neighbor-mask PyTorch/ONNX parity")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument(
        "--reference-logits",
        type=Path,
        help="Optional canonical PyTorch view logits; omit for ONNX-only outcome evaluation",
    )
    parser.add_argument("--evaluation-exclude-image-ids", type=int, nargs="*", default=())
    parser.add_argument("--provider", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--output-arrays", type=Path, required=True)
    parser.add_argument("--cross-provider-reference-arrays", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_rows = [
        json.loads(line)
        for line in args.detector_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    args.image_width_by_id = {int(row["image_id"]): int(row["width"]) for row in manifest_rows}
    args.image_height_by_id = {int(row["image_id"]): int(row["height"]) for row in manifest_rows}
    args.difficulty_by_id = {
        int(row["image_id"]): str(row["difficulty"]).upper() for row in manifest_rows
    }
    evaluate(args)


if __name__ == "__main__":
    main()
