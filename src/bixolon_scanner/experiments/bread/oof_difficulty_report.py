from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...evaluation.detector import _metrics
from ...training.data import read_manifest
from .zero_error_classifier import _guarded_threshold, policy_candidates, select_policy


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _classifier_oof_decisions(
    logits_path: Path,
    records_path: Path,
    *,
    guard_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    payload = np.load(logits_path)
    targets = payload["targets"].astype(np.int64)
    logits = {name: payload[name].astype(np.float32) for name in payload.files if name != "targets"}
    rows = _load_jsonl(records_path)
    if len(rows) != len(targets):
        raise ValueError("classifier records and logits have different lengths")
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    candidates = policy_candidates(logits)
    approved = np.zeros(len(targets), dtype=bool)
    predictions = np.zeros(len(targets), dtype=np.int64)
    top3 = np.zeros((len(targets), 3), dtype=np.int64)
    safety_scores = np.zeros(len(targets), dtype=np.float32)
    for held_out_fold in sorted(set(folds.tolist())):
        calibration = folds != held_out_fold
        held_out = folds == held_out_fold
        policy, approval_threshold, _, _ = select_policy(
            candidates,
            targets,
            calibration,
            guard_samples,
        )
        approved[held_out] = policy.approval_score[held_out] >= approval_threshold
        predictions[held_out] = policy.predictions[held_out]
        top3[held_out] = policy.top3[held_out]
        safety_scores[held_out] = policy.top3_safety_score[held_out]
    top3_correct = np.any(top3 == targets[:, None], axis=1)
    pooled_threshold = _guarded_threshold(
        safety_scores,
        ~approved & ~top3_correct,
        np.ones(len(targets), dtype=bool),
        0,
    )
    unknown = ~approved & (safety_scores >= pooled_threshold)
    segment_recapture = ~approved & ~unknown
    return rows, {
        "targets": targets,
        "approved": approved,
        "predictions": predictions,
        "top3_correct": top3_correct,
        "unknown": unknown,
        "segment_recapture": segment_recapture,
        "pooled_safety_threshold": np.asarray(pooled_threshold),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_difficulty_bucket(
    *,
    image_count: int,
    image_recapture_count: int,
    ground_truth_count: int,
    accepted_detector_metrics: dict[str, Any],
    raw_detector_metrics: dict[str, Any],
    approved_count: int,
    approved_error_count: int,
    unknown_count: int,
    unknown_top3_miss_count: int,
    segment_recapture_count: int,
) -> dict[str, Any]:
    segmentation_image_count = image_count - image_recapture_count
    segment_count = approved_count + unknown_count + segment_recapture_count
    return {
        "images": {
            "total": image_count,
            "segmentation": segmentation_image_count,
            "segmentation_rate": _rate(segmentation_image_count, image_count),
            "image_recapture": image_recapture_count,
            "image_recapture_rate": _rate(image_recapture_count, image_count),
        },
        "detector": {
            "raw": raw_detector_metrics,
            "accepted": accepted_detector_metrics,
            "fn_per_segmentation_image": _rate(
                int(accepted_detector_metrics["false_negative_count"]),
                segmentation_image_count,
            ),
            "fp_per_segmentation_image": _rate(
                int(accepted_detector_metrics["false_positive_count"]),
                segmentation_image_count,
            ),
        },
        "classifier": {
            "segment_count": segment_count,
            "ground_truth_count": ground_truth_count,
            "approved": approved_count,
            "end_to_end_approved_rate": _rate(approved_count, ground_truth_count),
            "accepted_matched_approved_rate_diagnostic": _rate(approved_count, segment_count),
            "approved_error_count": approved_error_count,
            "approved_misrecognition_rate": _rate(approved_error_count, ground_truth_count),
            "unknown": unknown_count,
            "unknown_rate": _rate(unknown_count, segment_count),
            "unknown_top3_miss_count": unknown_top3_miss_count,
            "unknown_candidate_out_rate": _rate(unknown_top3_miss_count, ground_truth_count),
            "segment_recapture": segment_recapture_count,
            "segment_recapture_rate": _rate(segment_recapture_count, segment_count),
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_rows = [
        row
        for row in read_manifest(args.detector_manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and row["evaluation_set"] == "multi_object_scenes"
    ]
    detector_report = _load_json(args.detector_report)
    predictions_by_id = {
        int(row["image_id"]): row for row in _load_jsonl(args.detector_predictions)
    }
    recaptured_ids = {
        int(value)
        for value in detector_report["disagreement_recapture_diagnostic"]["recaptured_image_ids"]
    }
    classifier_rows, decisions = _classifier_oof_decisions(
        args.classifier_logits,
        args.classifier_records,
        guard_samples=args.guard_samples,
    )
    difficulty_by_image = {
        int(row["image_id"]): str(row["difficulty"])[0].upper() for row in manifest_rows
    }
    classifier_indices = {
        difficulty: np.asarray(
            [
                difficulty_by_image.get(int(row["image_id"])) == difficulty
                for row in classifier_rows
            ],
            dtype=bool,
        )
        for difficulty in ("E", "M", "H")
    }
    latency_baseline = _load_json(args.latency_baseline)
    by_difficulty: dict[str, Any] = {}
    for difficulty in ("E", "M", "H"):
        records = [row for row in manifest_rows if str(row["difficulty"])[0].upper() == difficulty]
        predictions = [predictions_by_id[int(row["image_id"])] for row in records]
        raw_metrics = _metrics(
            records,
            predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        accepted_records = [row for row in records if int(row["image_id"]) not in recaptured_ids]
        accepted_predictions = [predictions_by_id[int(row["image_id"])] for row in accepted_records]
        accepted_metrics = _metrics(
            accepted_records,
            accepted_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        mask = classifier_indices[difficulty]
        approved = mask & decisions["approved"]
        unknown = mask & decisions["unknown"]
        segment_recapture = mask & decisions["segment_recapture"]
        summary = summarize_difficulty_bucket(
            image_count=len(records),
            image_recapture_count=sum(int(row["image_id"]) in recaptured_ids for row in records),
            ground_truth_count=sum(len(row.get("annotations", [])) for row in records),
            accepted_detector_metrics=accepted_metrics,
            raw_detector_metrics=raw_metrics,
            approved_count=int(approved.sum()),
            approved_error_count=int(
                np.count_nonzero(approved & (decisions["predictions"] != decisions["targets"]))
            ),
            unknown_count=int(unknown.sum()),
            unknown_top3_miss_count=int(np.count_nonzero(unknown & ~decisions["top3_correct"])),
            segment_recapture_count=int(segment_recapture.sum()),
        )
        summary["latency"] = {
            "worker_1_1_0_cuda": None,
            "worker_1_0_0_cuda_baseline": latency_baseline["by_difficulty"][difficulty][
                "end_to_end_latency_ms"
            ],
        }
        by_difficulty[difficulty] = summary

    report = {
        "schema_version": "1.0",
        "experiment": "bread-zero-error-1.1.0",
        "evaluation": "multi-object grouped-OOF difficulty report",
        "dataset": {
            "path": str(args.dataset_root.resolve()),
            "image_count": 300,
            "ground_truth_count": 1410,
            "exact_sha_overlap_with_development": "300/300",
            "independent_test": False,
        },
        "policy": {
            "classifier_source": "single_objects",
            "guard_samples": args.guard_samples,
            "pooled_safety_threshold": float(decisions["pooled_safety_threshold"]),
        },
        "by_difficulty": by_difficulty,
        "promotion": {
            "requested": True,
            "status": "blocked",
            "blockers": [
                "No versioned bread-worker-1.1.0 package exists.",
                "The count model and image-quality policy are not integrated into Worker inference.",
                "The OOF count models are not serialized as one final inference model.",
                "Worker 1.1.0 CPU/CUDA state parity has not been measured.",
                "Worker 1.1.0 E/M/H latency has not been measured.",
                "The same 300 images participate in development and policy selection.",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose bread 1.1 grouped-OOF metrics by E/M/H difficulty"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--detector-report", type=Path, required=True)
    parser.add_argument("--detector-predictions", type=Path, required=True)
    parser.add_argument("--classifier-logits", type=Path, required=True)
    parser.add_argument("--classifier-records", type=Path, required=True)
    parser.add_argument("--latency-baseline", type=Path, required=True)
    parser.add_argument("--guard-samples", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
