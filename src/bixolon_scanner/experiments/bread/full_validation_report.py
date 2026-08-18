from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ...evaluation.detector import _metrics
from ...training.data import read_manifest


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_gate_summary(
    *,
    raw_detector_metrics: dict[str, Any],
    accepted_detector_metrics: dict[str, Any],
    classifier_metrics: dict[str, Any],
    image_quality_metrics: dict[str, Any],
    total_image_count: int,
    annotated_image_count: int,
    total_ground_truth_object_count: int,
    total_image_recapture_count: int,
    false_annotated_image_recapture_count: int,
    minimum_segmentation_image_rate: float,
    minimum_approved_rate: float,
    target_end_to_end_approved_rate: float,
    maximum_segmentation_image_false_negative_rate: float,
    maximum_segmentation_image_false_positive_rate: float,
    maximum_approved_misrecognition_rate: float,
    maximum_unknown_top3_candidate_out_rate: float,
) -> dict[str, Any]:
    overall_image_recapture_rate = _rate(total_image_recapture_count, total_image_count)
    segmentation_image_count = total_image_count - total_image_recapture_count
    segmentation_image_rate = _rate(segmentation_image_count, total_image_count)
    false_annotated_image_recapture_rate = _rate(
        false_annotated_image_recapture_count, annotated_image_count
    )
    false_negative_image_rate = _rate(
        accepted_detector_metrics["false_negative_image_count"], segmentation_image_count
    )
    false_positive_image_rate = _rate(
        accepted_detector_metrics["false_positive_image_count"], segmentation_image_count
    )
    approved_rate = _rate(classifier_metrics["approved_count"], total_ground_truth_object_count)
    approved_misrecognition_rate = _rate(
        classifier_metrics["approved_misrecognition_count"], total_ground_truth_object_count
    )
    candidate_out_rate = _rate(
        classifier_metrics["unknown_top3_candidate_out_count"],
        total_ground_truth_object_count,
    )
    official_gates = {
        "segmentation_image_rate": segmentation_image_rate >= minimum_segmentation_image_rate,
        "end_to_end_approved_object_rate": approved_rate >= minimum_approved_rate,
        "segmentation_image_false_negative_rate": (
            false_negative_image_rate <= maximum_segmentation_image_false_negative_rate
        ),
        "segmentation_image_false_positive_rate": (
            false_positive_image_rate <= maximum_segmentation_image_false_positive_rate
        ),
        "approved_object_misrecognition_rate": (
            approved_misrecognition_rate <= maximum_approved_misrecognition_rate
        ),
        "unknown_top3_candidate_out_rate": (
            candidate_out_rate <= maximum_unknown_top3_candidate_out_rate
        ),
    }
    return {
        "counts": {
            "total_image_count": total_image_count,
            "annotated_image_count": annotated_image_count,
            "total_ground_truth_object_count": total_ground_truth_object_count,
            "segmentation_image_count": segmentation_image_count,
            "total_image_recapture_count": total_image_recapture_count,
            "false_annotated_image_recapture_count": (false_annotated_image_recapture_count),
        },
        "rates": {
            "segmentation_image_rate": segmentation_image_rate,
            "overall_image_recapture_rate": overall_image_recapture_rate,
            "false_annotated_image_recapture_rate": (false_annotated_image_recapture_rate),
            "segmentation_image_false_negative_rate": false_negative_image_rate,
            "segmentation_image_false_positive_rate": false_positive_image_rate,
            "end_to_end_approved_object_rate": approved_rate,
            "approved_object_misrecognition_rate": approved_misrecognition_rate,
            "unknown_top3_candidate_out_rate": candidate_out_rate,
            "unknown_rate_diagnostic_only": classifier_metrics["unknown_rate"],
            "segment_recapture_rate": classifier_metrics["segment_recapture_rate"],
        },
        "limits": {
            "minimum_segmentation_image_rate": minimum_segmentation_image_rate,
            "minimum_approved_rate": minimum_approved_rate,
            "target_end_to_end_approved_rate": target_end_to_end_approved_rate,
            "maximum_segmentation_image_false_negative_rate": (
                maximum_segmentation_image_false_negative_rate
            ),
            "maximum_segmentation_image_false_positive_rate": (
                maximum_segmentation_image_false_positive_rate
            ),
            "maximum_approved_misrecognition_rate": maximum_approved_misrecognition_rate,
            "maximum_unknown_top3_candidate_out_rate": (maximum_unknown_top3_candidate_out_rate),
        },
        "official_gates": official_gates,
        "diagnostics": {
            "raw_detector_false_positive_count": raw_detector_metrics["false_positive_count"],
            "raw_detector_false_negative_count": raw_detector_metrics["false_negative_count"],
            "expected_image_recapture_recall": image_quality_metrics["recapture_recall"],
            "unknown_and_segment_recapture_rates_are_not_promotion_gates": True,
        },
        "raw_detector_aspiration_met": bool(
            raw_detector_metrics["false_positive_count"] == 0
            and raw_detector_metrics["false_negative_count"] == 0
        ),
        "operational_gates_met": all(official_gates.values()),
        "final_end_to_end_approved_goal_met": approved_rate >= target_end_to_end_approved_rate,
        "finite_development_pipeline_goals_met": (
            all(official_gates.values()) and approved_rate >= target_end_to_end_approved_rate
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    metadata = _load_json(args.metadata)
    detector_report = _load_json(args.detector_report)
    image_quality_report = _load_json(args.image_quality_report)
    classifier_report = _load_json(args.classifier_report)
    records = [
        record
        for record in read_manifest(args.detector_manifest)
        if record["record_type"] == "detection" and record["split"] == "development"
    ]
    if len(records) != int(metadata["detector"]["image_count"]):
        raise ValueError("detector manifest image count does not match metadata")

    annotated_records = [
        record for record in records if record["expected_image_status"] == "ANNOTATED"
    ]
    expected_recapture_records = [
        record for record in records if record["expected_image_status"] == "RECAPTURE"
    ]
    annotated_ids = {int(record["image_id"]) for record in annotated_records}
    expected_recapture_ids = {int(record["image_id"]) for record in expected_recapture_records}
    quality_metrics = image_quality_report["reason_conjunction_policy"]["pooled_oof"]
    quality_missed_ids = {
        int(image_id) for image_id in quality_metrics["missed_recapture_image_ids"]
    }
    quality_false_ids = {int(image_id) for image_id in quality_metrics["false_recapture_image_ids"]}
    if not quality_missed_ids <= expected_recapture_ids:
        raise ValueError("image quality misses contain a non-recapture image")
    if not quality_false_ids <= annotated_ids:
        raise ValueError("image quality false recaptures contain an unannotated image")
    quality_recapture_ids = (expected_recapture_ids - quality_missed_ids) | quality_false_ids

    disagreement = detector_report["disagreement_recapture_diagnostic"]
    detector_recapture_ids = {int(image_id) for image_id in disagreement["recaptured_image_ids"]}
    if not detector_recapture_ids <= annotated_ids:
        raise ValueError("detector disagreement gate contains an unannotated image")
    if quality_recapture_ids & detector_recapture_ids:
        raise ValueError("quality and detector recapture routes overlap")

    predictions = _load_jsonl(args.detector_predictions)
    predictions_by_id = {int(row["image_id"]): row for row in predictions}
    if len(predictions_by_id) != len(predictions):
        raise ValueError("detector predictions contain duplicate image ids")
    if set(predictions_by_id) != annotated_ids:
        raise ValueError("detector prediction ids do not match annotated manifest ids")

    excluded_annotated_ids = quality_false_ids | detector_recapture_ids
    accepted_records = [
        record
        for record in annotated_records
        if int(record["image_id"]) not in excluded_annotated_ids
    ]
    accepted_predictions = [
        predictions_by_id[int(record["image_id"])] for record in accepted_records
    ]
    accepted_detector_metrics = _metrics(
        accepted_records,
        accepted_predictions,
        score_threshold=0.0,
        nms_iou_threshold=1.0,
        match_iou_threshold=0.5,
        max_queries=600,
    )
    per_image_metrics = [
        _metrics(
            [record],
            [prediction],
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        for record, prediction in zip(accepted_records, accepted_predictions)
    ]
    accepted_detector_metrics["false_positive_image_count"] = sum(
        metrics["false_positive_count"] > 0 for metrics in per_image_metrics
    )
    accepted_detector_metrics["false_negative_image_count"] = sum(
        metrics["false_negative_count"] > 0 for metrics in per_image_metrics
    )

    classifier_excluded_ids = {
        int(image_id) for image_id in classifier_report["evaluation_excluded_image_ids"]
    }
    if classifier_excluded_ids != quality_false_ids:
        raise ValueError("classifier subset does not match image quality exclusions")
    classifier_selection = classifier_report["selected"]
    classifier_metrics = classifier_selection["grouped_oof"]["metrics"]
    if classifier_metrics["sample_count"] != accepted_detector_metrics["matched_count"]:
        raise ValueError("classifier sample count does not match accepted detector matches")

    total_image_recapture_ids = quality_recapture_ids | detector_recapture_ids
    false_annotated_image_recapture_ids = quality_false_ids | detector_recapture_ids
    summary = build_gate_summary(
        raw_detector_metrics=detector_report["selected"]["metrics"],
        accepted_detector_metrics=accepted_detector_metrics,
        classifier_metrics=classifier_metrics,
        image_quality_metrics=quality_metrics,
        total_image_count=len(records),
        annotated_image_count=len(annotated_records),
        total_ground_truth_object_count=sum(
            len(record.get("annotations", [])) for record in records
        ),
        total_image_recapture_count=len(total_image_recapture_ids),
        false_annotated_image_recapture_count=len(false_annotated_image_recapture_ids),
        minimum_segmentation_image_rate=args.minimum_segmentation_image_rate,
        minimum_approved_rate=args.minimum_approved_rate,
        target_end_to_end_approved_rate=args.target_end_to_end_approved_rate,
        maximum_segmentation_image_false_negative_rate=(
            args.maximum_segmentation_image_false_negative_rate
        ),
        maximum_segmentation_image_false_positive_rate=(
            args.maximum_segmentation_image_false_positive_rate
        ),
        maximum_approved_misrecognition_rate=args.maximum_approved_misrecognition_rate,
        maximum_unknown_top3_candidate_out_rate=(args.maximum_unknown_top3_candidate_out_rate),
    )
    decisions = []
    for record in records:
        image_id = int(record["image_id"])
        if image_id in quality_recapture_ids:
            predicted_status = "IMAGE_RECAPTURE"
            decision_source = "pooled_reason_quality_policy"
        elif image_id in detector_recapture_ids:
            predicted_status = "IMAGE_RECAPTURE"
            decision_source = "detector_candidate_disagreement"
        else:
            predicted_status = "SEGMENTATION"
            decision_source = "accepted_detector"
        decisions.append(
            {
                "image_id": image_id,
                "fold": int(record["fold"]),
                "expected_image_status": record["expected_image_status"],
                "expected_reason_codes": record.get("expected_reason_codes", []),
                "predicted_status": predicted_status,
                "decision_source": decision_source,
                "ground_truth_object_count": len(record.get("annotations", [])),
            }
        )

    artifact_paths = {
        "metadata": args.metadata,
        "detector_manifest": args.detector_manifest,
        "detector_report": args.detector_report,
        "detector_predictions": args.detector_predictions,
        "image_quality_report": args.image_quality_report,
        "classifier_report": args.classifier_report,
    }
    report = {
        "schema_version": "1.0",
        "experiment": "bread-zero-error-1.1.0",
        "lifecycle": "active",
        "dataset_version": metadata["dataset_version"],
        "evaluation_scope": {
            "held_out_test_set": False,
            "selection": "all-available grouped OOF with pooled policy selection",
            "classifier_source": metadata["classifier"]["selected_source"],
            "mixed_classifier_sources": metadata["classifier"]["mixed_sources"],
        },
        "image_quality_gate": {
            **{
                key: quality_metrics[key]
                for key in (
                    "recapture_sample_count",
                    "normal_sample_count",
                    "true_recapture_count",
                    "recapture_recall",
                    "false_recapture_count",
                    "false_recapture_rate",
                    "total_image_recapture_count",
                    "total_image_recapture_rate",
                    "by_reason",
                    "missed_recapture_image_ids",
                    "false_recapture_image_ids",
                )
            },
            "nested_diagnostic_pass": image_quality_report["target_gate"][
                "reason_nested_diagnostic_pass"
            ],
        },
        "detector": {
            "raw_metrics": detector_report["selected"]["metrics"],
            "disagreement_recapture": disagreement,
            "accepted_after_all_image_gates": accepted_detector_metrics,
        },
        "classifier": {
            "policy": classifier_selection["policy"],
            "selection": classifier_report["selection"],
            "after_all_image_gates": classifier_metrics,
        },
        "pipeline_summary": summary,
        "routing": {
            "quality_image_recapture_ids": sorted(quality_recapture_ids),
            "detector_disagreement_recapture_ids": sorted(detector_recapture_ids),
            "false_annotated_image_recapture_ids": sorted(false_annotated_image_recapture_ids),
            "segmentation_image_count": len(records) - len(total_image_recapture_ids),
        },
        "promotion": {
            "ready": False,
            "blockers": [
                "raw detector still has three false negatives before the disagreement gate",
                "grouped development OOF was used for classifier policy selection",
                "the nested image-quality diagnostic does not meet the target",
                "the selected policies are not integrated into a versioned ONNX Worker package",
                "CPU/CUDA final-state parity and RTX 5080 full-path latency are not validated",
            ],
        },
        "evidence": {
            name: {"path": path.as_posix(), "sha256": _sha256_file(path)}
            for name, path in artifact_paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.decisions_output:
        args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
        args.decisions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in decisions),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose the bread zero-error full development validation report"
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--detector-report", type=Path, required=True)
    parser.add_argument("--detector-predictions", type=Path, required=True)
    parser.add_argument("--image-quality-report", type=Path, required=True)
    parser.add_argument("--classifier-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path)
    parser.add_argument("--minimum-segmentation-image-rate", type=float, default=0.90)
    parser.add_argument("--minimum-approved-rate", type=float, default=0.90)
    parser.add_argument("--target-end-to-end-approved-rate", type=float, default=0.99)
    parser.add_argument(
        "--maximum-segmentation-image-false-negative-rate", type=float, default=0.001
    )
    parser.add_argument(
        "--maximum-segmentation-image-false-positive-rate", type=float, default=0.001
    )
    parser.add_argument("--maximum-approved-misrecognition-rate", type=float, default=0.001)
    parser.add_argument("--maximum-unknown-top3-candidate-out-rate", type=float, default=0.001)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
