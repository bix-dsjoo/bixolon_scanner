from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from ...contracts.model_package import load_model_package, sha256_file
from ...training.calibration import binomial_rate_upper_bound


def build(args: argparse.Namespace) -> dict[str, Any]:
    base = load_model_package(args.base_package)
    base_metadata = json.loads((args.base_package / "metadata.json").read_text(encoding="utf-8"))
    dataset_metadata = json.loads(args.dataset_metadata.read_text(encoding="utf-8"))
    classifier_dataset = dataset_metadata["classifier"]
    if classifier_dataset["selected_source"] != "single_objects":
        raise ValueError("neighbor-mask candidate requires the locked single_objects source")
    if classifier_dataset["mixed_sources"]:
        raise ValueError("neighbor-mask candidate cannot mix classifier sources")
    policy_report = json.loads(args.policy_report.read_text(encoding="utf-8"))
    policy = policy_report["selected"]
    expected_policy = (
        "normalized_bias0.000@0.75+normalized_bias0.250@0.25:"
        "approval=margin:top3=weighted_reciprocal_rank:safety=inverse_entropy"
    )
    if policy["policy"] != expected_policy:
        raise ValueError("package builder and selected neighbor-mask policy differ")
    if float(policy_report.get("ranking_tie_break_bias_span", 0.0)) != 0.0002:
        raise ValueError("package builder and ranking tie-break policy differ")
    metrics = policy["evaluation_after_image_gates"]
    if metrics["approved_error_count"] != 0 or metrics["unknown_top3_miss_count"] != 0:
        raise ValueError("neighbor-mask policy does not satisfy the zero-error count gates")
    if metrics["segment_recapture_rate"] > 0.05:
        raise ValueError("neighbor-mask policy exceeds the five-percent recapture target")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    classifier_sha256 = sha256_file(args.classifier_onnx)
    metadata = dict(base_metadata)
    metadata["schema_version"] = "1.1"
    metadata["worker_version"] = args.worker_version
    metadata.pop("package_version", None)
    metadata["promotion_status"] = "development"
    metadata.pop("promotion", None)
    metadata["dataset_version"] = dataset_metadata["dataset_version"]
    classifier = dict(metadata["classifier"])
    classifier.update(
        {
            "version": args.classifier_version,
            "approval_threshold": float(policy["approval_threshold"]),
            "temperature": 1.0,
            "crop_mode": "box_resize",
            "neighbor_mask_inference": {
                "views": [
                    {
                        "name": "normalized_bias0.000",
                        "distance_bias": 0.0,
                        "weight": 0.75,
                        "shared_scale": False,
                    },
                    {
                        "name": "normalized_bias0.250",
                        "distance_bias": 0.25,
                        "weight": 0.25,
                        "shared_scale": False,
                    },
                ],
                "approval_metric": "margin",
                "ranking_aggregation": "weighted_reciprocal_rank",
                "top3_safety_metric": "inverse_entropy",
                "top3_safety_threshold": float(policy["top3_safety_threshold"]),
                "logit_quantum": None,
                "logit_phase": 0.0,
                "tie_break_bias_span": 0.04,
                "ranking_tie_break_bias_span": 0.0002,
            },
        }
    )
    classifier.pop("staged_inference", None)
    metadata["classifier"] = classifier
    metadata["sources"] = dict(metadata.get("sources", {}))
    metadata["sources"]["classifier"] = {
        "architecture": "DINOv3 ConvNeXt-Tiny + detector-neighbor ownership masks",
        "revision": base.metadata.sources["classifier"].revision,
        "weight_filename": args.checkpoint.name,
        "weight_sha256": checkpoint_sha256,
        "training_dataset_version": classifier_dataset["source_dataset_version"],
        "training_manifest_sha256": classifier_dataset["manifest_sha256"],
    }
    approved_count = int(metrics["approved_count"])
    false_upper = binomial_rate_upper_bound(0, approved_count)
    metadata["calibration"] = {
        "sample_count": int(metrics["sample_count"]),
        "approved_precision": 1.0,
        "approval_coverage": float(metrics["approved_rate"]),
        "false_approval_rate_upper_95": false_upper,
        "risk_control_satisfied": false_upper <= args.maximum_false_approval_rate,
    }
    metadata["checksums"] = dict(metadata["checksums"])
    metadata["checksums"][classifier["filename"]] = classifier_sha256

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base.detector_path, args.output_dir / base.detector_path.name)
    shutil.copy2(args.classifier_onnx, args.output_dir / classifier["filename"])
    if base.count_verifier_path is not None:
        shutil.copy2(base.count_verifier_path, args.output_dir / base.count_verifier_path.name)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    package = load_model_package(args.output_dir)
    promotion_blockers = []
    if not metadata["calibration"]["risk_control_satisfied"]:
        promotion_blockers.append("approved_false_rate_upper_95_exceeds_0.1_percent")
    if package.metadata.detector.version != args.detector_version:
        promotion_blockers.append("deployable_detector_1.1_policy_not_packaged")
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_neighbor_mask_candidate_package",
        "package": str(args.output_dir),
        "worker_version": package.metadata.worker_version,
        "detector_version": package.metadata.detector.version,
        "classifier_version": package.metadata.classifier.version,
        "classifier_source": classifier_dataset["selected_source"],
        "mixed_classifier_sources": classifier_dataset["mixed_sources"],
        "classifier_onnx_sha256": classifier_sha256,
        "classifier_checkpoint_sha256": checkpoint_sha256,
        "selected_policy_metrics": metrics,
        "calibration": metadata["calibration"],
        "promotion_status": package.metadata.promotion_status,
        "inherited_base_detector": True,
        "selected_development_detector_equivalent": False,
        "promotion_blockers": promotion_blockers,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a development neighbor-mask package")
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--classifier-onnx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-metadata", type=Path, required=True)
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--worker-version", default="1.1.0")
    parser.add_argument("--classifier-version", default="1.1.0")
    parser.add_argument("--detector-version", default="1.1.0")
    parser.add_argument("--maximum-false-approval-rate", type=float, default=0.001)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
