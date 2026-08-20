from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

from ..configuration import load_json_config
from ..package import load_model_package, sha256_file
from .fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
    wrap_inference_classifier,
)
from .models import require_torch
from .ten_shot_candidates import verify_experiment_lock


def _frozen_detector_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "detector": metadata["detector"],
        "quality": metadata["quality"],
        "input": metadata["input"],
    }


def verify_frozen_detector_package(
    base_package: Path,
    *,
    expected_detector_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    package = load_model_package(base_package)
    metadata = json.loads((base_package / "metadata.json").read_text(encoding="utf-8"))
    digest = sha256_file(package.detector_path)
    recorded = metadata.get("checksums", {}).get(package.detector_path.name)
    if recorded != digest:
        raise ValueError("base package detector checksum metadata does not match the file")
    if expected_detector_sha256 is not None and digest != expected_detector_sha256:
        raise ValueError("base package detector does not match the locked detector checksum")
    return metadata, digest


def build_ten_shot_metadata(
    *,
    base_metadata: dict[str, Any],
    manifest_metadata: dict[str, Any],
    calibration: dict[str, Any],
    classifier_sha256: str,
    classifier_version: str,
    package_version: str,
    checkpoint: dict[str, Any],
    promotion_status: str = "experiment_only",
    manual_waiver_approved: bool = False,
    inference_center_crop_scale: float | None = None,
    inference_logit_recipe: str | None = None,
) -> dict[str, Any]:
    if checkpoint.get("dataset_version") != manifest_metadata.get("dataset_version"):
        raise ValueError("classifier checkpoint dataset version does not match manifest")
    if checkpoint.get("manifest_sha256") != manifest_metadata.get("manifest_sha256"):
        raise ValueError("classifier checkpoint manifest checksum does not match manifest")
    labels = [
        {
            "class_id": str(label["class_id"]),
            "class_name": str(label["class_name"]),
            "recapture": False,
        }
        for label in sorted(manifest_metadata["labels"], key=lambda row: int(row["category_id"]))
    ]
    if len(labels) != int(checkpoint["num_classes"]):
        raise ValueError("classifier class count does not match manifest labels")
    required_calibration = (
        "approval_threshold",
        "temperature",
        "sample_count",
        "approved_precision",
        "approval_coverage",
        "approved_false_rate_upper_95",
        "risk_control_satisfied",
    )
    missing = [key for key in required_calibration if key not in calibration]
    if missing:
        raise ValueError(f"calibration report is missing: {missing}")
    metadata = copy.deepcopy(base_metadata)
    metadata["package_version"] = package_version
    if promotion_status not in {"production", "manual_waiver", "experiment_only"}:
        raise ValueError("invalid 0.2.0 promotion status")
    # The public package schema remains unchanged: a waiver candidate is kept
    # as development until a human creates the existing PromotionMetadata.
    deployable = promotion_status == "production" or (
        promotion_status == "manual_waiver" and manual_waiver_approved
    )
    metadata["promotion_status"] = "production" if deployable else "development"
    metadata["dataset_version"] = manifest_metadata["dataset_version"]
    metadata["classifier"] = {
        "filename": "classifier.onnx",
        "version": classifier_version,
        "input_name": "pixel_values",
        "logits_output": "logits",
        "input_size": [int(checkpoint["image_size"]), int(checkpoint["image_size"])],
        "color_order": "RGB",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "crop_margin_ratio": base_metadata["classifier"]["crop_margin_ratio"],
        "approval_threshold": float(calibration["approval_threshold"]),
        "temperature": float(calibration["temperature"]),
        "labels": labels,
        "resize_reducing_gap": base_metadata["classifier"]["resize_reducing_gap"],
        "warmup_batch_sizes": base_metadata["classifier"]["warmup_batch_sizes"],
    }
    detector_filename = metadata["detector"]["filename"]
    metadata["checksums"] = {
        detector_filename: base_metadata["checksums"][detector_filename],
        "classifier.onnx": classifier_sha256,
    }
    metadata["sources"] = copy.deepcopy(base_metadata.get("sources", {}))
    architecture = checkpoint["architecture"]
    if inference_center_crop_scale is not None:
        architecture += f"+center_crop_resize_{inference_center_crop_scale:.2f}"
    if inference_logit_recipe is not None:
        architecture += f"+{inference_logit_recipe}"
    metadata["sources"]["classifier"] = {
        "architecture": architecture,
        "revision": checkpoint["source_revision"],
        "weight_filename": checkpoint["source_weight_filename"],
        "weight_sha256": checkpoint["source_weight_sha256"],
    }
    metadata["calibration"] = {
        "sample_count": int(calibration["sample_count"]),
        "approved_precision": float(calibration["approved_precision"]),
        "approval_coverage": float(calibration["approval_coverage"]),
        "false_approval_rate_upper_95": float(calibration["approved_false_rate_upper_95"]),
        "risk_control_satisfied": bool(calibration["risk_control_satisfied"]),
    }
    metadata.pop("promotion", None)
    if promotion_status == "production":
        from datetime import UTC, datetime

        metadata["promotion"] = {
            "decision": "approved",
            "method": "all_gates",
            "decided_on": datetime.now(UTC).date().isoformat(),
            "waivers": [],
            "remaining_limitations": [
                "test 및 bread_project_2는 독립 평가셋이 아닌 회귀 평가셋이다"
            ],
        }
    elif promotion_status == "manual_waiver" and manual_waiver_approved:
        from datetime import UTC, datetime

        metadata["promotion"] = {
            "decision": "approved",
            "method": "manual_waiver",
            "decided_on": datetime.now(UTC).date().isoformat(),
            "waivers": [
                {
                    "gate": "evaluation_set_independence",
                    "observed": 0.0,
                    "target": 1.0,
                    "sample_count": 1,
                    "correct_count": 0,
                    "reason": "test 및 bread_project_2가 독립 평가셋이 아님을 인지한 0.2.0 수동 승인",
                }
            ],
            "remaining_limitations": [
                "Top-1 97% 목표 미달이나 95% 배포 하한과 안전 gate는 통과했다",
                "test 및 bread_project_2는 독립 평가셋이 아니다",
            ],
        }
    if _frozen_detector_fields(metadata) != _frozen_detector_fields(base_metadata):
        raise RuntimeError("0.2.0 metadata construction changed the frozen detector contract")
    return metadata


def export_ten_shot_package(args: argparse.Namespace) -> None:
    torch = require_torch()
    verify_experiment_lock(
        args.pretest_lock,
        config=args.config,
        manifest=args.manifest,
        manifest_metadata=args.manifest_metadata,
        checkpoint=args.head_checkpoint,
        calibration=args.calibration_report,
    )
    base_metadata, detector_sha256 = verify_frozen_detector_package(
        args.base_package,
        expected_detector_sha256=args.expected_detector_sha256,
    )
    checkpoint = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") not in {
        "ten_shot_residual_cosine",
        "ten_shot_residual_cosine_challenger",
    }:
        raise ValueError("unsupported ten-shot classifier checkpoint architecture")
    if sha256_file(args.backbone_weights) != checkpoint.get("source_weight_sha256"):
        raise ValueError("DINO backbone weights do not match the checkpoint provenance")
    spec = adapter_spec_from_dict(checkpoint["adapter_spec"])
    classifier = build_ten_shot_classifier(
        backbone_kind=checkpoint["backbone_kind"],
        weights_path=args.backbone_weights,
        hub_repository=(
            f"facebookresearch/dinov3:{checkpoint['source_revision']}"
            if checkpoint.get("source_revision")
            else "facebookresearch/dinov3"
        ),
        spec=spec,
    )
    if checkpoint["architecture"] == "ten_shot_residual_cosine_challenger":
        classifier.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    else:
        classifier.classifier.load_state_dict(
            compatible_proxy_state_dict(checkpoint["head_state_dict"])
        )
    classifier.eval()
    experiment_config = load_json_config(args.config)
    crop_scale_value = experiment_config.get("inference", {}).get("center_crop_scale")
    crop_scale = None if crop_scale_value is None else float(crop_scale_value)
    inference = experiment_config.get("inference", {})
    classifier = wrap_inference_classifier(
        classifier,
        input_size=int(checkpoint["image_size"]),
        crop_scale=crop_scale,
        num_classes=int(checkpoint["num_classes"]),
        logit_quantum=(
            None if inference.get("logit_quantum") is None else float(inference["logit_quantum"])
        ),
        logit_phase=float(inference.get("logit_phase", 0.0)),
        tie_break_bias_span=float(inference.get("tie_break_bias_span", 0.0)),
        logit_divisor=float(inference.get("logit_divisor", 1.0)),
    ).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = args.output_dir / "classifier.onnx"
    dummy = torch.zeros(
        1,
        3,
        int(checkpoint["image_size"]),
        int(checkpoint["image_size"]),
        dtype=torch.float32,
    )
    torch.onnx.export(
        classifier,
        (dummy,),
        classifier_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(classifier_path))
    base_package = load_model_package(args.base_package)
    detector_target = args.output_dir / base_package.detector_path.name
    shutil.copy2(base_package.detector_path, detector_target)
    if sha256_file(detector_target) != detector_sha256:
        raise RuntimeError("copied detector checksum changed")
    manifest_metadata = json.loads(args.manifest_metadata.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    promotion_status = "experiment_only"
    if args.promotion_report is not None:
        promotion = json.loads(args.promotion_report.read_text(encoding="utf-8"))
        promotion_status = str(promotion["promotion_status"])
    metadata = build_ten_shot_metadata(
        base_metadata=base_metadata,
        manifest_metadata=manifest_metadata,
        calibration=calibration,
        classifier_sha256=sha256_file(classifier_path),
        classifier_version=args.classifier_version,
        package_version=args.package_version,
        checkpoint=checkpoint,
        promotion_status=promotion_status,
        manual_waiver_approved=bool(args.approve_manual_waiver),
        inference_center_crop_scale=crop_scale,
        inference_logit_recipe=(
            None
            if inference.get("logit_quantum") is None
            else (
                f"stable_logits_q{float(inference['logit_quantum']):.2f}"
                f"_d{float(inference.get('logit_divisor', 1.0)):g}"
            )
        ),
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    load_model_package(args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a 10-shot classifier while copying the production detector byte-for-byte"
    )
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--pretest-lock", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-weights", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--promotion-report", type=Path)
    parser.add_argument("--approve-manual-waiver", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-detector-sha256")
    parser.add_argument("--package-version", default="0.2.0")
    parser.add_argument("--classifier-version", default="0.2.0")
    parser.add_argument("--opset", type=int, default=18)
    export_ten_shot_package(parser.parse_args())


if __name__ == "__main__":
    main()
