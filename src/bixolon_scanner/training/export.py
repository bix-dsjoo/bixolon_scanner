from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..package import load_model_package, sha256_file
from .config_file import parse_args_with_config
from .models import build_dino_classifier, require_torch
from .pipeline_contract import canonical_contract_sha256, load_pipeline_contract


def _copy_reused_classifier(package_dir: Path, classifier_path: Path) -> dict:
    reused_package = load_model_package(package_dir)
    reused_metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    shutil.copy2(reused_package.classifier_path, classifier_path)
    if sha256_file(classifier_path) != sha256_file(reused_package.classifier_path):
        raise RuntimeError("reused classifier checksum mismatch")
    return reused_metadata


def _export_worker_version(args: argparse.Namespace) -> str:
    worker_version = getattr(args, "worker_version", None)
    package_version = getattr(args, "package_version", None)
    if worker_version and package_version and worker_version != package_version:
        raise ValueError("--worker-version and --package-version must match")
    return str(worker_version or package_version or "0.1.0")


def export_models(args: argparse.Namespace) -> None:
    torch = require_torch()
    from transformers import RTDetrV2ForObjectDetection

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_version = _export_worker_version(args)
    detector_contract = (
        load_pipeline_contract(args.detector_contract)
        if args.detector_contract is not None
        else None
    )
    classifier_contract = (
        load_pipeline_contract(args.classifier_contract)
        if args.classifier_contract is not None
        else None
    )
    if not worker_version.startswith("0.") and (
        detector_contract is None or classifier_contract is None
    ):
        raise ValueError("official package export requires both component contracts")
    detector_evaluation = json.loads(args.detector_evaluation_report.read_text(encoding="utf-8"))
    classifier_path = args.output_dir / "classifier.onnx"
    reused_metadata = None
    reused_classifier_source_version = None
    reused_classifier_source_sha256 = None
    if args.reuse_classifier_package is not None:
        reused_metadata = _copy_reused_classifier(args.reuse_classifier_package, classifier_path)
        reused_classifier_source_version = str(reused_metadata["classifier"]["version"])
        reused_classifier_source_sha256 = sha256_file(classifier_path)
    else:
        if args.classifier_checkpoint is None or args.calibration_report is None:
            raise ValueError(
                "classifier checkpoint and calibration report are required unless a package is reused"
            )
        classifier_checkpoint = torch.load(
            args.classifier_checkpoint, map_location="cpu", weights_only=False
        )
        calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
        classifier = build_dino_classifier(
            classifier_checkpoint.get("backbone_kind", "dinov2"),
            classifier_checkpoint["num_classes"],
            pretrained_name=classifier_checkpoint["pretrained_name"],
            hub_repository=(
                f"facebookresearch/dinov3:{classifier_checkpoint['source_revision']}"
                if classifier_checkpoint.get("source_revision")
                else "facebookresearch/dinov3"
            ),
            classifier_head_kind=classifier_checkpoint.get("classifier_head_kind", "linear"),
            cosine_scale=float(classifier_checkpoint.get("cosine_scale", 16.0)),
        )
        classifier.load_state_dict(classifier_checkpoint["state_dict"])
        classifier.eval()
        classifier_dummy = torch.zeros(
            1,
            3,
            classifier_checkpoint["image_size"],
            classifier_checkpoint["image_size"],
            dtype=torch.float32,
        )
        torch.onnx.export(
            classifier,
            (classifier_dummy,),
            classifier_path,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=args.opset,
            dynamo=False,
        )

    detector_model = RTDetrV2ForObjectDetection.from_pretrained(args.detector_checkpoint).eval()

    class DetectorExport(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            output = self.model(pixel_values=pixel_values)
            return output.logits, output.pred_boxes

    detector_path = args.output_dir / "detector.onnx"
    detector_dummy = torch.zeros(1, 3, args.detector_size, args.detector_size, dtype=torch.float32)
    torch.onnx.export(
        DetectorExport(detector_model),
        (detector_dummy,),
        detector_path,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes=None,
        opset_version=args.opset,
        dynamo=False,
    )

    import onnx

    onnx.checker.check_model(onnx.load(detector_path))
    onnx.checker.check_model(onnx.load(classifier_path))

    manifest_metadata = json.loads(args.manifest_metadata.read_text(encoding="utf-8"))
    labels = [
        {"class_id": label["class_id"], "class_name": label["class_name"], "recapture": False}
        for label in manifest_metadata["labels"]
    ]
    if reused_metadata is not None:
        classifier_metadata = dict(reused_metadata["classifier"])
        classifier_metadata["filename"] = classifier_path.name
        if classifier_metadata["version"] != args.classifier_version:
            if bool(getattr(args, "relabel_reused_classifier", False)):
                classifier_metadata["version"] = args.classifier_version
            else:
                raise ValueError("reused classifier version does not match --classifier-version")
        if [label["class_id"] for label in classifier_metadata["labels"]] != [
            label["class_id"] for label in labels
        ]:
            raise ValueError("reused classifier labels do not match dataset metadata")
        classifier_metadata["labels"] = labels
    else:
        classifier_metadata = {
            "filename": classifier_path.name,
            "version": args.classifier_version,
            "input_name": "pixel_values",
            "logits_output": "logits",
            "input_size": [
                classifier_checkpoint["image_size"],
                classifier_checkpoint["image_size"],
            ],
            "color_order": "RGB",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_margin_ratio": args.crop_margin,
            "crop_mode": args.classifier_crop_mode,
            "approval_threshold": calibration["approval_threshold"],
            "temperature": calibration["temperature"],
            "labels": labels,
            "resize_reducing_gap": args.resize_reducing_gap,
            "warmup_batch_sizes": list(range(1, args.classifier_warmup_max_batch + 1)),
        }
    metadata = {
        "schema_version": "2.1" if not worker_version.startswith("0.") else "1.1",
        ("worker_version" if not worker_version.startswith("0.") else "package_version"): (
            worker_version
        ),
        "promotion_status": "development",
        "dataset_version": manifest_metadata["dataset_version"],
        "input": {"jpeg_draft_size": args.jpeg_draft_size},
        "detector": {
            "filename": detector_path.name,
            "version": args.detector_version,
            "input_name": "pixel_values",
            "logits_output": "logits",
            "boxes_output": "pred_boxes",
            "input_size": [args.detector_size, args.detector_size],
            "color_order": "RGB",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "score_threshold": detector_evaluation["selected_score_threshold"],
            "uncertainty_score_threshold": args.uncertainty_score_threshold,
            "uncertainty_min_area_ratio": args.uncertainty_min_area_ratio,
            "uncertainty_match_iou_threshold": args.uncertainty_match_iou_threshold,
            "nms_iou_threshold": detector_evaluation["nms_iou_threshold"],
            "nms_containment_threshold": getattr(args, "nms_containment_threshold", None),
            "max_object_aspect_ratio": getattr(args, "max_object_aspect_ratio", None),
            "max_queries": int(detector_model.config.num_queries),
            "box_format": "normalized_cxcywh",
            "resize_reducing_gap": args.resize_reducing_gap,
        },
        "classifier": classifier_metadata,
        "quality": {
            "min_object_area_ratio": args.min_object_area_ratio,
            "border_margin_ratio": args.border_margin_ratio,
            "border_policy": args.border_policy,
            "min_sharpness": args.min_sharpness,
            "min_mean_luminance": args.min_mean_luminance,
            "max_mean_luminance": args.max_mean_luminance,
        },
        "checksums": {
            detector_path.name: sha256_file(detector_path),
            classifier_path.name: sha256_file(classifier_path),
        },
        "licenses": {
            "detector": "Apache-2.0: https://huggingface.co/PekingU/rtdetr_v2_r18vd",
            "classifier": reused_metadata["licenses"]["classifier"]
            if reused_metadata
            else (
                "DINOv3 License (authorized by model recipient): https://github.com/facebookresearch/dinov3"
                if classifier_checkpoint.get("backbone_kind") == "dinov3_convnext_tiny"
                else "Apache-2.0: https://github.com/facebookresearch/dinov2"
            ),
        },
        "sources": reused_metadata.get("sources", {})
        if reused_metadata
        else {
            "classifier": {
                "architecture": classifier_checkpoint.get("backbone_architecture"),
                "revision": classifier_checkpoint.get("source_revision"),
                "weight_filename": classifier_checkpoint.get("source_weight_filename"),
                "weight_sha256": classifier_checkpoint.get("source_weight_sha256"),
            }
        },
        "calibration": reused_metadata.get("calibration")
        if reused_metadata
        else {
            "sample_count": calibration["sample_count"],
            "approved_precision": calibration["approved_precision"],
            "approval_coverage": calibration["approval_coverage"],
            "false_approval_rate_upper_95": calibration["approved_false_rate_upper_95"],
            "risk_control_satisfied": calibration["risk_control_satisfied"],
        },
        "detector_evaluation": {
            "recall": detector_evaluation["metrics"]["recall"],
            "precision": detector_evaluation["metrics"]["precision"],
            "count_accuracy": detector_evaluation["metrics"]["count_accuracy"],
            "target_recall_satisfied": detector_evaluation["target_recall_satisfied"],
        },
    }
    target_provenance = getattr(args, "detector_target_provenance", None)
    if target_provenance is not None:
        if reused_metadata is None:
            raise ValueError("detector target mode requires a frozen reused classifier")
        metadata["bundle_provenance"] = {
            "target_mode": "detector_safety_first_0.2.5",
            "model_version": worker_version,
            "classifier_source_version": reused_classifier_source_version,
            "classifier_source_sha256": reused_classifier_source_sha256,
            "detector_selection_sha256": target_provenance["selection_report_sha256"],
            "evaluation_dataset_versions": target_provenance["evaluation_dataset_versions"],
        }
    for component, contract, model_path, checkpoint_path in (
        ("detector", detector_contract, detector_path, args.detector_checkpoint),
        ("classifier", classifier_contract, classifier_path, args.classifier_checkpoint),
    ):
        if contract is None:
            continue
        if contract.component != component:
            raise ValueError(f"{component} contract component mismatch")
        if contract.onnx.sha256 != sha256_file(model_path):
            raise ValueError(f"{component} ONNX does not match its contract")
        if checkpoint_path is not None and checkpoint_path.is_file():
            if contract.checkpoint.sha256 != sha256_file(checkpoint_path):
                raise ValueError(f"{component} checkpoint does not match its contract")
        source = metadata.setdefault("sources", {}).setdefault(component, {})
        source.update(
            {
                "revision": contract.pretrained.revision,
                "weight_filename": Path(contract.checkpoint.path).name,
                "weight_sha256": contract.checkpoint.sha256,
                "training_pipeline_version": contract.pipeline_version,
                "training_contract_sha256": canonical_contract_sha256(contract),
                "training_dataset_version": contract.dataset.dataset_version,
                "training_manifest_sha256": sha256_file(Path(contract.dataset.manifest_path)),
            }
        )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    load_model_package(args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export detector and classifier to a Worker model package"
    )
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument(
        "--reuse-classifier-package",
        type=Path,
        help="Copy a validated classifier ONNX and its metadata byte-for-byte from a package.",
    )
    parser.add_argument("--detector-evaluation-report", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector-contract", type=Path)
    parser.add_argument("--classifier-contract", type=Path)
    parser.add_argument("--worker-version")
    parser.add_argument("--package-version", help="Deprecated 0.x compatibility alias")
    parser.add_argument("--detector-version", default="0.1.0")
    parser.add_argument("--classifier-version", default="0.1.0")
    parser.add_argument(
        "--relabel-reused-classifier",
        action="store_true",
        help="Expose the reused classifier under the package-wide inference version.",
    )
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--uncertainty-score-threshold", type=float)
    parser.add_argument("--uncertainty-min-area-ratio", type=float, default=0.0)
    parser.add_argument("--uncertainty-match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-object-aspect-ratio", type=float)
    parser.add_argument("--nms-containment-threshold", type=float)
    parser.add_argument("--crop-margin", type=float, default=0.05)
    parser.add_argument(
        "--classifier-crop-mode",
        choices=("box_resize", "square_context"),
        default="box_resize",
    )
    parser.add_argument("--resize-reducing-gap", type=float, default=1.0)
    parser.add_argument("--classifier-warmup-max-batch", type=int, default=7)
    parser.add_argument("--jpeg-draft-size", type=int, default=1500)
    parser.add_argument("--min-object-area-ratio", type=float, default=0.005)
    parser.add_argument("--border-margin-ratio", type=float, default=0.002)
    parser.add_argument(
        "--border-policy",
        choices=("always_recapture", "classifier_confidence"),
        default="classifier_confidence",
    )
    parser.add_argument("--min-sharpness", type=float)
    parser.add_argument("--min-mean-luminance", type=float)
    parser.add_argument("--max-mean-luminance", type=float)
    parser.add_argument("--opset", type=int, default=18)
    export_models(parse_args_with_config(parser, section="export"))


if __name__ == "__main__":
    main()
