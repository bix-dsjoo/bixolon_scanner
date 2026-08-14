from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..contracts.model_package import load_model_package, sha256_file
from .pipeline_contract import canonical_contract_sha256, load_pipeline_contract
from .staged_classifier_export import view_affine


def dfine_package_metadata(
    base_metadata: dict[str, Any],
    *,
    worker_version: str,
    detector_version: str,
    detector_sha256: str,
    score_threshold: float,
    input_size: tuple[int, int],
    detector_evaluation: dict[str, Any],
    source_revision: str,
    checkpoint_filename: str,
    checkpoint_sha256: str,
    training_pipeline_version: str | None = None,
    training_contract_sha256: str | None = None,
    training_dataset_version: str | None = None,
    training_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    metadata = deepcopy(base_metadata)
    detector_filename = str(metadata["detector"]["filename"])
    classifier_filename = str(metadata["classifier"]["filename"])
    metadata["worker_version"] = worker_version
    metadata.pop("package_version", None)
    metadata["promotion_status"] = "development"
    metadata.pop("promotion", None)
    metadata["detector"].update(
        {
            "version": detector_version,
            "input_name": "pixel_values",
            "logits_output": "logits",
            "boxes_output": "pred_boxes",
            "input_size": list(input_size),
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "score_threshold": score_threshold,
            "uncertainty_score_threshold": None,
            "uncertainty_min_area_ratio": 0.0,
            "uncertainty_match_iou_threshold": 0.5,
            "nms_iou_threshold": 0.7,
            "nms_containment_threshold": None,
            "max_object_aspect_ratio": 5.0,
            "max_queries": 300,
            "box_format": "normalized_cxcywh",
            "resize_reducing_gap": 1.0,
        }
    )
    metadata["checksums"] = {
        detector_filename: detector_sha256,
        classifier_filename: metadata["checksums"][classifier_filename],
    }
    metadata.setdefault("licenses", {})["detector"] = (
        "Apache-2.0: https://github.com/Peterande/D-FINE"
    )
    source = {
        "architecture": "D-FINE-N HGNetv2",
        "revision": source_revision,
        "weight_filename": checkpoint_filename,
        "weight_sha256": checkpoint_sha256,
    }
    _apply_training_pipeline_provenance(
        source,
        version=training_pipeline_version,
        contract_sha256=training_contract_sha256,
        dataset_version=training_dataset_version,
        manifest_sha256=training_manifest_sha256,
    )
    metadata.setdefault("sources", {})["detector"] = source
    metrics = detector_evaluation["metrics"]
    metadata["detector_evaluation"] = {
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "count_accuracy": metrics["count_accuracy"],
        "target_recall_satisfied": detector_evaluation["target_recall_satisfied"],
    }
    return metadata


def apply_staged_classifier_metadata(
    metadata: dict[str, Any],
    *,
    classifier_version: str,
    classifier_sha256: str,
    policy: dict[str, Any],
    checkpoint_filename: str,
    checkpoint_sha256: str,
    training_pipeline_version: str | None = None,
    training_contract_sha256: str | None = None,
    training_dataset_version: str | None = None,
    training_manifest_sha256: str | None = None,
) -> None:
    final_policy = policy["final_policy"]
    staged_policy = policy["staged_policy"]
    selected_views = list(final_policy["top3_views"])
    classifier = metadata["classifier"]
    classifier.update(
        {
            "version": classifier_version,
            "approval_threshold": float(final_policy["threshold"]),
            "temperature": 1.0,
            "crop_mode": "box_resize",
            "resize_reducing_gap": 1.0,
            "warmup_batch_sizes": [1, 2, 3, 4, 5, 6, 7],
            "staged_inference": {
                "affine_input_name": "view_affine",
                "center_crop_scale": 0.855,
                "views": [
                    {"name": name, "affine": view_affine(name).tolist()} for name in selected_views
                ],
                "first_view": staged_policy["first_view"],
                "early_approval_threshold": float(staged_policy["early_approval_threshold"]),
                "final_views": list(final_policy["views"]),
                "top3_views": selected_views,
                "ranking_aggregation": final_policy.get("top3_aggregation", "mean_logits"),
            },
        }
    )
    classifier_filename = str(classifier["filename"])
    metadata["checksums"][classifier_filename] = classifier_sha256
    source = {
        "architecture": "DINOv3 ConvNeXt-Tiny staged affine-view classifier",
        "revision": "6876159a11b4df116f30f667f8c9888617df0751",
        "weight_filename": checkpoint_filename,
        "weight_sha256": checkpoint_sha256,
    }
    _apply_training_pipeline_provenance(
        source,
        version=training_pipeline_version,
        contract_sha256=training_contract_sha256,
        dataset_version=training_dataset_version,
        manifest_sha256=training_manifest_sha256,
    )
    metadata.setdefault("sources", {})["classifier"] = source
    metadata.pop("calibration", None)


def _apply_training_pipeline_provenance(
    source: dict[str, Any],
    *,
    version: str | None,
    contract_sha256: str | None,
    dataset_version: str | None = None,
    manifest_sha256: str | None = None,
) -> None:
    if (version is None) != (contract_sha256 is None):
        raise ValueError("training pipeline version and contract checksum are required together")
    if version is not None:
        source["training_pipeline_version"] = version
        source["training_contract_sha256"] = contract_sha256
    if (dataset_version is None) != (manifest_sha256 is None):
        raise ValueError("training dataset version and manifest checksum are required together")
    if dataset_version is not None:
        source["training_dataset_version"] = dataset_version
        source["training_manifest_sha256"] = manifest_sha256


def package_dfine_detector(args: argparse.Namespace) -> None:
    base_package = load_model_package(args.base_package)
    base_metadata = json.loads((args.base_package / "metadata.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    detector_output = args.output / base_package.metadata.detector.filename
    classifier_output = args.output / base_package.metadata.classifier.filename
    shutil.copy2(args.detector, detector_output)
    classifier_source = args.classifier or base_package.classifier_path
    shutil.copy2(classifier_source, classifier_output)
    selection = json.loads(args.detector_evaluation.read_text(encoding="utf-8"))
    detector_contract = None
    classifier_contract = None
    if args.detector_contract is not None:
        detector_contract = load_pipeline_contract(args.detector_contract)
        if detector_contract.component != "detector":
            raise ValueError("--detector-contract must be a Detector contract")
        if detector_contract.checkpoint.sha256 != sha256_file(args.checkpoint):
            raise ValueError("Detector checkpoint does not match its training contract")
        if detector_contract.onnx.sha256 != sha256_file(detector_output):
            raise ValueError("Detector ONNX does not match its training contract")
    if args.classifier_contract is not None:
        classifier_contract = load_pipeline_contract(args.classifier_contract)
        if classifier_contract.component != "classifier":
            raise ValueError("--classifier-contract must be a Classifier contract")
        if classifier_contract.onnx.sha256 != sha256_file(classifier_output):
            raise ValueError("Classifier ONNX does not match its training contract")
    if not args.worker_version.startswith("0.") and (
        detector_contract is None or classifier_contract is None
    ):
        raise ValueError("official packages require Detector and Classifier contracts")
    metadata = dfine_package_metadata(
        base_metadata,
        worker_version=args.worker_version,
        detector_version=args.detector_version,
        detector_sha256=sha256_file(detector_output),
        score_threshold=float(selection["selected_score_threshold"]),
        input_size=(args.input_height, args.input_width),
        detector_evaluation=selection,
        source_revision=args.source_revision,
        checkpoint_filename=args.checkpoint.name,
        checkpoint_sha256=sha256_file(args.checkpoint),
        training_pipeline_version=(
            detector_contract.pipeline_version
            if detector_contract is not None
            else getattr(args, "detector_training_pipeline_version", None)
        ),
        training_contract_sha256=(
            canonical_contract_sha256(detector_contract)
            if detector_contract is not None
            else getattr(args, "detector_training_contract_sha256", None)
        ),
        training_dataset_version=(
            detector_contract.dataset.dataset_version if detector_contract is not None else None
        ),
        training_manifest_sha256=(
            sha256_file(Path(detector_contract.dataset.manifest_path))
            if detector_contract is not None
            else None
        ),
    )
    metadata["checksums"][classifier_output.name] = sha256_file(classifier_output)
    metadata["quality"]["border_margin_ratio"] = 0.0
    metadata["quality"]["duplicate_review_containment_threshold"] = getattr(
        args, "duplicate_review_containment_threshold", None
    )
    metadata.setdefault("input", {})["jpeg_draft_size"] = args.jpeg_draft_size
    if args.manifest_metadata is not None:
        dataset_metadata = json.loads(args.manifest_metadata.read_text(encoding="utf-8"))
        metadata["dataset_version"] = str(dataset_metadata["dataset_version"])
    if args.classifier is not None:
        if args.classifier_policy is None or args.classifier_checkpoint is None:
            raise ValueError(
                "classifier policy and checkpoint are required with a classifier model"
            )
        policy = json.loads(args.classifier_policy.read_text(encoding="utf-8"))
        apply_staged_classifier_metadata(
            metadata,
            classifier_version=args.classifier_version,
            classifier_sha256=sha256_file(classifier_output),
            policy=policy,
            checkpoint_filename=args.classifier_checkpoint.name,
            checkpoint_sha256=sha256_file(args.classifier_checkpoint),
            training_pipeline_version=getattr(args, "classifier_training_pipeline_version", None),
            training_contract_sha256=getattr(args, "classifier_training_contract_sha256", None),
        )
    if classifier_contract is not None:
        classifier_source = metadata.setdefault("sources", {}).setdefault("classifier", {})
        classifier_source["training_pipeline_version"] = classifier_contract.pipeline_version
        classifier_source["training_contract_sha256"] = canonical_contract_sha256(
            classifier_contract
        )
        classifier_source["training_dataset_version"] = classifier_contract.dataset.dataset_version
        classifier_source["training_manifest_sha256"] = sha256_file(
            Path(classifier_contract.dataset.manifest_path)
        )
        if classifier_source.get("revision") != classifier_contract.pretrained.revision:
            raise ValueError("Classifier source revision does not match its contract")
        if classifier_source.get("weight_sha256") != classifier_contract.checkpoint.sha256:
            raise ValueError("Classifier checkpoint does not match its contract")
    if detector_contract is not None and classifier_contract is not None:
        metadata["schema_version"] = "2.1"
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    load_model_package(args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a development D-FINE Worker package")
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--detector-evaluation", type=Path, required=True)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--classifier-policy", type=Path)
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument("--classifier-version", default="1.0.0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-version", required=True)
    parser.add_argument("--detector-version", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--input-height", type=int, default=640)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--jpeg-draft-size", type=int, default=1500)
    parser.add_argument("--duplicate-review-containment-threshold", type=float)
    parser.add_argument("--manifest-metadata", type=Path)
    parser.add_argument("--detector-contract", type=Path)
    parser.add_argument("--classifier-contract", type=Path)
    parser.add_argument("--detector-training-pipeline-version")
    parser.add_argument("--detector-training-contract-sha256")
    parser.add_argument("--classifier-training-pipeline-version")
    parser.add_argument("--classifier-training-contract-sha256")
    package_dfine_detector(parser.parse_args())


if __name__ == "__main__":
    main()
