from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import onnx

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.contracts.runtime_package_v2 import (
    RuntimePackageV2Metadata,
    load_runtime_package_v2,
)
from bixolon_scanner.runtime.catalog import verification_runtime_package


def _copy_files(output_dir: Path, files: dict[str, Path]) -> dict[str, str]:
    output_dir.mkdir(parents=True)
    for relative, source in files.items():
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {name: sha256_file(output_dir / name) for name in sorted(files)}


def _write_metadata(output_dir: Path, payload: dict) -> RuntimePackageV2Metadata:
    metadata = RuntimePackageV2Metadata.model_validate(payload)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            metadata.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def _detector_query_count(model_path: Path, output_name: str = "logits") -> int:
    model = onnx.load_model(model_path, load_external_data=False)
    output = next((value for value in model.graph.output if value.name == output_name), None)
    if output is None:
        raise ValueError(f"detector output is missing: {output_name}")
    dimensions = output.type.tensor_type.shape.dim
    if len(dimensions) != 3 or dimensions[1].dim_value <= 0:
        raise ValueError("detector logits must have a static [batch, queries, classes] shape")
    return int(dimensions[1].dim_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Store 2 next Runtime package")
    parser.add_argument("--version", required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--primary-embedder", type=Path, required=True)
    parser.add_argument("--detail-embedder", type=Path, required=True)
    parser.add_argument("--verifier-embedder", type=Path, required=True)
    parser.add_argument("--classifier-report", type=Path, required=True)
    parser.add_argument("--detector-report", type=Path, required=True)
    parser.add_argument("--detector-score-threshold", type=float, required=True)
    parser.add_argument("--detector-uncertainty-score-threshold", type=float, required=True)
    parser.add_argument("--classifier-approval-margin", type=float, required=True)
    parser.add_argument("--bread-04-approval-margin", type=float, required=True)
    parser.add_argument("--jpeg-draft-size", type=int, default=640)
    parser.add_argument("--verifier-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verifier-runtime-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.verifier_runtime_dir.exists():
        raise FileExistsError("Runtime output directories must not exist")

    repository_root = Path(__file__).resolve().parents[1]
    classifier_report = load_json_config(args.classifier_report)
    detector_report = load_json_config(args.detector_report)
    verifier_report = load_json_config(args.verifier_report)
    detector_query_count = _detector_query_count(args.detector)
    version = args.version
    license_files = [
        "licenses/TORCHVISION-LICENSE.txt",
        "licenses/DINOV3-LICENSE.md",
        "licenses/THIRD_PARTY_MODELS.md",
    ]
    files = {
        "detector.onnx": args.detector,
        "embedder-primary-192.onnx": args.primary_embedder,
        "embedder-detail-224.onnx": args.detail_embedder,
        "classifier-verifier-160.onnx": args.verifier_embedder,
        **{name: repository_root / name for name in license_files},
    }
    checksums = _copy_files(args.output_dir, files)
    class_thresholds: list[float | None] = [None] * 20
    class_thresholds[3] = args.bread_04_approval_margin
    embedder_common = {
        "embedder_id": "dinov3-convnext-tiny-trained-cosine-head",
        "version": version,
        "input_name": "pixel_values",
        "output_name": "embeddings",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "crop_margin_ratio": 0.0,
        "crop_mode": "box_resize",
        "embedding_dimension": 20,
        "l2_normalized": True,
        "resize_reducing_gap": 3.0,
        "warmup_batch_sizes": [1, 3, 5, 8],
        "horizontal_flip_tta": False,
        "rotation_180_tta": False,
        "neighbor_mask": True,
        "neighbor_distance_bias": -0.1,
        "neighbor_shared_scale": False,
    }
    payload = {
        "schema_version": "2.0",
        "worker_version": version,
        "dataset_version": "bread-store2-next-202609",
        "detector_policy_version": version,
        "detector_class_mode": "class_agnostic",
        "detector_class_count": 1,
        "detector": {
            "filename": "detector.onnx",
            "version": version,
            "input_name": "pixel_values",
            "logits_output": "logits",
            "boxes_output": "pred_boxes",
            "input_size": [320, 320],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "score_threshold": args.detector_score_threshold,
            "uncertainty_score_threshold": args.detector_uncertainty_score_threshold,
            "uncertainty_min_area_ratio": 0.001,
            "uncertainty_match_iou_threshold": 0.5,
            "nms_iou_threshold": 0.4,
            "nms_containment_threshold": 0.9,
            "nms_class_aware_containment": False,
            "max_object_aspect_ratio": 8.0,
            "max_queries": detector_query_count,
        },
        "embedder": {
            **embedder_common,
            "filename": "embedder-primary-192.onnx",
            "input_size": [192, 192],
        },
        "metric_projection": {
            "input_dimension": 20,
            "output_dimension": 20,
            "residual_weight": 1.0,
            "projection_weight": 0.0,
        },
        "classifier_policy": {
            "version": version,
            "prototype_weight": 0.5,
            "support_top_k": 3,
            "approval_minimum_similarity": 0.2,
            "approval_minimum_margin": args.classifier_approval_margin,
            "ood_maximum_similarity": -0.2,
            "top3_minimum_similarity": -0.5,
            "catalog_conflict_similarity": 0.9999,
            "ridge_approval_metric": "l2_normalized_logit_margin",
            "ridge_approval_minimum_margin": args.classifier_approval_margin,
            "ridge_approval_thresholds": class_thresholds,
            "ridge_disagreement_minimum_margin": args.classifier_approval_margin,
            "ridge_top3_minimum_inverse_entropy": -10.0,
            "ridge_require_retrieval_agreement": False,
            "ridge_retrieval_minimum_similarity": -0.2,
            "ridge_alpha": 0.01,
            "support_augmentation": {"views_per_source": 0, "output_size": 192},
        },
        "classifier_verification": {
            "ambiguity_maximum_approval_score": 0.55,
            "verify_all_approved_candidates": False,
            "unknown_recapture_on_dual_verifier_rejection": False,
            "unknown_recapture_on_any_verifier_rejection": False,
            "rotation_degrees": 180,
            "independent_embedder": {
                "filename": "classifier-verifier-160.onnx",
                "embedder_id": "dinov3-vitb16-frozen",
                "version": version,
                "input_size": [160, 160],
                "crop_margin_ratio": 0.0,
                "crop_mode": "box_resize",
                "embedding_dimension": 768,
                "l2_normalized": True,
                "fixed_batch_size": 1,
                "warmup_batch_sizes": [1],
                "neighbor_mask": True,
                "neighbor_distance_bias": -0.1,
                "neighbor_shared_scale": False,
            },
            "independent_metric_projection": {
                "input_dimension": 768,
                "output_dimension": 768,
                "residual_weight": 1.0,
                "projection_weight": 0.0,
            },
        },
        "classifier_resolution_fallback": {
            "embedder": {
                **embedder_common,
                "filename": "embedder-detail-224.onnx",
                "input_size": [224, 224],
            },
            "fallback_on_unknown": True,
            "fallback_on_unsafe": True,
            "selective_roi_only": True,
            "fuse_unapproved_top3": True,
            "minimum_fallback_approval_score": args.classifier_approval_margin,
            "minimum_detector_support": 3,
            "approval_disagreement_rules": [
                {
                    "minimum_detection_count": 7,
                    "maximum_approval_score": 0.55,
                    "require_detector_disagreement": False,
                },
                {
                    "minimum_detection_count": 1,
                    "maximum_approval_score": 0.55,
                    "require_detector_disagreement": False,
                    "minimum_box_aspect_ratio": 2.5,
                },
            ],
        },
        "input": {"jpeg_draft_size": args.jpeg_draft_size},
        "quality": {
            "min_object_area_ratio": 0.001,
            "border_margin_ratio": 0.002,
            "border_policy": "classifier_confidence",
            "duplicate_review_containment_threshold": 0.9,
        },
        "checksums": checksums,
        "licenses": {
            "detector": "torchvision BSD-3-Clause",
            "classifier": "DINOv3 License",
        },
        "license_files": license_files,
        "sources": {
            "detector": {
                "architecture": detector_report["experiment"],
                "revision": verifier_report["source_revision"],
            },
            "embedder": {
                "architecture": classifier_report["experiment"],
                "revision": verifier_report["source_revision"],
                "weight_filename": classifier_report["settings"]["weights"].split("\\")[-1],
                "weight_sha256": classifier_report["settings"]["weights_sha256"],
                "training_dataset_version": "bread-store2-support-200",
                "training_manifest_sha256": sha256_file(
                    Path(classifier_report["settings"]["manifest"])
                ),
            },
            "classifier_verifier": {
                "architecture": verifier_report["training_architecture"],
                "revision": verifier_report["source_revision"],
                "weight_filename": verifier_report["source_weight_filename"],
                "weight_sha256": verifier_report["source_weight_sha256"],
            },
        },
    }
    _write_metadata(args.output_dir, payload)
    runtime = load_runtime_package_v2(args.output_dir)

    verifier_view = verification_runtime_package(runtime)
    verifier_files = {
        "detector.onnx": args.detector,
        "classifier-verifier-160.onnx": args.verifier_embedder,
        **{name: repository_root / name for name in license_files},
    }
    _copy_files(args.verifier_runtime_dir, verifier_files)
    verifier_payload = verifier_view.metadata.model_dump(mode="json", exclude_none=True)
    verifier_payload["checksums"] = {
        name: sha256_file(args.verifier_runtime_dir / name) for name in sorted(verifier_files)
    }
    _write_metadata(args.verifier_runtime_dir, verifier_payload)

    report = {
        "schema_version": "1.0",
        "version": version,
        "runtime": args.output_dir.resolve().as_posix(),
        "verifier_runtime": args.verifier_runtime_dir.resolve().as_posix(),
        "classifier_checkpoint_sha256": sha256_file(
            Path(classifier_report["settings"]["output_dir"]) / classifier_report["checkpoint"]
        ),
        "detector_onnx_sha256": sha256_file(args.detector),
        "primary_embedder_onnx_sha256": sha256_file(args.primary_embedder),
        "detail_embedder_onnx_sha256": sha256_file(args.detail_embedder),
        "verifier_embedder_onnx_sha256": sha256_file(args.verifier_embedder),
        "metadata_sha256": sha256_file(args.output_dir / "metadata.json"),
        "detector_policy": {
            "score_threshold": args.detector_score_threshold,
            "uncertainty_score_threshold": args.detector_uncertainty_score_threshold,
            "query_count": detector_query_count,
        },
        "classifier_policy": {
            "default_approval_margin": args.classifier_approval_margin,
            "bread_04_approval_margin": class_thresholds[3],
        },
    }
    (args.output_dir / "build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
