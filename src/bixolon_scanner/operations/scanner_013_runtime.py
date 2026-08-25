from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import (
    RuntimePackageV2Metadata,
    load_runtime_package_v2,
)


def candidate_metadata(
    base: RuntimePackageV2Metadata,
    *,
    version: str,
    dataset_version: str,
    detector_sha256: str,
    detector_checkpoint_name: str,
    detector_checkpoint_sha256: str,
    detector_manifest_sha256: str,
    license_checksums: dict[str, str],
    horizontal_flip_tta: bool = False,
    rotation_180_tta: bool = False,
    crop_margin_ratio: float = 0.0,
    crop_mode: str = "box_resize",
    support_views_per_source: int = 0,
    ridge_alpha: float = 0.01,
    neighbor_mask: bool = True,
    neighbor_distance_bias: float = 0.0,
) -> RuntimePackageV2Metadata:
    """Build the class-agnostic 0.1.3 contract without classifier-side detector labels."""

    payload = base.model_dump(mode="json")
    payload.pop("promotion_status", None)
    payload.update(
        {
            "worker_version": version,
            "dataset_version": dataset_version,
            "detector_policy_version": version,
            "detector_class_count": 1,
            "detector_refinement": None,
            "count_verifier": None,
        }
    )
    payload["detector"].update(
        {
            "filename": "detector.onnx",
            "version": version,
            "input_size": [640, 640],
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "score_threshold": 0.65,
            "uncertainty_score_threshold": None,
            "uncertainty_min_area_ratio": 0.0,
            "uncertainty_match_iou_threshold": 0.5,
            "nms_iou_threshold": 0.4,
            "nms_containment_threshold": 0.8,
            "nms_class_aware_containment": False,
            "max_object_aspect_ratio": 20.0,
            "max_queries": 8400,
            "resize_reducing_gap": None,
            "ensemble": None,
        }
    )
    payload["embedder"]["version"] = version
    # The selected single_objects_3 robust-training recipe uses the unbiased ownership mask.
    # Keep deployment geometry identical instead of carrying a development-scene calibration.
    payload["embedder"]["neighbor_distance_bias"] = neighbor_distance_bias
    payload["embedder"]["horizontal_flip_tta"] = horizontal_flip_tta
    payload["embedder"]["rotation_180_tta"] = rotation_180_tta
    payload["embedder"]["crop_margin_ratio"] = crop_margin_ratio
    payload["embedder"]["crop_mode"] = crop_mode
    payload["embedder"]["neighbor_mask"] = neighbor_mask
    policy = payload["classifier_policy"]
    policy["version"] = version
    policy["ridge_approval_thresholds"] = None
    policy["detector_corroboration_minimum_score"] = None
    policy["detector_corroboration_maximum_approval_score"] = None
    policy["detector_corroboration_low_similarity_minimum_score"] = None
    policy["detector_corroboration_low_similarity_maximum_retrieval"] = None
    policy["detector_corroboration_low_similarity_minimum_approval_score"] = None
    policy["support_augmentation"]["views_per_source"] = support_views_per_source
    policy["ridge_alpha"] = ridge_alpha
    payload["checksums"] = {"detector.onnx": detector_sha256, **license_checksums}
    payload["licenses"] = {
        "detector": (
            "GNU Affero General Public License v3.0 or later: "
            "https://github.com/ultralytics/ultralytics"
        ),
        "classifier": (
            "DINOv3 License: https://ai.meta.com/resources/models-and-libraries/dinov3-license/"
        ),
    }
    payload["license_files"] = sorted(
        filename for filename in license_checksums if filename.startswith("licenses/")
    )
    payload["sources"] = {
        "detector": {
            "architecture": "YOLO26n one-to-many one-class objectness",
            "revision": "ultralytics-8.4.102",
            "weight_filename": detector_checkpoint_name,
            "weight_sha256": detector_checkpoint_sha256,
            "training_pipeline_version": None,
            "training_contract_sha256": None,
            "training_dataset_version": dataset_version,
            "training_manifest_sha256": detector_manifest_sha256,
        },
        "embedder": payload["sources"]["embedder"],
    }
    return RuntimePackageV2Metadata.model_validate(payload)


def assemble_runtime(
    *,
    base_runtime_dir: Path,
    detector_onnx: Path,
    detector_export_report: Path,
    detector_manifest: Path,
    output_dir: Path,
    version: str,
    dataset_version: str,
    license_sources: dict[str, Path],
    embedder_onnx: Path | None = None,
    embedder_export_report: Path | None = None,
    embedder_training_report: Path | None = None,
    horizontal_flip_tta: bool = False,
    rotation_180_tta: bool = False,
    crop_margin_ratio: float = 0.0,
    crop_mode: str = "box_resize",
    support_views_per_source: int = 0,
    ridge_alpha: float = 0.01,
    neighbor_mask: bool = True,
    neighbor_distance_bias: float = 0.0,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    base = load_runtime_package_v2(base_runtime_dir)
    report = json.loads(detector_export_report.read_text(encoding="utf-8"))
    if report.get("architecture_contract") != "one-class-product-independent-objectness":
        raise ValueError("detector export is not the product-independent objectness contract")
    if report.get("product_labels_exported") is not False:
        raise ValueError("detector export must not contain product labels")
    detector_sha256 = sha256_file(detector_onnx)
    if report.get("onnx_sha256") != detector_sha256:
        raise ValueError("detector export report checksum mismatch")
    checkpoint_sha256 = str(report.get("checkpoint_sha256", ""))
    if len(checkpoint_sha256) != 64:
        raise ValueError("detector checkpoint checksum is missing")

    if (embedder_onnx is None) != (embedder_export_report is None):
        raise ValueError("embedder override requires both the ONNX and export report")
    if embedder_training_report is not None and embedder_onnx is None:
        raise ValueError("embedder training report requires an embedder override")
    selected_embedder = base.embedder_path if embedder_onnx is None else embedder_onnx
    embedder_report = None
    training_report = None
    if embedder_export_report is not None:
        embedder_report = json.loads(embedder_export_report.read_text(encoding="utf-8"))
        if embedder_report.get("onnx_sha256") != sha256_file(selected_embedder):
            raise ValueError("embedder export report checksum mismatch")
        if embedder_report.get("training_scope") == "frozen_backbone":
            if embedder_training_report is not None:
                raise ValueError("frozen embedder override must not have a training report")
        else:
            if embedder_training_report is None:
                raise ValueError("trained embedder override requires a training report")
            training_report = json.loads(embedder_training_report.read_text(encoding="utf-8"))
            if (
                training_report.get("training_source") != "single_objects_3"
                or training_report.get("mixed_support_sources") is not False
            ):
                raise ValueError("embedder override must be trained from single_objects_3 only")

    output_dir.mkdir(parents=True)
    shutil.copy2(detector_onnx, output_dir / "detector.onnx")
    shutil.copy2(selected_embedder, output_dir / base.metadata.embedder.filename)
    license_checksums: dict[str, str] = {}
    for filename, source in license_sources.items():
        destination = output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        license_checksums[filename] = sha256_file(destination)
    license_checksums[base.metadata.embedder.filename] = sha256_file(
        output_dir / base.metadata.embedder.filename
    )
    metadata = candidate_metadata(
        base.metadata,
        version=version,
        dataset_version=dataset_version,
        detector_sha256=detector_sha256,
        detector_checkpoint_name=Path(str(report["checkpoint"])).name,
        detector_checkpoint_sha256=checkpoint_sha256,
        detector_manifest_sha256=sha256_file(detector_manifest),
        license_checksums=license_checksums,
        horizontal_flip_tta=horizontal_flip_tta,
        rotation_180_tta=rotation_180_tta,
        crop_margin_ratio=crop_margin_ratio,
        crop_mode=crop_mode,
        support_views_per_source=support_views_per_source,
        ridge_alpha=ridge_alpha,
        neighbor_mask=neighbor_mask,
        neighbor_distance_bias=neighbor_distance_bias,
    )
    if embedder_report is not None:
        payload = metadata.model_dump(mode="json")
        frozen = embedder_report.get("training_scope") == "frozen_backbone"
        payload["embedder"].update(
            {
                "embedder_id": str(embedder_report["backbone_kind"]).replace("_", "-"),
                "input_size": [
                    int(embedder_report["image_size"]),
                    int(embedder_report["image_size"]),
                ],
                "embedding_dimension": int(embedder_report["embedding_dimension"]),
                "l2_normalized": bool(embedder_report["l2_normalized"]),
            }
        )
        payload["metric_projection"].update(
            {
                "input_dimension": int(embedder_report["embedding_dimension"]),
                "output_dimension": int(embedder_report["embedding_dimension"]),
            }
        )
        payload["sources"]["embedder"] = {
            "architecture": embedder_report["training_architecture"],
            "revision": embedder_report["source_revision"],
            "weight_filename": (
                embedder_report["source_weight_filename"]
                if frozen
                else Path(str(embedder_report["checkpoint"])).name
            ),
            "weight_sha256": (
                embedder_report["source_weight_sha256"]
                if frozen
                else embedder_report["checkpoint_sha256"]
            ),
            "training_pipeline_version": None,
            "training_contract_sha256": None,
            "training_dataset_version": embedder_report["training_dataset_version"],
            "training_manifest_sha256": embedder_report["training_manifest_sha256"],
        }
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
    loaded = load_runtime_package_v2(output_dir)
    return {
        "schema_version": "1.0",
        "operation": "assemble_scanner_0_1_3_runtime",
        "runtime": output_dir.resolve().as_posix(),
        "version": loaded.metadata.worker_version,
        "detector_architecture": loaded.metadata.sources["detector"].architecture,
        "detector_class_count": loaded.metadata.detector_class_count,
        "classifier_source": "single_objects_3",
        "mixed_classifier_sources": False,
        "count_verifier": False,
        "detector_corroboration": False,
        "horizontal_flip_tta": loaded.metadata.embedder.horizontal_flip_tta,
        "rotation_180_tta": loaded.metadata.embedder.rotation_180_tta,
        "crop_margin_ratio": loaded.metadata.embedder.crop_margin_ratio,
        "crop_mode": loaded.metadata.embedder.crop_mode,
        "support_views_per_source": (
            loaded.metadata.classifier_policy.support_augmentation.views_per_source
        ),
        "ridge_alpha": loaded.metadata.classifier_policy.ridge_alpha,
        "neighbor_mask": loaded.metadata.embedder.neighbor_mask,
        "neighbor_distance_bias": loaded.metadata.embedder.neighbor_distance_bias,
        "embedder_checkpoint_sha256": (
            None
            if embedder_report is None
            else embedder_report.get(
                "checkpoint_sha256", embedder_report.get("source_weight_sha256")
            )
        ),
        "metadata_sha256": sha256_file(output_dir / "metadata.json"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assemble the Scanner 0.1.3 Runtime candidate")
    parser.add_argument("--base-runtime", type=Path, required=True)
    parser.add_argument("--detector-onnx", type=Path, required=True)
    parser.add_argument("--detector-export-report", type=Path, required=True)
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="0.1.3")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--apache-license", type=Path, required=True)
    parser.add_argument("--dinov3-license", type=Path, required=True)
    parser.add_argument("--agpl-license", type=Path, required=True)
    parser.add_argument("--third-party-notice", type=Path, required=True)
    parser.add_argument("--embedder-onnx", type=Path)
    parser.add_argument("--embedder-export-report", type=Path)
    parser.add_argument("--embedder-training-report", type=Path)
    parser.add_argument("--horizontal-flip-tta", action="store_true")
    parser.add_argument("--rotation-180-tta", action="store_true")
    parser.add_argument("--crop-margin-ratio", type=float, default=0.0)
    parser.add_argument(
        "--crop-mode", choices=("box_resize", "square_context"), default="box_resize"
    )
    parser.add_argument("--support-views-per-source", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--neighbor-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neighbor-distance-bias", type=float, default=0.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result = assemble_runtime(
        base_runtime_dir=args.base_runtime,
        detector_onnx=args.detector_onnx,
        detector_export_report=args.detector_export_report,
        detector_manifest=args.detector_manifest,
        output_dir=args.output_dir,
        version=args.version,
        dataset_version=args.dataset_version,
        license_sources={
            "licenses/APACHE-2.0.txt": args.apache_license,
            "licenses/DINOV3-LICENSE.md": args.dinov3_license,
            "licenses/AGPL-3.0.txt": args.agpl_license,
            "licenses/THIRD_PARTY_MODELS.md": args.third_party_notice,
        },
        embedder_onnx=args.embedder_onnx,
        embedder_export_report=args.embedder_export_report,
        embedder_training_report=args.embedder_training_report,
        horizontal_flip_tta=args.horizontal_flip_tta,
        rotation_180_tta=args.rotation_180_tta,
        crop_margin_ratio=args.crop_margin_ratio,
        crop_mode=args.crop_mode,
        support_views_per_source=args.support_views_per_source,
        ridge_alpha=args.ridge_alpha,
        neighbor_mask=args.neighbor_mask,
        neighbor_distance_bias=args.neighbor_distance_bias,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
