from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Literal

from ...contracts.catalog import sha256_file
from ...contracts.model_package import DetectorMetadata, InputMetadata, ModelSource, QualityMetadata
from ...contracts.runtime_package_v2 import (
    CatalogDecisionPolicy,
    CatalogSupportAugmentationMetadata,
    DetectorAmbiguityPolicyMetadata,
    DetectorRefinementMetadata,
    EmbedderMetadata,
    MetricProjectionMetadata,
    RuntimePackageV2Metadata,
)


def build_runtime_package(
    source_metadata_path: Path,
    detector_path: Path,
    detector_refinement_path: Path | None,
    embedder_path: Path,
    output_dir: Path,
    *,
    version: str,
    dataset_version: str = "bread-scanner-2.0.0-development",
    embedder_report_path: Path | None = None,
    approval_threshold: float | None = None,
    approval_metric: Literal[
        "l2_normalized_logit_margin", "top2_pair_probability"
    ] = "l2_normalized_logit_margin",
    disagreement_approval_threshold: float | None = None,
    ood_minimum_similarity: float = -1.0,
    top3_safety_threshold: float | None = None,
    jpeg_draft_size: int | None = 1000,
    promotion_status: Literal["development", "independent_test_pending"] = "development",
    ridge_alpha: float = 0.01,
    support_views_per_source: int = 0,
    detector_member_paths: list[Path] | None = None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True)
    repository_root = Path(__file__).resolve().parents[4]
    license_files = ["licenses/APACHE-2.0.txt", "licenses/THIRD_PARTY_MODELS.md"]
    source_detector = DetectorMetadata.model_validate(source["detector"])
    if detector_member_paths:
        if source_detector.ensemble is None:
            raise ValueError("detector members require ensemble metadata")
        if len(detector_member_paths) != len(source_detector.ensemble.members):
            raise ValueError("detector member paths must match ensemble members")
        files = {
            member.filename: path
            for member, path in zip(
                source_detector.ensemble.members,
                detector_member_paths,
                strict=True,
            )
        }
        files["embedder.onnx"] = embedder_path
        detector = source_detector.model_copy(update={"version": version})
        detector_refinement = None
    else:
        if detector_refinement_path is None:
            raise ValueError("cross-scale detector requires a refinement model")
        files = {
            "detector.onnx": detector_path,
            "detector-refinement.onnx": detector_refinement_path,
            "embedder.onnx": embedder_path,
        }
        detector = source_detector.model_copy(
            update={
                "filename": "detector.onnx",
                "version": version,
                "score_threshold": 0.23,
                "nms_iou_threshold": 0.5,
                "nms_containment_threshold": 0.95,
                "nms_class_aware_containment": True,
                "ensemble": None,
            }
        )
        detector_refinement = DetectorRefinementMetadata(
            filename="detector-refinement.onnx",
            input_size=(768, 768),
            score_threshold=0.12,
            nms_iou_threshold=0.5,
            containment_threshold=0.9,
            group_minimum=2,
            agreement_iou_threshold=0.65,
        )
    files.update({filename: repository_root / filename for filename in license_files})
    for filename, path in files.items():
        destination = output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    source_models = {
        name: ModelSource.model_validate(payload) for name, payload in source["sources"].items()
    }
    embedder_report = (
        None
        if embedder_report_path is None
        else json.loads(embedder_report_path.read_text(encoding="utf-8"))
    )
    embedder_kind = (
        "dinov3_convnext_tiny" if embedder_report is None else embedder_report["backbone_kind"]
    )
    if embedder_kind == "dinov2_base":
        embedder_id = "dinov2-base-frozen"
        embedder_mean = (0.485, 0.456, 0.406)
        embedder_std = (0.229, 0.224, 0.225)
        source_architecture = "DINOv2 ViT-Base/14 frozen CLS embedding backbone"
        source_license = "Apache-2.0: https://github.com/facebookresearch/dinov2"
        selected_approval = approval_threshold
        selected_top3 = top3_safety_threshold
        if selected_approval is None or selected_top3 is None:
            raise ValueError("DINOv2 runtime requires locked approval and Top-3 thresholds")
        embedder_source = ModelSource(
            architecture=source_architecture,
            revision=embedder_report.get("source_revision"),
            weight_filename=embedder_report.get("source_weight_filename"),
            weight_sha256=embedder_report.get("source_weight_sha256"),
            training_pipeline_version=version,
            training_contract_sha256=sha256_file(embedder_report_path),
            training_dataset_version="bread-catalog-10shot-dinov2-probe",
            training_manifest_sha256=(
                "afe4b679c806847267e8dc8c9a2c89a48de479ec1c66e455333e757a304f4ddd"
            ),
        )
    elif embedder_kind == "dinov3_convnext_tiny":
        embedder_id = "dinov3-convnext-tiny-frozen"
        embedder_mean = (0.485, 0.456, 0.406)
        embedder_std = (0.229, 0.224, 0.225)
        source_architecture = "DINOv3 ConvNeXt-Tiny frozen embedding backbone"
        source_license = source["licenses"]["classifier"]
        selected_approval = 0.4449983835220337 if approval_threshold is None else approval_threshold
        selected_top3 = (
            -2.960296392440796 if top3_safety_threshold is None else top3_safety_threshold
        )
        embedder_source = source_models["classifier"].model_copy(
            update={"architecture": source_architecture}
        )
    else:
        raise ValueError(f"unsupported 2.0 embedder kind: {embedder_kind}")
    metadata = RuntimePackageV2Metadata(
        worker_version=version,
        promotion_status=promotion_status,
        dataset_version=dataset_version,
        detector_policy_version=version,
        detector_class_count=int(
            source.get("detector_class_count") or len(source["classifier"]["labels"])
        ),
        detector=detector,
        detector_refinement=detector_refinement,
        detector_ambiguity=DetectorAmbiguityPolicyMetadata(
            mode="selective" if detector_member_paths else "all",
            high_aspect_ratio_minimum=1.9,
            dense_selected_count_minimum=6,
            dense_selected_count_maximum=6,
            dense_agreement_count_minimum=4,
            dense_aspect_ratio_minimum=1.5,
        ),
        embedder=EmbedderMetadata(
            filename="embedder.onnx",
            embedder_id=embedder_id,
            version=version,
            input_size=(224, 224),
            mean=embedder_mean,
            std=embedder_std,
            crop_margin_ratio=0.05,
            crop_mode="box_resize",
            embedding_dimension=768,
            resize_reducing_gap=1.0,
            warmup_batch_sizes=[1, 2, 3, 4, 5, 6, 7, 8],
            neighbor_mask=True,
        ),
        metric_projection=MetricProjectionMetadata(
            filename=None,
            input_dimension=768,
            output_dimension=768,
            residual_weight=1.0,
            projection_weight=0.0,
        ),
        classifier_policy=CatalogDecisionPolicy(
            version=version,
            prototype_weight=0.5,
            support_top_k=3,
            approval_minimum_similarity=1.0,
            approval_minimum_margin=0.1,
            ood_maximum_similarity=ood_minimum_similarity,
            top3_minimum_similarity=-1.0,
            catalog_conflict_similarity=0.95,
            ridge_approval_metric=approval_metric,
            ridge_approval_minimum_margin=(
                selected_approval if approval_metric == "l2_normalized_logit_margin" else None
            ),
            ridge_approval_minimum_pair_probability=(
                selected_approval if approval_metric == "top2_pair_probability" else None
            ),
            ridge_disagreement_minimum_pair_probability=disagreement_approval_threshold,
            ridge_top3_minimum_inverse_entropy=selected_top3,
            ridge_alpha=ridge_alpha,
            support_augmentation=CatalogSupportAugmentationMetadata(
                views_per_source=support_views_per_source
            ),
        ),
        input=InputMetadata(jpeg_draft_size=jpeg_draft_size),
        quality=QualityMetadata.model_validate(source["quality"]),
        checksums={filename: sha256_file(output_dir / filename) for filename in sorted(files)},
        licenses={**source["licenses"], "classifier": source_license},
        license_files=license_files,
        sources={
            "detector": source_models["detector"].model_copy(
                update={
                    "architecture": (
                        "fixed four-model D-FINE ensemble with selective ambiguity gate"
                        if detector_member_paths
                        else "single D-FINE checkpoint with 640/768 cross-scale disagreement gate"
                    )
                }
            ),
            "embedder": embedder_source,
        },
    )
    # `jpeg_draft_size=None` is an explicit policy value. Omitting it would make
    # the loader restore InputMetadata's default and silently change inference.
    payload = metadata.model_dump(mode="json")
    (output_dir / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the Bread Scanner 2.0 runtime candidate")
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--detector-refinement", type=Path)
    parser.add_argument(
        "--detector-members",
        type=Path,
        nargs="+",
        help="Ordered ensemble paths matching source metadata members.",
    )
    parser.add_argument("--embedder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="2.0.0-rc.1")
    parser.add_argument("--dataset-version", default="bread-scanner-2.0.0-development")
    parser.add_argument("--embedder-report", type=Path)
    parser.add_argument("--approval-threshold", type=float)
    parser.add_argument(
        "--approval-metric",
        choices=("l2_normalized_logit_margin", "top2_pair_probability"),
        default="l2_normalized_logit_margin",
    )
    parser.add_argument("--disagreement-approval-threshold", type=float)
    parser.add_argument("--ood-minimum-similarity", type=float, default=-1.0)
    parser.add_argument("--top3-safety-threshold", type=float)
    parser.add_argument("--jpeg-draft-size", type=int, default=1000)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--support-views-per-source", type=int, default=0)
    parser.add_argument(
        "--promotion-status",
        choices=("development", "independent_test_pending"),
        default="development",
        help="Use independent_test_pending only after every development gate passes.",
    )
    args = parser.parse_args(argv)
    payload = build_runtime_package(
        args.source_metadata,
        args.detector,
        args.detector_refinement,
        args.embedder,
        args.output_dir,
        version=args.version,
        dataset_version=args.dataset_version,
        embedder_report_path=args.embedder_report,
        approval_threshold=args.approval_threshold,
        approval_metric=args.approval_metric,
        disagreement_approval_threshold=args.disagreement_approval_threshold,
        ood_minimum_similarity=args.ood_minimum_similarity,
        top3_safety_threshold=args.top3_safety_threshold,
        jpeg_draft_size=args.jpeg_draft_size,
        promotion_status=args.promotion_status,
        ridge_alpha=args.ridge_alpha,
        support_views_per_source=args.support_views_per_source,
        detector_member_paths=args.detector_members,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
