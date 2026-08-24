from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Literal

from ...contracts.catalog import sha256_file
from ...contracts.model_package import (
    CountVerifierMetadata,
    DetectorMetadata,
    InputMetadata,
    ModelSource,
    QualityMetadata,
)
from ...contracts.runtime_package_v2 import (
    CatalogDecisionPolicy,
    CatalogSupportAugmentationMetadata,
    DetectorAmbiguityPolicyMetadata,
    DetectorRefinementMetadata,
    EmbedderMetadata,
    MetricProjectionMetadata,
    RuntimePackageV2Metadata,
)


def _embedder_source(source_models: dict[str, ModelSource]) -> ModelSource:
    for name in ("embedder", "classifier"):
        if name in source_models:
            return source_models[name]
    raise ValueError("source metadata must describe an embedder model")


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
    ridge_alpha: float = 0.01,
    support_views_per_source: int = 0,
    detector_corroboration_minimum_score: float | None = None,
    detector_corroboration_maximum_approval_score: float | None = None,
    detector_corroboration_low_similarity_minimum_score: float | None = None,
    detector_corroboration_low_similarity_maximum_retrieval: float | None = None,
    detector_corroboration_low_similarity_minimum_approval_score: float | None = None,
    neighbor_mask: bool = True,
    neighbor_distance_bias: float = 0.0,
    approval_thresholds: list[float] | None = None,
    detector_member_paths: list[Path] | None = None,
    detector_score_threshold: float | None = None,
    detector_source_path: Path | None = None,
    count_verifier_path: Path | None = None,
    count_verifier_report_path: Path | None = None,
    low_agreement_count_maximum: int | None = None,
    low_agreement_aspect_ratio_minimum: float | None = None,
    cascade_primary_member_filename: str | None = None,
    cascade_secondary_trigger_selected_counts: list[int] | None = None,
    cascade_secondary_trigger_minimum_score_maximum: float | None = None,
    cascade_secondary_trigger_minimum_score_minimum: float | None = None,
    cascade_secondary_trigger_on_uncertain: bool = False,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if (count_verifier_path is None) != (count_verifier_report_path is None):
        raise ValueError("count verifier model and report must be provided together")
    count_verifier_report = (
        None
        if count_verifier_report_path is None
        else json.loads(count_verifier_report_path.read_text(encoding="utf-8"))
    )
    output_dir.mkdir(parents=True)
    repository_root = Path(__file__).resolve().parents[4]
    license_files = [
        "licenses/APACHE-2.0.txt",
        "licenses/DINOV3-LICENSE.md",
        "licenses/THIRD_PARTY_MODELS.md",
    ]
    source_detector = DetectorMetadata.model_validate(source["detector"])
    if detector_member_paths:
        if source_detector.ensemble is None:
            raise ValueError("detector members require ensemble metadata")
        member_by_filename = {
            member.filename: member for member in source_detector.ensemble.members
        }
        selected_filenames = [path.name for path in detector_member_paths]
        if len(selected_filenames) != len(set(selected_filenames)):
            raise ValueError("detector member paths must be unique")
        if len(selected_filenames) < 2 or any(
            filename not in member_by_filename for filename in selected_filenames
        ):
            raise ValueError("detector member paths must select at least two known members")
        ensemble_payload = source_detector.ensemble.model_dump(mode="json")
        ensemble_payload["members"] = [
            member_by_filename[filename].model_dump(mode="json") for filename in selected_filenames
        ]
        consensus = ensemble_payload.get("policy_consensus")
        if consensus is not None:
            consensus["policies"] = [
                policy
                for policy in consensus["policies"]
                if policy["member_filename"] in selected_filenames
            ]
            if not consensus["policies"]:
                ensemble_payload["policy_consensus"] = None
                ensemble_payload["draft_refinement"] = None
            else:
                consensus["minimum_agreeing_policy_count"] = min(
                    consensus["minimum_agreeing_policy_count"], len(consensus["policies"])
                )
                draft = ensemble_payload.get("draft_refinement")
                if draft is not None:
                    draft["maximum_agreeing_policy_count"] = min(
                        draft["maximum_agreeing_policy_count"], len(consensus["policies"])
                    )
        selector = ensemble_payload["class_verified_selector"]
        selector["candidate_minimum_support"] = min(
            selector["candidate_minimum_support"], len(selected_filenames)
        )
        if (cascade_primary_member_filename is None) != (
            cascade_secondary_trigger_selected_counts is None
        ):
            raise ValueError(
                "detector cascade primary and trigger counts must be configured together"
            )
        if cascade_primary_member_filename is not None:
            ensemble_payload["parallel_execution"] = False
            ensemble_payload["selective_cascade"] = {
                "primary_member_filename": cascade_primary_member_filename,
                "secondary_trigger_selected_counts": (cascade_secondary_trigger_selected_counts),
                "secondary_trigger_minimum_score_maximum": (
                    cascade_secondary_trigger_minimum_score_maximum
                ),
                "secondary_trigger_minimum_score_minimum": (
                    cascade_secondary_trigger_minimum_score_minimum
                ),
                "secondary_trigger_on_uncertain": cascade_secondary_trigger_on_uncertain,
            }
        selected_ensemble = type(source_detector.ensemble).model_validate(ensemble_payload)
        files = {
            filename: path
            for filename, path in zip(selected_filenames, detector_member_paths, strict=True)
        }
        files["embedder.onnx"] = embedder_path
        detector = DetectorMetadata.model_validate(
            {
                **source_detector.model_dump(mode="json"),
                "filename": selected_filenames[0],
                "version": version,
                "ensemble": selected_ensemble.model_dump(mode="json"),
            }
        )
        detector_refinement = None
    elif detector_refinement_path is None:
        files = {
            "detector.onnx": detector_path,
            "embedder.onnx": embedder_path,
        }
        detector = source_detector.model_copy(
            update={
                "filename": "detector.onnx",
                "version": version,
                "score_threshold": (
                    source_detector.score_threshold
                    if detector_score_threshold is None
                    else detector_score_threshold
                ),
                "ensemble": None,
            }
        )
        detector_refinement = None
    else:
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
    if count_verifier_path is not None:
        files["count-verifier.onnx"] = count_verifier_path
    for filename, path in files.items():
        destination = output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    source_models = {
        name: ModelSource.model_validate(payload) for name, payload in source["sources"].items()
    }
    source_embedder = _embedder_source(source_models)
    detector_source = (
        source_models["detector"]
        if detector_source_path is None
        else ModelSource.model_validate(
            json.loads(detector_source_path.read_text(encoding="utf-8"))
        )
    )
    embedder_report = (
        None
        if embedder_report_path is None
        else json.loads(embedder_report_path.read_text(encoding="utf-8"))
    )
    embedder_kind = (
        "dinov3_convnext_tiny" if embedder_report is None else embedder_report["backbone_kind"]
    )
    embedding_dimension = (
        768 if embedder_report is None else int(embedder_report["embedding_dimension"])
    )
    embedder_image_size = 224 if embedder_report is None else int(embedder_report["image_size"])
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
    elif embedder_kind in {"dinov3_convnext_tiny", "dinov3_vits16"}:
        training_scope = (
            "frozen_backbone"
            if embedder_report is None
            else str(embedder_report.get("training_scope") or "frozen_backbone")
        )
        finetuned = training_scope != "frozen_backbone"
        embedding_space = (
            None if embedder_report is None else embedder_report.get("embedding_space")
        )
        adapted = embedding_space in {"trained_adapter", "trained_adapter_proxy_fusion"}
        proxy_fusion = embedding_space == "trained_adapter_proxy_fusion"
        if embedder_kind == "dinov3_vits16":
            if finetuned or adapted:
                raise ValueError("the DINOv3 ViT-S/16 runtime candidate must stay frozen")
            embedder_id = "dinov3-vits16-frozen"
        else:
            embedder_id = (
                "dinov3-convnext-tiny-adapted-proxy-fusion"
                if proxy_fusion
                else "dinov3-convnext-tiny-adapted"
                if adapted
                else "dinov3-convnext-tiny-last-stage-finetuned"
                if finetuned
                else "dinov3-convnext-tiny-frozen"
            )
        embedder_mean = (0.485, 0.456, 0.406)
        embedder_std = (0.229, 0.224, 0.225)
        source_architecture = (
            "DINOv3 ViT-S/16 frozen CLS embedding backbone"
            if embedder_kind == "dinov3_vits16"
            else "DINOv3 ConvNeXt-Tiny last-stage fine-tuned embedding backbone"
            if finetuned
            else "DINOv3 ConvNeXt-Tiny frozen embedding backbone"
        )
        source_license = source["licenses"]["classifier"]
        selected_approval = 0.4449983835220337 if approval_threshold is None else approval_threshold
        selected_top3 = (
            -2.960296392440796 if top3_safety_threshold is None else top3_safety_threshold
        )
        embedder_source = (
            source_embedder.model_copy(update={"architecture": source_architecture})
            if embedder_report is None
            else ModelSource(
                architecture=source_architecture,
                revision=embedder_report.get("source_revision"),
                weight_filename=embedder_report.get("source_weight_filename"),
                weight_sha256=embedder_report.get("source_weight_sha256"),
                training_pipeline_version=None,
                training_contract_sha256=None,
                training_dataset_version=embedder_report.get("training_dataset_version"),
                training_manifest_sha256=embedder_report.get("training_manifest_sha256"),
            )
        )
    else:
        raise ValueError(f"unsupported 2.0 embedder kind: {embedder_kind}")
    metadata = RuntimePackageV2Metadata(
        worker_version=version,
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
            dense_agreement_count_minimum=(
                min(4, len(detector_member_paths)) if detector_member_paths else 4
            ),
            dense_aspect_ratio_minimum=1.5,
            low_agreement_count_maximum=low_agreement_count_maximum,
            low_agreement_aspect_ratio_minimum=low_agreement_aspect_ratio_minimum,
        ),
        count_verifier=(
            None
            if count_verifier_report is None
            else CountVerifierMetadata(
                filename="count-verifier.onnx",
                version=version,
                input_size=(
                    int(count_verifier_report["image_size"]),
                    int(count_verifier_report["image_size"]),
                ),
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                count_labels=[0, 1],
                comparison_mode="object_presence",
                confidence_threshold=float(count_verifier_report.get("confidence_threshold", 0.5)),
                temperature=float(count_verifier_report.get("temperature", 1.0)),
                resize_reducing_gap=1.0,
            )
        ),
        embedder=EmbedderMetadata(
            filename="embedder.onnx",
            embedder_id=embedder_id,
            version=version,
            input_size=(embedder_image_size, embedder_image_size),
            mean=embedder_mean,
            std=embedder_std,
            crop_margin_ratio=0.05,
            crop_mode="box_resize",
            embedding_dimension=embedding_dimension,
            l2_normalized=bool(embedder_report and embedder_report.get("l2_normalized")),
            resize_reducing_gap=1.0,
            warmup_batch_sizes=[1, 2, 3, 4, 5, 6, 7, 8],
            neighbor_mask=neighbor_mask,
            neighbor_distance_bias=neighbor_distance_bias,
        ),
        metric_projection=MetricProjectionMetadata(
            filename=None,
            input_dimension=embedding_dimension,
            output_dimension=embedding_dimension,
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
            ridge_approval_thresholds=approval_thresholds,
            ridge_disagreement_minimum_margin=(
                disagreement_approval_threshold
                if approval_metric == "l2_normalized_logit_margin"
                else None
            ),
            ridge_disagreement_minimum_pair_probability=(
                disagreement_approval_threshold
                if approval_metric == "top2_pair_probability"
                else None
            ),
            ridge_top3_minimum_inverse_entropy=selected_top3,
            detector_corroboration_minimum_score=(detector_corroboration_minimum_score),
            detector_corroboration_maximum_approval_score=(
                detector_corroboration_maximum_approval_score
            ),
            detector_corroboration_low_similarity_minimum_score=(
                detector_corroboration_low_similarity_minimum_score
            ),
            detector_corroboration_low_similarity_maximum_retrieval=(
                detector_corroboration_low_similarity_maximum_retrieval
            ),
            detector_corroboration_low_similarity_minimum_approval_score=(
                detector_corroboration_low_similarity_minimum_approval_score
            ),
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
            "detector": detector_source.model_copy(
                update={
                    "architecture": (
                        f"fixed {len(detector_member_paths)}-model D-FINE ensemble with selective ambiguity gate"
                        if detector_member_paths
                        else "single D-FINE detector"
                        if detector_refinement_path is None
                        else "single D-FINE checkpoint with 640/768 cross-scale disagreement gate"
                    )
                }
            ),
            "embedder": embedder_source,
            **(
                {}
                if count_verifier_report is None or count_verifier_report_path is None
                else {
                    "count_verifier": ModelSource(
                        architecture=("frozen DINOv3 ViT-S/16 with linear object-presence head"),
                        revision=count_verifier_report.get("source_revision"),
                        weight_filename=count_verifier_report.get("source_weight_filename"),
                        weight_sha256=count_verifier_report.get("source_weight_sha256"),
                        training_pipeline_version=version,
                        training_contract_sha256=sha256_file(count_verifier_report_path),
                        training_dataset_version=count_verifier_report.get(
                            "training_dataset_version"
                        ),
                        training_manifest_sha256=count_verifier_report.get(
                            "training_manifest_sha256"
                        ),
                    )
                }
            ),
        },
    )
    # `jpeg_draft_size=None` is an explicit policy value. Omitting it would make
    # the loader restore InputMetadata's default and silently change inference.
    payload = metadata.model_dump(mode="json")
    payload.pop("promotion_status", None)
    (output_dir / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the Bread Scanner 2.0 runtime candidate")
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--detector-refinement", type=Path)
    parser.add_argument("--detector-score-threshold", type=float)
    parser.add_argument(
        "--detector-source",
        type=Path,
        help="JSON ModelSource provenance for the selected detector checkpoint.",
    )
    parser.add_argument("--count-verifier", type=Path)
    parser.add_argument("--count-verifier-report", type=Path)
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
    parser.add_argument("--detector-corroboration-minimum-score", type=float)
    parser.add_argument("--detector-corroboration-maximum-approval-score", type=float)
    parser.add_argument("--detector-corroboration-low-similarity-minimum-score", type=float)
    parser.add_argument("--detector-corroboration-low-similarity-maximum-retrieval", type=float)
    parser.add_argument(
        "--detector-corroboration-low-similarity-minimum-approval-score", type=float
    )
    parser.add_argument("--disable-neighbor-mask", action="store_true")
    parser.add_argument("--neighbor-distance-bias", type=float, default=0.0)
    parser.add_argument(
        "--approval-thresholds",
        type=float,
        nargs="+",
        help="Optional per-class Ridge approval thresholds in Catalog label order.",
    )
    parser.add_argument("--low-agreement-count-maximum", type=int)
    parser.add_argument("--low-agreement-aspect-ratio-minimum", type=float)
    parser.add_argument("--cascade-primary-member-filename")
    parser.add_argument(
        "--cascade-secondary-trigger-selected-counts",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--cascade-secondary-trigger-minimum-score-maximum",
        type=float,
    )
    parser.add_argument(
        "--cascade-secondary-trigger-minimum-score-minimum",
        type=float,
    )
    parser.add_argument(
        "--cascade-secondary-trigger-on-uncertain",
        action="store_true",
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
        ridge_alpha=args.ridge_alpha,
        support_views_per_source=args.support_views_per_source,
        detector_corroboration_minimum_score=(args.detector_corroboration_minimum_score),
        detector_corroboration_maximum_approval_score=(
            args.detector_corroboration_maximum_approval_score
        ),
        detector_corroboration_low_similarity_minimum_score=(
            args.detector_corroboration_low_similarity_minimum_score
        ),
        detector_corroboration_low_similarity_maximum_retrieval=(
            args.detector_corroboration_low_similarity_maximum_retrieval
        ),
        detector_corroboration_low_similarity_minimum_approval_score=(
            args.detector_corroboration_low_similarity_minimum_approval_score
        ),
        neighbor_mask=not args.disable_neighbor_mask,
        neighbor_distance_bias=args.neighbor_distance_bias,
        approval_thresholds=args.approval_thresholds,
        detector_member_paths=args.detector_members,
        detector_score_threshold=args.detector_score_threshold,
        detector_source_path=args.detector_source,
        count_verifier_path=args.count_verifier,
        count_verifier_report_path=args.count_verifier_report,
        low_agreement_count_maximum=args.low_agreement_count_maximum,
        low_agreement_aspect_ratio_minimum=args.low_agreement_aspect_ratio_minimum,
        cascade_primary_member_filename=args.cascade_primary_member_filename,
        cascade_secondary_trigger_selected_counts=(args.cascade_secondary_trigger_selected_counts),
        cascade_secondary_trigger_minimum_score_maximum=(
            args.cascade_secondary_trigger_minimum_score_maximum
        ),
        cascade_secondary_trigger_minimum_score_minimum=(
            args.cascade_secondary_trigger_minimum_score_minimum
        ),
        cascade_secondary_trigger_on_uncertain=(args.cascade_secondary_trigger_on_uncertain),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
