from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ...contracts.runtime_package_v2 import load_runtime_package_v2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict:
    if args.class_agnostic and args.enable_detector_class_consensus:
        raise ValueError("class-agnostic detector cannot enable detector-class consensus")
    if args.drop_inherited_fallback_approval_rules and not args.class_agnostic:
        raise ValueError("dropping class-coupled fallback rules requires a class-agnostic detector")
    if (
        args.drop_inherited_fallback_approval_rules
        and args.minimum_inherited_fallback_detection_count is not None
    ):
        raise ValueError(
            "cannot both drop and selectively retain inherited fallback approval rules"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    shutil.copytree(args.source_runtime, args.output_dir)
    output = args.output_dir
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    source_detector = metadata["detector"]
    detector_filenames = {source_detector["filename"]}
    ensemble = source_detector.get("ensemble")
    if ensemble is not None:
        detector_filenames.update(member["filename"] for member in ensemble["members"])
    for filename in detector_filenames:
        path = output / filename
        if path.exists():
            path.unlink()
    destination = output / "detector.onnx"
    shutil.copy2(args.detector, destination)

    for obsolete_license in ("D-FINE-LICENSE.txt", "APACHE-2.0.txt"):
        path = output / "licenses" / obsolete_license
        if path.exists():
            path.unlink()
    shutil.copy2(args.torchvision_license, output / "licenses" / "TORCHVISION-LICENSE.txt")
    shutil.copy2(args.third_party_notice, output / "licenses" / "THIRD_PARTY_MODELS.md")

    metadata["detector"] = {
        "filename": destination.name,
        "version": args.version,
        "input_name": "pixel_values",
        "logits_output": "logits",
        "boxes_output": "pred_boxes",
        "input_size": [320, 320],
        "color_order": "RGB",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "score_threshold": args.score_threshold,
        "nms_iou_threshold": args.nms_threshold,
        "nms_containment_threshold": None,
        "nms_class_aware_containment": False,
        "max_object_aspect_ratio": 5.0,
        "max_queries": args.maximum_queries,
        "box_format": "normalized_cxcywh",
        "resize_reducing_gap": 1.0,
        "ensemble": None,
    }
    metadata["detector_class_mode"] = "class_agnostic" if args.class_agnostic else "class_aware"
    metadata["detector_class_count"] = 1 if args.class_agnostic else 20
    metadata["detector_policy_version"] = args.version
    metadata["embedder"]["crop_margin_ratio"] = args.crop_margin_ratio
    fallback = metadata.get("classifier_resolution_fallback")
    if fallback is not None:
        fallback["embedder"]["crop_margin_ratio"] = args.crop_margin_ratio
        fallback["selective_roi_only"] = args.selective_classifier_fallback
    metadata["quality"].pop("detector_classifier_consensus", None)
    if args.enable_detector_class_consensus:
        metadata["quality"]["detector_classifier_consensus"] = {
            "minimum_detector_score": args.consensus_minimum_detector_score,
            "approved_disagreement_maximum_classifier_score": (
                args.consensus_disagreement_maximum_classifier_score
            ),
            "promote_safe_unknown_on_top3_consensus": True,
            "promote_recapture_on_top1_consensus": True,
            "inject_detector_class_into_unknown_top3": True,
        }
    if args.class_agnostic and fallback is not None:
        if args.drop_inherited_fallback_approval_rules:
            fallback["approval_disagreement_rules"] = []
        else:
            inherited_rules = fallback.get("approval_disagreement_rules", [])
            if args.minimum_inherited_fallback_detection_count is not None:
                inherited_rules = [
                    rule
                    for rule in inherited_rules
                    if int(rule["minimum_detection_count"])
                    >= args.minimum_inherited_fallback_detection_count
                ]
                fallback["approval_disagreement_rules"] = inherited_rules
            for rule in inherited_rules:
                rule["require_detector_disagreement"] = False
        if args.fallback_minimum_box_aspect_ratio is not None:
            fallback["approval_disagreement_rules"].append(
                {
                    "minimum_detection_count": 1,
                    "maximum_approval_score": args.fallback_maximum_approval_score,
                    "require_detector_disagreement": False,
                    "minimum_box_aspect_ratio": args.fallback_minimum_box_aspect_ratio,
                    "maximum_approval_score_decrease": (
                        args.fallback_maximum_approval_score_decrease
                    ),
                }
            )
        fallback["fuse_unapproved_top3"] = args.fuse_unapproved_top3
        fallback["minimum_fallback_approval_score"] = args.minimum_fallback_approval_score
    metadata.pop("detector_ambiguity", None)
    metadata["detector_crowding"] = None
    metadata["licenses"]["detector"] = "BSD 3-Clause: https://github.com/pytorch/vision"
    metadata["license_files"] = [
        "licenses/TORCHVISION-LICENSE.txt",
        "licenses/DINOV3-LICENSE.md",
        "licenses/THIRD_PARTY_MODELS.md",
    ]
    metadata["sources"]["detector"] = {
        "architecture": (
            "torchvision SSDLite320 MobileNetV3-Large "
            + (
                "class-agnostic bread/object detector, "
                if args.class_agnostic
                else "class-aware detector, "
            )
            + "random initialization without pretrained weights"
        ),
        "revision": args.torchvision_revision,
        "weight_filename": args.checkpoint.name,
        "weight_sha256": _sha256(args.checkpoint),
    }
    removed_count_verifier = None
    if args.disable_count_verifier and metadata.get("count_verifier") is not None:
        removed_count_verifier = metadata["count_verifier"]["filename"]
        count_verifier_path = output / removed_count_verifier
        if count_verifier_path.exists():
            count_verifier_path.unlink()
        metadata["count_verifier"] = None
        metadata["sources"].pop("count_verifier", None)
    metadata["checksums"] = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    runtime = load_runtime_package_v2(output)
    if runtime.metadata.detector.ensemble is not None:
        raise RuntimeError("SSDLite diagnostic Runtime must contain one detector")

    return {
        "schema_version": "1.0",
        "operation": "build_ssdlite_runtime_variant",
        "source_runtime": args.source_runtime.resolve().as_posix(),
        "output_runtime": output.resolve().as_posix(),
        "detector_sha256": _sha256(destination),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "training_report_sha256": _sha256(args.training_report),
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "maximum_queries": args.maximum_queries,
        "crop_margin_ratio": args.crop_margin_ratio,
        "selective_classifier_fallback": args.selective_classifier_fallback,
        "detector_class_mode": metadata["detector_class_mode"],
        "detector_class_count": metadata["detector_class_count"],
        "detector_class_consensus": metadata["quality"].get("detector_classifier_consensus"),
        "fallback_minimum_box_aspect_ratio": args.fallback_minimum_box_aspect_ratio,
        "fallback_maximum_approval_score": args.fallback_maximum_approval_score,
        "fallback_maximum_approval_score_decrease": (args.fallback_maximum_approval_score_decrease),
        "fuse_unapproved_top3": args.fuse_unapproved_top3,
        "minimum_fallback_approval_score": args.minimum_fallback_approval_score,
        "drop_inherited_fallback_approval_rules": (args.drop_inherited_fallback_approval_rules),
        "minimum_inherited_fallback_detection_count": (
            args.minimum_inherited_fallback_detection_count
        ),
        "pretrained_weights_used": False,
        "removed_detector_files": sorted(detector_filenames),
        "removed_count_verifier": removed_count_verifier,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single-model SSDLite Runtime variant")
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--torchvision-license", type=Path, required=True)
    parser.add_argument("--third-party-notice", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--maximum-queries", type=int, default=3234)
    parser.add_argument("--crop-margin-ratio", type=float, default=0.0)
    parser.add_argument("--disable-count-verifier", action="store_true")
    parser.add_argument("--selective-classifier-fallback", action="store_true")
    parser.add_argument("--class-agnostic", action="store_true")
    parser.add_argument(
        "--drop-inherited-fallback-approval-rules",
        action="store_true",
        help=(
            "For a class-agnostic detector, remove fallback rules that depended on "
            "detector/classifier disagreement instead of broadening them to every ROI"
        ),
    )
    parser.add_argument(
        "--minimum-inherited-fallback-detection-count",
        type=int,
        help=(
            "For a class-agnostic detector, retain only inherited fallback approval "
            "rules whose minimum detection count is at least this value"
        ),
    )
    parser.add_argument("--fallback-minimum-box-aspect-ratio", type=float)
    parser.add_argument("--fallback-maximum-approval-score", type=float, default=0.85)
    parser.add_argument(
        "--fallback-maximum-approval-score-decrease",
        type=float,
        default=0.2,
    )
    parser.add_argument("--fuse-unapproved-top3", action="store_true")
    parser.add_argument("--minimum-fallback-approval-score", type=float)
    parser.add_argument("--enable-detector-class-consensus", action="store_true")
    parser.add_argument("--consensus-minimum-detector-score", type=float, default=0.735)
    parser.add_argument(
        "--consensus-disagreement-maximum-classifier-score",
        type=float,
        default=0.8,
    )
    parser.add_argument("--torchvision-revision", default="v0.28.0")
    parser.add_argument("--version", default="0.1.7")
    args = parser.parse_args()
    report = build(args)
    rendered = json.dumps(report, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
