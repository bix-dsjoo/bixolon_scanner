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


def build(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    shutil.copytree(args.source_runtime, args.output_dir)
    output = args.output_dir
    (output / "detector.onnx").unlink()
    agpl = output / "licenses" / "AGPL-3.0.txt"
    if agpl.exists():
        agpl.unlink()

    member_names = [
        "detector-fold0.onnx",
        "detector-fold1.onnx",
        "detector-fold2.onnx",
        "detector-production.onnx",
    ]
    for source, name in zip(args.detector_members, member_names, strict=True):
        shutil.copy2(source, output / name)
    shutil.copy2(args.count_verifier, output / "count-verifier.onnx")
    shutil.copy2(args.dfine_license, output / "licenses" / "D-FINE-LICENSE.txt")
    shutil.copy2(args.third_party_notice, output / "licenses" / "THIRD_PARTY_MODELS.md")

    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    historic = json.loads(args.ensemble_metadata.read_text(encoding="utf-8"))
    detector = historic["detector"]
    detector["filename"] = member_names[0]
    detector["version"] = args.version
    detector["ensemble"]["members"] = [
        {"filename": name, "weight": 1.0, "score_threshold": 0.02} for name in member_names
    ]
    detector["ensemble"]["parallel_execution"] = True
    detector["ensemble"]["cuda_graph_execution"] = False
    detector["ensemble"]["fusion"]["class_agnostic_output"] = True
    detector["ensemble"]["base_selection"] = {
        "score_threshold": 0.4,
        "nms_iou_threshold": 0.5,
        "containment_threshold": 0.85,
        "group_minimum": 2,
    }
    detector["ensemble"]["draft_refinement"] = None
    detector["ensemble"]["class_verified_selector"] = {
        "candidate_minimum_score": 0.03,
        "candidate_minimum_support": 3,
        "candidate_duplicate_iou": 0.9,
        "base_match_iou": 0.9,
        "group_relation_iou": 0.3,
        "group_area_ratio": 0.8,
        "group_margin_ratio": 0.5,
        "group_novel_margin": 0.2,
        "group_minimum_score": 0.04,
        "independent_maximum_iou": 0.3,
        "independent_margin": 0.5,
        "independent_minimum_score": 0.04,
        "classifier_batch_size": 96,
        "unique_class_per_image_contract": True,
        "low_resolution_maximum_dimension": 1000,
        "low_resolution_score_threshold": 0.58,
        "count_assistance_minimum_dimension": 1000,
        "count_assistance_on_count_mismatch": True,
        "zero_count_minimum_confidence": 0.6,
        "refinement_candidate_maximum_support": 3,
        "wide_pair_detector_class_index": 1,
        "wide_pair_classifier_class_index": 2,
        "wide_pair_target_aspect_ratio": 4.5,
        "unknown_consensus_promotion": True,
        "unknown_promotion_minimum_single_view_approval": 0.1,
        "single_view_top3_fusion": True,
    }
    metadata["detector"] = detector
    metadata["detector_class_count"] = 20
    metadata["detector_policy_version"] = args.version
    metadata["detector_crowding"] = None
    count_metadata = metadata["count_verifier"]
    count_metadata.update(
        {
            "version": args.version,
            "count_labels": [0, 1, 3, 4, 5, 6, 7, 8],
            "comparison_mode": "exact_count",
            "confidence_threshold": 0.0,
            "temperature": 1.4640856959456257,
        }
    )
    metadata["licenses"]["detector"] = "Apache License 2.0: https://github.com/Peterande/D-FINE"
    metadata["license_files"] = [
        "licenses/APACHE-2.0.txt",
        "licenses/D-FINE-LICENSE.txt",
        "licenses/DINOV3-LICENSE.md",
        "licenses/THIRD_PARTY_MODELS.md",
    ]
    metadata["sources"]["detector"] = {
        "architecture": "D-FINE-N fixed four-model ensemble with HGNetv2 backbone",
        "revision": "267a6da",
    }
    metadata["sources"]["count_verifier"] = {
        "architecture": "DINOv3 ViT-S/16 frozen exact-count auxiliary verifier",
        "revision": "6876159a11b4df116f30f667f8c9888617df0751",
    }
    metadata["checksums"] = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    load_runtime_package_v2(output)
    print(json.dumps({"output": str(output), "file_count": len(metadata["checksums"])}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the YOLO-free 0.1.7 diagnostic runtime")
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--ensemble-metadata", type=Path, required=True)
    parser.add_argument("--detector-members", type=Path, nargs=4, required=True)
    parser.add_argument("--count-verifier", type=Path, required=True)
    parser.add_argument("--dfine-license", type=Path, required=True)
    parser.add_argument("--third-party-notice", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="0.1.7")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
