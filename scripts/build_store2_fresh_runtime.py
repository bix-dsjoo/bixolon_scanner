from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.contracts.runtime_package_v2 import RuntimePackageV2Metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an isolated 0.1.12-policy Runtime from fresh Store 2 ONNX files"
    )
    parser.add_argument("--policy-metadata", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--embedder", type=Path, required=True)
    parser.add_argument("--fallback-embedder", type=Path, required=True)
    parser.add_argument("--detector-report", type=Path, required=True)
    parser.add_argument("--embedder-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    payload = load_json_config(args.policy_metadata)
    detector_report = load_json_config(args.detector_report)
    embedder_report = load_json_config(args.embedder_report)
    version = "0.1.12"
    payload["worker_version"] = version
    payload["dataset_version"] = "bread-store2-single4-fresh"
    payload["detector_policy_version"] = version
    payload["detector_class_count"] = 20
    detector = payload["detector"]
    detector.update(
        {
            "filename": "detector.onnx",
            "version": version,
            "input_size": [640, 640],
            "score_threshold": 0.735,
            "nms_iou_threshold": 0.4,
            "max_queries": 100,
            "ensemble": None,
        }
    )
    embedder = payload["embedder"]
    embedder.update(
        {
            "filename": "embedder.onnx",
            "embedder_id": "dinov3-vitb16-store2-pair-rule",
            "version": version,
            "input_size": [192, 192],
            "embedding_dimension": 20,
            "crop_margin_ratio": 0.0,
            "neighbor_mask": True,
            "neighbor_distance_bias": -0.1,
            "neighbor_shared_scale": False,
        }
    )
    payload["metric_projection"].update(
        {
            "filename": None,
            "input_dimension": 20,
            "output_dimension": 20,
            "residual_weight": 1.0,
            "projection_weight": 0.0,
        }
    )
    payload["classifier_policy"]["version"] = version
    verification = payload.get("classifier_verification")
    if verification is not None:
        verification["independent_embedder"].update(
            {
                "filename": "classifier-verifier.onnx",
                "embedder_id": "dinov3-vitb16-store2-pair-rule",
                "version": version,
                "input_size": [192, 192],
                "embedding_dimension": 20,
                "fixed_batch_size": 1,
            }
        )
        verification["independent_metric_projection"].update(
            {
                "filename": None,
                "input_dimension": 20,
                "output_dimension": 20,
                "residual_weight": 1.0,
                "projection_weight": 0.0,
            }
        )
    fallback = payload.get("classifier_resolution_fallback")
    if fallback is not None:
        fallback["embedder"].update(
            {
                "filename": "embedder-fallback-224.onnx",
                "embedder_id": "dinov3-vitb16-store2-pair-rule",
                "version": version,
                "input_size": [224, 224],
                "embedding_dimension": 20,
            }
        )
    payload["sources"] = {
        "detector": {
            "architecture": "DINOv3 ConvNeXt-Tiny frozen backbone with fresh FPN/FCOS objectness head",
            "revision": "6876159a11b4df116f30f667f8c9888617df0751",
            "weight_filename": Path(detector_report["official_weights"]).name,
            "weight_sha256": "21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54",
            "training_dataset_version": "bread-store2-single4-synthetic-v1",
            "training_manifest_sha256": sha256_file(
                Path(
                    "artifacts/experiments/bread-store2-single4-0.1.12/synthetic-v1/manifest.jsonl"
                )
            ),
        },
        "embedder": {
            "architecture": "DINOv3 ViT-B/16 frozen dual-rotation LDA/hybrid pair-rule embedder",
            "revision": "6876159a11b4df116f30f667f8c9888617df0751",
            "weight_filename": Path(embedder_report["official_weights"]).name,
            "weight_sha256": embedder_report["official_weights_sha256"],
            "training_dataset_version": "bread-store2-single4-support-200",
            "training_manifest_sha256": embedder_report["support_manifest_sha256"],
        },
        "classifier_verifier": {
            "architecture": "Fresh Store 2 DINOv3 ViT-B/16 pair-rule verifier view",
            "revision": "6876159a11b4df116f30f667f8c9888617df0751",
            "weight_filename": Path(embedder_report["official_weights"]).name,
            "weight_sha256": embedder_report["official_weights_sha256"],
            "training_dataset_version": "bread-store2-single4-support-200",
            "training_manifest_sha256": embedder_report["support_manifest_sha256"],
        },
    }
    payload["licenses"] = {
        "detector": "DINOv3 License and torchvision BSD 3-Clause",
        "classifier": "DINOv3 License",
    }

    repository_root = Path(__file__).resolve().parents[1]
    files = {
        "detector.onnx": args.detector,
        "embedder.onnx": args.embedder,
        "classifier-verifier.onnx": args.embedder,
        "embedder-fallback-224.onnx": args.fallback_embedder,
        "licenses/TORCHVISION-LICENSE.txt": repository_root / "licenses/TORCHVISION-LICENSE.txt",
        "licenses/DINOV3-LICENSE.md": repository_root / "licenses/DINOV3-LICENSE.md",
        "licenses/THIRD_PARTY_MODELS.md": repository_root / "licenses/THIRD_PARTY_MODELS.md",
    }
    args.output_dir.mkdir(parents=True)
    for filename, source in files.items():
        destination = args.output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    payload["license_files"] = [
        "licenses/TORCHVISION-LICENSE.txt",
        "licenses/DINOV3-LICENSE.md",
        "licenses/THIRD_PARTY_MODELS.md",
    ]
    payload["checksums"] = {
        filename: sha256_file(args.output_dir / filename) for filename in sorted(files)
    }
    metadata = RuntimePackageV2Metadata.model_validate(payload)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata.model_dump(mode="json", exclude_none=True), indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "1.0",
        "version": version,
        "runtime": str(args.output_dir),
        "policy_metadata": str(args.policy_metadata),
        "policy_metadata_sha256": sha256_file(args.policy_metadata),
        "fresh_detector_sha256": sha256_file(args.detector),
        "fresh_embedder_sha256": sha256_file(args.embedder),
        "fresh_fallback_embedder_sha256": sha256_file(args.fallback_embedder),
        "existing_model_checkpoint_onnx_catalog_reused": False,
        "checksums": payload["checksums"],
    }
    (args.output_dir / "build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
