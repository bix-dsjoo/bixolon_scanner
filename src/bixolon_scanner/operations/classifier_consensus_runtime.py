from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import RuntimePackageV2Metadata, load_runtime_package_v2


def add_classifier_verification(
    runtime_dir: Path,
    verifier_onnx: Path,
    verifier_report_path: Path,
    output_dir: Path,
    *,
    ambiguity_maximum_approval_score: float = 0.5,
) -> dict:
    """Add a fixed-batch independent verifier to a Scanner 0.1.3 Runtime."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    runtime = load_runtime_package_v2(runtime_dir)
    report = json.loads(verifier_report_path.read_text(encoding="utf-8"))
    if report.get("training_scope") != "frozen_backbone":
        raise ValueError("classifier verifier must use a frozen product-independent backbone")
    if report.get("onnx_sha256") != sha256_file(verifier_onnx):
        raise ValueError("classifier verifier ONNX checksum differs from its report")
    fixed_batch_size = report.get("fixed_batch_size")
    if not isinstance(fixed_batch_size, int) or fixed_batch_size < 1:
        raise ValueError("classifier verifier must use a fixed positive batch size")
    if report.get("hardware_specific_optimization") is not False:
        raise ValueError("classifier verifier graph must use portable ONNX optimization")

    verifier_filename = "classifier-verifier.onnx"
    payload = runtime.metadata.model_dump(mode="json")
    dimension = int(report["embedding_dimension"])
    verifier_embedder_id = str(report["backbone_kind"]).replace("_", "-")
    if report.get("training_scope") == "frozen_backbone" and not verifier_embedder_id.endswith(
        "-frozen"
    ):
        verifier_embedder_id += "-frozen"
    payload["classifier_verification"] = {
        "ambiguity_maximum_approval_score": ambiguity_maximum_approval_score,
        "rotation_degrees": 180,
        "independent_embedder": {
            "filename": verifier_filename,
            "embedder_id": verifier_embedder_id,
            "version": runtime.metadata.classifier_policy.version,
            "input_name": "pixel_values",
            "output_name": "embeddings",
            "input_size": [int(report["image_size"]), int(report["image_size"])],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_margin_ratio": runtime.metadata.embedder.crop_margin_ratio,
            "crop_mode": runtime.metadata.embedder.crop_mode,
            "embedding_dimension": dimension,
            "l2_normalized": bool(report["l2_normalized"]),
            "resize_reducing_gap": runtime.metadata.embedder.resize_reducing_gap,
            "warmup_batch_sizes": [fixed_batch_size],
            "fixed_batch_size": fixed_batch_size,
            "horizontal_flip_tta": False,
            "rotation_180_tta": False,
            "neighbor_mask": runtime.metadata.embedder.neighbor_mask,
            "neighbor_distance_bias": runtime.metadata.embedder.neighbor_distance_bias,
            "neighbor_shared_scale": runtime.metadata.embedder.neighbor_shared_scale,
        },
        "independent_metric_projection": {
            "filename": None,
            "input_dimension": dimension,
            "output_dimension": dimension,
            "residual_weight": 1.0,
            "projection_weight": 0.0,
        },
    }
    payload["checksums"][verifier_filename] = sha256_file(verifier_onnx)
    payload["sources"]["classifier_verifier"] = {
        "architecture": report["training_architecture"],
        "revision": report["source_revision"],
        "weight_filename": report["source_weight_filename"],
        "weight_sha256": report["source_weight_sha256"],
        "training_pipeline_version": None,
        "training_contract_sha256": None,
        "training_dataset_version": None,
        "training_manifest_sha256": None,
    }
    metadata = RuntimePackageV2Metadata.model_validate(payload)
    shutil.copytree(runtime.root, output_dir)
    shutil.copy2(verifier_onnx, output_dir / verifier_filename)
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
        "operation": "add_selective_classifier_verification",
        "runtime": output_dir.resolve().as_posix(),
        "ambiguity_maximum_approval_score": (
            loaded.metadata.classifier_verification.ambiguity_maximum_approval_score
        ),
        "verifier_fixed_batch_size": (
            loaded.metadata.classifier_verification.independent_embedder.fixed_batch_size
        ),
        "verifier_onnx_sha256": sha256_file(loaded.verification_embedder_path),
        "metadata_sha256": sha256_file(output_dir / "metadata.json"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Add the Scanner 0.1.3 classifier verifier")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--verifier-onnx", type=Path, required=True)
    parser.add_argument("--verifier-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ambiguity-maximum-approval-score", type=float, default=0.5)
    args = parser.parse_args(argv)
    report = add_classifier_verification(
        args.runtime,
        args.verifier_onnx,
        args.verifier_report,
        args.output_dir,
        ambiguity_maximum_approval_score=args.ambiguity_maximum_approval_score,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
