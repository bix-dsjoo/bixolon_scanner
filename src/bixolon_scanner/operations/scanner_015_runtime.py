from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from ..configuration import load_json_config
from ..contracts.artifact import directory_content_manifest
from ..contracts.catalog import sha256_file
from ..contracts.model_package import CountVerifierMetadata, ModelSource
from ..contracts.runtime_package_v2 import (
    DetectorCrowdingPolicyMetadata,
    RuntimePackageV2Metadata,
    load_runtime_package_v2,
)
from .scanner_014_runtime import add_detector_crowding_policy, default_crowding_policy

PRESENCE_VERIFIER_FILENAME = "count-verifier.onnx"


def corroborated_large_proposal_policy() -> DetectorCrowdingPolicyMetadata:
    """Keep the 0.1.4 crowding policy but require evidence beyond proposal size."""
    payload = default_crowding_policy().model_dump(mode="json")
    payload["large_proposal_corroboration"] = {
        "query_containment_surplus_minimum": 1,
        "selected_center_minimum": 2,
        "selected_count_maximum": 5,
    }
    return DetectorCrowdingPolicyMetadata.model_validate(payload)


def object_presence_verifier_metadata(
    report: dict[str, Any],
    *,
    version: str,
    confidence_threshold: float,
) -> CountVerifierMetadata:
    """Create the Runtime contract for the detector-independent presence verifier."""
    if report.get("model_role") != "generic_object_presence_count_verifier":
        raise ValueError("presence verifier report has an unsupported model role")
    if report.get("comparison_mode") != "object_presence":
        raise ValueError("presence verifier report must use object_presence comparison")
    image_size = int(report["image_size"])
    return CountVerifierMetadata(
        filename=PRESENCE_VERIFIER_FILENAME,
        version=version,
        input_size=(image_size, image_size),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        count_labels=[0, 1],
        comparison_mode="object_presence",
        confidence_threshold=confidence_threshold,
        temperature=float(report.get("temperature", 1.0)),
        resize_reducing_gap=1.0,
    )


def _add_object_presence_verifier(
    runtime_dir: Path,
    *,
    verifier_path: Path,
    verifier_report_path: Path,
    verifier_conversion_report_path: Path | None,
    version: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    report = load_json_config(verifier_report_path)
    verifier_sha256 = sha256_file(verifier_path)
    conversion_report = (
        None
        if verifier_conversion_report_path is None
        else load_json_config(verifier_conversion_report_path)
    )
    if conversion_report is None:
        if report.get("onnx_sha256") != verifier_sha256:
            raise ValueError("presence verifier ONNX checksum does not match its report")
    else:
        conversion_operation = conversion_report.get("operation")
        origin_sha256 = (
            conversion_report.get("source_onnx_sha256")
            if conversion_operation == "fix_public_onnx_batch_and_infer_shapes"
            else conversion_report.get("origin_onnx_sha256")
            if conversion_operation == "optimize_fixed_batch_onnx"
            else None
        )
        if (
            conversion_report.get("fixed_batch_size") != 1
            or origin_sha256 != report.get("onnx_sha256")
            or conversion_report.get("onnx_sha256") != verifier_sha256
            or conversion_report.get("weights_modified") is not False
        ):
            raise ValueError("presence verifier fixed-batch conversion evidence is invalid")

    runtime = load_runtime_package_v2(runtime_dir)
    if runtime.metadata.count_verifier is not None:
        raise ValueError("runtime already contains a count verifier")
    existing_payload_sha256 = {
        filename: sha256_file(runtime.root / filename) for filename in runtime.metadata.checksums
    }
    destination = runtime.root / PRESENCE_VERIFIER_FILENAME
    shutil.copy2(verifier_path, destination)

    payload = runtime.metadata.model_dump(mode="json")
    payload["count_verifier"] = object_presence_verifier_metadata(
        report,
        version=version,
        confidence_threshold=confidence_threshold,
    ).model_dump(mode="json")
    payload["checksums"][PRESENCE_VERIFIER_FILENAME] = verifier_sha256
    report_schema_version = str(report.get("schema_version", ""))
    training_pipeline_version = (
        f"{report_schema_version}.0"
        if report_schema_version.count(".") == 1
        else report_schema_version
    )
    payload["sources"]["count_verifier"] = ModelSource(
        architecture=(
            f"{report.get('backbone_kind', 'DINOv3')} frozen generic object-presence verifier"
        ),
        revision=report.get("source_revision"),
        weight_filename=report.get("source_weight_filename"),
        weight_sha256=report.get("source_weight_sha256"),
        training_pipeline_version=training_pipeline_version,
        training_contract_sha256=sha256_file(verifier_report_path),
        training_dataset_version=report.get("training_dataset_version"),
        training_manifest_sha256=report.get("training_manifest_sha256"),
    ).model_dump(mode="json", exclude_none=True)
    metadata = RuntimePackageV2Metadata.model_validate(payload)
    metadata_path = runtime.root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_runtime_package_v2(runtime.root)
    output_existing_payload_sha256 = {
        filename: sha256_file(loaded.root / filename) for filename in existing_payload_sha256
    }
    if output_existing_payload_sha256 != existing_payload_sha256:
        raise ValueError("presence verifier conversion modified existing Runtime payloads")
    return {
        "filename": PRESENCE_VERIFIER_FILENAME,
        "sha256": verifier_sha256,
        "report": verifier_report_path.resolve().as_posix(),
        "report_sha256": sha256_file(verifier_report_path),
        "conversion_report": (
            None
            if verifier_conversion_report_path is None
            else verifier_conversion_report_path.resolve().as_posix()
        ),
        "conversion_report_sha256": (
            None
            if verifier_conversion_report_path is None
            else sha256_file(verifier_conversion_report_path)
        ),
        "confidence_threshold": confidence_threshold,
        "training_image_count": report.get("training_image_count"),
        "training_empty_image_count": report.get("training_empty_image_count"),
        "training_nonempty_image_count": report.get("training_nonempty_image_count"),
    }


def create_scanner_015_runtime(
    source_runtime_dir: Path,
    output_dir: Path,
    *,
    version: str = "0.1.6",
    presence_verifier_path: Path | None = None,
    presence_verifier_report_path: Path | None = None,
    presence_verifier_conversion_report_path: Path | None = None,
    presence_confidence_threshold: float = 0.54,
) -> dict[str, Any]:
    if (presence_verifier_path is None) != (presence_verifier_report_path is None):
        raise ValueError("presence verifier ONNX and report must be supplied together")
    if presence_verifier_path is None and presence_verifier_conversion_report_path is not None:
        raise ValueError("presence verifier conversion report requires the verifier ONNX")
    report = add_detector_crowding_policy(
        source_runtime_dir,
        output_dir,
        version=version,
        crowding_policy=corroborated_large_proposal_policy(),
    )
    if presence_verifier_path is None or presence_verifier_report_path is None:
        return report
    report["operation"] = "add_corroborated_crowding_and_object_presence_verifier"
    report["object_presence_verifier"] = _add_object_presence_verifier(
        output_dir,
        verifier_path=presence_verifier_path,
        verifier_report_path=presence_verifier_report_path,
        verifier_conversion_report_path=presence_verifier_conversion_report_path,
        version=version,
        confidence_threshold=presence_confidence_threshold,
    )
    loaded = load_runtime_package_v2(output_dir)
    report["payload_sha256"] = dict(sorted(loaded.metadata.checksums.items()))
    report["metadata_sha256"] = sha256_file(output_dir / "metadata.json")
    report["manifest_sha256"] = directory_content_manifest(output_dir)["manifest_sha256"]
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Add corroboration to the Scanner 0.1.6 large-proposal crowding gate"
    )
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="0.1.6")
    parser.add_argument("--presence-verifier", type=Path)
    parser.add_argument("--presence-verifier-report", type=Path)
    parser.add_argument("--presence-verifier-conversion-report", type=Path)
    parser.add_argument("--presence-confidence-threshold", type=float, default=0.54)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = create_scanner_015_runtime(
        args.source_runtime,
        args.output_dir,
        version=args.version,
        presence_verifier_path=args.presence_verifier,
        presence_verifier_report_path=args.presence_verifier_report,
        presence_verifier_conversion_report_path=args.presence_verifier_conversion_report,
        presence_confidence_threshold=args.presence_confidence_threshold,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
