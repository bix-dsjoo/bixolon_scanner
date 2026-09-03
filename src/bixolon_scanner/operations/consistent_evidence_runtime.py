from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from ..contracts.artifact import directory_content_manifest
from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import RuntimePackageV2Metadata, load_runtime_package_v2


def build_consistent_evidence_runtime(
    source_runtime_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create the 0.1.13 policy source without changing model payloads."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    source = load_runtime_package_v2(source_runtime_dir)
    metadata = source.metadata
    fallback = metadata.classifier_resolution_fallback
    verification = metadata.classifier_verification
    if metadata.detector_class_mode != "class_agnostic":
        raise ValueError("consistent evidence Runtime requires a class-agnostic detector")
    if metadata.detector_primary_classifier_routing is not None:
        raise ValueError("consistent evidence Runtime forbids detector-primary classification")
    if metadata.embedder.input_size != (192, 192):
        raise ValueError("consistent evidence Runtime requires a 192x192 primary embedder")
    if fallback is None or fallback.embedder.input_size != (224, 224):
        raise ValueError("consistent evidence Runtime requires a 224x224 selective detail path")
    if verification is None or verification.independent_embedder.input_size != (160, 160):
        raise ValueError("consistent evidence Runtime requires a 160x160 independent verifier")

    payload = metadata.model_dump(mode="json", exclude_none=True)
    verifier_payload = payload["classifier_verification"]
    verifier_payload["verify_all_approved_candidates"] = False
    verifier_payload["unknown_recapture_on_dual_verifier_rejection"] = False
    verifier_payload["unknown_recapture_on_any_verifier_rejection"] = False
    candidate_metadata = RuntimePackageV2Metadata.model_validate(payload)

    shutil.copytree(source.root, output_dir)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            candidate_metadata.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    candidate = load_runtime_package_v2(output_dir)
    candidate_verification = candidate.metadata.classifier_verification
    if candidate_verification is None:
        raise ValueError("candidate Runtime lost classifier verification metadata")
    if candidate_verification.verify_all_approved_candidates:
        raise ValueError("candidate Runtime must keep verifier selection risk-based")
    if candidate_verification.unknown_recapture_on_any_verifier_rejection:
        raise ValueError("candidate Runtime must not turn one verifier rejection into recapture")

    source_payloads = {
        path.relative_to(source.root).as_posix(): sha256_file(path)
        for path in source.root.rglob("*")
        if path.is_file() and path.name != "metadata.json"
    }
    candidate_payloads = {
        path.relative_to(candidate.root).as_posix(): sha256_file(path)
        for path in candidate.root.rglob("*")
        if path.is_file() and path.name != "metadata.json"
    }
    if candidate_payloads != source_payloads:
        raise ValueError("consistent evidence packaging modified an immutable Runtime payload")

    return {
        "schema_version": "1.0",
        "operation": "build_consistent_evidence_runtime",
        "source_runtime": source.root.as_posix(),
        "candidate_runtime": candidate.root.as_posix(),
        "detector_class_mode": candidate.metadata.detector_class_mode,
        "evidence_order": [
            "ssdlite320_objectness",
            "dinov3_convnext_tiny_192_primary",
            "dinov3_convnext_tiny_224_selective_detail",
            "dinov3_vitb16_160_selective_risk_verifier",
            "decision_pipeline_safety_arbiter",
        ],
        "policy": {
            "classifier_verification.verify_all_approved_candidates": False,
            "classifier_verification.unknown_recapture_on_any_verifier_rejection": False,
            "detector_primary_classifier_routing": None,
        },
        "model_graph_or_weight_changed": False,
        "source_manifest_sha256": directory_content_manifest(source.root)["manifest_sha256"],
        "candidate_manifest_sha256": directory_content_manifest(candidate.root)["manifest_sha256"],
        "metadata_sha256": sha256_file(metadata_path),
        "payload_sha256": dict(sorted(candidate.metadata.checksums.items())),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the class-agnostic 192 -> selective 224 -> ViT 160 consistent evidence Runtime"
        )
    )
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = build_consistent_evidence_runtime(args.source_runtime, args.output_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
