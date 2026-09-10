"""Assemble an audited detector-only variant while preserving classifier payloads."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..configuration import load_json_config
from ..contracts.artifact import directory_content_manifest
from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2
from ..training.three_bakery_data import read_jsonl, write_json


def assemble_detector_variant(
    baseline: Path,
    source: Path,
    output: Path,
    training_report: Path,
    training_contract: Path,
    original_manifest: Path,
    detector_graph: Path | None = None,
) -> dict:
    baseline_model = load_runtime_package_v2(baseline / "runtime")
    source_model = load_runtime_package_v2(source)
    report = load_json_config(training_report)
    contract = load_json_config(training_contract)
    required = {r["image_sha256"] for r in read_jsonl(original_manifest)}
    if report["contract_sha256"] != sha256_file(training_contract):
        raise ValueError("detector training contract changed")
    if contract["original_manifest_sha256"] != sha256_file(original_manifest):
        raise ValueError("detector was trained on different originals")
    if set(report["observed_input_counts"]) != required or any(
        count <= 0 for count in report["observed_input_counts"].values()
    ):
        raise ValueError("detector did not consume every authorized original")
    source_graph = detector_graph or (source / source_model.metadata.detector.filename)
    if sha256_file(source_graph) != report["output_sha256"]["detector.onnx"]:
        raise ValueError("detector graph differs from its training output")
    identity = {
        "baseline_runtime": directory_content_manifest(baseline / "runtime"),
        "baseline_catalog": directory_content_manifest(baseline / "catalog"),
        "detector_sha256": sha256_file(source_graph),
        "source_metadata_sha256": sha256_file(source / "metadata.json"),
        "training_report_sha256": sha256_file(training_report),
        "training_contract_sha256": sha256_file(training_contract),
        "original_manifest_sha256": sha256_file(original_manifest),
        "classifier_and_catalog_unchanged": True,
    }
    if (output / "variant.json").exists():
        if load_json_config(output / "variant.json") != identity:
            raise ValueError("detector variant input changed")
        load_runtime_package_v2(output / "runtime")
        return identity
    if output.exists():
        raise ValueError("incomplete variant directory requires inspection")
    shutil.copytree(baseline / "runtime", output / "runtime")
    shutil.copytree(baseline / "catalog", output / "catalog")
    metadata = load_json_config(output / "runtime/metadata.json")
    filename = metadata["detector"]["filename"]
    metadata["detector"] = source_model.metadata.detector.model_dump(mode="json")
    metadata["detector"]["filename"] = filename
    metadata["detector"]["version"] = baseline_model.metadata.detector.version
    shutil.copy2(source_graph, output / "runtime" / filename)
    metadata["checksums"][filename] = sha256_file(output / "runtime" / filename)
    write_json(output / "runtime/metadata.json", metadata)
    load_runtime_package_v2(output / "runtime")
    write_json(output / "variant.json", identity)
    return identity
