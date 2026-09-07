"""Assemble an experimental Runtime and Catalog from fixed-budget training outputs."""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import RuntimePackageV2Metadata, load_runtime_package_v2
from ..evaluation.routing_audit import audit_routing
from ..training.limited_source import read_jsonl, verify_sources, write_json, write_jsonl
from ..training.ten_shot_manifest import inspect_image
from .catalog_activation import build_catalog
from .classifier_consensus_catalog import build_consensus_catalog


def _package(output: Path, payload: dict, files: dict[str, Path]) -> None:
    if output.exists():
        raise FileExistsError(output)
    payload = copy.deepcopy(payload)
    payload["checksums"] = {name: sha256_file(path) for name, path in files.items()}
    metadata = RuntimePackageV2Metadata.model_validate(payload)
    output.mkdir(parents=True)
    for name, path in files.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write_json(output / "metadata.json", metadata.model_dump(mode="json", exclude_none=True))
    load_runtime_package_v2(output)


def assemble(work: Path, template: Path, output: Path) -> dict:
    source = verify_sources(work / "sources")
    if sha256_file(work / "plan.json") != (work / "plan.sha256").read_text().strip():
        raise ValueError("execution plan changed")
    plan = load_json_config(work / "plan.json")
    if (
        Path(plan["work_dir"]).resolve() != work.resolve()
        or plan["source_manifest_sha256"] != source["source_manifest_sha256"]
    ):
        raise ValueError("assembly plan belongs to another experiment")
    config = load_json_config(Path(plan["config"]))
    if sha256_file(Path(plan["config"])) != plan["config_sha256"]:
        raise ValueError("experiment configuration changed")
    if output.exists():
        raise FileExistsError(output)
    for stage in (
        "train-detector",
        "train-classifier",
        "export-primary",
        "export-detail",
        "export-verifier",
    ):
        evidence = load_json_config(work / f"stages/{stage}.json")
        expected_stage = next(row for row in plan["stages"] if row["name"] == stage)
        if (
            evidence["returncode"] != 0
            or evidence["source_manifest_sha256"] != source["source_manifest_sha256"]
            or set(evidence["output_sha256s"]) != set(expected_stage["outputs"])
            or evidence["argv"] != expected_stage["argv"]
        ):
            raise ValueError(
                "model training/export evidence is missing or belongs to another source"
            )
        for name, digest in evidence["output_sha256s"].items():
            if sha256_file(Path(name)) != digest:
                raise ValueError("trained model or report changed")
    payload = load_json_config(template)
    payload.pop("promotion_status", None)
    version = config["product_version"]
    payload["worker_version"] = version
    payload["detector_policy_version"] = version
    payload["dataset_version"] = "limited220-" + source["source_manifest_sha256"][:12]
    payload["detector"]["version"] = version
    payload["embedder"]["version"] = version
    payload["classifier_policy"]["version"] = version
    payload["classifier_policy"]["ridge_approval_thresholds"] = None
    policy = config["runtime_policy_candidates"]
    margin = policy["approval_margin"][0]
    payload["classifier_policy"]["ridge_approval_minimum_margin"] = margin
    payload["classifier_policy"]["ridge_disagreement_minimum_margin"] = margin
    payload["classifier_policy"]["approval_minimum_margin"] = margin
    payload["classifier_policy"]["support_augmentation"]["views_per_source"] = 0
    verification = payload["classifier_verification"]
    verification["independent_embedder"]["version"] = version
    verification["ambiguity_maximum_approval_score"] = policy[
        "verification_maximum_approval_score"
    ][0]
    verification["verify_all_approved_candidates"] = False
    verification["unknown_recapture_on_any_verifier_rejection"] = False
    verification["unknown_recapture_on_dual_verifier_rejection"] = False
    fallback = payload["classifier_resolution_fallback"]
    fallback["embedder"]["version"] = version
    fallback["minimum_fallback_approval_score"] = margin
    for rule in fallback["approval_disagreement_rules"]:
        rule["maximum_approval_score"] = policy["detail_maximum_approval_score"][0]
        rule["require_detector_disagreement"] = False
    payload["input"]["jpeg_draft_size"] = policy["jpeg_draft_size"][0]
    revision = "6876159a11b4df116f30f667f8c9888617df0751"
    payload["sources"] = {
        "detector": {
            "architecture": "ssdlite320_mobilenet_v3_objectness_220_originals",
            "revision": "sha256:"
            + sha256_file(
                Path(__file__).resolve().parents[1] / "training/ssdlite_objectness_detector.py"
            ),
            "weight_filename": "detector.onnx",
            "weight_sha256": sha256_file(work / "detector/detector.onnx"),
            "training_contract_sha256": plan["config_sha256"],
            "training_pipeline_version": version,
            "training_dataset_version": payload["dataset_version"],
            "training_manifest_sha256": source["source_manifest_sha256"],
        },
        "embedder": {
            "architecture": "dinov3_convnext_tiny_220_originals",
            "revision": revision,
            "training_contract_sha256": plan["config_sha256"],
            "training_pipeline_version": version,
            "training_dataset_version": payload["dataset_version"],
            "training_manifest_sha256": source["source_manifest_sha256"],
        },
        "classifier_verifier": {"architecture": "dinov3_vitb16_frozen", "revision": revision},
    }
    metadata = RuntimePackageV2Metadata.model_validate(payload)
    routing = audit_routing(metadata)
    if not routing["reachable"]:
        raise ValueError(f"experimental Runtime has unreachable routes: {routing['issues']}")
    repository = Path(__file__).resolve().parents[3]
    licenses = {name: repository / name for name in payload["license_files"]}
    files = {
        payload["detector"]["filename"]: work / "detector/detector.onnx",
        payload["embedder"]["filename"]: work / "exports/primary.onnx",
        fallback["embedder"]["filename"]: work / "exports/detail.onnx",
        verification["independent_embedder"]["filename"]: work / "exports/verifier.onnx",
        **licenses,
    }
    _package(output / "runtime", payload, files)
    supports = read_jsonl(work / "sources/classifier.jsonl")
    root = Path(source["dataset_root"])
    # Perceptual duplicate identity is distinct from the physical-item split group.
    for row in supports:
        row["perceptual_group_id"] = (
            "average_hash:" + inspect_image(root / row["image_path"]).average_hash
        )
    support_manifest = output / "catalog-support.jsonl"
    write_jsonl(support_manifest, supports)
    view_names = ("primary", "rotation", "independent")
    for name in view_names:
        view = copy.deepcopy(payload)
        view["classifier_verification"] = None
        view["classifier_resolution_fallback"] = None
        if name == "rotation":
            view["embedder"]["rotation_180_tta"] = True
        elif name == "independent":
            view["embedder"] = copy.deepcopy(verification["independent_embedder"])
            view["metric_projection"] = copy.deepcopy(verification["independent_metric_projection"])
        view_files = {
            view["detector"]["filename"]: files[view["detector"]["filename"]],
            view["embedder"]["filename"]: files[view["embedder"]["filename"]],
            **licenses,
        }
        view_dir = output / f"runtime-{name}"
        _package(view_dir, view, view_files)
        build_catalog(
            view_dir,
            root,
            support_manifest,
            output / f"catalog-{name}",
            store_id="limited220",
            catalog_version=version,
            signing_key=None,
            key_id=None,
            authentication="CHECKSUM-SHA256",
            supports_per_class=10,
            provider="cpu",
            cuda_dll_dir=None,
            decision_head="ridge_adapter" if name == "independent" else "classifier_logits",
        )
    build_consensus_catalog(
        *(output / f"catalog-{name}" for name in view_names), output / "catalog"
    )
    report = {
        "schema_version": "1.0",
        "source_manifest_sha256": source["source_manifest_sha256"],
        "original_count": source["original_count"],
        "runtime": str((output / "runtime").resolve()),
        "catalog": str((output / "catalog").resolve()),
        "routing_audit": routing,
        "template_sha256": sha256_file(template),
        "candidate_policy": "first_prespecified_candidate",
        "evaluation_role": source["evaluation_role"],
        "independent_validation_available": False,
        "product_version": version,
        "production_bundle_modified": False,
    }
    write_json(output / "assembly-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble a fixed-budget experimental Worker Runtime"
    )
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(assemble(args.work_dir.resolve(), args.template.resolve(), args.output_dir.resolve()))


if __name__ == "__main__":
    main()
