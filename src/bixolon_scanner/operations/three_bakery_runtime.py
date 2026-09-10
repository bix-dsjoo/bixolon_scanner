"""Assemble fresh Runtime/Catalog payloads from audited three_bakery runs."""

from __future__ import annotations

import copy
import shutil
from collections import Counter
from pathlib import Path

from ..configuration import load_json_config
from ..contracts.catalog import load_store_catalog_package, sha256_file
from ..contracts.runtime_package_v2 import RuntimePackageV2Metadata, load_runtime_package_v2
from ..evaluation.routing_audit import audit_routing
from ..training.ten_shot_manifest import inspect_image
from ..training.three_bakery_data import read_jsonl, write_json, write_jsonl
from ..training.three_bakery_preparation import verify_annotations
from .catalog_activation import build_catalog
from .classifier_consensus_catalog import build_consensus_catalog


def package(output: Path, payload: dict, files: dict[str, Path]) -> None:
    payload = copy.deepcopy(payload)
    payload["checksums"] = {name: sha256_file(path) for name, path in files.items()}
    metadata = RuntimePackageV2Metadata.model_validate(payload)
    serialized = metadata.model_dump(mode="json", exclude_none=True)
    if output.exists():
        if load_json_config(output / "metadata.json") != serialized:
            raise ValueError("existing runtime belongs to different assembly inputs")
        load_runtime_package_v2(output)
        return
    output.mkdir(parents=True)
    for name, path in files.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write_json(output / "metadata.json", serialized)
    load_runtime_package_v2(output)


def verify_training(directory: Path, source: dict, annotation_sha256: str) -> dict:
    report = load_json_config(directory / "report.json")
    contract = load_json_config(directory / "contract.json")
    if (
        report["contract_sha256"] != sha256_file(directory / "contract.json")
        or contract["source_manifest_sha256"] != source["source_manifest_sha256"]
        or contract["config_sha256"] != source["config_sha256"]
        or contract["annotation_sha256"] != annotation_sha256
        or report["epochs"] != contract["settings"]["epochs"]
    ):
        raise ValueError(
            "trained model evidence does not satisfy the source/config/annotation contract"
        )
    for name, digest in report["output_sha256"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError("trained output checksum mismatch")
    return contract


def assemble(
    work: Path,
    architecture: str,
    method: str,
    recipe: str,
    seed: int,
    template: Path,
    *,
    provider: str = "cpu",
    cuda_dll_dir: Path | None = None,
    roi_integrity_config: Path | None = None,
) -> Path:
    source, _ = verify_annotations(work / "sources")
    annotation_sha256 = sha256_file(work / "sources/annotations.jsonl")
    config = load_json_config(Path(source["config_path"]))
    detector_dir = work / f"models/{architecture}-{recipe}-{seed}"
    classifier_dir = work / f"models/{method}-{recipe}-{seed}"
    detector_contract = verify_training(detector_dir, source, annotation_sha256)
    classifier_contract = verify_training(classifier_dir, source, annotation_sha256)
    verifier_dir = work / "models/verifier"
    verifier_report = load_json_config(verifier_dir / "report.json")
    if verifier_report["onnx_sha256"] != sha256_file(verifier_dir / "verifier.onnx"):
        raise ValueError("frozen verifier checksum mismatch")
    candidate_id = f"{architecture}-{method}-{recipe}"
    output = work / f"candidates/{candidate_id}-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    payload = load_json_config(template)
    version = config["product_version"]
    policy = config["runtime_policy"]
    payload["dataset_version"] = "three_bakery-" + source["source_manifest_sha256"][:12]
    payload["worker_version"] = payload["detector_policy_version"] = version
    payload["detector_class_mode"], payload["detector_class_count"] = "class_agnostic", 1
    detector = payload["detector"]
    detector["version"] = version
    detector["input_size"] = [config["training"][architecture]["image_size"]] * 2
    detector["score_threshold"] = policy["detector_score_threshold"]
    detector["nms_iou_threshold"] = policy["detector_nms_threshold"]
    if architecture == "dfine":
        detector["mean"], detector["std"], detector["max_queries"] = [0.0] * 3, [1.0] * 3, 300
        payload["licenses"]["detector"] = "Apache-2.0 (D-FINE)"
    payload["embedder"]["version"] = version
    # Different trained feature spaces must not accept each other's valid Catalogs.
    payload["embedder"]["embedder_id"] = (
        "dinov3-convnext-tiny-" + sha256_file(classifier_dir / "primary.onnx")[:24]
    )
    payload["classifier_policy"]["version"] = version
    for key in (
        "approval_minimum_margin",
        "ridge_approval_minimum_margin",
        "ridge_disagreement_minimum_margin",
    ):
        payload["classifier_policy"][key] = policy["approval_margin"]
    payload["classifier_policy"]["support_augmentation"]["views_per_source"] = 0
    verification = payload["classifier_verification"]
    verification["independent_embedder"]["version"] = version
    verification["ambiguity_maximum_approval_score"] = policy["ambiguity_maximum_approval_score"]
    verification["verify_all_approved_candidates"] = False
    verification["unknown_recapture_on_any_verifier_rejection"] = False
    verification["unknown_recapture_on_dual_verifier_rejection"] = False
    fallback = payload["classifier_resolution_fallback"]
    fallback["embedder"]["version"] = version
    fallback["embedder"]["embedder_id"] = payload["embedder"]["embedder_id"]
    fallback["minimum_fallback_approval_score"] = policy["approval_margin"]
    for rule in fallback["approval_disagreement_rules"]:
        rule["maximum_approval_score"] = policy["ambiguity_maximum_approval_score"]
        rule["require_detector_disagreement"] = False
    payload["input"]["jpeg_draft_size"] = policy["jpeg_draft_size"]
    payload["sources"] = {
        "detector": {
            "architecture": architecture + "_class_agnostic",
            "revision": "sha256:" + detector_contract["code_sha256"],
            "weight_filename": detector["filename"],
            "weight_sha256": sha256_file(detector_dir / "detector.onnx"),
            "training_contract_sha256": sha256_file(detector_dir / "contract.json"),
            "training_pipeline_version": version,
            "training_dataset_version": payload["dataset_version"],
            "training_manifest_sha256": source["source_manifest_sha256"],
        },
        "embedder": {
            "architecture": "dinov3_convnext_tiny_" + method,
            "revision": "sha256:" + classifier_contract["code_sha256"],
            "training_contract_sha256": sha256_file(classifier_dir / "contract.json"),
            "training_pipeline_version": version,
            "training_dataset_version": payload["dataset_version"],
            "training_manifest_sha256": source["source_manifest_sha256"],
        },
        "classifier_verifier": {
            "architecture": "dinov3_vitb16_frozen",
            "revision": "sha256:" + verifier_report["weights_sha256"],
        },
    }
    repository = Path(__file__).resolve().parents[3]
    licenses = {name: repository / name for name in payload["license_files"]}
    files = {
        detector["filename"]: detector_dir / "detector.onnx",
        payload["embedder"]["filename"]: classifier_dir / "primary.onnx",
        fallback["embedder"]["filename"]: classifier_dir / "detail.onnx",
        verification["independent_embedder"]["filename"]: verifier_dir / "verifier.onnx",
        **licenses,
    }
    if roi_integrity_config is not None:
        from ..training.roi_integrity import train as train_roi_integrity

        settings = load_json_config(roi_integrity_config)
        if Path(settings["source_work"]).resolve() != work.resolve():
            raise ValueError("ROI integrity training and candidate work directories differ")
        integrity = train_roi_integrity(
            roi_integrity_config, method, recipe, seed, embedder_metadata=payload["embedder"]
        )
        integrity_dir = work / f"roi-integrity/{method}-{recipe}-{seed}"
        files[payload["embedder"]["filename"]] = integrity_dir / "primary.onnx"
        payload["embedder"]["multi_object_output_name"] = "multi_object_probabilities"
        payload["embedder"]["embedder_id"] = "dinov3-convnext-tiny-" + integrity["onnx_sha256"][:24]
        fallback["embedder"]["embedder_id"] = payload["embedder"]["embedder_id"]
        payload["quality"]["multi_object_recapture_threshold"] = settings[
            "multi_object_probability_threshold"
        ]
        payload["sources"]["roi_integrity"] = {
            "architecture": "shared_convnext_roi_object_count_head",
            "revision": "sha256:" + integrity["head_sha256"],
            "training_manifest_sha256": integrity["samples_sha256"],
            "training_dataset_version": payload["dataset_version"],
            "weight_filename": payload["embedder"]["filename"],
            "weight_sha256": integrity["onnx_sha256"],
        }
    package(output / "runtime", payload, files)
    metadata = load_runtime_package_v2(output / "runtime").metadata
    routing = audit_routing(metadata)
    if not routing["reachable"]:
        raise ValueError("assembled runtime has unreachable classifier routes")
    supports = read_jsonl(work / "prepared/original_crops.jsonl")
    seen, selected = set(), []
    for row in supports:
        path = Path(row["image_path"])
        perceptual = inspect_image(path).average_hash
        if perceptual in seen:
            continue
        seen.add(perceptual)
        selected.append(
            {
                **row,
                "image_path": path.relative_to((work / "prepared").resolve()).as_posix(),
                "perceptual_group_id": "average_hash:" + perceptual,
            }
        )
    per_class = config["shots_per_class"]
    if min(Counter(row["class_id"] for row in selected).values()) < per_class:
        raise ValueError("not enough distinct reviewed object crops for the Catalog")
    if len({row["class_id"] for row in selected}) != config["class_count"]:
        raise ValueError("Catalog duplicate filtering removed a class")
    balanced = []
    for class_id in sorted({row["class_id"] for row in selected}):
        balanced.extend(
            sorted(
                (row for row in selected if row["class_id"] == class_id),
                key=lambda row: row["image_sha256"],
            )[:per_class]
        )
    support_manifest = output / "catalog-support.jsonl"
    write_jsonl(support_manifest, balanced)
    for name in ("primary", "rotation", "independent"):
        view = copy.deepcopy(payload)
        view["classifier_verification"] = view["classifier_resolution_fallback"] = None
        if name == "rotation":
            view["embedder"]["rotation_180_tta"] = True
        elif name == "independent":
            view["embedder"] = copy.deepcopy(verification["independent_embedder"])
            view["metric_projection"] = copy.deepcopy(verification["independent_metric_projection"])
            view["quality"].pop("multi_object_recapture_threshold", None)
            view["sources"].pop("roi_integrity", None)
        view_files = {
            detector["filename"]: files[detector["filename"]],
            view["embedder"]["filename"]: files[view["embedder"]["filename"]],
            **licenses,
        }
        package(output / f"runtime-{name}", view, view_files)
        catalog_dir = output / f"catalog-{name}"
        if catalog_dir.exists():
            load_store_catalog_package(catalog_dir, expected_store_id="three_bakery")
        else:
            build_catalog(
                output / f"runtime-{name}",
                work / "prepared",
                support_manifest,
                catalog_dir,
                store_id="three_bakery",
                catalog_version=version,
                signing_key=None,
                key_id=None,
                authentication="CHECKSUM-SHA256",
                supports_per_class=per_class,
                provider=provider,
                cuda_dll_dir=cuda_dll_dir,
                decision_head="ridge_adapter" if name == "independent" else "classifier_logits",
            )
    if not (output / "catalog").exists():
        build_consensus_catalog(
            *(output / f"catalog-{name}" for name in ("primary", "rotation", "independent")),
            output / "catalog",
        )
    load_store_catalog_package(output / "catalog", expected_store_id="three_bakery")
    write_json(
        output / "assembly-report.json",
        {
            "candidate_id": candidate_id,
            "seed": seed,
            "source_manifest_sha256": source["source_manifest_sha256"],
            "template_sha256": sha256_file(template),
            "runtime_metadata_sha256": sha256_file(output / "runtime/metadata.json"),
            "catalog_metadata_sha256": sha256_file(output / "catalog/catalog.json"),
            "routing": routing,
            "catalog_supports_per_class": per_class,
            "all_originals_used_for_training": True,
            "evaluation_role": source["evaluation_role"],
            "product_version": version,
        },
    )
    return output
