from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..contracts.catalog import sha256_file
from .models import require_torch
from .ten_shot_candidates import create_weighted_parameter_soup


def build_seed_soup(
    members: list[Path], output_checkpoint: Path, report_path: Path
) -> dict[str, Any]:
    """Average equivalent source-only seeds without development-label selection."""
    if len(members) < 2:
        raise ValueError("seed soup requires at least two checkpoints")
    torch = require_torch()
    checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in members]
    reference = checkpoints[0]
    reference_contract = reference.get("robust_classifier_training") or {}
    invariant_keys = (
        "source_dataset",
        "mixed_support_sources",
        "manifest_sha256",
        "recipe_sha256",
        "views_per_source",
        "recipe_profile",
        "augmentation_seed",
        "neighbor_mask",
        "crop_margin_ratio",
        "neighbor_distance_bias",
        "trainable_scope",
        "final_training_epochs",
        "development_evaluation_used_for_training_or_selection",
    )
    expected = {key: reference_contract.get(key) for key in invariant_keys}
    if (
        expected["source_dataset"] != "single_objects_3"
        or expected["mixed_support_sources"] is not False
        or expected["development_evaluation_used_for_training_or_selection"] is not False
    ):
        raise ValueError("seed soup members must use only single_objects_3")
    for checkpoint in checkpoints[1:]:
        contract = checkpoint.get("robust_classifier_training") or {}
        if {key: contract.get(key) for key in invariant_keys} != expected:
            raise ValueError("seed soup members do not share one training contract")
        if checkpoint.get("dataset_version") != reference.get("dataset_version"):
            raise ValueError("seed soup members do not share one dataset version")
    weights = [1.0 / len(members)] * len(members)
    provenance = create_weighted_parameter_soup(
        members,
        output_checkpoint,
        weights=weights,
        selection_scope="equivalent_single_objects_3_source_validation_seeds",
        dataset_version=str(reference["dataset_version"]),
        manifest_sha256=str(reference["manifest_sha256"]),
    )
    report = {
        "schema_version": "1.0",
        "operation": "single_objects_3_equal_seed_parameter_soup",
        "training_source": "single_objects_3",
        "mixed_support_sources": False,
        "multi_object_product_labels_used": False,
        "selection_basis": "equal weights across equivalent source-validation seeds",
        "weights": weights,
        "members": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in members],
        "source_dataset_version": reference["dataset_version"],
        "source_manifest_sha256": reference["manifest_sha256"],
        "training_contract": expected,
        "output": str(output_checkpoint.resolve()),
        "output_sha256": sha256_file(output_checkpoint),
        "parameter_soup": provenance,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_source_recipe_soup(
    members: list[Path], output_checkpoint: Path, report_path: Path
) -> dict[str, Any]:
    """Average source-selected recipes without using multi-object product labels."""
    if len(members) < 2:
        raise ValueError("source recipe soup requires at least two checkpoints")
    torch = require_torch()
    checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in members]
    reference = checkpoints[0]
    source_manifest = str(reference.get("manifest_sha256", ""))
    dataset_version = str(reference.get("dataset_version", ""))
    contracts = []
    for checkpoint in checkpoints:
        contract = checkpoint.get("robust_classifier_training") or {}
        if (
            contract.get("source_dataset") != "single_objects_3"
            or contract.get("mixed_support_sources") is not False
            or contract.get("development_evaluation_used_for_training_or_selection") is not False
            or checkpoint.get("manifest_sha256") != source_manifest
            or checkpoint.get("dataset_version") != dataset_version
            or checkpoint.get("backbone_kind") != reference.get("backbone_kind")
            or checkpoint.get("adapter_spec") != reference.get("adapter_spec")
        ):
            raise ValueError("source recipe soup members are not compatible source-only models")
        contracts.append(contract)
    weights = [1.0 / len(members)] * len(members)
    provenance = create_weighted_parameter_soup(
        members,
        output_checkpoint,
        weights=weights,
        selection_scope="equal_complementary_single_objects_3_source_recipes",
        dataset_version=dataset_version,
        manifest_sha256=source_manifest,
    )
    report = {
        "schema_version": "1.0",
        "operation": "single_objects_3_equal_source_recipe_parameter_soup",
        "training_source": "single_objects_3",
        "mixed_support_sources": False,
        "multi_object_product_labels_used": False,
        "selection_basis": "equal weights across source-validation-selected recipes",
        "weights": weights,
        "members": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "training_contract": contract,
            }
            for path, contract in zip(members, contracts)
        ],
        "source_dataset_version": dataset_version,
        "source_manifest_sha256": source_manifest,
        "output": str(output_checkpoint.resolve()),
        "output_sha256": sha256_file(output_checkpoint),
        "parameter_soup": provenance,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build an equal-weight source-only classifier seed soup"
    )
    parser.add_argument("--member", type=Path, action="append", required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--complementary-source-recipes", action="store_true")
    args = parser.parse_args(argv)
    build = build_source_recipe_soup if args.complementary_source_recipes else build_seed_soup
    print(json.dumps(build(args.member, args.output_checkpoint, args.report), indent=2))


if __name__ == "__main__":
    main()
