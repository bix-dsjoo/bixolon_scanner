from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..contracts.catalog import sha256_file
from .models import require_torch
from .ten_shot_candidates import create_weighted_parameter_soup


def build_neutral_soup(
    moderate_checkpoint: Path,
    mild_checkpoint: Path,
    output_checkpoint: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Average two source-only recipes without using multi-object product labels."""

    torch = require_torch()
    members = [moderate_checkpoint, mild_checkpoint]
    checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in members]
    recipes = []
    for checkpoint in checkpoints:
        contract = checkpoint.get("robust_classifier_training") or {}
        if (
            contract.get("source_dataset") != "single_objects_3"
            or contract.get("mixed_support_sources") is not False
            or contract.get("development_evaluation_used_for_training_or_selection") is not False
        ):
            raise ValueError("neutral soup members must be source-only single_objects_3 candidates")
        recipes.append(str(contract.get("recipe_profile")))
    if recipes != ["moderate", "mild"]:
        raise ValueError("neutral soup requires the moderate and mild recipes in that order")
    manifest_sha256 = str(checkpoints[0].get("manifest_sha256"))
    dataset_version = str(checkpoints[0].get("dataset_version"))
    if any(
        checkpoint.get("manifest_sha256") != manifest_sha256
        or checkpoint.get("dataset_version") != dataset_version
        for checkpoint in checkpoints[1:]
    ):
        raise ValueError("neutral soup members must share the same source manifest")
    provenance = create_weighted_parameter_soup(
        members,
        output_checkpoint,
        weights=[0.5, 0.5],
        selection_scope="single_objects_3_source_validation_and_detector_geometry_only",
        dataset_version=dataset_version,
        manifest_sha256=manifest_sha256,
    )
    report = {
        "schema_version": "1.0",
        "operation": "single_objects_3_neutral_classifier_soup",
        "training_source": "single_objects_3",
        "mixed_support_sources": False,
        "multi_object_product_labels_used": False,
        "weights": [0.5, 0.5],
        "recipe_profiles": recipes,
        "selection_basis": (
            "both recipes passed source-only validation; equal weights avoid a development-label "
            "calibration and cover the OOF detector geometry range"
        ),
        "members": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in members],
        "source_dataset_version": dataset_version,
        "source_manifest_sha256": manifest_sha256,
        "output": str(output_checkpoint.resolve()),
        "output_sha256": sha256_file(output_checkpoint),
        "parameter_soup": provenance,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the neutral Scanner 0.1.3 classifier soup")
    parser.add_argument("--moderate-checkpoint", type=Path, required=True)
    parser.add_argument("--mild-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_neutral_soup(
        args.moderate_checkpoint,
        args.mild_checkpoint,
        args.output_checkpoint,
        args.report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
