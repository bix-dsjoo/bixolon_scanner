from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dfine_export import checkpoint_model_state

SCORE_PARAMETER_MARKERS = ("score_head", "denoising_class_embed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_dfine_score_head_states(
    reference: dict[str, Any], donor: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    if set(donor) - set(reference):
        raise ValueError("D-FINE score donor has keys absent from reference")
    merged = {key: value.detach().clone() for key, value in reference.items()}
    transferred = []
    for key, reference_value in reference.items():
        if not any(marker in key for marker in SCORE_PARAMETER_MARKERS):
            continue
        donor_value = donor.get(key)
        if donor_value is None:
            raise ValueError(f"D-FINE score donor is missing parameter: {key}")
        if donor_value.shape != reference_value.shape or donor_value.dtype != reference_value.dtype:
            raise ValueError(f"D-FINE score donor tensor contract differs: {key}")
        merged[key] = donor_value.detach().clone()
        transferred.append(key)
    if not transferred:
        raise ValueError("D-FINE score donor has no score parameters")
    return merged, transferred


def create_dfine_score_head_checkpoint(
    reference_checkpoint: Path,
    donor_checkpoint: Path,
    output: Path,
    *,
    donor_weight_source: str = "model",
) -> dict[str, Any]:
    import torch

    reference_path = reference_checkpoint.resolve()
    donor_path = donor_checkpoint.resolve()
    if reference_path == donor_path:
        raise ValueError("D-FINE score reference and donor must differ")
    reference_payload = torch.load(reference_path, map_location="cpu", weights_only=False)
    donor_payload = torch.load(donor_path, map_location="cpu", weights_only=False)
    reference = checkpoint_model_state(reference_payload, weight_source="auto")
    donor = checkpoint_model_state(donor_payload, weight_source=donor_weight_source)
    merged, transferred = merge_dfine_score_head_states(reference, donor)
    provenance = {
        "recipe": "production_reference_with_score_head_transfer",
        "reference": {
            "path": str(reference_path),
            "sha256": _sha256(reference_path),
            "weight_source": "auto",
        },
        "donor": {
            "path": str(donor_path),
            "sha256": _sha256(donor_path),
            "weight_source": donor_weight_source,
        },
        "transferred_parameter_count": len(transferred),
        "transferred_parameters": transferred,
        "selection_scope": "development_dataset_only",
        "independent_test_claimed": False,
    }
    result = {
        "date": datetime.now(timezone.utc).isoformat(),
        "last_epoch": -1,
        "model": merged,
        "ema": {"module": merged, "updates": 0},
        "detector_score_head_transfer": provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer only D-FINE score-head parameters onto a reference"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--donor-weight-source", choices=("auto", "ema", "model"), default="model")
    parser.add_argument("--source-output", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--training-dataset-version")
    parser.add_argument("--training-manifest", type=Path)
    args = parser.parse_args()
    provenance = create_dfine_score_head_checkpoint(
        args.reference,
        args.donor,
        args.output,
        donor_weight_source=args.donor_weight_source,
    )
    if args.source_output is not None:
        if (args.training_dataset_version is None) != (args.training_manifest is None):
            raise ValueError("training dataset version and manifest are required together")
        source = {
            "architecture": "D-FINE-N production reference with transferred score head",
            "revision": args.source_revision,
            "weight_filename": args.output.name,
            "weight_sha256": _sha256(args.output),
            "training_pipeline_version": None,
            "training_contract_sha256": None,
            "training_dataset_version": args.training_dataset_version,
            "training_manifest_sha256": (
                None if args.training_manifest is None else _sha256(args.training_manifest)
            ),
        }
        args.source_output.parent.mkdir(parents=True, exist_ok=True)
        args.source_output.write_text(
            json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        provenance["model_source"] = source
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
