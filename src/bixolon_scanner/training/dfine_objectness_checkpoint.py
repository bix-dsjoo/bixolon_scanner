from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dfine_export import checkpoint_model_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collapse_dfine_state_to_objectness(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert a multi-class D-FINE state into a one-class tuning checkpoint.

    Box, backbone, encoder, and decoder features are preserved.  Linear class
    heads are reduced to their class mean, while the bias uses log-sum-exp so
    the initial scalar logit approximates the union of class logits near the
    feature origin.  Subsequent one-class tuning learns the actual objectness
    decision; the collapsed checkpoint is never a release model by itself.
    """
    import torch

    result = {key: value.detach().clone() for key, value in state.items()}
    score_weights = sorted(key for key in state if "score_head" in key and key.endswith(".weight"))
    score_biases = sorted(key for key in state if "score_head" in key and key.endswith(".bias"))
    if not score_weights or not score_biases:
        raise ValueError("D-FINE checkpoint has no score heads")

    class_counts: set[int] = set()
    converted: list[str] = []
    for key in score_weights:
        value = state[key]
        if value.ndim != 2 or value.shape[0] < 2:
            raise ValueError(f"D-FINE score weight is not multi-class: {key}")
        class_counts.add(int(value.shape[0]))
        result[key] = value.mean(dim=0, keepdim=True)
        converted.append(key)
    for key in score_biases:
        value = state[key]
        if value.ndim != 1 or value.shape[0] < 2:
            raise ValueError(f"D-FINE score bias is not multi-class: {key}")
        class_counts.add(int(value.shape[0]))
        result[key] = torch.logsumexp(value, dim=0, keepdim=True)
        converted.append(key)
    if len(class_counts) != 1:
        raise ValueError(f"D-FINE score heads disagree on class count: {sorted(class_counts)}")
    source_class_count = class_counts.pop()

    denoising_key = "decoder.denoising_class_embed.weight"
    denoising = state.get(denoising_key)
    if denoising is None:
        raise ValueError("D-FINE checkpoint is missing the denoising class embedding")
    if denoising.ndim != 2 or denoising.shape[0] != source_class_count + 1:
        raise ValueError("D-FINE denoising class embedding disagrees with score heads")
    result[denoising_key] = torch.stack((denoising[:-1].mean(dim=0), denoising[-1]), dim=0)
    converted.append(denoising_key)
    return result, {
        "source_class_count": source_class_count,
        "target_class_count": 1,
        "score_weight_reduction": "arithmetic_mean",
        "score_bias_reduction": "logsumexp",
        "score_bias_union_offset": math.log(source_class_count),
        "denoising_object_embedding_reduction": "arithmetic_mean",
        "converted_parameters": converted,
        "requires_one_class_tuning": True,
    }


def create_dfine_objectness_tuning_checkpoint(
    source_checkpoint: Path,
    output: Path,
    *,
    weight_source: str = "auto",
) -> dict[str, Any]:
    import torch

    source = source_checkpoint.resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint_model_state(payload, weight_source=weight_source)
    converted, recipe = collapse_dfine_state_to_objectness(state)
    provenance = {
        "recipe": "multiclass_features_to_one_class_objectness_tuning",
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "weight_source": weight_source,
        },
        **recipe,
        "product_class_outputs_removed": True,
        "release_model": False,
        "independent_test_claimed": False,
    }
    checkpoint = {
        "date": datetime.now(timezone.utc).isoformat(),
        "last_epoch": -1,
        "model": converted,
        "ema": {"module": converted, "updates": 0},
        "detector_objectness_initialization": provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a one-class D-FINE objectness tuning checkpoint"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-source", choices=("auto", "ema", "model"), default="auto")
    args = parser.parse_args()
    report = create_dfine_objectness_tuning_checkpoint(
        args.source,
        args.output,
        weight_source=args.weight_source,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
