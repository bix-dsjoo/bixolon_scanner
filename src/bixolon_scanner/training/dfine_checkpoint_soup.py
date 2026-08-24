from __future__ import annotations

import argparse
import hashlib
import json
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


def _normalized_weights(count: int, weights: list[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError("D-FINE parameter soup weight count differs")
    if any(weight < 0 for weight in weights):
        raise ValueError("D-FINE parameter soup weights must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("D-FINE parameter soup weights must have a positive sum")
    return [weight / total for weight in weights]


def average_dfine_states(
    states: list[dict[str, Any]],
    weights: list[float] | None = None,
) -> dict[str, Any]:
    if len(states) < 2:
        raise ValueError("D-FINE parameter soup requires at least two checkpoints")
    normalized_weights = _normalized_weights(len(states), weights)
    keys = set(states[0])
    if any(set(state) - keys for state in states[1:]):
        raise ValueError("D-FINE parameter soup state has keys absent from reference")
    averaged: dict[str, Any] = {}
    for key in states[0]:
        tensors = [state[key] for state in states if key in state]
        reference = tensors[0]
        if len(tensors) != len(states):
            if reference.is_floating_point() or reference.is_complex():
                raise ValueError(f"D-FINE parameter soup floating tensor is missing: {key}")
            if any(not reference.equal(value) for value in tensors[1:]):
                raise ValueError(f"D-FINE parameter soup non-floating tensor differs: {key}")
            averaged[key] = reference.detach().clone()
            continue
        if any(
            value.shape != reference.shape or value.dtype != reference.dtype
            for value in tensors[1:]
        ):
            raise ValueError(f"D-FINE parameter soup tensor contract differs: {key}")
        if reference.is_floating_point() or reference.is_complex():
            value = reference.detach().clone().mul_(normalized_weights[0])
            for tensor, weight in zip(tensors[1:], normalized_weights[1:], strict=True):
                value.add_(tensor, alpha=weight)
            averaged[key] = value
        else:
            if any(not reference.equal(value) for value in tensors[1:]):
                raise ValueError(f"D-FINE parameter soup non-floating tensor differs: {key}")
            averaged[key] = reference.detach().clone()
    return averaged


def create_dfine_checkpoint_soup(
    checkpoints: list[Path],
    output: Path,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    import torch

    resolved = [path.resolve() for path in checkpoints]
    if len(set(resolved)) != len(resolved):
        raise ValueError("D-FINE parameter soup checkpoints must be unique")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in resolved]
    states = [checkpoint_model_state(payload) for payload in payloads]
    normalized_weights = _normalized_weights(len(states), weights)
    averaged = average_dfine_states(states, normalized_weights)
    members = [
        {
            "path": str(path),
            "sha256": _sha256(path),
            "last_epoch": payload.get("last_epoch"),
            "weight": weight,
        }
        for path, payload, weight in zip(resolved, payloads, normalized_weights, strict=True)
    ]
    uniform_weight = 1.0 / len(members)
    is_uniform = all(abs(weight - uniform_weight) < 1e-12 for weight in normalized_weights)
    provenance = {
        "recipe": (
            "uniform_inference_parameter_soup"
            if is_uniform
            else "weighted_inference_parameter_interpolation"
        ),
        "member_count": len(members),
        "members": members,
        "selection_scope": "development_dataset_only",
        "independent_test_claimed": False,
    }
    result = {
        "date": datetime.now(timezone.utc).isoformat(),
        "last_epoch": -1,
        "model": averaged,
        "ema": {"module": averaged, "updates": 0},
        "detector_parameter_soup": provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one deployable D-FINE checkpoint from locked fold members"
    )
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--weight",
        type=float,
        action="append",
        help="Optional checkpoint weight; repeat once per --checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            create_dfine_checkpoint_soup(args.checkpoint, args.output, args.weight),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
