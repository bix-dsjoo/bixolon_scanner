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


def average_dfine_states(states: list[dict[str, Any]]) -> dict[str, Any]:
    if len(states) < 2:
        raise ValueError("D-FINE parameter soup requires at least two checkpoints")
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("D-FINE parameter soup state keys differ")
    averaged: dict[str, Any] = {}
    for key in states[0]:
        tensors = [state[key] for state in states]
        reference = tensors[0]
        if any(
            value.shape != reference.shape or value.dtype != reference.dtype
            for value in tensors[1:]
        ):
            raise ValueError(f"D-FINE parameter soup tensor contract differs: {key}")
        if reference.is_floating_point() or reference.is_complex():
            value = reference.detach().clone()
            for tensor in tensors[1:]:
                value.add_(tensor)
            averaged[key] = value.div_(len(tensors))
        else:
            if any(not reference.equal(value) for value in tensors[1:]):
                raise ValueError(f"D-FINE parameter soup non-floating tensor differs: {key}")
            averaged[key] = reference.detach().clone()
    return averaged


def create_dfine_checkpoint_soup(
    checkpoints: list[Path],
    output: Path,
) -> dict[str, Any]:
    import torch

    resolved = [path.resolve() for path in checkpoints]
    if len(set(resolved)) != len(resolved):
        raise ValueError("D-FINE parameter soup checkpoints must be unique")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in resolved]
    states = [checkpoint_model_state(payload) for payload in payloads]
    averaged = average_dfine_states(states)
    members = [
        {
            "path": str(path),
            "sha256": _sha256(path),
            "last_epoch": payload.get("last_epoch"),
        }
        for path, payload in zip(resolved, payloads)
    ]
    provenance = {
        "recipe": "uniform_inference_weight_soup",
        "member_count": len(members),
        "members": members,
        "selection_scope": "locked_grouped_oof_members",
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            create_dfine_checkpoint_soup(args.checkpoint, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
