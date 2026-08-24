from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..package import sha256_file
from .models import require_torch

LOCK_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CandidateResult:
    recipe: str
    seed: int
    checkpoint: str
    checkpoint_sha256: str
    fold_top1: tuple[float, ...]

    @property
    def mean_top1(self) -> float:
        return float(np.mean(self.fold_top1))


def validate_seed_matrix(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in seeds)
    if values != (20260812, 20260813, 20260814):
        raise ValueError("strict 10-shot selection requires seeds 20260812, 20260813 and 20260814")
    return values


def select_candidate(results: Iterable[CandidateResult]) -> CandidateResult:
    values = list(results)
    if not values:
        raise ValueError("candidate selection requires at least one run")
    fold_counts = {len(value.fold_top1) for value in values}
    if fold_counts != {3}:
        raise ValueError("candidate selection requires development 3-fold metrics")
    # Deterministic tie-break: mean, worst fold, lower seed, recipe.
    return max(
        values,
        key=lambda value: (
            value.mean_top1,
            min(value.fold_top1),
            -value.seed,
            value.recipe,
        ),
    )


def challenger_required(main_results: Iterable[CandidateResult], floor: float = 0.95) -> bool:
    selected = select_candidate(main_results)
    return selected.mean_top1 < floor


def freeze_for_challenger(model) -> dict[str, Any]:
    """Unfreeze only the last ConvNeXt stage and norm; never the full backbone."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not hasattr(model.backbone, "stages") or not hasattr(model.backbone, "norm"):
        raise ValueError("challenger requires a DINOv3 ConvNeXt backbone")
    for parameter in model.backbone.stages[-1].parameters():
        parameter.requires_grad = True
    for parameter in model.backbone.norm.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    reference = {
        name: parameter.detach().clone()
        for name, parameter in model.backbone.stages[-1].named_parameters()
    }
    return {"scope": "backbone.stages[-1]", "reference": reference}


def l2_sp_penalty(model, reference: dict[str, Any]):
    torch = require_torch()
    penalty = torch.zeros((), device=next(model.parameters()).device)
    current = dict(model.backbone.stages[-1].named_parameters())
    if set(current) != set(reference):
        raise ValueError("L2-SP reference does not match the final stage")
    for name, initial in reference.items():
        penalty = penalty + torch.sum((current[name] - initial.to(current[name].device)) ** 2)
    return penalty


def create_uniform_parameter_soup(
    checkpoint_paths: Iterable[Path],
    output_path: Path,
    *,
    member_seeds: Iterable[int],
    selection_scope: str = "development_capture_session_3fold_only",
) -> dict[str, Any]:
    """Average compatible strict 10-shot checkpoints into one deployable model."""
    torch = require_torch()
    paths = tuple(Path(path) for path in checkpoint_paths)
    seeds = tuple(int(seed) for seed in member_seeds)
    if len(paths) < 2 or len(paths) != len(seeds):
        raise ValueError("parameter soup requires at least two aligned checkpoints")
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    required_equal = (
        "architecture",
        "adapter_spec",
        "dataset_version",
        "manifest_sha256",
        "backbone_kind",
        "backbone_revision",
        "backbone_weight_sha256",
        "image_size",
    )
    reference = checkpoints[0]
    if reference.get("architecture") != "ten_shot_residual_cosine_challenger":
        raise ValueError("parameter soup requires challenger checkpoints")
    for checkpoint in checkpoints[1:]:
        if any(checkpoint.get(key) != reference.get(key) for key in required_equal):
            raise ValueError("parameter soup checkpoints have incompatible provenance")
    states = [checkpoint.get("model_state_dict") for checkpoint in checkpoints]
    if any(not isinstance(state, dict) for state in states):
        raise ValueError("parameter soup checkpoint is missing model_state_dict")
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("parameter soup state keys differ")
    averaged = {}
    for key in sorted(keys):
        values = [state[key] for state in states]
        if any(
            value.shape != values[0].shape or value.dtype != values[0].dtype for value in values[1:]
        ):
            raise ValueError(f"parameter soup tensor contract differs: {key}")
        if values[0].is_floating_point():
            averaged[key] = (
                torch.stack([value.detach().float() for value in values])
                .mean(dim=0)
                .to(values[0].dtype)
            )
        else:
            if any(not torch.equal(values[0], value) for value in values[1:]):
                raise ValueError(f"parameter soup non-floating tensor differs: {key}")
            averaged[key] = values[0].detach().clone()
    member_hashes = [sha256_file(path) for path in paths]
    soup = {
        **{
            key: value
            for key, value in reference.items()
            if key not in {"model_state_dict", "history", "parameter_soup"}
        },
        "model_state_dict": averaged,
        "history": [],
        "parameter_soup": {
            "recipe": "uniform_full_model_parameter_soup",
            "selection_scope": selection_scope,
            "member_seeds": list(seeds),
            "member_checkpoint_sha256": member_hashes,
            "member_checkpoint_paths": [str(path) for path in paths],
            "floating_tensor_reduction": "float32_arithmetic_mean",
            "runtime_model_count": 1,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup, output_path)
    return soup["parameter_soup"]


def create_weighted_parameter_soup(
    checkpoint_paths: Iterable[Path],
    output_path: Path,
    *,
    weights: Iterable[float],
    selection_scope: str,
    dataset_version: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Interpolate compatible single-source candidates into one runtime model."""
    torch = require_torch()
    paths = tuple(Path(path) for path in checkpoint_paths)
    coefficients = np.asarray(tuple(float(value) for value in weights), dtype=np.float64)
    if len(paths) < 2 or coefficients.shape != (len(paths),):
        raise ValueError("weighted parameter soup requires aligned checkpoints and weights")
    if np.any(coefficients < 0.0) or not np.isclose(coefficients.sum(), 1.0):
        raise ValueError("weighted parameter soup weights must be non-negative and sum to one")
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    reference = checkpoints[0]
    required_equal = (
        "architecture",
        "adapter_spec",
        "backbone_kind",
        "source_revision",
        "source_weight_sha256",
        "image_size",
        "num_classes",
    )
    for checkpoint in checkpoints[1:]:
        if any(checkpoint.get(key) != reference.get(key) for key in required_equal):
            raise ValueError("weighted parameter soup checkpoints have incompatible models")
    states = [checkpoint.get("model_state_dict") for checkpoint in checkpoints]
    if any(not isinstance(state, dict) for state in states):
        raise ValueError("weighted parameter soup checkpoint is missing model_state_dict")
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("weighted parameter soup state keys differ")
    averaged = {}
    for key in sorted(keys):
        values = [state[key] for state in states]
        if any(
            value.shape != values[0].shape or value.dtype != values[0].dtype for value in values[1:]
        ):
            raise ValueError(f"weighted parameter soup tensor contract differs: {key}")
        if values[0].is_floating_point():
            averaged[key] = sum(
                value.detach().float() * float(weight)
                for value, weight in zip(values, coefficients, strict=True)
            ).to(values[0].dtype)
        else:
            if any(not torch.equal(values[0], value) for value in values[1:]):
                raise ValueError(f"weighted parameter soup non-floating tensor differs: {key}")
            averaged[key] = values[0].detach().clone()
    provenance = {
        "recipe": "weighted_full_model_parameter_soup",
        "selection_scope": selection_scope,
        "weights": coefficients.tolist(),
        "member_checkpoint_sha256": [sha256_file(path) for path in paths],
        "member_checkpoint_paths": [str(path) for path in paths],
        "floating_tensor_reduction": "float32_weighted_sum",
        "runtime_model_count": 1,
    }
    soup = {
        **{
            key: value
            for key, value in reference.items()
            if key not in {"model_state_dict", "history", "parameter_soup"}
        },
        "dataset_version": dataset_version,
        "manifest_sha256": manifest_sha256,
        "model_state_dict": averaged,
        "history": [],
        "parameter_soup": provenance,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup, output_path)
    return provenance


def create_experiment_lock(
    path: Path,
    *,
    config_path: Path,
    manifest_path: Path,
    manifest_metadata_path: Path,
    checkpoint_path: Path,
    calibration_path: Path,
    selected: CandidateResult,
) -> dict[str, Any]:
    if sha256_file(checkpoint_path) != selected.checkpoint_sha256:
        raise ValueError("selected checkpoint changed before lock")
    files = {
        "config": config_path,
        "manifest": manifest_path,
        "manifest_metadata": manifest_metadata_path,
        "checkpoint": checkpoint_path,
        "calibration": calibration_path,
    }
    hashes = {name: sha256_file(value) for name, value in files.items()}
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "test_accessed": False,
        "selected": asdict(selected),
        "hashes": hashes,
        "lock_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return lock


def verify_experiment_lock(lock_path: Path, **files: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for name, path in files.items():
        expected = lock.get("hashes", {}).get(name)
        if expected is None or sha256_file(path) != expected:
            raise ValueError(f"locked {name} changed before test evaluation")
    return lock
