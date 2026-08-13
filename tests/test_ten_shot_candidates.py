from __future__ import annotations

from pathlib import Path

import pytest

from bixolon_scanner.training.models import require_torch
from bixolon_scanner.training.ten_shot_candidates import (
    CandidateResult,
    challenger_required,
    create_uniform_parameter_soup,
    select_candidate,
    validate_seed_matrix,
)


def _run(seed: int, folds: tuple[float, ...]) -> CandidateResult:
    return CandidateResult("main", seed, "best.pt", "a" * 64, folds)


def test_three_locked_seeds_and_development_three_fold_selection():
    assert validate_seed_matrix([20260812, 20260813, 20260814]) == (20260812, 20260813, 20260814)
    with pytest.raises(ValueError, match="requires seeds"):
        validate_seed_matrix([1, 2, 3])
    runs = [_run(20260812, (0.95, 0.96, 0.94)), _run(20260813, (0.96, 0.96, 0.96))]
    assert select_candidate(runs).seed == 20260813
    assert challenger_required(runs) is False


def test_challenger_is_gated_below_95_percent_only():
    assert challenger_required([_run(20260812, (0.94, 0.94, 0.94))]) is True


def test_uniform_parameter_soup_averages_floating_state_and_records_members(
    tmp_path: Path,
):
    torch = require_torch()
    common = {
        "architecture": "ten_shot_residual_cosine_challenger",
        "adapter_spec": {"hidden_size": 2},
        "dataset_version": "data-v1",
        "manifest_sha256": "manifest",
        "backbone_kind": "dinov3_convnext_tiny",
        "backbone_revision": "revision",
        "backbone_weight_sha256": "weights",
        "image_size": 224,
    }
    paths = []
    for index, value in enumerate((1.0, 3.0)):
        path = tmp_path / f"member-{index}.pt"
        torch.save(
            common
            | {
                "model_state_dict": {
                    "weight": torch.tensor([value, value + 1]),
                    "counter": torch.tensor(1, dtype=torch.int64),
                },
                "history": [{"loss": value}],
            },
            path,
        )
        paths.append(path)

    output = tmp_path / "soup.pt"
    provenance = create_uniform_parameter_soup(paths, output, member_seeds=(20260813, 20260814))
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)

    assert torch.equal(checkpoint["model_state_dict"]["weight"], torch.tensor([2.0, 3.0]))
    assert provenance["member_seeds"] == [20260813, 20260814]
    assert provenance["runtime_model_count"] == 1


def test_uniform_parameter_soup_rejects_incompatible_provenance(tmp_path: Path):
    torch = require_torch()
    paths = []
    for index, dataset in enumerate(("a", "b")):
        path = tmp_path / f"member-{index}.pt"
        torch.save(
            {
                "architecture": "ten_shot_residual_cosine_challenger",
                "adapter_spec": {},
                "dataset_version": dataset,
                "manifest_sha256": "manifest",
                "backbone_kind": "dinov3_convnext_tiny",
                "backbone_revision": "revision",
                "backbone_weight_sha256": "weights",
                "image_size": 224,
                "model_state_dict": {"weight": torch.tensor([1.0])},
            },
            path,
        )
        paths.append(path)

    with pytest.raises(ValueError, match="incompatible provenance"):
        create_uniform_parameter_soup(
            paths,
            tmp_path / "soup.pt",
            member_seeds=(20260813, 20260814),
        )
