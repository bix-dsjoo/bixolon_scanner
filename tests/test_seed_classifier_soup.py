from pathlib import Path

import pytest

from bixolon_scanner.training.seed_classifier_soup import (
    build_seed_soup,
    build_source_recipe_soup,
)


def test_seed_classifier_soup_requires_two_members(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_seed_soup([tmp_path / "one.pt"], tmp_path / "out.pt", tmp_path / "report.json")


def test_source_recipe_soup_requires_two_members(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_source_recipe_soup(
            [tmp_path / "one.pt"], tmp_path / "out.pt", tmp_path / "report.json"
        )
