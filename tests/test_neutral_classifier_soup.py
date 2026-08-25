from pathlib import Path

import pytest

from bixolon_scanner.training.neutral_classifier_soup import build_neutral_soup


def test_neutral_soup_rejects_missing_members(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_neutral_soup(
            tmp_path / "moderate.pt",
            tmp_path / "mild.pt",
            tmp_path / "output.pt",
            tmp_path / "report.json",
        )
