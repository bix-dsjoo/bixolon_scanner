from __future__ import annotations

from pathlib import Path

import pytest

from bixolon_scanner.training.pipeline_contract import PIPELINE_STAGES
from bixolon_scanner.training.run_evidence import write_native_run_evidence


def test_native_run_evidence_is_ordered_and_immutable(tmp_path: Path) -> None:
    artifacts = {}
    for stage in PIPELINE_STAGES:
        path = tmp_path / f"{stage}.json"
        path.write_text(stage, encoding="utf-8")
        artifacts[stage] = path
    output = tmp_path / "run-evidence.json"

    payload = write_native_run_evidence(
        component="detector",
        pipeline_version="1.0.0",
        repository_root=tmp_path,
        stage_artifacts=artifacts,
        output=output,
    )

    assert [row["stage"] for row in payload["stages"]] == list(PIPELINE_STAGES)
    assert payload["stages"][1]["previous_stage_sha256"] == payload["stages"][0]["stage_sha256"]
    with pytest.raises(FileExistsError):
        write_native_run_evidence(
            component="detector",
            pipeline_version="1.0.0",
            repository_root=tmp_path,
            stage_artifacts=artifacts,
            output=output,
        )
