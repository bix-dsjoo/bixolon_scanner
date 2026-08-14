from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from ..contracts.model_package import sha256_file
from .pipeline_contract import PIPELINE_STAGES, ArtifactLock, stage_ledger_sha256


def write_native_run_evidence(
    *,
    component: Literal["detector", "classifier"],
    pipeline_version: str,
    repository_root: Path,
    stage_artifacts: dict[str, Path],
    output: Path,
) -> dict[str, object]:
    if tuple(stage_artifacts) != PIPELINE_STAGES:
        raise ValueError("native evidence requires every pipeline stage in canonical order")
    if output.exists():
        raise FileExistsError(f"native run evidence already exists: {output}")
    root = repository_root.resolve()
    previous: str | None = None
    stages = []
    for stage, artifact_path in stage_artifacts.items():
        resolved = artifact_path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("native evidence artifact is outside the repository") from exc
        artifact = ArtifactLock(path=relative, sha256=sha256_file(resolved))
        stage_hash = stage_ledger_sha256(
            stage=stage,
            passes=True,
            artifact=artifact,
            previous_stage_sha256=previous,
        )
        stages.append(
            {
                "stage": stage,
                "passes": True,
                "artifact": artifact.model_dump(mode="json"),
                "previous_stage_sha256": previous,
                "stage_sha256": stage_hash,
            }
        )
        previous = stage_hash
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "component": component,
        "pipeline_version": pipeline_version,
        "provenance_mode": "native",
        "stages": stages,
        "final_stage_sha256": previous,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a native training stage hash-chain")
    parser.add_argument("--component", choices=("detector", "classifier"), required=True)
    parser.add_argument("--pipeline-version", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        action="append",
        nargs=2,
        metavar=("NAME", "ARTIFACT"),
        required=True,
    )
    args = parser.parse_args()
    stage_artifacts = {name: Path(path) for name, path in args.stage}
    payload = write_native_run_evidence(
        component=args.component,
        pipeline_version=args.pipeline_version,
        repository_root=args.repository_root,
        stage_artifacts=stage_artifacts,
        output=args.output,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
