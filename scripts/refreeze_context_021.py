"""Archive the rejected unbuilt candidate and freeze the context-preserving repair."""

import shutil
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.training.three_bakery_data import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    work = (root / "artifacts/retraining/recapture-0.2.1").resolve()
    for key in ("log/cpu", "log/cuda", "original/cpu"):
        if not load_json_config(work / f"confirmation-context/{key}/comparison.json")["accepted"]:
            raise ValueError(f"context-preserving repair failed: {key}")
    old = (root / "artifacts/versions/0.2.1").resolve()
    old.relative_to((root / "artifacts/versions").resolve())
    rejected = work / "rejected-score040-version"
    rejected.relative_to(work)
    if rejected.exists():
        raise ValueError("rejected archive already exists")
    shutil.move(str(old), str(rejected))
    shutil.copy2(root / "configs/versions/0.2.1.json", rejected / "version-config.json")
    shutil.copy2(work / "selection.json", work / "selection-score040.json")
    target = root / "artifacts/versions/0.2.1/source"
    candidate = work / "candidates/context040"
    for name in ("runtime", "catalog"):
        shutil.copytree(candidate / name, target / name)
    evidence = target / "evidence"
    evidence.mkdir()
    files = {
        "log-inputs.jsonl": root
        / "artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl",
        "source-baseline.json": work / "source-baseline.json",
        "recapture-causes.json": work / "recapture-causes.json",
        "rejected-score040-final300.json": work / "final300/comparison.json",
        "rejected-score040-selection.json": work / "selection-score040.json",
        "context-development.json": work / "measurements/context-study/context040-cpu/report.json",
        "cpu-comparison.json": work / "confirmation-context/log/cpu/comparison.json",
        "cuda-comparison.json": work / "confirmation-context/log/cuda/comparison.json",
        "original-comparison.json": work / "confirmation-context/original/cpu/comparison.json",
    }
    for name, path in files.items():
        shutil.copy2(path, evidence / name)
    config = load_json_config(root / "configs/versions/0.2.1.json")
    config["source_candidate"] = "dfine-margin-dense-20260908-context-preserved-output040"
    for name in ("runtime", "catalog"):
        config[name] = {
            "path": f"artifacts/versions/0.2.1/source/{name}",
            "manifest_sha256": directory_content_manifest(target / name)["manifest_sha256"],
        }
    config["catalog"]["store_id"] = "three_bakery"
    config["evaluation_evidence"] = [
        {"path": f"artifacts/versions/0.2.1/source/evidence/{p.name}", "sha256": sha256_file(p)}
        for p in sorted(evidence.iterdir())
    ]
    write_json(root / "configs/versions/0.2.1.json", config)
    write_json(
        work / "selection-context.json",
        {
            "version": "0.2.1",
            "candidate": "context040",
            "runtime_sha256": sha256_file(candidate / "runtime/metadata.json"),
            "version_config_sha256": sha256_file(root / "configs/versions/0.2.1.json"),
            "changes": {
                "detector.score_threshold": 0.025,
                "quality.detector_output_score_threshold": 0.04,
            },
            "rule": "Retained ROI decisions must equal baseline; no per-image, per-SKU or final300 threshold tuning. Same frozen models and context.",
            "data_role": "Exposed log development; source regression; historical final300 regression. Previous final300 rejected the context-deleting candidate; general context-preservation fix uses no final300-derived thresholds or learned parameters.",
            "rejected_previous_candidate": str(rejected),
            "inference_code_sha256": {
                str(p.relative_to(root)): sha256_file(p)
                for d in ("contracts", "pipeline", "runtime", "worker")
                for p in sorted((root / "src/bixolon_scanner" / d).glob("*.py"))
            },
        },
    )
    print("Frozen context-preserving 0.2.1", flush=True)


if __name__ == "__main__":
    main()
