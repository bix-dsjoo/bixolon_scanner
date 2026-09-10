"""Freeze the user-selected three_bakery candidate without changing its payloads."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.contracts.catalog import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run = root / "artifacts/retraining/three-bakery-revised300-roi-integrity"
    candidate = run / "candidates/ssdlite-margin-dense-20260908"
    target = root / "artifacts/versions/0.1.17/source"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("runtime", "catalog"):
        destination = target / name
        if not destination.exists():
            shutil.copytree(candidate / name, destination)
        if directory_content_manifest(destination) != directory_content_manifest(candidate / name):
            raise ValueError(f"Release snapshot mismatch: {name}")
    evidence = target / "evidence"
    evidence.mkdir(exist_ok=True)
    sources = [
        candidate / "comparison-objects.json",
        candidate / "cpu-profile.json",
        candidate / "worker-cpu.json",
        candidate / "catalog-support.jsonl",
        run / "objective-revision.json",
        run / "policy-revision.json",
        run / "base-comparison.json",
        run / "sources/source-report.json",
    ]
    for source in sources:
        shutil.copy2(source, evidence / source.name)
    selection = {
        "product_version": "0.1.17",
        "candidate_id": "ssdlite-margin-dense-20260908",
        "selection": "user_requested_current_first_place_base_seed",
        "source_object_count": 649,
        "correct_approved_objects": 645,
        "wrong_approved": 0,
        "cpu_selected_threads": [8, 12],
        "benchmark_scope": "same_physical_objects_development_diagnostic",
        "final_300_evaluated_at_selection": False,
        "additional_seed_comparison": "continues_separately_without_replacing_this_release",
        "model_graph_or_weight_changed": False,
    }
    (evidence / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config = load_json_config(root / "configs/archive/versions/0.1.16.json")
    config.update(
        version="0.1.17",
        app_build=20,
        source_date_epoch=1788825600,
        source_candidate="three-bakery-revised300-roi-integrity/ssdlite-margin-dense-20260908",
    )
    for name in ("runtime", "catalog"):
        config[name] = {
            "path": (target / name).relative_to(root).as_posix(),
            "manifest_sha256": directory_content_manifest(target / name)["manifest_sha256"],
        }
    config["catalog"]["store_id"] = "three_bakery"
    config["evaluation_evidence"] = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(evidence.iterdir())
        if path.is_file()
    ]
    (root / "configs/versions/0.1.17.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
