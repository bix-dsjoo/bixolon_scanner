"""Ordered, resumable classifier trials with identical seeds and fixed budgets."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...configuration import load_json_config
from ...contracts.runtime_package_v2 import EmbedderMetadata
from ...evaluation.structural_benchmark import benchmark
from ...training.structural_student import build_student, export_student, train_student
from ...training.three_bakery_data import write_json
from ...training.three_bakery_detector import seed_everything


def run(config_path: Path, *, preflight_only: bool = False):
    config = load_json_config(config_path)
    root = Path(config["output"])
    metadata = EmbedderMetadata.model_validate(
        load_json_config(Path(config["runtime"]) / "metadata.json")["embedder"]
    )
    for name in config["students"]:
        directory = root / "preflight" / name.split(".")[0]
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "export.json").exists():
            seed_everything(config["seed"])
            model = build_student(name, config["classifier"])
            result = export_student(model, directory / "model.onnx", metadata)
            result["purpose"] = (
                "architecture timing only; random task heads are not accuracy candidates"
            )
            result["parameters"] = sum(p.numel() for p in model.parameters())
            write_json(directory / "export.json", result)
            del model
    if preflight_only:
        return
    if not (root / "student-cache/complete.json").is_file():
        raise ValueError("finish teacher cache before timing and training")
    for name, path in [("reference", Path(config["runtime"]) / metadata.filename)] + [
        (n.split(".")[0], root / "preflight" / n.split(".")[0] / "model.onnx")
        for n in config["students"]
    ]:
        destination = root / "preflight" / f"{name}-cpu.json"
        if not destination.exists():
            write_json(
                destination,
                benchmark(path, config["benchmark"], batches=config["benchmark"]["batches"]),
            )
    for name in config["students"]:
        for objective in config["objectives"]:
            train_student(config_path, name, objective)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.config, preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
