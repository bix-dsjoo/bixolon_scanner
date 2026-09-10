"""Reproducible log diagnosis and source-only improvement for the 0.1.18 release."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from ...configuration import load_json_config
from ...contracts.catalog import sha256_file
from ...evaluation.three_bakery_http import measure, provider_parity
from ...operations.detector_variant import assemble_detector_variant
from ...training.three_bakery_data import read_jsonl, source_path, write_json, write_jsonl


def prepare(config_path: Path) -> dict:
    config = load_json_config(config_path)
    work = Path(config["work"])
    root = Path(config["dataset_root"])
    annotation_path = source_path(root, config["log_annotations"])
    confirmation_path = source_path(root, config["log_confirmation"])
    confirmation = load_json_config(confirmation_path)
    if not confirmation["user_confirmed"] or confirmation["review_status"] != "confirmed":
        raise ValueError("log annotations require the recorded user confirmation")
    if sha256_file(annotation_path) != confirmation["annotation_sha256"]:
        raise ValueError("confirmed annotation checksum changed")
    originals = read_jsonl(Path(config["original_manifest"]))
    original_hashes = {r["image_sha256"] for r in originals}
    if (
        len(originals) != config["expected_training_images"]
        or sum(len(r["annotations"]) for r in originals) != config["expected_training_objects"]
    ):
        raise ValueError("training source count changed")
    for row in originals:
        if sha256_file(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError("training source checksum changed")
    logs = read_jsonl(annotation_path)
    records, seen, classes = [], set(), Counter()
    for row in logs:
        path = source_path(root, row["image_path"])
        digest = sha256_file(path)
        if digest != row["image_sha256"] or digest in seen or digest in original_hashes:
            raise ValueError("log checksum or source isolation violation")
        seen.add(digest)
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.size != (row["width"], row["height"]):
                raise ValueError("log annotation dimensions differ from decoded image")
        annotations = []
        for value in row["annotations"]:
            x1, y1, x2, y2 = value["bbox_xyxy"]
            if not (0 <= x1 < x2 <= row["width"] and 0 <= y1 < y2 <= row["height"]):
                raise ValueError("log annotation bbox out of bounds")
            if not value["reviewed"] or value["category_id"] not in range(1, 21):
                raise ValueError("unreviewed or unsupported log object")
            annotations.append(
                {
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": value["category_id"],
                    "object_number": value["object_number"],
                }
            )
            classes[value["category_id"]] += 1
        records.append(
            {
                "image_id": row["image_id"],
                "image_path": str(path),
                "image_sha256": digest,
                "width": row["width"],
                "height": row["height"],
                "annotations": annotations,
                "source_group": row["source_group"],
                "capture_session_id": row["capture_session_id"],
                "split": "untrained_log_diagnostic",
                "difficulty": f"objects_{len(annotations):02d}",
            }
        )
    if (
        len(records) != config["expected_log_images"]
        or sum(classes.values()) != config["expected_log_objects"]
    ):
        raise ValueError("log count changed")
    audit = {
        "config_sha256": sha256_file(config_path),
        "annotation_sha256": sha256_file(annotation_path),
        "confirmation_sha256": sha256_file(confirmation_path),
        "original_manifest_sha256": sha256_file(Path(config["original_manifest"])),
        "log_image_count": len(records),
        "log_object_count": sum(classes.values()),
        "training_image_count": len(originals),
        "sha256_overlap": 0,
        "log_groups": sorted({r["source_group"] for r in records}),
        "log_sessions": sorted({r["capture_session_id"] for r in records}),
        "shares_physical_group_with_training": bool(
            {r["source_group"] for r in records} & {r["source_group"] for r in originals}
        ),
        "object_count_histogram": {
            str(k): v for k, v in sorted(Counter(len(r["annotations"]) for r in records).items())
        },
        "classes": {str(k): v for k, v in sorted(classes.items())},
        "data_policy": config["data_policy"],
    }
    previous = work / "data-audit.json"
    if previous.exists() and load_json_config(previous) != audit:
        raise ValueError("frozen data/config audit changed; use a new work directory")
    write_json(previous, audit)
    manifest = work / "log-inputs.jsonl"
    if manifest.exists() and read_jsonl(manifest) != records:
        raise ValueError("frozen log manifest changed")
    write_jsonl(manifest, records)
    return audit


def baseline(config_path: Path) -> dict:
    audit = prepare(config_path)
    config = load_json_config(config_path)
    work = Path(config["work"])
    return measure(
        Path(config["baseline_candidate"]),
        read_jsonl(work / "log-inputs.jsonl"),
        work / "baseline/log-cpu",
        python=Path(config["cpu_python"]),
        provider="cpu",
        warmup=config["warmup"],
        repetitions=config["repetitions"],
        match_iou=config["match_iou"],
        input_identity=audit,
        cpu_profile=tuple(config["cpu_profile"]),
        worker_executable=Path(config["baseline_worker"]),
    )


def detector_candidates(config_path: Path) -> dict:
    settings = load_json_config(config_path)
    study_path = Path(settings["study_config"])
    audit = prepare(study_path)
    config = load_json_config(study_path)
    work = Path(config["work"])
    results = {}
    for item in settings["candidates"]:
        candidate = work / "candidates" / item["id"]
        identity = assemble_detector_variant(
            Path(config["baseline_candidate"]),
            Path(item["source"]),
            candidate,
            Path(item["training_report"]),
            Path(item["training_contract"]),
            Path(config["original_manifest"]),
            Path(item["detector_graph"]) if item.get("detector_graph") else None,
        )
        for name, records in (
            ("log", read_jsonl(work / "log-inputs.jsonl")),
            ("source", read_jsonl(Path(config["original_manifest"]))),
        ):
            result = measure(
                candidate,
                records,
                work / "measurements" / item["id"] / f"{name}-cpu",
                python=Path(config["cpu_python"]),
                provider="cpu",
                warmup=config["warmup"],
                repetitions=config["repetitions"] if name == "log" else 1,
                match_iou=config["match_iou"],
                input_identity={"audit": audit, "variant": identity},
                cpu_profile=tuple(config["cpu_profile"]),
                worker_executable=Path(config["baseline_worker"]),
            )
            results[f"{item['id']}/{name}"] = result["summary"]
            print(item["id"], name, result["summary"], flush=True)
    write_json(work / f"detector-comparison-{config_path.stem}.json", results)
    return results


def cpu_sleep(config_path: Path) -> dict:
    settings = load_json_config(config_path)
    study_path = Path(settings["study_config"])
    audit = prepare(study_path)
    config = load_json_config(study_path)
    work = Path(config["work"])
    results = {}
    for name, records in (
        ("log", read_jsonl(work / "log-inputs.jsonl")),
        ("source", read_jsonl(Path(config["original_manifest"]))),
    ):
        base_dir = work / "baseline" / f"{name}-cpu"
        if name == "source":
            measure(
                Path(config["baseline_candidate"]),
                records,
                base_dir,
                python=Path(config["cpu_python"]),
                provider="cpu",
                warmup=config["warmup"],
                repetitions=config["repetitions"],
                input_identity=audit,
                cpu_profile=tuple(config["cpu_profile"]),
                worker_executable=Path(config["baseline_worker"]),
            )
        output = work / "measurements/cpu-sleep" / f"{name}-cpu"
        report = measure(
            Path(config["baseline_candidate"]),
            records,
            output,
            python=Path(config["cpu_python"]),
            provider="cpu",
            warmup=config["warmup"],
            repetitions=config["repetitions"],
            input_identity={"audit": audit, "cpu_config_sha256": sha256_file(config_path)},
            cpu_profile=tuple(config["cpu_profile"]),
        )
        parity = provider_parity(base_dir / "responses.jsonl", output / "responses.jsonl")
        baseline_report = load_json_config(base_dir / "report.json")
        ratios = [
            new["latency"]["all"]["p95_ms"] / old["latency"]["all"]["p95_ms"]
            for new, old in zip(report["repetitions"], baseline_report["repetitions"], strict=True)
        ]
        results[name] = {"parity": parity, "p95_ratios": ratios, "summary": report["summary"]}
        print("cpu-sleep", name, results[name], flush=True)
    write_json(work / "cpu-sleep-comparison.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "baseline", "detectors", "cpu-sleep"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    stages = {
        "prepare": prepare,
        "baseline": baseline,
        "detectors": detector_candidates,
        "cpu-sleep": cpu_sleep,
    }
    result = stages[args.stage](args.config)
    print(result.get("summary", result), flush=True)


if __name__ == "__main__":
    main()
