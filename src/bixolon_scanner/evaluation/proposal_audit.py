"""Diagnose class-blind proposal recall without treating it as recognition accuracy."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2
from ..pipeline.ports import Detection
from ..runtime.geometry import box_iou_matrix, nms
from ..runtime.onnx import prepare_rgb, sigmoid
from ..runtime.onnx_session import OrtRunner
from ..training.three_bakery_data import read_jsonl, write_json
from .three_bakery import match_boxes


def audit(config_path: Path) -> dict:
    config = load_json_config(config_path)
    output = Path(config["work"]) / "proposal-audit"
    output.mkdir(parents=True, exist_ok=True)
    package = load_runtime_package_v2(Path(config["baseline"]) / "runtime")
    metadata = package.metadata.detector
    manifest = Path(config["log_manifest"])
    records = read_jsonl(manifest)
    identity = {
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest),
        "detector_sha256": sha256_file(package.detector_path),
        "script_sha256": sha256_file(Path(__file__)),
    }
    contract = output / "contract.json"
    if contract.exists() and load_json_config(contract) != identity:
        raise ValueError("proposal audit inputs changed")
    write_json(contract, identity)
    policy = config["proposal_diagnosis"]
    settings = list(
        product(
            policy["score_thresholds"],
            policy["nms_iou_thresholds"],
            policy["containment_thresholds"],
        )
    )
    summaries = [
        {"score": s, "nms_iou": i, "containment": c, "missed": 0, "extra": 0, "images": []}
        for s, i, c in settings
    ]
    runner = OrtRunner(package.detector_path, "cpu", cpu_intra_op_threads=4)
    raw_misses = []
    try:
        for position, record in enumerate(records):
            path = Path(record["image_path"])
            if sha256_file(path) != record["image_sha256"]:
                raise ValueError("log source changed")
            cache = output / f"raw-{record['image_id']:03d}.npz"
            if not cache.exists():
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    tensor = prepare_rgb(image, metadata.input_size, metadata.mean, metadata.std)[
                        None
                    ]
                logits, boxes = runner.run(
                    [metadata.logits_output, metadata.boxes_output], metadata.input_name, tensor
                )
                np.savez(cache, logits=logits[0], boxes=boxes[0])
            with np.load(cache) as values:
                scores = sigmoid(values["logits"]).max(axis=-1)
                boxes = values["boxes"]
            width, height = record["width"], record["height"]
            raw = np.column_stack(
                (
                    np.maximum(0, (boxes[:, 0] - boxes[:, 2] / 2) * width),
                    np.maximum(0, (boxes[:, 1] - boxes[:, 3] / 2) * height),
                    np.minimum(width, (boxes[:, 0] + boxes[:, 2] / 2) * width),
                    np.minimum(height, (boxes[:, 1] + boxes[:, 3] / 2) * height),
                )
            )
            targets = []
            for annotation in record["annotations"]:
                x, y, w, h = annotation["bbox_xywh"]
                targets.append([x, y, x + w, y + h])
            overlaps = box_iou_matrix(raw, targets)
            for target_index in range(len(targets)):
                choices = np.flatnonzero(overlaps[:, target_index] >= 0.5)
                best = None if not len(choices) else int(choices[np.argmax(scores[choices])])
                if best is None or scores[best] < 0.5:
                    raw_misses.append(
                        {
                            "image_id": record["image_id"],
                            "target_index": target_index,
                            "best_matching_score": None if best is None else float(scores[best]),
                            "best_matching_iou": None
                            if best is None
                            else float(overlaps[best, target_index]),
                        }
                    )
            for summary, (score, iou, containment) in zip(summaries, settings, strict=True):
                proposals = [
                    Detection(*map(float, raw[index]), float(scores[index]))
                    for index in np.flatnonzero(scores >= score)
                    if raw[index, 2] > raw[index, 0] and raw[index, 3] > raw[index, 1]
                ]
                selected = nms(proposals, iou, containment)
                matched = match_boxes([[b.x1, b.y1, b.x2, b.y2] for b in selected], targets, 0.5)
                missed, extra = len(targets) - len(matched), len(selected) - len(matched)
                summary["missed"] += missed
                summary["extra"] += extra
                if missed or extra:
                    summary["images"].append(
                        {"image_id": record["image_id"], "missed": missed, "extra": extra}
                    )
            if (position + 1) % 25 == 0:
                print(f"proposal audit {position + 1}/{len(records)}", flush=True)
    finally:
        runner.close()
    result = {
        "identity": identity,
        "raw_low_score_targets": raw_misses,
        "policies": summaries,
        "scope": "proposal recall only; no classification or approval claim",
    }
    write_json(output / "report.json", result)
    print([{k: v for k, v in row.items() if k != "images"} for row in summaries])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    audit(parser.parse_args().config)


if __name__ == "__main__":
    main()
