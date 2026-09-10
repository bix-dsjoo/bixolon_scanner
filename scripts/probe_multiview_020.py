"""Fixed rotation proposal diagnostic; classification is intentionally not scored."""

import itertools
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.contracts.runtime_package_v2 import load_runtime_package_v2
from bixolon_scanner.evaluation.three_bakery import match_boxes
from bixolon_scanner.pipeline.ports import Detection
from bixolon_scanner.runtime.geometry import nms
from bixolon_scanner.runtime.onnx import prepare_rgb, sigmoid
from bixolon_scanner.runtime.onnx_session import OrtRunner
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    config = load_json_config(root / "configs/experiments/bread/n100_020.json")
    out = root / config["work"] / "rotation-probe"
    package = load_runtime_package_v2(root / config["baseline"] / "runtime")
    meta = package.metadata.detector
    settings = list(itertools.product([(0, 1), (0, 2), (0, 1, 2, 3)], [0.3, 0.5], [0.1, 0.5]))
    protocol = {"settings": settings, "detector_sha256": sha256_file(package.detector_path)}
    write_json(out / "protocol.json", protocol)
    summaries = [
        dict(turns=t, nms=i, score=s, missed=0, extra=0, failures=[]) for t, i, s in settings
    ]
    runner = OrtRunner(package.detector_path, "cpu", cpu_intra_op_threads=4)
    try:
        for record in read_jsonl(root / config["log_manifest"]):
            width, height = record["width"], record["height"]
            views = []
            with Image.open(record["image_path"]) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                for turn in range(4):
                    cache = out / f"{record['image_id']:03d}-{turn}.npz"
                    if not cache.exists():
                        view = Image.fromarray(np.rot90(np.asarray(image), turn).copy())
                        tensor = prepare_rgb(view, meta.input_size, meta.mean, meta.std)[None]
                        logits, boxes = runner.run(
                            [meta.logits_output, meta.boxes_output], meta.input_name, tensor
                        )
                        np.savez(cache, logits=logits[0], boxes=boxes[0])
                    with np.load(cache) as value:
                        scores = sigmoid(value["logits"]).max(axis=-1)
                        b = value["boxes"]
                    xy = np.column_stack(
                        (
                            b[:, 0] - b[:, 2] / 2,
                            b[:, 1] - b[:, 3] / 2,
                            b[:, 0] + b[:, 2] / 2,
                            b[:, 1] + b[:, 3] / 2,
                        )
                    )
                    xy = np.clip(xy, 0, 1)
                    for _ in range(turn):
                        xy = np.column_stack((1 - xy[:, 3], xy[:, 0], 1 - xy[:, 1], xy[:, 2]))
                    xy *= np.array([width, height, width, height])
                    views.append(
                        [
                            Detection(*map(float, box), float(score))
                            for box, score in zip(xy, scores, strict=True)
                            if score >= 0.1 and box[2] > box[0] and box[3] > box[1]
                        ]
                    )
            targets = []
            for a in record["annotations"]:
                x, y, w, h = a["bbox_xywh"]
                targets.append([x, y, x + w, y + h])
            for result, (turns, threshold, score) in zip(summaries, settings, strict=True):
                rows = nms([b for t in turns for b in views[t] if b.score >= score], threshold)
                matches = match_boxes([[b.x1, b.y1, b.x2, b.y2] for b in rows], targets, 0.5)
                missed = len(targets) - len(matches)
                extra = len(rows) - len(matches)
                result["missed"] += missed
                result["extra"] += extra
                if missed:
                    result["failures"].append({"image_id": record["image_id"], "missed": missed})
            if record["image_id"] % 25 == 0:
                print(record["image_id"], flush=True)
    finally:
        runner.close()
    write_json(out / "report.json", {"policies": summaries})
    print(summaries)


if __name__ == "__main__":
    main()
