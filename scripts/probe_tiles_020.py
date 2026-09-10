"""Fixed overlapping-tile proposal experiment; recall is not recognition."""

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
    config = load_json_config(Path("configs/experiments/bread/n100_020.json"))
    work = Path(config["work"]) / "tile-probe"
    package = load_runtime_package_v2(Path(config["baseline"]) / "runtime")
    metadata = package.metadata.detector
    runner = OrtRunner(package.detector_path, "cpu", cpu_intra_op_threads=4)
    rows = read_jsonl(Path(config["log_manifest"]))
    protocol = {
        "model": sha256_file(package.detector_path),
        "manifest": sha256_file(Path(config["log_manifest"])),
        "tile_scales": [0.6, 0.75],
        "code": sha256_file(Path(__file__)),
    }
    work.mkdir(parents=True, exist_ok=True)
    contract = work / "protocol.json"
    if contract.exists() and load_json_config(contract) != protocol:
        raise ValueError("tile probe inputs changed")
    write_json(contract, protocol)
    summaries = []
    for scale in protocol["tile_scales"]:
        settings = [(score, overlap) for score in [0.25, 0.5] for overlap in [0.3, 0.5]]
        results = [
            {"score": s, "nms_iou": n, "tile_scale": scale, "missed": 0, "extra": 0, "images": []}
            for s, n in settings
        ]
        for row in rows:
            cache = work / f"{scale}-{row['image_id']}.npz"
            if not cache.exists():
                if sha256_file(Path(row["image_path"])) != row["image_sha256"]:
                    raise ValueError("log source changed")
                with Image.open(row["image_path"]) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                w, h = image.size
                tw, th = round(w * scale), round(h * scale)
                views = [(0, 0, w, h)] + [
                    (x, y, x + tw, y + th) for x in [0, w - tw] for y in [0, h - th]
                ]
                all_boxes, all_scores = [], []
                for x, y, x2, y2 in views:
                    crop = image.crop((x, y, x2, y2))
                    tensor = prepare_rgb(crop, metadata.input_size, metadata.mean, metadata.std)[
                        None
                    ]
                    logits, boxes = runner.run(
                        [metadata.logits_output, metadata.boxes_output], metadata.input_name, tensor
                    )
                    b = boxes[0]
                    cw, ch = x2 - x, y2 - y
                    coords = np.column_stack(
                        (
                            (b[:, 0] - b[:, 2] / 2) * cw + x,
                            (b[:, 1] - b[:, 3] / 2) * ch + y,
                            (b[:, 0] + b[:, 2] / 2) * cw + x,
                            (b[:, 1] + b[:, 3] / 2) * ch + y,
                        )
                    )
                    scores = sigmoid(logits[0]).max(axis=-1)
                    if (x, y, x2, y2) != (0, 0, w, h):
                        # Do not add artificial crop-border fragments.
                        valid = (
                            (coords[:, 0] > x + 2)
                            & (coords[:, 1] > y + 2)
                            & (coords[:, 2] < x2 - 2)
                            & (coords[:, 3] < y2 - 2)
                        )
                        scores = np.where(valid, scores, 0)
                    all_boxes.append(coords)
                    all_scores.append(scores)
                    crop.close()
                image.close()
                np.savez(cache, boxes=np.concatenate(all_boxes), scores=np.concatenate(all_scores))
            with np.load(cache) as values:
                boxes, scores = values["boxes"], values["scores"]
            targets = []
            for a in row["annotations"]:
                x, y, w, h = a["bbox_xywh"]
                targets.append([x, y, x + w, y + h])
            for result, (score, overlap) in zip(results, settings, strict=True):
                selected = nms(
                    [
                        Detection(*map(float, boxes[i]), float(scores[i]))
                        for i in np.flatnonzero(scores >= score)
                    ],
                    overlap,
                    0.9,
                )
                matches = match_boxes([[d.x1, d.y1, d.x2, d.y2] for d in selected], targets, 0.5)
                missed, extra = len(targets) - len(matches), len(selected) - len(matches)
                result["missed"] += missed
                result["extra"] += extra
                if missed or extra:
                    result["images"].append(
                        {"image_id": row["image_id"], "missed": missed, "extra": extra}
                    )
            if row["image_id"] % 25 == 0:
                print(scale, row["image_id"], flush=True)
        summaries.extend(results)
        print(results, flush=True)
    runner.close()
    write_json(work / "report.json", {"protocol": protocol, "policies": summaries})


if __name__ == "__main__":
    main()
