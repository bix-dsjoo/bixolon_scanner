"""Render every log132 wrong approval and miss for rejected detector candidates."""

import html
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.evaluation.structural_diagnostics import (
    BASELINES,
    MANIFESTS,
    compare,
    render_region,
)
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main():
    root = Path("artifacts/n100/structural-r4")
    gallery = root / "failure-images-rejected"
    gallery.mkdir(exist_ok=True)
    records = {row["image_id"]: row for row in read_jsonl(Path(MANIFESTS["log"]))}
    baseline = {row["image_id"]: row for row in read_jsonl(Path(BASELINES["log"]))}
    jobs = []
    candidates = [
        *sorted((root / "detectors").glob("*/regression/log.json")),
        *sorted((root / "quantization").glob("*/regression/log.json")),
    ]
    for candidate in candidates:
        name = candidate.parent.parent.name
        rows = load_json_config(candidate)["rows"]
        if name == "repvit-qat":
            indexed = {row["image_id"]: row for row in rows}
            for loss in compare(list(baseline.values()), rows)["lost_correct"]:
                record = records[loss["image_id"]]
                jobs.append(
                    (
                        name,
                        indexed[loss["image_id"]],
                        record["annotations"][loss["target_index"]]["bbox_xywh"],
                        "lost correct approval",
                    )
                )
        for row in rows:
            record = records[row["image_id"]]
            targets = {item["target_index"] for item in row["metrics"]["items"]}
            for index, annotation in enumerate(record["annotations"]):
                if index not in targets:
                    jobs.append((name, row, annotation["bbox_xywh"], f"missed GT {index}"))
            for item, segment in zip(
                row["metrics"]["items"], row["response"]["segmentations"], strict=True
            ):
                if (
                    item["status"] == "APPROVED"
                    and item["predicted_class_id"] != item["target_class_id"]
                ):
                    box = segment["bbox"]
                    jobs.append(
                        (
                            name,
                            row,
                            [box[k] for k in ["x", "y", "width", "height"]],
                            "wrong approval",
                        )
                    )
        crowded = max(rows, key=lambda row: row["metrics"]["extra_count"])
        record = records[crowded["image_id"]]
        jobs.append(
            (name, crowded, [0, 0, record["width"], record["height"]], "most extra detections")
        )
    entries, details = [], []
    for index, (name, row, focus, reason) in enumerate(jobs):
        filename = f"detector-{name}-{row['image_id']}-{index}.png"
        render_region(
            records[row["image_id"]], baseline[row["image_id"]], row, focus, gallery / filename
        )
        details.append(
            {"candidate": name, "image_id": row["image_id"], "reason": reason, "file": filename}
        )
        entries.append(
            f"<p>{html.escape(name)} / image {row['image_id']} / {html.escape(reason)}<br>"
            f'<a href="{filename}"><img src="{filename}" width="1100"></a></p>'
        )
    (gallery / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Rejected candidate failures</title>'
        "<h1>제외한 후보: 원본 · GT · 기준 · 후보</h1>" + "\n".join(entries),
        encoding="utf-8",
    )
    write_json(gallery / "regions.json", details)
    print(f"Rendered {len(details)} rejected detector regions")


if __name__ == "__main__":
    main()
