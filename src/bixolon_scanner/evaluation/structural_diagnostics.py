"""Keep individual gains, losses and failures visible beside aggregate model scores."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from ..training.three_bakery_data import read_jsonl, write_json

BASELINES = {
    "log": "artifacts/retraining/recapture-0.2.1/packaged/cpu/0.2.1/responses.jsonl",
    "original": "artifacts/retraining/recapture-0.2.1/confirmation-context/original/cpu/pair-1/candidate/responses.jsonl",
    "final": "artifacts/retraining/recapture-0.2.1/final300-context/cpu/0.2.1/responses.jsonl",
}
MANIFESTS = {
    "log": "artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl",
    "original": "artifacts/retraining/three-bakery-revised300/prepared/original_detection.jsonl",
    "final": "artifacts/retraining/recapture-0.2.1/final300-context/final-inputs.jsonl",
}


def object_results(rows):
    objects, wrong = {}, set()
    for row in rows:
        for item, segmentation in zip(
            row["metrics"]["items"], row["response"]["segmentations"], strict=True
        ):
            key = (row["image_id"], item["target_index"])
            entry = {
                "status": item["status"],
                "predicted": item["predicted_class_id"],
                "target": item["target_class_id"],
                "segmentation": segmentation,
            }
            if item["target_index"] is not None:
                objects[key] = entry
            if (
                item["status"] == "APPROVED"
                and item["predicted_class_id"] != item["target_class_id"]
            ):
                location = (
                    item["target_index"]
                    if item["target_index"] is not None
                    else tuple(segmentation["bbox"].values())
                )
                wrong.add((row["image_id"], location, item["predicted_class_id"]))
    return objects, wrong


def compare(baseline, candidate):
    before, old_wrong = object_results(baseline)
    after, new_wrong = object_results(candidate)

    def correct(row):
        return row is not None and row["status"] == "APPROVED" and row["predicted"] == row["target"]

    gains, losses, transitions = [], [], []
    for key in sorted(before.keys() | after.keys()):
        left, right = before.get(key), after.get(key)
        record = {"image_id": key[0], "target_index": key[1], "before": left, "after": right}
        if correct(left) and not correct(right):
            losses.append(record)
        if correct(right) and not correct(left):
            gains.append(record)
        if (
            left is None
            or right is None
            or (left["status"], left["predicted"]) != (right["status"], right["predicted"])
        ):
            transitions.append(record)
    return {
        "lost_correct": losses,
        "gained_correct": gains,
        "transitions": transitions,
        "new_wrong_approvals": sorted(new_wrong - old_wrong, key=str),
        "fixed_wrong_approvals": sorted(old_wrong - new_wrong, key=str),
    }


def render_region(record, baseline, candidate, focus, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    if sha256_file(Path(record["image_path"])) != record["image_sha256"]:
        raise ValueError("failure image checksum mismatch")
    with Image.open(record["image_path"]) as opened:
        image = opened.convert("RGB")
    x, y, width, height = focus
    margin = max(width, height) * 0.2
    bounds = [
        max(0, x - margin),
        min(image.width, x + width + margin),
        min(image.height, y + height + margin),
        max(0, y - margin),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    colors = {"APPROVED": "#14944b", "UNKNOWN": "#db8c00", "SEGMENT_RECAPTURE": "#d92828"}
    for ax, title in zip(axes, ["Source", "GT", "Reference", "Candidate"], strict=True):
        ax.imshow(image)
        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])
        ax.set_title(title)
        ax.axis("off")

    def draw(ax, box, label, color):
        left, top, w, h = box
        if left > bounds[1] or left + w < bounds[0] or top > bounds[2] or top + h < bounds[3]:
            return
        ax.add_patch(Rectangle((left, top), w, h, fill=False, edgecolor=color, linewidth=2))
        ax.text(
            left,
            max(top, bounds[3]),
            label,
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.85, "pad": 1},
            clip_on=True,
        )

    for annotation in record["annotations"]:
        draw(
            axes[1],
            annotation.get("bbox_xywh", annotation.get("bbox")),
            f"bread_{annotation['category_id']:02d}",
            "#186cd0",
        )
    for ax, row in [(axes[2], baseline), (axes[3], candidate)]:
        for segmentation in row["response"]["segmentations"]:
            box = segmentation["bbox"]
            predicted = segmentation["prediction"] or (
                segmentation["top3"][0] if segmentation["top3"] else {}
            )
            label = segmentation["status"] + " " + predicted.get("class_id", "")
            draw(
                ax,
                [box["x"], box["y"], box["width"], box["height"]],
                label,
                colors[segmentation["status"]],
            )
    fig.suptitle(f"Image {record['image_id']} | original pixels, GT and public scan outputs")
    fig.savefig(output, dpi=120)
    plt.close(fig)


def run(root: Path):
    report = {}
    for candidate in sorted(root.glob("students/*")):
        for dataset, source in BASELINES.items():
            path = candidate / "regression" / f"{dataset}.json"
            if not path.exists():
                continue
            data = load_json_config(path)
            result = compare(read_jsonl(Path(source)), data["rows"])
            report[candidate.name + "/" + dataset] = {"summary": data["summary"], **result}
    write_json(root / "individual-regression.json", report)
    selected = root / "students/repvit_m0_9-supervised/regression"
    gallery = root / "failure-images"
    gallery.mkdir(exist_ok=True)
    entries = []
    for dataset in ["log", "final"]:
        records = {r["image_id"]: r for r in read_jsonl(Path(MANIFESTS[dataset]))}
        before = {r["image_id"]: r for r in read_jsonl(Path(BASELINES[dataset]))}
        after = {r["image_id"]: r for r in load_json_config(selected / f"{dataset}.json")["rows"]}
        jobs = []
        if dataset == "log":
            for row in after.values():
                for index, segmentation in enumerate(row["response"]["segmentations"]):
                    if segmentation["status"] != "APPROVED":
                        box = segmentation["bbox"]
                        jobs.append(
                            (
                                row["image_id"],
                                index,
                                [box["x"], box["y"], box["width"], box["height"]],
                                segmentation["status"],
                            )
                        )
        else:
            for index, row in enumerate(report["repvit_m0_9-supervised/final"]["lost_correct"]):
                jobs.append(
                    (
                        row["image_id"],
                        index,
                        records[row["image_id"]]["annotations"][row["target_index"]].get(
                            "bbox_xywh",
                            records[row["image_id"]]["annotations"][row["target_index"]].get(
                                "bbox"
                            ),
                        ),
                        "lost correct approval",
                    )
                )
        for image_id, index, focus, reason in jobs:
            filename = f"{dataset}-{image_id}-{index}.png"
            render_region(
                records[image_id], before[image_id], after[image_id], focus, gallery / filename
            )
            entries.append(
                f'<p>{html.escape(dataset)} {image_id}: {html.escape(reason)}<br><a href="{filename}"><img src="{filename}" width="1100"></a></p>'
            )
    (gallery / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>R4 failure regions</title><h1>원본 · GT · 기준 · 후보</h1>'
        + "\n".join(entries),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)


if __name__ == "__main__":
    main()
