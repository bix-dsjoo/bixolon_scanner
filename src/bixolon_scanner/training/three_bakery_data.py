"""Audited, source-only data and annotation preparation for three_bakery."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_path(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    result.relative_to(root.resolve())
    return result


def audit_sources(config: dict) -> tuple[list[dict], list[dict]]:
    root = Path(config["dataset_root"]).resolve()
    directories = {"single": "single_object", "multi": "multi_object", "background": "background"}
    if {p.name for p in root.iterdir()} != set(directories.values()):
        raise ValueError("unexpected entries in the training source root")
    rows, labels, hashes = [], [], set()
    identities = {}
    for snapshot in config.get("source_revision", {}).get("snapshots", []):
        path = Path(snapshot["path"])
        if sha256_file(path) != snapshot["sha256"]:
            raise ValueError("source revision snapshot checksum mismatch")
        for row in read_jsonl(path):
            if row["image_sha256"] in identities:
                raise ValueError("duplicate source revision identity")
            identities[row["image_sha256"]] = row
    for category, directory in enumerate(sorted((root / "single_object").iterdir()), 1):
        prefix = f"bread_{category:02d}_"
        if not directory.is_dir() or not directory.name.startswith(prefix):
            raise ValueError("invalid class directory")
        if len(list(directory.iterdir())) != config["shots_per_class"]:
            raise ValueError("unexpected source count per class")
        labels.append(
            {
                "category_id": category,
                "class_id": f"bread_{category:02d}",
                "class_name": directory.name[len(prefix) :].replace("_", " ").title(),
            }
        )
    if len(labels) != config["class_count"]:
        raise ValueError("unexpected class count")
    for kind, dirname in directories.items():
        files = sorted((root / dirname).rglob("*"))
        files = [p for p in files if p.is_file()]
        if len(files) != config["expected_counts"][kind]:
            raise ValueError(f"unexpected {kind} source count")
        for path in files:
            path.resolve().relative_to(root)
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("unexpected input format")
            digest = sha256_file(path)
            identity = identities.get(digest)
            if identities and (
                identity is None
                or identity["image_path"] != path.relative_to(root).as_posix()
                or identity["kind"] != kind
            ):
                raise ValueError("source differs from the authorized revision")
            if digest in hashes:
                raise ValueError("duplicate original content")
            hashes.add(digest)
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.load()
            category = int(path.parent.name.split("_")[1]) if kind == "single" else None
            rows.append(
                {
                    "image_id": identity["image_id"] if identity else len(rows) + 1,
                    "kind": kind,
                    "image_path": path.relative_to(root).as_posix(),
                    "image_sha256": digest,
                    "width": image.width,
                    "height": image.height,
                    "category_id": category,
                    "capture_session_id": identity["capture_session_id"]
                    if identity
                    else "three_bakery:20260907",
                    "source_group": config["source_group"],
                    "split": "train",
                    "fold": None,
                    "physical_item_ids": []
                    if category is None
                    else [f"three_bakery:bread_{category:02d}:item"],
                }
            )
    if identities and hashes != set(identities):
        raise ValueError("source revision coverage mismatch")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate source IDs")
    return sorted(rows, key=lambda r: r["image_id"]), labels


def freeze_sources(config_path: Path, output: Path) -> dict:
    config = load_json_config(config_path)
    rows, labels = audit_sources(config)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "originals.jsonl"
    if manifest.exists() and read_jsonl(manifest) != rows:
        raise ValueError("original manifest changed")
    write_jsonl(manifest, rows)
    report = {
        "schema_version": "1.0",
        "dataset_root": str(Path(config["dataset_root"]).resolve()),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(manifest),
        "original_count": len(rows),
        "labels": labels,
        "independent_validation_available": False,
        "evaluation_role": "same_physical_item_development_diagnostic",
    }
    write_json(output / "source-report.json", report)
    return report


def verify_sources(output: Path) -> tuple[dict, list[dict]]:
    report = load_json_config(output / "source-report.json")
    if sha256_file(output / "originals.jsonl") != report["source_manifest_sha256"]:
        raise ValueError("source manifest checksum mismatch")
    if sha256_file(Path(report["config_path"])) != report["config_sha256"]:
        raise ValueError("configuration changed")
    rows = read_jsonl(output / "originals.jsonl")
    for row in rows:
        if (
            sha256_file(source_path(Path(report["dataset_root"]), row["image_path"]))
            != row["image_sha256"]
        ):
            raise ValueError("source image checksum mismatch")
    return report, rows


def verify_prepared(work: Path, recipe: str, seed: int) -> None:
    """Reject stale or edited derivatives before a trainer consumes any input."""
    from .three_bakery_preparation import verify_annotations

    report, _ = verify_annotations(work / "sources")
    annotation_sha = sha256_file(work / "sources/annotations.jsonl")
    prepared = load_json_config(work / "prepared/originals-prepared.json")
    if prepared["annotation_sha256"] != annotation_sha:
        raise ValueError("prepared crops/boxes refer to stale annotations")
    for name, digest in prepared["manifest_sha256"].items():
        if sha256_file(work / f"prepared/{name}.jsonl") != digest:
            raise ValueError("prepared original manifest changed")
    synthetic = work / f"prepared/{recipe}-{seed}"
    evidence = load_json_config(synthetic / "report.json")
    if (
        evidence["annotation_sha256"] != annotation_sha
        or evidence["config_sha256"] != report["config_sha256"]
        or evidence["recipe"] != recipe
        or evidence["seed"] != seed
        or evidence["manifest_sha256"] != sha256_file(synthetic / "manifest.jsonl")
    ):
        raise ValueError("synthetic derivatives refer to stale or changed inputs")


def contact_sheet(paths: list[Path], output: Path, *, columns: int = 5, width: int = 400) -> None:
    height = round(width * 1714 / 2048) + 30
    canvas = Image.new("RGB", (columns * width, math.ceil(len(paths) / columns) * height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, path in enumerate(paths):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((width, height - 30))
        x, y = (i % columns) * width, (i // columns) * height
        canvas.paste(image, (x, y + 30))
        draw.text((x + 5, y + 5), path.stem, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def propose_boxes(image: Image.Image, settings: dict, single: bool) -> list[list[int]]:
    import cv2

    resized = image.copy()
    resized.thumbnail((settings["proposal_size"], settings["proposal_size"]))
    rgb = np.asarray(resized).astype(np.float32)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    foreground = (
        (hsv[:, :, 1] >= settings["saturation_minimum"])
        & (rgb[:, :, 0] > rgb[:, :, 2] * settings["red_blue_ratio"])
        & (rgb[:, :, 0] > rgb[:, :, 1] * settings["red_green_ratio"])
    )
    binary = cv2.morphologyEx(
        foreground.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    components = [r for r in stats[1:n] if r[4] >= settings["minimum_component_area"]]
    components.sort(key=lambda r: -r[4])
    if single:
        components = components[:1]
    sx, sy = image.width / resized.width, image.height / resized.height
    boxes = []
    for x, y, w, h, _ in components:
        margin = max(w, h) * settings["prompt_margin"]
        boxes.append(
            [
                max(0, round((x - margin) * sx)),
                max(0, round((y - margin) * sy)),
                min(image.width, round((x + w + margin) * sx)),
                min(image.height, round((y + h + margin) * sy)),
            ]
        )
    return sorted(boxes, key=lambda b: (b[1], b[0]))


class SamMaskProposer:
    def __init__(self, name: str):
        import torch
        from transformers import SamModel, SamProcessor

        torch.set_num_threads(4)
        self.torch = torch
        self.processor = SamProcessor.from_pretrained(name, local_files_only=True)
        self.model = SamModel.from_pretrained(name, local_files_only=True).cuda().eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def masks(self, image: Image.Image, boxes: list[list[int]]) -> list[np.ndarray]:
        if not boxes:
            return []
        inputs = self.processor(image, input_boxes=[boxes], return_tensors="pt")
        original, reshaped = inputs.pop("original_sizes"), inputs.pop("reshaped_input_sizes")
        with self.torch.inference_mode():
            outputs = self.model(**{k: v.cuda() for k, v in inputs.items()}, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), original, reshaped
        )[0]
        scores = outputs.iou_scores.cpu()[0]
        result = []
        for i, box in enumerate(boxes):
            candidates = []
            left, top, right, bottom = box
            for j in range(masks.shape[1]):
                mask = masks[i, j].numpy().astype(bool)
                inside = mask[top:bottom, left:right].sum()
                total = mask.sum()
                if total and inside / total >= 0.85:
                    candidates.append((float(scores[i, j]), j, mask))
            if not candidates:
                raise ValueError(f"SAM mask escaped its prompt box: {box}")
            result.append(max(candidates, key=lambda x: (x[0], -x[1]))[2])
        return result


def draft_annotations(source: Path, overrides: Path | None = None) -> None:
    report, rows = verify_sources(source)
    config = load_json_config(Path(report["config_path"]))
    settings = config["annotation"]
    override_data = load_json_config(overrides) if overrides else {}
    proposer = SamMaskProposer(settings["model"])
    draft_root = source / "draft"
    draft_root.mkdir(exist_ok=True)
    for row in rows:
        image_id = row["image_id"]
        destination = draft_root / f"{image_id:04d}.json"
        manual = override_data.get(str(image_id))
        if destination.exists() and manual is None:
            continue
        path = source_path(Path(report["dataset_root"]), row["image_path"])
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        boxes = (
            []
            if row["kind"] == "background"
            else (
                [v["bbox_xyxy"] for v in manual]
                if manual
                else propose_boxes(image, settings, row["kind"] == "single")
            )
        )
        masks = proposer.masks(image, boxes)
        annotations = []
        overlay = np.asarray(image).copy()
        for index, mask in enumerate(masks):
            ys, xs = np.nonzero(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            mask_path = source / "masks" / f"{image_id:04d}_{index:02d}.png"
            mask_path.parent.mkdir(exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            category = manual[index].get("category_id") if manual else row["category_id"]
            annotations.append(
                {
                    "category_id": category,
                    "bbox_xyxy": bbox,
                    "mask_path": mask_path.relative_to(source).as_posix(),
                    "mask_sha256": sha256_file(mask_path),
                    "prompt_box": boxes[index],
                }
            )
            overlay[mask] = (overlay[mask] * 0.85 + np.array([0, 255, 100]) * 0.15).astype(np.uint8)
        qa = Image.fromarray(overlay)
        draw = ImageDraw.Draw(qa)
        for i, annotation in enumerate(annotations):
            box = annotation["bbox_xyxy"]
            draw.rectangle(box, outline="red", width=5)
            draw.text(
                (box[0] + 5, box[1] + 5),
                f"{i}: {annotation['category_id']}",
                fill="black",
                stroke_width=2,
                stroke_fill="white",
                font_size=32,
            )
        qa_path = source / "overlays" / f"{image_id:04d}.jpg"
        qa_path.parent.mkdir(exist_ok=True)
        qa.save(qa_path, quality=94)
        write_json(
            destination,
            {
                **row,
                "annotations": annotations,
                "reviewed": False,
                "sam_revision": proposer.revision,
            },
        )
        print(json.dumps({"draft_image": image_id, "objects": len(annotations)}), flush=True)
    for start in range(0, 200, 10):
        contact_sheet(
            [source / "overlays" / f"{i:04d}.jpg" for i in range(start + 1, start + 11)],
            source / "qa" / f"single_{start // 10 + 1:02d}.jpg",
        )
    for start in range(200, 250, 6):
        contact_sheet(
            [source / "overlays" / f"{i:04d}.jpg" for i in range(start + 1, min(251, start + 7))],
            source / "qa" / f"multi_{start + 1:04d}.jpg",
            columns=3,
            width=640,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "draft"])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/bread/three_bakery.json")
    )
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/retraining/three-bakery/sources")
    )
    parser.add_argument("--overrides", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(freeze_sources(args.config, args.source), ensure_ascii=False))
    else:
        draft_annotations(args.source, args.overrides)


if __name__ == "__main__":
    main()
