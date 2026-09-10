"""Reviewed source derivatives; every generated pixel has an allowed parent."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from .three_bakery_data import read_jsonl, source_path, verify_sources, write_json, write_jsonl


def validate_annotation(row: dict, class_count: int) -> None:
    annotations = row["annotations"]
    if (row["kind"] == "background") != (len(annotations) == 0):
        raise ValueError("background/foreground annotation mismatch")
    if row["kind"] == "single" and len(annotations) != 1:
        raise ValueError("single source must have one object")
    for annotation in annotations:
        x1, y1, x2, y2 = annotation["bbox_xyxy"]
        if not (0 <= x1 < x2 <= row["width"] and 0 <= y1 < y2 <= row["height"]):
            raise ValueError("annotation outside native image coordinates")
        if not 1 <= annotation["category_id"] <= class_count:
            raise ValueError("invalid class annotation")
        if row["kind"] == "single" and annotation["category_id"] != row["category_id"]:
            raise ValueError("single annotation disagrees with directory label")


def accept_review(source: Path, review_path: Path) -> None:
    report, originals = verify_sources(source)
    review = load_json_config(review_path)
    if review["source_manifest_sha256"] != report["source_manifest_sha256"]:
        raise ValueError("review refers to different source images")
    if set(review["reviewed_image_ids"]) != {r["image_id"] for r in originals}:
        raise ValueError("all originals require visual annotation review")
    overrides = review.get("bbox_overrides", {})
    applied = set()
    rows = []
    for original in originals:
        draft_path = source / "draft" / f"{original['image_id']:04d}.json"
        if review["draft_sha256"][str(original["image_id"])] != sha256_file(draft_path):
            raise ValueError("annotation changed after review")
        row = load_json_config(draft_path)
        if any(row[k] != v for k, v in original.items()):
            raise ValueError("annotation source identity mismatch")
        for index, annotation in enumerate(row["annotations"]):
            key = f"{row['image_id']}:{index}"
            if key in overrides:
                correction = overrides[key]
                if correction["source_sha256"] != row["image_sha256"]:
                    raise ValueError("bbox correction source identity mismatch")
                if correction["original_bbox_xyxy"] != annotation["bbox_xyxy"]:
                    raise ValueError("bbox correction does not match reviewed SAM proposal")
                annotation["sam_bbox_xyxy"] = annotation["bbox_xyxy"]
                annotation["bbox_xyxy"] = correction["bbox_xyxy"]
                annotation["bbox_review_reason"] = correction["reason"]
                applied.add(key)
            mask_path = source_path(source, annotation["mask_path"])
            if sha256_file(mask_path) != annotation["mask_sha256"]:
                raise ValueError("reviewed mask changed")
            with Image.open(mask_path) as mask:
                if mask.size != (row["width"], row["height"]):
                    raise ValueError("mask must use original image coordinates")
            annotation["copy_paste_eligible"] = (
                row["kind"] == "single" and row["image_id"] not in review["copy_paste_excluded_ids"]
            )
        validate_annotation(row, len(report["labels"]))
        row.update(reviewed=True, review_sha256=sha256_file(review_path))
        row["physical_item_ids"] = sorted(
            {f"three_bakery:bread_{a['category_id']:02d}:item" for a in row["annotations"]}
        )
        rows.append(row)
    if applied != set(overrides):
        raise ValueError("bbox correction refers to an absent object")
    write_jsonl(source / "annotations.jsonl", rows)
    write_json(
        source / "annotation-report.json",
        {
            "source_manifest_sha256": report["source_manifest_sha256"],
            "annotation_sha256": sha256_file(source / "annotations.jsonl"),
            "review_sha256": sha256_file(review_path),
            "images": len(rows),
            "objects": sum(len(r["annotations"]) for r in rows),
            "review_method": review["review_method"],
        },
    )


def verify_annotations(source: Path) -> tuple[dict, list[dict]]:
    report, originals = verify_sources(source)
    annotation_report = load_json_config(source / "annotation-report.json")
    if (
        annotation_report["source_manifest_sha256"] != report["source_manifest_sha256"]
        or sha256_file(source / "annotations.jsonl") != annotation_report["annotation_sha256"]
    ):
        raise ValueError("annotation provenance mismatch")
    rows = read_jsonl(source / "annotations.jsonl")
    if {r["image_sha256"] for r in rows} != {r["image_sha256"] for r in originals}:
        raise ValueError("annotation source coverage mismatch")
    return report, rows


def validate_parents(rows: list[dict], allowed: set[str]) -> None:
    for row in rows:
        parents = row.get("parent_sha256", [])
        if not parents or not set(parents) <= allowed:
            raise ValueError("derivative missing provenance or contains external sources")
        for annotation in row.get("annotations", []):
            if annotation["source_sha256"] not in parents:
                raise ValueError("object parent missing from scene provenance")


def prepare_originals(source: Path, output: Path) -> None:
    report, rows = verify_annotations(source)
    output.mkdir(parents=True, exist_ok=True)
    detection, classification, support = [], [], []
    for row in rows:
        path = source_path(Path(report["dataset_root"]), row["image_path"])
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        annotations = []
        for index, annotation in enumerate(row["annotations"]):
            x1, y1, x2, y2 = annotation["bbox_xyxy"]
            annotations.append(
                {
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": annotation["category_id"],
                    "source_sha256": row["image_sha256"],
                }
            )
            margin = round(max(x2 - x1, y2 - y1) * 0.08)
            box = [
                max(0, x1 - margin),
                max(0, y1 - margin),
                min(image.width, x2 + margin),
                min(image.height, y2 + margin),
            ]
            crop_path = output / "crops" / f"{row['image_id']:04d}_{index:02d}.jpg"
            crop_path.parent.mkdir(exist_ok=True)
            image.crop(box).save(crop_path, quality=96)
            crop_row = {
                "image_id": f"{row['image_id']}_{index}",
                "record_type": "classification",
                "image_path": str(crop_path.resolve()),
                "image_sha256": sha256_file(crop_path),
                "category_id": annotation["category_id"],
                "class_id": f"bread_{annotation['category_id']:02d}",
                "class_name": report["labels"][annotation["category_id"] - 1]["class_name"],
                "parent_sha256": [row["image_sha256"]],
                "parent_bbox_xyxy": box,
                "source_group": row["source_group"],
                "split": "train",
                "capture_session_id": row["capture_session_id"],
                "perceptual_group_id": f"source_view:{row['image_sha256']}",
                "physical_item_id": f"three_bakery:bread_{annotation['category_id']:02d}:item",
            }
            classification.append(crop_row)
            if row["kind"] == "single":
                support.append(crop_row)
        detection.append(
            {
                **row,
                "image_path": str(path),
                "record_type": "detection",
                "parent_sha256": [row["image_sha256"]],
                "annotations": annotations,
                "difficulty": row["kind"],
            }
        )
    for name, records in (
        ("original_detection", detection),
        ("original_crops", classification),
        ("catalog_support", support),
    ):
        validate_parents(records, {r["image_sha256"] for r in rows})
        write_jsonl(output / f"{name}.jsonl", records)
    write_json(
        output / "originals-prepared.json",
        {
            "annotation_sha256": sha256_file(source / "annotations.jsonl"),
            "detection_images": len(detection),
            "object_crops": len(classification),
            "catalog_support": len(support),
            "manifest_sha256": {
                name: sha256_file(output / f"{name}.jsonl")
                for name in ("original_detection", "original_crops", "catalog_support")
            },
        },
    )


def generate_scenes(
    source: Path, output: Path, *, recipe: str, seed: int, settings_path: Path | None = None
) -> None:
    report, rows = verify_annotations(source)
    config = load_json_config(Path(report["config_path"]))
    settings = config["synthetic"] if settings_path is None else load_json_config(settings_path)
    variant = settings[recipe]
    rng = random.Random(seed)
    size = settings["image_size"]
    out = output / f"{recipe}-{seed}"
    key = {
        "annotation_sha256": sha256_file(source / "annotations.jsonl"),
        "config_sha256": report["config_sha256"],
        "recipe": recipe,
        "seed": seed,
        "generator_sha256": sha256_file(Path(__file__)),
        "settings_sha256": None if settings_path is None else sha256_file(settings_path),
    }
    if (out / "report.json").exists():
        previous = load_json_config(out / "report.json")
        if any(previous[k] != v for k, v in key.items()):
            raise ValueError("synthetic resume inputs changed")
        if sha256_file(out / "manifest.jsonl") != previous["manifest_sha256"]:
            raise ValueError("synthetic manifest changed")
        for row in read_jsonl(out / "manifest.jsonl"):
            if sha256_file(Path(row["image_path"])) != row["image_sha256"]:
                raise ValueError("synthetic image changed")
        return
    (out / "images").mkdir(parents=True, exist_ok=True)
    cutouts, backgrounds = [], []
    for row in rows:
        if row["kind"] == "background" or any(a["copy_paste_eligible"] for a in row["annotations"]):
            with Image.open(source_path(Path(report["dataset_root"]), row["image_path"])) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            if row["kind"] == "background":
                backgrounds.append((image.resize((size, size)), row["image_sha256"]))
            for annotation in row["annotations"]:
                if not annotation["copy_paste_eligible"]:
                    continue
                box = annotation["bbox_xyxy"]
                rgba = image.crop(box).convert("RGBA")
                mask_path = source_path(source, annotation["mask_path"])
                if sha256_file(mask_path) != annotation["mask_sha256"]:
                    raise ValueError("copy-paste mask checksum mismatch")
                with Image.open(mask_path) as mask:
                    rgba.putalpha(mask.crop(box))
                cutouts.append((rgba, annotation["category_id"], row["image_sha256"]))
    if {c[1] for c in cutouts} != {v["category_id"] for v in report["labels"]}:
        raise ValueError("copy-paste library must cover every class")
    records = []
    empty_count = round(variant["image_count"] * settings["empty_probability"])
    empty_indices = set(rng.sample(range(variant["image_count"]), empty_count))
    for index in range(variant["image_count"]):
        bg, bg_parent = rng.choice(backgrounds)
        parents = {bg_parent}
        if rng.random() < 0.5:
            canvas = bg.copy()
        else:
            # This procedural background derives its color palette from an allowed empty source.
            color = np.asarray(bg)[rng.randrange(size), rng.randrange(size)].astype(float)
            gradient = np.linspace(rng.uniform(0.7, 1), rng.uniform(1, 1.15), size)
            values = np.clip(color[None, None, :] * gradient[:, None, None], 0, 255)
            canvas = Image.fromarray(np.repeat(values.astype(np.uint8), size, axis=1))
        count = (
            0
            if index in empty_indices
            else rng.randint(settings.get("minimum_objects", 1), settings["maximum_objects"])
        )
        masks, placed = [], []
        for _ in range(count):
            cutout, category, digest = rng.choice(cutouts)
            rotated = cutout.rotate(
                rng.uniform(-180, 180), expand=True, resample=Image.Resampling.BICUBIC
            )
            minimum_scale, maximum_scale = settings.get(
                "object_scale_range", [0.11, 0.31 if variant["dense"] else 0.24]
            )
            scale = rng.uniform(minimum_scale, maximum_scale) * size / max(rotated.size)
            rgba = rotated.resize(
                tuple(max(1, round(v * scale)) for v in rotated.size), Image.Resampling.LANCZOS
            )
            for attempt in range(200):
                x, y = rng.randrange(size - rgba.width + 1), rng.randrange(size - rgba.height + 1)
                proposed = np.zeros((size, size), dtype=bool)
                proposed[y : y + rgba.height, x : x + rgba.width] = (
                    np.asarray(rgba.getchannel("A")) >= 128
                )
                if not proposed.any():
                    raise ValueError("empty transformed object mask")
                min_visible = settings["minimum_visible_fraction"] if variant["dense"] else 1.0
                if all(
                    np.count_nonzero(mask & ~proposed) / original_area >= min_visible
                    for mask, original_area in masks
                ):
                    break
            else:
                raise RuntimeError("could not place the configured object count")
            rgb = ImageEnhance.Color(rgba.convert("RGB")).enhance(rng.uniform(0.75, 1.2))
            rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.7, 1.2))
            canvas.paste(rgb, (x, y), rgba.getchannel("A"))
            masks = [(mask & ~proposed, area) for mask, area in masks]
            masks.append((proposed, int(proposed.sum())))
            placed.append((category, digest))
            parents.add(digest)
        annotations = []
        for (mask, area), (category, digest) in zip(masks, placed, strict=True):
            ys, xs = np.nonzero(mask)
            annotations.append(
                {
                    "category_id": category,
                    "source_sha256": digest,
                    "bbox_xywh": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1),
                    ],
                    "visible_fraction": float(mask.sum() / area),
                }
            )
        path = out / "images" / f"{index:05d}.jpg"
        canvas.save(path, quality=93)
        records.append(
            {
                "image_id": index + 1,
                "record_type": "detection",
                "image_path": str(path.resolve()),
                "image_sha256": sha256_file(path),
                "width": size,
                "height": size,
                "annotations": annotations,
                "parent_sha256": sorted(parents),
                "source_group": config["source_group"],
                "recipe": recipe,
                "seed": seed,
                "split": "development_diagnostic" if recipe == "diagnostic" else "train",
                "difficulty": "synthetic_empty" if not count else "synthetic_objects",
            }
        )
        if (index + 1) % 100 == 0:
            print(f"{recipe} {seed}: {index + 1}/{variant['image_count']}", flush=True)
    validate_parents(records, {r["image_sha256"] for r in rows})
    write_jsonl(out / "manifest.jsonl", records)
    write_json(
        out / "report.json",
        {
            **key,
            "images": len(records),
            "empty_images": empty_count,
            "manifest_sha256": sha256_file(out / "manifest.jsonl"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["review", "originals", "synthetic"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--recipe", choices=["basic", "dense", "diagnostic"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--settings", type=Path)
    args = parser.parse_args()
    if args.command == "review":
        accept_review(args.source, args.review)
    elif args.command == "originals":
        prepare_originals(args.source, args.output)
    else:
        generate_scenes(
            args.source,
            args.output,
            recipe=args.recipe,
            seed=args.seed,
            settings_path=args.settings,
        )


if __name__ == "__main__":
    main()
