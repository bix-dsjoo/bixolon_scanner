"""Import hash-bound source corrections and inspect new native-resolution SAM masks."""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..configuration import load_json_config
from ..contracts.catalog import sha256_file
from .three_bakery_data import (
    SamMaskProposer,
    read_jsonl,
    source_path,
    verify_sources,
    write_json,
    write_jsonl,
)
from .three_bakery_preparation import validate_annotation


def validate_corrected_labels(rows: list[dict]) -> None:
    by_id = {row["image_id"]: row for row in rows}
    for image_id, object_index, category in ((223, 3, 7), (284, 2, 3), (271, 5, 5)):
        objects = by_id[image_id]["annotations"]
        if len(objects) <= object_index or objects[object_index]["category_id"] != category:
            raise ValueError("authorized annotation correction is missing")
    if len(by_id[271]["annotations"]) != 6:
        raise ValueError("source 271 requires the six-object corrected annotation")


def draft_revision(source: Path) -> None:
    report, originals = verify_sources(source)
    config = load_json_config(Path(report["config_path"]))
    revision = config["source_revision"]
    imported = {}
    for snapshot in revision["snapshots"]:
        path = Path(snapshot["path"])
        if sha256_file(path) != snapshot["sha256"]:
            raise ValueError("revision annotation checksum mismatch")
        for row in read_jsonl(path):
            imported[row["image_sha256"]] = (row, snapshot.get("mask_root"))
    proposer = None
    rows = []
    native_review = []
    for original in originals:
        previous, mask_root = imported[original["image_sha256"]]
        row = {**original, "annotations": copy.deepcopy(previous["annotations"])}
        row["annotation_origin"] = {
            "previous_reviewed": previous.get("reviewed", False),
            "previous_review_status": previous.get("review_status"),
            "training_authorization": revision["authorization"],
        }
        validate_annotation(row, len(report["labels"]))
        for a in row["annotations"]:
            a["copy_paste_eligible"] = (
                bool(a.get("copy_paste_eligible")) and row["kind"] == "single"
            )
        if mask_root:
            for a in row["annotations"]:
                old_mask = source_path(Path(mask_root), a["mask_path"])
                if sha256_file(old_mask) != a["mask_sha256"]:
                    raise ValueError("previous source mask changed")
                new_mask = source_path(source, a["mask_path"])
                new_mask.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_mask, new_mask)
        elif row["annotations"]:
            path = source_path(Path(report["dataset_root"]), row["image_path"])
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            draft = source / "revision-drafts" / f"{row['image_id']:04d}.json"
            if draft.exists():
                cached = load_json_config(draft)
                if cached["image_sha256"] != row["image_sha256"]:
                    raise ValueError("revision mask draft source changed")
                for a, cached_a in zip(row["annotations"], cached["annotations"], strict=True):
                    if (a["bbox_xyxy"], a["category_id"]) != (
                        cached_a["bbox_xyxy"],
                        cached_a["category_id"],
                    ):
                        raise ValueError("revision mask draft annotation changed")
                    if (
                        sha256_file(source_path(source, cached_a["mask_path"]))
                        != cached_a["mask_sha256"]
                    ):
                        raise ValueError("revision draft mask changed")
                row = cached
            else:
                if proposer is None:
                    proposer = SamMaskProposer(config["annotation"]["model"])
                masks = proposer.masks(image, [a["bbox_xyxy"] for a in row["annotations"]])
                for i, (a, mask) in enumerate(zip(row["annotations"], masks, strict=True)):
                    mask_path = source / "masks" / f"{row['image_id']:04d}_{i:02d}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                    a.update(
                        mask_path=mask_path.relative_to(source).as_posix(),
                        mask_sha256=sha256_file(mask_path),
                        sam_revision=proposer.revision,
                    )
                write_json(draft, row)
            overlay = np.asarray(image).copy()
            for i, a in enumerate(row["annotations"]):
                with Image.open(source_path(source, a["mask_path"])) as opened:
                    mask = np.asarray(opened) > 0
                if mask.shape != (image.height, image.width) or not mask.any():
                    raise ValueError("empty or non-native revision mask")
                color = np.array([(i * 71 + 40) % 255, (i * 107 + 150) % 255, (i * 139 + 80) % 255])
                overlay[mask] = (overlay[mask] * 0.65 + color * 0.35).astype(np.uint8)
                crop = Image.new("RGB", image.size, "#303030")
                crop.paste(image, mask=Image.fromarray(mask.astype(np.uint8) * 255))
                crop_path = source / "mask-review" / f"{row['image_id']:04d}_{i + 1:02d}.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                # Native pixels retained; whitespace outside the prompt is removed only for display.
                crop.crop(a["bbox_xyxy"]).save(crop_path)
            view = Image.fromarray(overlay)
            draw = ImageDraw.Draw(view)
            for i, a in enumerate(row["annotations"]):
                b = a["bbox_xyxy"]
                draw.rectangle(b, outline="#00ff00", width=3)
                draw.text(
                    (b[0] + 4, b[1] + 4),
                    f"{i + 1}: C{a['category_id']:02d}",
                    font_size=32,
                    fill="black",
                    stroke_width=2,
                    stroke_fill="white",
                )
            overlay_path = source / "mask-review" / f"{row['image_id']:04d}.jpg"
            view.save(overlay_path, quality=95)
            native_review.append(
                {
                    "image_id": row["image_id"],
                    "overlay_sha256": sha256_file(overlay_path),
                    "mask_sha256": [a["mask_sha256"] for a in row["annotations"]],
                }
            )
        row["physical_item_ids"] = sorted(
            {f"three_bakery:bread_{a['category_id']:02d}:item" for a in row["annotations"]}
        )
        rows.append(row)
        print(f"Revision source {row['image_id']}: {len(row['annotations'])} objects", flush=True)
    if (
        len(rows) != revision["expected_images"]
        or sum(len(r["annotations"]) for r in rows) != revision["expected_objects"]
    ):
        raise ValueError("revision annotation counts mismatch")
    if (
        sum(a["copy_paste_eligible"] for r in rows for a in r["annotations"])
        != revision["expected_cutouts"]
    ):
        raise ValueError("revision cutout coverage mismatch")
    write_jsonl(source / "revision-annotations.jsonl", rows)
    write_json(
        source / "revision-mask-inputs.json",
        {
            "source_manifest_sha256": report["source_manifest_sha256"],
            "annotation_sha256": sha256_file(source / "revision-annotations.jsonl"),
            "new_images": native_review,
        },
    )


def accept_revision(source: Path, review_path: Path) -> None:
    report, _ = verify_sources(source)
    review = load_json_config(review_path)
    inputs = load_json_config(source / "revision-mask-inputs.json")
    if review["mask_inputs_sha256"] != sha256_file(source / "revision-mask-inputs.json"):
        raise ValueError("revision mask review differs from current inputs")
    if set(review["reviewed_image_ids"]) != {r["image_id"] for r in inputs["new_images"]}:
        raise ValueError("new revision masks require complete visual inspection")
    if sha256_file(source / "revision-annotations.jsonl") != inputs["annotation_sha256"]:
        raise ValueError("revision annotation changed after inspection")
    rows = read_jsonl(source / "revision-annotations.jsonl")
    validate_corrected_labels(rows)
    for row in rows:
        for a in row["annotations"]:
            path = source_path(source, a["mask_path"])
            if sha256_file(path) != a["mask_sha256"]:
                raise ValueError("revision mask changed after inspection")
            with Image.open(path) as mask:
                if mask.size != (row["width"], row["height"]):
                    raise ValueError("revision mask coordinates mismatch")
        row.update(reviewed=True, review_sha256=sha256_file(review_path))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.review:
        accept_revision(args.source, args.review)
    else:
        draft_revision(args.source)


if __name__ == "__main__":
    main()
