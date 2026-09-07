from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from bixolon_scanner.configuration import load_json_config


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_bix_bakery_multi_object(
    dataset_root: Path,
    annotation_config: Path,
    output_root: Path,
) -> dict[str, Any]:
    root = dataset_root.resolve()
    if root.name != "bix_bakery_dataset":
        raise ValueError("dataset root must be named bix_bakery_dataset")
    image_root = root / "multi_object"
    if not image_root.is_dir():
        raise ValueError("bix_bakery_dataset requires multi_object")
    if output_root.exists():
        raise FileExistsError(output_root)

    config = load_json_config(annotation_config)
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported multi-object annotation schema")
    configured = config.get("images")
    if not isinstance(configured, list) or len(configured) != 20:
        raise ValueError("multi-object annotation config requires exactly 20 images")
    actual_paths = sorted(path for path in image_root.iterdir() if path.is_file())
    if len(actual_paths) != 20 or any(
        path.suffix.lower() not in {".jpg", ".jpeg"} for path in actual_paths
    ):
        raise ValueError("multi_object must contain exactly 20 JPEG images")
    if [str(row.get("image")) for row in configured] != [path.name for path in actual_paths]:
        raise ValueError("multi-object annotation image list does not match the source directory")

    output_root.mkdir(parents=True)
    qa_root = output_root / "qa"
    qa_root.mkdir()
    crop_root = output_root / "crops"
    crop_root.mkdir()
    records: list[dict[str, Any]] = []
    classifier_records: list[dict[str, Any]] = []
    source_identity: list[dict[str, str]] = []
    annotation_id = 1
    category_counts = {category_id: 0 for category_id in range(1, 21)}
    for image_id, (path, configured_row) in enumerate(
        zip(actual_paths, configured, strict=True), start=1
    ):
        image_sha256 = _sha256(path)
        if configured_row.get("sha256") != image_sha256:
            raise ValueError(f"source image SHA-256 mismatch: {path.name}")
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        configured_objects = configured_row.get("objects")
        if not isinstance(configured_objects, list) or not configured_objects:
            raise ValueError(f"multi-object image has no annotations: {path.name}")
        annotations = []
        physical_items = set()
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for object_index, value in enumerate(configured_objects, start=1):
            category_id = int(value["category_id"])
            if category_id not in category_counts:
                raise ValueError(f"invalid category_id in {path.name}: {category_id}")
            box = [int(coordinate) for coordinate in value["bbox_xyxy"]]
            if len(box) != 4:
                raise ValueError(f"invalid bbox in {path.name}")
            left, top, right, bottom = box
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError(f"bbox is outside {path.name}")
            annotations.append(
                {
                    "annotation_id": annotation_id,
                    "category_id": category_id,
                    "bbox_xywh": [left, top, right - left, bottom - top],
                    "area": (right - left) * (bottom - top),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            category_counts[category_id] += 1
            physical_items.add(f"bread_{category_id:02d}:bix_bakery:item")
            crop_path = crop_root / f"{image_id:02d}_{object_index:02d}.png"
            image.crop((left, top, right, bottom)).save(crop_path)
            classifier_records.append(
                {
                    "record_type": "classification",
                    "source": "bix_bakery_multi_object_reviewed_crop",
                    "source_dataset": "bix_bakery_dataset",
                    "image_path": crop_path.relative_to(output_root).as_posix(),
                    "image_sha256": _sha256(crop_path),
                    "original_image_path": path.relative_to(root).as_posix(),
                    "original_image_sha256": image_sha256,
                    "category_id": category_id,
                    "class_id": f"bread_{category_id:02d}",
                    "capture_session_id": "bix_bakery:multi_object:session",
                    "physical_item_id": f"bread_{category_id:02d}:bix_bakery:item",
                    "source_group": "bix_bakery:multi_object:session",
                    "capture_index": image_id - 1,
                    "width": right - left,
                    "height": bottom - top,
                    "split": "train_support",
                    "fold": None,
                }
            )
            draw.rectangle((left, top, right - 1, bottom - 1), outline=(255, 0, 0), width=5)
            draw.text(
                (left + 6, top + 6),
                f"{category_id:02d}",
                fill=(255, 0, 0),
                stroke_width=2,
                stroke_fill="white",
            )
        qa_path = qa_root / f"{image_id:02d}.jpg"
        overlay.thumbnail((960, 720), Image.Resampling.LANCZOS)
        overlay.save(qa_path, quality=92)
        source_identity.append({"image": path.name, "sha256": image_sha256})
        records.append(
            {
                "record_type": "detection",
                "source": "bix_bakery_multi_object_manual_annotation",
                "source_dataset": "bix_bakery_dataset",
                "image_id": image_id,
                "image_path": path.relative_to(root).as_posix(),
                "image_sha256": image_sha256,
                "capture_session_id": "bix_bakery:multi_object:session",
                "physical_item_ids": sorted(physical_items),
                "split": "train_operational",
                "fold": None,
                "width": width,
                "height": height,
                "annotations": annotations,
            }
        )

    missing_categories = [
        category_id for category_id, count in category_counts.items() if count == 0
    ]
    if missing_categories:
        raise ValueError(f"multi-object annotations omit categories: {missing_categories}")
    manifest = "".join(_canonical_json(row) + "\n" for row in records)
    (output_root / "manifest.jsonl").write_text(manifest, encoding="utf-8", newline="\n")
    classifier_manifest = "".join(_canonical_json(row) + "\n" for row in classifier_records)
    (output_root / "classifier-manifest.jsonl").write_text(
        classifier_manifest,
        encoding="utf-8",
        newline="\n",
    )
    source_digest = hashlib.sha256(_canonical_json(source_identity).encode()).hexdigest()
    metadata = {
        "schema_version": "1.0",
        "source_dataset": "bix_bakery_dataset",
        "source_directory": "multi_object",
        "source_image_set_sha256": source_digest,
        "image_count": len(records),
        "annotation_count": sum(len(row["annotations"]) for row in records),
        "category_counts": {str(key): value for key, value in category_counts.items()},
        "annotation_method": "human_visual_bbox_review",
        "annotation_config_sha256": _sha256(annotation_config),
        "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "classifier_manifest_sha256": hashlib.sha256(classifier_manifest.encode()).hexdigest(),
        "external_training_images_accessed": False,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reviewed BIX bakery multi-object labels")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_bix_bakery_multi_object(
                args.dataset_root,
                args.annotation_config,
                args.output_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
