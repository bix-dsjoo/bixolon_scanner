from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _label_text(row: dict[str, Any]) -> str:
    width = float(row["width"])
    height = float(row["height"])
    lines = []
    for annotation in row["annotations"]:
        x, y, box_width, box_height = [float(value) for value in annotation["bbox_xywh"]]
        center_x = (x + box_width * 0.5) / width
        center_y = (y + box_height * 0.5) / height
        lines.append(
            f"0 {center_x:.10f} {center_y:.10f} {box_width / width:.10f} "
            f"{box_height / height:.10f}\n"
        )
    return "".join(lines)


def _objectness_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": int(row["image_id"]),
        "image_path": str(row["image_path"]),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "fold": row.get("fold"),
        "annotations": [
            {"bbox_xywh": [float(value) for value in annotation["bbox_xywh"]]}
            for annotation in row["annotations"]
        ],
    }


def export_yolo_objectness_dataset(
    manifest: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    validation_fold: int | None,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    rows = _read_manifest(manifest)
    if validation_fold is None:
        splits = {"train": rows, "validation": rows}
    else:
        splits = {
            "train": [row for row in rows if int(row["fold"]) != validation_fold],
            "validation": [row for row in rows if int(row["fold"]) == validation_fold],
        }
    if any(not split_rows for split_rows in splits.values()):
        raise ValueError("YOLO objectness export requires non-empty train and validation splits")

    transfer_modes: set[str] = set()
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for row in split_rows:
            source = (dataset_root / str(row["image_path"])).resolve()
            source.relative_to(dataset_root)
            if not source.is_file():
                raise FileNotFoundError(source)
            stem = f"{int(row['image_id']):09d}"
            image_path = image_dir / f"{stem}{source.suffix.lower()}"
            if not image_path.exists():
                transfer_modes.add(_link_or_copy(source, image_path))
            (label_dir / f"{stem}.txt").write_text(_label_text(row), encoding="utf-8")
        (manifest_dir / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(_objectness_row(row), separators=(",", ":")) + "\n" for row in split_rows
            ),
            encoding="utf-8",
        )

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/validation\n"
        "names:\n"
        "  0: bread_object\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "1.0",
        "architecture_contract": "one-class-product-independent-objectness",
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256(manifest),
        "validation_fold": validation_fold,
        "training_image_count": len(splits["train"]),
        "validation_image_count": len(splits["validation"]),
        "training_annotation_count": sum(len(row["annotations"]) for row in splits["train"]),
        "validation_annotation_count": sum(len(row["annotations"]) for row in splits["validation"]),
        "class_count": 1,
        "product_labels_exported": False,
        "difficulty_metadata_exported": False,
        "transfer_modes": sorted(transfer_modes),
        "independent_test_claimed": False,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export class-agnostic YOLO Detector data")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            export_yolo_objectness_dataset(
                args.manifest,
                args.dataset_root,
                args.output_dir,
                validation_fold=args.validation_fold,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
