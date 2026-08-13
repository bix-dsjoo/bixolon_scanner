from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..inference import build_onnx_adapters
from ..package import load_model_package, sha256_file


@dataclass(frozen=True)
class ScanLogSource:
    scan_id: str
    recorded_at: datetime
    log_path: Path
    image_path: Path
    payload: dict[str, Any]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _corrections(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "clear_annotations": [],
            "add_annotations": {},
            "attention": {},
            "confirmed_recapture": {},
        }
    value = _json(path)
    if value.get("schema_version") != "1.0":
        raise ValueError("unsupported review correction schema")
    return value


def _timestamp(payload: dict[str, Any]) -> datetime:
    raw = payload.get("recorded_at") or payload.get("confirmed_at") or payload.get("analyzed_at")
    if not isinstance(raw, str) or not raw:
        raise ValueError("scan log is missing its timestamp")
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def load_latest_logs(log_dir: Path, limit: int) -> list[ScanLogSource]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if not log_dir.is_dir():
        raise FileNotFoundError(log_dir)
    found: list[ScanLogSource] = []
    for log_path in sorted(log_dir.glob("*.json")):
        payload = _json(log_path)
        scan_id = payload.get("scan_id")
        stored_name = payload.get("original_image")
        if not isinstance(scan_id, str) or len(scan_id) != 32:
            raise ValueError(f"invalid scan id: {log_path}")
        if (
            not isinstance(stored_name, str)
            or not stored_name
            or Path(stored_name).name != stored_name
            or Path(stored_name).is_absolute()
        ):
            raise ValueError(f"unsafe original image name: {log_path}")
        image_path = log_dir / stored_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        found.append(
            ScanLogSource(
                scan_id=scan_id,
                recorded_at=_timestamp(payload),
                log_path=log_path,
                image_path=image_path,
                payload=payload,
            )
        )
    if len(found) < limit:
        raise ValueError(f"requested {limit} logs, found {len(found)}")
    selected = sorted(found, key=lambda row: (row.recorded_at, row.scan_id), reverse=True)[:limit]
    return sorted(selected, key=lambda row: (row.recorded_at, row.scan_id))


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max(axis=1, keepdims=True)
    exponential = np.exp(values)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _logged_annotations(payload: dict[str, Any], label_ids: dict[str, int]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for detection in payload.get("detections") or []:
        if not isinstance(detection, dict):
            continue
        bbox = detection.get("bbox")
        product = detection.get("final_product") or detection.get("initial_ai_prediction")
        if not isinstance(bbox, dict) or not isinstance(product, dict):
            continue
        class_id = product.get("class_id")
        if class_id not in label_ids:
            continue
        x = float(bbox["x"])
        y = float(bbox["y"])
        width = float(bbox["width"])
        height = float(bbox["height"])
        if width <= 0 or height <= 0:
            continue
        annotations.append(
            {
                "category_id": label_ids[class_id],
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
                "source": "scan_log_final",
                "review_status": "pending_user_review",
                "detector_score": None,
                "classifier_confidence": float(detection.get("initial_confidence", 0.0)),
                "user_modified": bool(detection.get("user_modified", False)),
            }
        )
    return annotations


def _model_annotations(
    image: Image.Image,
    detector,
    classifier,
    temperature: float,
) -> list[dict[str, Any]]:
    detections = sorted(detector.detect(image).detections, key=lambda box: (box.y1, box.x1))
    if not detections:
        return []
    probabilities = _softmax(classifier.classify(image, detections), temperature)
    annotations: list[dict[str, Any]] = []
    for detection, scores in zip(detections, probabilities):
        category_id = int(np.argmax(scores)) + 1
        width = detection.x2 - detection.x1
        height = detection.y2 - detection.y1
        annotations.append(
            {
                "category_id": category_id,
                "bbox": [detection.x1, detection.y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "source": "model_draft_detector",
                "review_status": "pending_user_review",
                "detector_score": float(detection.score),
                "classifier_confidence": float(scores[category_id - 1]),
                "user_modified": False,
            }
        )
    return annotations


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _render_review(
    image: Image.Image,
    sequence: int,
    status: str,
    reason_codes: Iterable[str],
    annotations: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    output: Path,
) -> None:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    scale = max(1, round(min(canvas.size) / 450))
    font = _font(max(22, round(min(canvas.size) / 70)))
    header = f"{sequence:03d}  {status or 'LEGACY'}  {'|'.join(reason_codes) or '-'}"
    box = draw.textbbox((12, 10), header, font=font, stroke_width=1)
    header_fill = (170, 25, 25) if status == "RECAPTURE (CONFIRMED)" else (25, 25, 25)
    draw.rectangle((0, 0, canvas.width, box[3] + 20), fill=header_fill)
    draw.text((12, 10), header, fill="white", font=font, stroke_width=1, stroke_fill="black")
    palette = ((0, 220, 255), (255, 190, 0), (80, 255, 120), (255, 80, 170))
    for offset, annotation in enumerate(annotations, start=1):
        x, y, width, height = annotation["bbox"]
        color = palette[(offset - 1) % len(palette)]
        draw.rectangle((x, y, x + width, y + height), outline=color, width=scale * 3)
        label = labels[annotation["category_id"] - 1]
        confidence = annotation.get("classifier_confidence")
        text = f"#{offset} {label['class_id']} {label['class_name']}"
        if confidence is not None:
            text += f" cls={confidence:.3f}"
        text_y = max(box[3] + 24, y - font.size - 8)
        draw.text((x, text_y), text, fill=color, font=font, stroke_width=2, stroke_fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=93, optimize=True)


def _contact_sheets(review_paths: list[Path], output_dir: Path) -> None:
    cell_width, cell_height, margin = 640, 480, 16
    for start in range(0, len(review_paths), 10):
        selected = review_paths[start : start + 10]
        sheet = Image.new(
            "RGB",
            (cell_width * 2 + margin * 3, cell_height * 5 + margin * 6),
            (18, 18, 18),
        )
        for offset, path in enumerate(selected):
            with Image.open(path) as source:
                tile = source.convert("RGB")
                tile.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
            column, row = offset % 2, offset // 2
            x = margin + column * (cell_width + margin) + (cell_width - tile.width) // 2
            y = margin + row * (cell_height + margin) + (cell_height - tile.height) // 2
            sheet.paste(tile, (x, y))
        sheet.save(
            output_dir / f"sheet_{start + 1:03d}_{start + len(selected):03d}.jpg", quality=92
        )


def export(args: argparse.Namespace) -> dict[str, Any]:
    selected = load_latest_logs(args.log_dir, args.limit)
    corrections = _corrections(args.corrections)
    clear_annotations = set(corrections.get("clear_annotations") or [])
    add_annotations = corrections.get("add_annotations") or {}
    attention = corrections.get("attention") or {}
    confirmed_recapture = corrections.get("confirmed_recapture") or {}
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    image_dir = output / "images"
    annotation_dir = output / "annotations"
    review_dir = output / "review"
    sheet_dir = review_dir / "sheets"
    for directory in (image_dir, annotation_dir, review_dir, sheet_dir):
        directory.mkdir(parents=True, exist_ok=True)

    package = load_model_package(args.package_dir)
    draft_detector, classifier, provider = build_onnx_adapters(package, args.provider)
    labels = [label.model_dump() for label in package.metadata.classifier.labels]
    label_ids = {label["class_id"]: index for index, label in enumerate(labels, start=1)}
    categories = [
        {"id": index, "name": label["class_name"], "class_id": label["class_id"]}
        for index, label in enumerate(labels, start=1)
    ]

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    review_paths: list[Path] = []
    review_index: list[dict[str, Any]] = []
    annotation_id = 1
    for sequence, source in enumerate(selected, start=1):
        with Image.open(source.image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        file_name = f"{sequence:03d}_{source.scan_id}.jpg"
        exported_path = image_dir / file_name
        image.save(exported_path, quality=96, optimize=True)
        logged = _logged_annotations(source.payload, label_ids)
        annotations = logged or _model_annotations(
            image,
            draft_detector,
            classifier,
            package.metadata.classifier.temperature,
        )
        if source.scan_id in clear_annotations:
            annotations = []
        if source.scan_id in confirmed_recapture:
            annotations = []
        for addition in add_annotations.get(source.scan_id, []):
            class_id = addition["class_id"]
            if class_id not in label_ids:
                raise ValueError(f"unknown correction class id: {class_id}")
            x, y, width, height = [float(value) for value in addition["bbox"]]
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(f"invalid correction bbox: {source.scan_id}")
            if x + width > image.width or y + height > image.height:
                raise ValueError(f"correction bbox is outside image: {source.scan_id}")
            annotations.append(
                {
                    "category_id": label_ids[class_id],
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "source": "assistant_manual_review",
                    "review_status": "pending_user_review",
                    "detector_score": None,
                    "classifier_confidence": float(addition.get("confidence", 1.0)),
                    "user_modified": True,
                }
            )
        image_record = {
            "id": sequence,
            "file_name": file_name,
            "width": image.width,
            "height": image.height,
            "scan_id": source.scan_id,
            "recorded_at": source.recorded_at.isoformat(),
            "worker_status": source.payload.get("worker_status"),
            "reason_codes": source.payload.get("reason_codes") or [],
            "review_decision": (
                "confirmed_recapture" if source.scan_id in confirmed_recapture else "pending"
            ),
            "expected_status": ("RECAPTURE" if source.scan_id in confirmed_recapture else None),
            "expected_reason_codes": (
                confirmed_recapture[source.scan_id].get("reason_codes", [])
                if source.scan_id in confirmed_recapture
                else []
            ),
            "exclude_from_detector_training": source.scan_id in confirmed_recapture,
            "source_image_sha256": sha256_file(source.image_path),
            "exported_image_sha256": sha256_file(exported_path),
        }
        coco_images.append(image_record)
        row_annotations: list[dict[str, Any]] = []
        for annotation in annotations:
            annotation = dict(annotation)
            annotation.update({"id": annotation_id, "image_id": sequence})
            annotation["bbox"] = [round(float(value), 2) for value in annotation["bbox"]]
            annotation["area"] = round(float(annotation["area"]), 2)
            if annotation["detector_score"] is not None:
                annotation["detector_score"] = round(float(annotation["detector_score"]), 6)
            annotation["classifier_confidence"] = round(
                float(annotation["classifier_confidence"]), 6
            )
            coco_annotations.append(annotation)
            row_annotations.append(
                {
                    "annotation_id": annotation_id,
                    "category_id": annotation["category_id"],
                    "bbox_xywh": annotation["bbox"],
                    "area": annotation["area"],
                    "iscrowd": 0,
                    "source": annotation["source"],
                    "review_status": annotation["review_status"],
                    "draft_detector_score": annotation["detector_score"],
                    "draft_classifier_confidence": annotation["classifier_confidence"],
                }
            )
            annotation_id += 1
        manifest_rows.append(
            {
                "record_type": "detection",
                "source": "product_scanner_scan_log",
                "image_id": sequence,
                "scan_id": source.scan_id,
                "image_path": f"images/{file_name}",
                "image_sha256": image_record["exported_image_sha256"],
                "width": image.width,
                "height": image.height,
                "capture_time": source.recorded_at.isoformat(),
                "capture_session_id": source.payload.get("capture_session_id"),
                "worker_status": source.payload.get("worker_status"),
                "observed_reason_codes": source.payload.get("reason_codes") or [],
                "annotation_review_status": "pending_user_review",
                "assistant_reviewed": True,
                "review_decision": (
                    "confirmed_recapture" if source.scan_id in confirmed_recapture else "pending"
                ),
                "expected_status": ("RECAPTURE" if source.scan_id in confirmed_recapture else None),
                "expected_reason_codes": (
                    confirmed_recapture[source.scan_id].get("reason_codes", [])
                    if source.scan_id in confirmed_recapture
                    else []
                ),
                "exclude_from_detector_training": source.scan_id in confirmed_recapture,
                "review_attention": attention.get(source.scan_id),
                "annotations": row_annotations,
            }
        )
        review_path = review_dir / f"{sequence:03d}_{source.scan_id}_review.jpg"
        _render_review(
            image,
            sequence,
            (
                "RECAPTURE (CONFIRMED)"
                if source.scan_id in confirmed_recapture
                else str(source.payload.get("worker_status") or "")
            ),
            (
                confirmed_recapture[source.scan_id].get("reason_codes", [])
                if source.scan_id in confirmed_recapture
                else source.payload.get("reason_codes") or []
            ),
            annotations,
            labels,
            review_path,
        )
        review_paths.append(review_path)
        review_index.append(
            {
                "sequence": sequence,
                "scan_id": source.scan_id,
                "worker_status": source.payload.get("worker_status") or "LEGACY",
                "reason_codes": "|".join(source.payload.get("reason_codes") or []),
                "annotation_count": len(annotations),
                "annotation_source": (
                    "none_confirmed_recapture"
                    if source.scan_id in confirmed_recapture
                    else "scan_log_final"
                    if logged
                    else "model_draft_detector"
                ),
                "assistant_action": (
                    "confirmed_recapture_no_annotations"
                    if source.scan_id in confirmed_recapture
                    else "cleared_false_positive"
                    if source.scan_id in clear_annotations
                    else "manual_box_added"
                    if source.scan_id in add_annotations
                    else "visually_checked"
                ),
                "attention": attention.get(source.scan_id, ""),
                "review_decision": (
                    "confirmed_recapture" if source.scan_id in confirmed_recapture else "pending"
                ),
                "review_file": review_path.name,
            }
        )

    coco = {
        "info": {
            "description": "Product Scanner latest logs; draft pending user review",
            "annotation_review_status": "pending_user_review",
            "selected_count": len(selected),
            "selection": "latest_by_recorded_at",
            "package_version": package.metadata.package_version,
            "provider": provider,
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    coco_path = annotation_dir / "instances_draft.json"
    coco_text = json.dumps(coco, ensure_ascii=False, indent=2) + "\n"
    coco_path.write_text(coco_text, encoding="utf-8")
    manifest_path = output / "manifest_draft.jsonl"
    manifest_text = (
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            for row in manifest_rows
        )
        + "\n"
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")
    with (review_dir / "review_index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_index[0]))
        writer.writeheader()
        writer.writerows(review_index)
    _contact_sheets(review_paths, sheet_dir)
    recapture_paths = [
        review_paths[index]
        for index, source in enumerate(selected)
        if source.scan_id in confirmed_recapture
    ]
    recapture_sheet_dir = review_dir / "confirmed_recapture_sheets"
    recapture_sheet_dir.mkdir(parents=True, exist_ok=True)
    _contact_sheets(recapture_paths, recapture_sheet_dir)
    metadata = {
        "schema_version": "1.0",
        "annotation_review_status": "pending_user_review",
        "selected_count": len(selected),
        "selection": "latest_by_recorded_at",
        "selection_start_utc": selected[0].recorded_at.isoformat(),
        "selection_end_utc": selected[-1].recorded_at.isoformat(),
        "image_count": len(coco_images),
        "annotation_count": len(coco_annotations),
        "zero_annotation_image_count": sum(
            not any(annotation["image_id"] == image["id"] for annotation in coco_annotations)
            for image in coco_images
        ),
        "logged_annotation_image_count": sum(
            row["annotation_source"] == "scan_log_final" for row in review_index
        ),
        "model_draft_image_count": sum(
            row["annotation_source"] == "model_draft_detector" for row in review_index
        ),
        "assistant_reviewed_image_count": len(selected),
        "assistant_cleared_false_positive_count": len(clear_annotations),
        "assistant_manual_box_image_count": len(add_annotations),
        "user_attention_image_count": len(attention),
        "user_confirmed_recapture_count": len(confirmed_recapture),
        "package_version": package.metadata.package_version,
        "detector_version": package.metadata.detector.version,
        "classifier_version": package.metadata.classifier.version,
        "provider": provider,
        "coco_sha256": hashlib.sha256(coco_text.encode("utf-8")).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export recent Product Scanner logs for visual review"
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--corrections", type=Path)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
