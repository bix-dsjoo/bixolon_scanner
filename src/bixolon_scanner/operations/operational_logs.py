from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..contracts.model_package import load_model_package, sha256_file
from ..pipeline.ports import Detection
from ..runtime.imaging import decode_image
from ..runtime.onnx import OnnxDetector, build_onnx_adapters
from ..training.data import read_manifest
from ..training.manifest import _assign_folds, _canonical_json, _manifest_version

SCHEMA_VERSION = "1.0"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _validated_contract(path: Path) -> dict[str, Any]:
    contract = _load_json(path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported operational decision schema")
    ordered_ids = contract.get("ordered_scan_ids")
    if not isinstance(ordered_ids, list) or not ordered_ids:
        raise ValueError("ordered_scan_ids must be a non-empty list")
    if any(not isinstance(value, str) or len(value) != 32 for value in ordered_ids):
        raise ValueError("every scan id must be a 32-character identifier")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("ordered_scan_ids contains duplicates")
    expected_count = int(contract.get("expected_log_count", -1))
    if expected_count != len(ordered_ids):
        raise ValueError("expected_log_count does not match ordered_scan_ids")
    empty_ids = set(contract.get("empty_recapture_scan_ids", []))
    blur_ids = set(contract.get("blur_recapture_scan_ids", []))
    if empty_ids & blur_ids:
        raise ValueError("empty and blur decisions overlap")
    if not empty_ids | blur_ids <= set(ordered_ids):
        raise ValueError("recapture decision references an unexpected scan id")
    for field in ("capture_session_id", "physical_target_group_id", "camera"):
        if not isinstance(contract.get(field), str) or not contract[field]:
            raise ValueError(f"{field} is required")
    return contract


def _load_scan_logs(log_dirs: list[Path], contract: dict[str, Any]) -> list[dict[str, Any]]:
    expected = set(contract["ordered_scan_ids"])
    discovered: dict[str, dict[str, Any]] = {}
    for root in log_dirs:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.glob("*.json")):
            raw = _load_json(path)
            scan_id = raw.get("scan_id")
            if scan_id not in expected:
                continue
            if scan_id in discovered:
                raise ValueError(f"duplicate scan id across log directories: {scan_id}")
            if raw.get("log_schema_version") != 2:
                raise ValueError(f"scan log {scan_id} is not schema v2")
            if raw.get("worker_status") != "RECAPTURE":
                raise ValueError(f"scan log {scan_id} is not a RECAPTURE record")
            reasons = raw.get("reason_codes")
            if not isinstance(reasons, list) or "DETECTOR_UNCERTAIN_OBJECT" not in reasons:
                raise ValueError(f"scan log {scan_id} is not an uncertain-object record")
            stored_name = raw.get("original_image")
            if (
                not isinstance(stored_name, str)
                or not stored_name
                or Path(stored_name).name != stored_name
                or Path(stored_name).is_absolute()
            ):
                raise ValueError(f"scan log {scan_id} has an unsafe image name")
            image_path = root / stored_name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            try:
                with Image.open(image_path) as source:
                    source.verify()
                with Image.open(image_path) as source:
                    width, height = ImageOps.exif_transpose(source).size
            except Exception as exc:
                raise ValueError(f"scan log {scan_id} has a corrupt image") from exc
            discovered[scan_id] = {
                "log": raw,
                "log_path": path,
                "image_path": image_path,
                "width": int(width),
                "height": int(height),
                "image_sha256": sha256_file(image_path),
            }
    missing = expected - set(discovered)
    if missing:
        raise ValueError(f"missing operational logs: {sorted(missing)}")
    ordered = [discovered[scan_id] for scan_id in contract["ordered_scan_ids"]]
    hashes = [record["image_sha256"] for record in ordered]
    duplicates = [value for value, count in Counter(hashes).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate operational images: {duplicates}")
    times = [record["log"].get("recorded_at") for record in ordered]
    if times != sorted(times):
        raise ValueError("ordered_scan_ids is not in recorded_at order")
    return ordered


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = logits.astype(np.float64) / temperature
    values -= values.max(axis=1, keepdims=True)
    exponential = np.exp(values)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _draft_annotations(
    image: Image.Image,
    detections: list[Detection],
    classifier,
    package,
    annotation_id: int,
) -> tuple[list[dict[str, Any]], int]:
    ordered = sorted(detections, key=lambda item: (item.y1, item.x1))
    if not ordered:
        return [], annotation_id
    logits = classifier.classify(image, ordered)
    probabilities = _softmax(logits, package.metadata.classifier.temperature)
    annotations: list[dict[str, Any]] = []
    for detection, scores in zip(ordered, probabilities):
        class_index = int(np.argmax(scores))
        width = detection.x2 - detection.x1
        height = detection.y2 - detection.y1
        annotations.append(
            {
                "annotation_id": annotation_id,
                "category_id": class_index + 1,
                "bbox_xywh": [
                    round(detection.x1, 2),
                    round(detection.y1, 2),
                    round(width, 2),
                    round(height, 2),
                ],
                "area": round(width * height, 2),
                "iscrowd": 0,
                "draft_confidence": round(float(scores[class_index]), 6),
                "draft_detector_score": round(detection.score, 6),
            }
        )
        annotation_id += 1
    return annotations, annotation_id


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _box_iou(left: Detection, right: Detection) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _render_review(
    image_path: Path,
    record: dict[str, Any],
    labels: list[dict[str, Any]],
    shadow: list[Detection],
    output_path: Path,
) -> None:
    with Image.open(image_path) as source:
        canvas = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line_width = max(5, round(min(canvas.size) / 300))
    font = _font(max(24, round(min(canvas.size) / 75)))
    expected = record["expected_detector_action"]
    header_color = (38, 180, 90) if expected == "CONTINUE" else (220, 55, 65)
    header = f"{record['sequence_index']:02d} {record['scan_id'][:8]} {expected}"
    if record["expected_reason_codes"]:
        header += " " + "|".join(record["expected_reason_codes"])
    header_box = draw.textbbox((16, 12), header, font=font, stroke_width=1)
    draw.rectangle((0, 0, canvas.width, header_box[3] + 24), fill=header_color)
    draw.text((16, 12), header, fill="white", font=font, stroke_width=1, stroke_fill=(0, 0, 0))

    accepted_boxes: list[Detection] = []
    for annotation in record["annotations"]:
        x, y, width, height = annotation["bbox_xywh"]
        accepted_boxes.append(
            Detection(x, y, x + width, y + height, annotation["draft_detector_score"])
        )
        color = (0, 170, 255)
        draw.rectangle((x, y, x + width, y + height), outline=color, width=line_width)
        label = labels[annotation["category_id"] - 1]["class_id"]
        text = (
            f"GT-DRAFT {label} det={annotation['draft_detector_score']:.3f} "
            f"cls={annotation['draft_confidence']:.3f}"
        )
        text_y = max(header_box[3] + 28, y - font.size - 10)
        draw.text((x, text_y), text, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    for detection in shadow:
        if any(_box_iou(detection, accepted) >= 0.5 for accepted in accepted_boxes):
            continue
        color = (255, 170, 0)
        draw.rectangle(
            (detection.x1, detection.y1, detection.x2, detection.y2),
            outline=color,
            width=line_width,
        )
        draw.text(
            (detection.x1, max(header_box[3] + 28, detection.y1 - font.size - 10)),
            f"SHADOW score={detection.score:.3f}",
            fill=color,
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, optimize=True)


def _contact_sheets(paths: list[Path], output_dir: Path) -> None:
    width, height = 720, 540
    margin = 20
    for start in range(0, len(paths), 6):
        selected = paths[start : start + 6]
        sheet = Image.new("RGB", (width * 2 + margin * 3, height * 3 + margin * 4), (22, 22, 22))
        for offset, path in enumerate(selected):
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((width, height), Image.Resampling.LANCZOS)
            column, row = offset % 2, offset // 2
            x = margin + column * (width + margin)
            y = margin + row * (height + margin)
            sheet.paste(image, (x + (width - image.width) // 2, y + (height - image.height) // 2))
        sheet.save(
            output_dir / f"contact_sheet_{start // 6 + 1:02d}.jpg", quality=90, optimize=True
        )


def ingest(args: argparse.Namespace) -> dict[str, Any]:
    contract = _validated_contract(args.decisions)
    logs = _load_scan_logs(args.log_dir, contract)
    package = load_model_package(args.package_dir)
    detector, classifier, provider = build_onnx_adapters(
        package, args.provider, cuda_dll_dir=args.cuda_dll_dir
    )
    shadow_metadata = package.metadata.detector.model_copy(
        update={
            "score_threshold": package.metadata.detector.uncertainty_score_threshold,
            "uncertainty_score_threshold": None,
            "uncertainty_min_area_ratio": 0.0,
        }
    )
    shadow_detector = OnnxDetector(
        package.detector_path, shadow_metadata, provider, args.cuda_dll_dir
    )
    shadow_detector.warmup()

    combined_root = args.dataset_root.resolve()
    base_root = args.base_dataset_root.resolve()
    try:
        base_prefix = base_root.relative_to(combined_root)
    except ValueError as exc:
        raise ValueError("base dataset root must be inside the combined dataset root") from exc
    operational_root = combined_root / args.operational_subdir
    image_dir = operational_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)

    base_records = [
        copy.deepcopy(record)
        for record in read_manifest(args.base_manifest)
        if record["record_type"] == "detection"
    ]
    for record in base_records:
        record["image_path"] = (base_prefix / record["image_path"]).as_posix()
    next_image_id = max(int(record["image_id"]) for record in base_records) + 1
    next_annotation_id = (
        max(
            int(annotation["annotation_id"])
            for record in base_records
            for annotation in record["annotations"]
        )
        + 1
    )
    labels = _load_json(args.base_metadata)["labels"]
    empty_ids = set(contract["empty_recapture_scan_ids"])
    blur_ids = set(contract["blur_recapture_scan_ids"])
    operational_records: list[dict[str, Any]] = []
    review_paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    for index, source_record in enumerate(logs):
        scan_id = source_record["log"]["scan_id"]
        suffix = source_record["image_path"].suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"unsupported stored image suffix: {suffix}")
        target_name = f"{scan_id}{'.png' if suffix == '.png' else '.jpg'}"
        target_path = image_dir / target_name
        shutil.copy2(source_record["image_path"], target_path)
        if sha256_file(target_path) != source_record["image_sha256"]:
            raise RuntimeError(f"copied image checksum mismatch: {scan_id}")
        image = decode_image(
            target_path.read_bytes(),
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        result = detector.detect(image)
        shadow = shadow_detector.detect(image).detections
        if scan_id in empty_ids:
            expected_action = "RECAPTURE"
            expected_reasons = ["DETECTOR_NO_OBJECT"]
            role = "detector_hard_negative"
            annotations: list[dict[str, Any]] = []
        else:
            annotations, next_annotation_id = _draft_annotations(
                image, result.detections, classifier, package, next_annotation_id
            )
            if not annotations:
                raise ValueError(f"non-empty decision has no accepted detector boxes: {scan_id}")
            if scan_id in blur_ids:
                expected_action = "RECAPTURE"
                expected_reasons = ["DETECTOR_BLUR"]
                role = "quality_regression"
            else:
                expected_action = "CONTINUE"
                expected_reasons = []
                role = "detector_positive"
        record = {
            "record_type": "detection",
            "source": "operational_scan_log_v2",
            "image_path": (Path(args.operational_subdir) / "images" / target_name).as_posix(),
            "image_sha256": source_record["image_sha256"],
            "image_id": next_image_id + index,
            "scan_id": scan_id,
            "sequence_index": index,
            "width": source_record["width"],
            "height": source_record["height"],
            "capture_time": source_record["log"]["recorded_at"],
            "capture_session_id": contract["capture_session_id"],
            "physical_target_group_id": contract["physical_target_group_id"],
            "camera": contract["camera"],
            "split": "development",
            "fold": None,
            "training_role": role,
            "exclude_from_detector_training": role == "quality_regression",
            "expected_detector_action": expected_action,
            "expected_reason_codes": expected_reasons,
            "observed_reason_codes": source_record["log"]["reason_codes"],
            "source_model_versions": source_record["log"]["model_versions"],
            "annotations": annotations,
        }
        operational_records.append(record)
        review_path = args.review_dir / f"{index:02d}_{scan_id}_review.jpg"
        _render_review(target_path, record, labels, shadow, review_path)
        review_paths.append(review_path)
        rows.append(
            {
                "sequence_index": index,
                "scan_id": scan_id,
                "expected_detector_action": expected_action,
                "expected_reason_codes": "|".join(expected_reasons),
                "training_role": role,
                "annotation_count": len(annotations),
                "annotation_summary": " | ".join(
                    (
                        f"{labels[annotation['category_id'] - 1]['class_id']} "
                        f"({labels[annotation['category_id'] - 1]['class_name']}); "
                        f"bbox={','.join(str(round(value, 2)) for value in annotation['bbox_xywh'])}; "
                        f"det={annotation['draft_detector_score']:.4f}; "
                        f"cls={annotation['draft_confidence']:.4f}"
                    )
                    for annotation in annotations
                ),
                "image_sha256": source_record["image_sha256"],
            }
        )

    merged = base_records + operational_records
    _assign_folds(merged, int(args.fold_count))
    lines = [_canonical_json(record) for record in merged]
    dataset_version = _manifest_version(lines)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_dir / "manifest.jsonl"
    manifest_text = "\n".join(lines) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    metadata = {
        "schema_version": "1.1",
        "dataset_version": dataset_version,
        "base_dataset_version": _load_json(args.base_metadata)["dataset_version"],
        "fold_count": int(args.fold_count),
        "record_count": len(merged),
        "detection_record_count": len(merged),
        "operational_record_count": len(operational_records),
        "operational_capture_session_count": 1,
        "operational_decisions": {
            "continue_count": sum(
                record["expected_detector_action"] == "CONTINUE" for record in operational_records
            ),
            "empty_recapture_count": len(empty_ids),
            "blur_recapture_count": len(blur_ids),
            "promotion_evidence": False,
            "note": "Single-session fit data; never use as production promotion evidence.",
        },
        "labels": labels,
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "decision_contract_sha256": sha256_file(args.decisions),
        "draft_label_package": {
            "package_version": package.metadata.package_version,
            "detector_version": package.metadata.detector.version,
            "classifier_version": package.metadata.classifier.version,
            "detector_sha256": sha256_file(package.detector_path),
            "classifier_sha256": sha256_file(package.classifier_path),
            "provider": provider,
        },
        "annotation_review_status": args.annotation_review_status,
    }
    (args.manifest_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.review_dir / "review_index.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _contact_sheets(review_paths, args.review_dir)
    report = {
        "dataset_version": dataset_version,
        "manifest": str(manifest_path),
        "combined_dataset_root": str(combined_root),
        "operational_image_root": str(image_dir),
        "review_dir": str(args.review_dir),
        "record_count": len(merged),
        "operational_record_count": len(operational_records),
        "continue_count": metadata["operational_decisions"]["continue_count"],
        "recapture_count": len(empty_ids) + len(blur_ids),
        "annotation_review_status": args.annotation_review_status,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest reviewed Scan Log v2 records into an external detector dataset"
    )
    parser.add_argument("--log-dir", type=Path, action="append", required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--base-dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--operational-subdir", type=Path, default=Path("bixolon_operational_0.1.2")
    )
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument(
        "--provider",
        choices=("auto", "cuda", "cpu"),
        default="cpu",
        help="CPU is the default so model-assisted draft annotations are reproducible.",
    )
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument(
        "--annotation-review-status",
        choices=("draft", "approved"),
        default="draft",
    )
    ingest(parser.parse_args())


if __name__ == "__main__":
    main()
