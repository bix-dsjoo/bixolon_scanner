from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from ...training.data import read_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detection_row(image_id: int, detections: Any) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "boxes_xyxy": detections.xyxy.astype(float).tolist(),
        "scores": detections.confidence.astype(float).tolist(),
        "class_ids": detections.class_id.astype(int).tolist(),
    }


def extract_predictions(
    manifest_path: Path,
    dataset_root: Path,
    checkpoint: Path,
    output: Path,
    *,
    fold: int,
    resolution: int,
    threshold: float,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    from rfdetr import RFDETRLarge

    records = [
        row
        for row in read_manifest(manifest_path)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) == fold
    ]
    if not records:
        raise ValueError(f"RF-DETR prediction fold {fold} has no records")
    model = RFDETRLarge(
        pretrain_weights=str(checkpoint),
        num_classes=20,
        device=device,
        resolution=resolution,
    )
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        image_paths = [str(dataset_root / row["image_path"]) for row in batch_records]
        batch_detections = model.predict(
            image_paths,
            threshold=threshold,
            shape=(resolution, resolution),
            include_source_image=False,
        )
        if not isinstance(batch_detections, list):
            batch_detections = [batch_detections]
        if len(batch_detections) != len(batch_records):
            raise RuntimeError("RF-DETR prediction batch size mismatch")
        rows.extend(
            detection_row(int(record["image_id"]), detections)
            for record, detections in zip(batch_records, batch_detections)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "rfdetr_bread_fold_raw_predictions",
        "fold": fold,
        "record_count": len(records),
        "annotated_record_count": sum(
            row.get("expected_image_status") == "ANNOTATED" for row in records
        ),
        "recapture_record_count": sum(
            row.get("expected_image_status") == "RECAPTURE" for row in records
        ),
        "prediction_count": sum(len(row["scores"]) for row in rows),
        "threshold": threshold,
        "resolution": resolution,
        "batch_size": batch_size,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "output": str(output),
        "output_sha256": _sha256(output),
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Extract raw RF-DETR bread predictions")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=704)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = extract_predictions(
        args.manifest,
        args.dataset_root,
        args.checkpoint,
        args.output,
        fold=args.fold,
        resolution=args.resolution,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
