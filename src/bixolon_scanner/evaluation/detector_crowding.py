from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..configuration import load_json_config
from ..contracts.artifact import directory_content_manifest
from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2
from ..pipeline.ports import Detector
from ..runtime.detector_v2 import build_detector_v2
from ..runtime.imaging import decode_image
from ..runtime.onnx_session import ExecutionProvider


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(row)
    return rows


def _crowding_flags(
    detector: Detector,
    rows: list[dict[str, Any]],
    *,
    image_root: Path,
    jpeg_draft_size: int | None,
) -> list[bool]:
    flags: list[bool] = []
    for row in rows:
        image_path = image_root / str(row["image_path"])
        if sha256_file(image_path) != row["image_sha256"]:
            raise ValueError(f"image checksum mismatch: {image_path}")
        image = decode_image(
            image_path.read_bytes(),
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=jpeg_draft_size,
        )
        try:
            result = detector.detect(image)
        finally:
            image.close()
        flags.append(result.uncertain_candidate_count > 0)
    return flags


def evaluate_detector_crowding(
    *,
    runtime_dir: Path,
    operational_manifest: Path,
    operational_review: Path,
    validation_manifest: Path,
    validation_image_root: Path,
    provider: ExecutionProvider = "cpu",
    cuda_dll_dir: Path | None = None,
) -> dict[str, Any]:
    runtime = load_runtime_package_v2(runtime_dir)
    if runtime.metadata.detector_crowding is None:
        raise ValueError("runtime does not configure detector crowding")
    if runtime.metadata.detector.uncertainty_score_threshold is not None:
        raise ValueError("crowding evaluation requires the shadow uncertainty gate to be disabled")
    detector = build_detector_v2(runtime, provider, cuda_dll_dir)

    operational_rows = _read_jsonl(operational_manifest)
    review = load_json_config(operational_review)
    reviewed_scan_ids = {
        scan_id
        for scan_id, decision in review.get("confirmed_recapture", {}).items()
        if "DETECTOR_UNCERTAIN_OBJECT" in decision.get("reason_codes", [])
    }
    confirmed_segmentation_scan_ids = set(review.get("confirmed_segmentation", {}))
    if reviewed_scan_ids & confirmed_segmentation_scan_ids:
        raise ValueError("review config cannot confirm both recapture and segmentation")
    operational_flags = _crowding_flags(
        detector,
        operational_rows,
        image_root=operational_manifest.parent,
        jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
    )
    operational_scan_ids = {str(row["scan_id"]) for row in operational_rows}
    if not (reviewed_scan_ids | confirmed_segmentation_scan_ids) <= operational_scan_ids:
        raise ValueError("review config references scans outside the operational manifest")
    expected_flags = [str(row["scan_id"]) in reviewed_scan_ids for row in operational_rows]
    confirmed_segmentation_flags = [
        str(row["scan_id"]) in confirmed_segmentation_scan_ids for row in operational_rows
    ]
    caught_rows = [
        row
        for row, flagged, expected in zip(
            operational_rows, operational_flags, expected_flags, strict=True
        )
        if flagged and expected
    ]
    false_rows = [
        row
        for row, flagged, confirmed in zip(
            operational_rows,
            operational_flags,
            confirmed_segmentation_flags,
            strict=True,
        )
        if flagged and confirmed
    ]
    unreviewed_rows = [
        row
        for row, flagged, expected, confirmed in zip(
            operational_rows,
            operational_flags,
            expected_flags,
            confirmed_segmentation_flags,
            strict=True,
        )
        if flagged and not expected and not confirmed
    ]
    target_count = sum(expected_flags)
    confirmed_segmentation_count = sum(confirmed_segmentation_flags)
    unreviewed_count = len(operational_rows) - target_count - confirmed_segmentation_count

    validation_rows = _read_jsonl(validation_manifest)
    if any(row.get("expected_image_status") != "ANNOTATED" for row in validation_rows):
        raise ValueError("validation regression set must contain accepted annotated scenes only")
    validation_flags = _crowding_flags(
        detector,
        validation_rows,
        image_root=validation_image_root,
        jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
    )
    validation_false_rows = [
        row for row, flagged in zip(validation_rows, validation_flags, strict=True) if flagged
    ]

    provided_sequences = {50, 52, 55, 56, 57}
    provided_flags = [
        flagged
        for row, flagged in zip(operational_rows, operational_flags, strict=True)
        if int(row["image_id"]) in provided_sequences
    ]
    if len(provided_flags) != len(provided_sequences):
        raise ValueError("operational manifest does not contain all provided examples")

    return {
        "schema_version": "1.0",
        "product_version": runtime.metadata.worker_version,
        "provider": provider,
        "runtime": {
            "path": runtime.root.as_posix(),
            "manifest_sha256": directory_content_manifest(runtime.root)["manifest_sha256"],
            "detector_onnx_sha256": sha256_file(runtime.detector_path),
            "detector_crowding": runtime.metadata.detector_crowding.model_dump(mode="json"),
        },
        "provided_examples": {
            "sequence_ids": sorted(provided_sequences),
            "sample_count": len(provided_flags),
            "recapture_count": sum(provided_flags),
        },
        "operational_review": {
            "manifest": operational_manifest.resolve().as_posix(),
            "manifest_sha256": sha256_file(operational_manifest),
            "review_config": operational_review.resolve().as_posix(),
            "review_config_sha256": sha256_file(operational_review),
            "review_config_confirmed_count": len(reviewed_scan_ids),
            "sample_count": len(operational_rows),
            "confirmed_uncertain_object_count": target_count,
            "caught_count": len(caught_rows),
            "caught_sequence_ids": sorted(int(row["image_id"]) for row in caught_rows),
            "recall": len(caught_rows) / target_count if target_count else None,
            "confirmed_segmentation_count": confirmed_segmentation_count,
            "false_recapture_count": len(false_rows),
            "false_recapture_sequence_ids": sorted(int(row["image_id"]) for row in false_rows),
            "false_recapture_rate": (
                len(false_rows) / confirmed_segmentation_count
                if confirmed_segmentation_count
                else None
            ),
            "unreviewed_count": unreviewed_count,
            "unreviewed_recapture_count": len(unreviewed_rows),
            "unreviewed_recapture_sequence_ids": sorted(
                int(row["image_id"]) for row in unreviewed_rows
            ),
        },
        "accepted_detector_regression": {
            "manifest": validation_manifest.resolve().as_posix(),
            "manifest_sha256": sha256_file(validation_manifest),
            "sample_count": len(validation_rows),
            "false_recapture_count": len(validation_false_rows),
            "false_recapture_image_ids": sorted(
                int(row["image_id"]) for row in validation_false_rows
            ),
            "false_recapture_rate": (
                len(validation_false_rows) / len(validation_rows) if validation_rows else None
            ),
        },
        "limitations": [
            "운영 표본과 detector 검증 표본은 독립 일반화 성능을 입증하지 않는다.",
            "운영 표본의 미확정 행은 false-recapture 분모에서 제외한다.",
            "정책은 raw query 기하와 회전 입력의 객체 수 복구를 사용하며 물리적 정답 수를 직접 추정하지 않는다.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the detector crowding policy")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--operational-manifest", type=Path, required=True)
    parser.add_argument("--operational-review", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-image-root", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda", "openvino"), default="cpu")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_detector_crowding(
        runtime_dir=args.runtime,
        operational_manifest=args.operational_manifest,
        operational_review=args.operational_review,
        validation_manifest=args.validation_manifest,
        validation_image_root=args.validation_image_root,
        provider=args.provider,
        cuda_dll_dir=args.cuda_dll_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
