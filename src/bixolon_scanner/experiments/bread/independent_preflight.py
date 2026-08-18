from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ...contracts.model_package import sha256_file
from ...training.bread_cv import difference_hash, hamming_distance

_FINAL_REVIEW_STATUSES = {"approved", "finalized", "locked", "user_review_complete"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _resolve_image(dataset_root: Path, annotation_path: Path, file_name: str) -> Path:
    candidates = (
        dataset_root / file_name,
        dataset_root / "images" / file_name,
        annotation_path.parent / file_name,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(dataset_root) and resolved.is_file():
            return resolved
    raise FileNotFoundError(file_name)


def _structural_audit(payload: dict[str, Any]) -> dict[str, int]:
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = {int(row["id"]) for row in payload.get("categories", [])}
    image_by_id = {int(row["id"]): row for row in images}
    annotation_ids = [int(row["id"]) for row in annotations]
    orphan_count = 0
    invalid_bbox_count = 0
    unknown_category_count = 0
    for annotation in annotations:
        image = image_by_id.get(int(annotation["image_id"]))
        if image is None:
            orphan_count += 1
            continue
        bbox = [float(value) for value in annotation["bbox"]]
        if len(bbox) != 4:
            invalid_bbox_count += 1
            continue
        x, y, width, height = bbox
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > float(image["width"]) + 1e-6
            or y + height > float(image["height"]) + 1e-6
        ):
            invalid_bbox_count += 1
        if categories and int(annotation["category_id"]) not in categories:
            unknown_category_count += 1
    return {
        "image_count": len(images),
        "annotation_count": len(annotations),
        "category_count": len(categories),
        "duplicate_image_id_count": len(images) - len(image_by_id),
        "duplicate_annotation_id_count": len(annotation_ids) - len(set(annotation_ids)),
        "orphan_annotation_count": orphan_count,
        "invalid_bbox_count": invalid_bbox_count,
        "unknown_category_count": unknown_category_count,
    }


def audit_independent_dataset(
    *,
    dataset_root: Path,
    annotation_path: Path,
    metadata_path: Path,
    record_manifest_path: Path,
    source_manifest_paths: list[Path],
    dataset_version: str,
    candidate_id: str,
    candidate_commit: str,
    maximum_hamming_distance: int = 2,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    annotation_path = annotation_path.resolve()
    metadata_path = metadata_path.resolve()
    record_manifest_path = record_manifest_path.resolve()
    if maximum_hamming_distance < 0:
        raise ValueError("maximum Hamming distance must be non-negative")
    if not re.fullmatch(r"[0-9a-f]{7,40}", candidate_commit):
        raise ValueError("candidate commit must be a lowercase Git object id")
    payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = _read_jsonl(record_manifest_path)
    record_by_id = {int(row["image_id"]): row for row in records}
    source_rows = [row for path in source_manifest_paths for row in _read_jsonl(path.resolve())]
    known_by_sha: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    known_perceptual: dict[tuple[str, int], dict[str, Any]] = {}
    for row in source_rows:
        image_sha = str(row["image_sha256"]).lower()
        known_by_sha[image_sha].append(row)
        if row.get("perceptual_hash") is not None:
            known_perceptual[(image_sha, int(row["perceptual_hash"]))] = row

    exact_overlap_image_ids: list[int] = []
    near_overlap_image_ids: list[int] = []
    overlap_examples: list[dict[str, Any]] = []
    minimum_cross_set_hamming_distance = 64
    missing_record_count = 0
    pending_record_review_count = 0
    missing_capture_session_count = 0
    image_identity_mismatch_count = 0
    image_dimension_mismatch_count = 0
    image_manifest_lines: list[str] = []
    for image in sorted(payload["images"], key=lambda row: int(row["id"])):
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        path = _resolve_image(dataset_root, annotation_path, file_name)
        image_sha = sha256_file(path).lower()
        declared_sha = image.get("exported_image_sha256") or image.get("image_sha256")
        image_identity_mismatch_count += bool(
            declared_sha is not None and str(declared_sha).lower() != image_sha
        )
        with Image.open(path) as source:
            image_dimension_mismatch_count += ImageOps.exif_transpose(source).size != (
                int(image["width"]),
                int(image["height"]),
            )
        perceptual_hash = difference_hash(path)
        image_manifest_lines.append(f"{image_id:06d} {image_sha} {file_name}\n")
        exact_rows = known_by_sha.get(image_sha, [])
        if exact_rows:
            exact_overlap_image_ids.append(image_id)
            if len(overlap_examples) < 20:
                overlap_examples.append(
                    {
                        "image_id": image_id,
                        "file_name": file_name,
                        "overlap_type": "exact_sha256",
                        "source_image_path": exact_rows[0].get("image_path"),
                        "source_evaluation_set": exact_rows[0].get("evaluation_set"),
                    }
                )
        nearest: tuple[int, dict[str, Any]] | None = None
        for (_, known_hash), known_row in known_perceptual.items():
            distance = hamming_distance(perceptual_hash, known_hash)
            minimum_cross_set_hamming_distance = min(minimum_cross_set_hamming_distance, distance)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, known_row)
        if not exact_rows and nearest is not None and nearest[0] <= maximum_hamming_distance:
            near_overlap_image_ids.append(image_id)
            if len(overlap_examples) < 20:
                overlap_examples.append(
                    {
                        "image_id": image_id,
                        "file_name": file_name,
                        "overlap_type": "difference_hash",
                        "hamming_distance": nearest[0],
                        "source_image_path": nearest[1].get("image_path"),
                        "source_evaluation_set": nearest[1].get("evaluation_set"),
                    }
                )
        record = record_by_id.get(image_id)
        if record is None:
            missing_record_count += 1
        else:
            pending_record_review_count += (
                str(record.get("annotation_review_status", "")).lower()
                not in _FINAL_REVIEW_STATUSES
            )
            missing_capture_session_count += not bool(record.get("capture_session_id"))

    image_manifest_sha256 = hashlib.sha256(
        "".join(image_manifest_lines).encode("utf-8")
    ).hexdigest()
    structural = _structural_audit(payload)
    dataset_review_status = str(metadata.get("annotation_review_status", "")).lower()
    dataset_review_final = dataset_review_status in _FINAL_REVIEW_STATUSES
    failure_reasons = []
    checks = {
        "candidate_fixed_before_preflight": True,
        "dataset_review_final": dataset_review_final,
        "record_manifest_complete": missing_record_count == 0,
        "record_reviews_final": pending_record_review_count == 0,
        "capture_session_provenance_complete": missing_capture_session_count == 0,
        "image_identity_valid": image_identity_mismatch_count == 0,
        "image_dimensions_valid": image_dimension_mismatch_count == 0,
        "coco_structure_valid": all(
            structural[key] == 0
            for key in (
                "duplicate_image_id_count",
                "duplicate_annotation_id_count",
                "orphan_annotation_count",
                "invalid_bbox_count",
                "unknown_category_count",
            )
        ),
        "no_exact_development_overlap": not exact_overlap_image_ids,
        "no_near_development_overlap": not near_overlap_image_ids,
    }
    labels = {
        "dataset_review_final": "annotation review is not finalized",
        "record_manifest_complete": "record manifest does not cover every COCO image",
        "record_reviews_final": "one or more image annotations remain pending review",
        "capture_session_provenance_complete": "capture session provenance is incomplete",
        "image_identity_valid": "declared and actual image SHA-256 differ",
        "image_dimensions_valid": "declared and actual image dimensions differ",
        "coco_structure_valid": "COCO structure or bounding boxes are invalid",
        "no_exact_development_overlap": "images exactly overlap candidate development data",
        "no_near_development_overlap": "images perceptually overlap candidate development data",
    }
    failure_reasons.extend(
        labels[key] for key, passed in checks.items() if not passed and key in labels
    )
    eligible = all(checks.values())
    return {
        "schema_version": "1.0",
        "evaluation": "bread_1_1_independent_dataset_preflight",
        "dataset_version": dataset_version,
        "candidate": {"candidate_id": candidate_id, "git_commit": candidate_commit},
        "dataset": {
            "directory_name": dataset_root.name,
            "annotation_file": annotation_path.name,
            "annotation_sha256": sha256_file(annotation_path),
            "metadata_sha256": sha256_file(metadata_path),
            "record_manifest_sha256": sha256_file(record_manifest_path),
            "image_manifest_sha256": image_manifest_sha256,
            "annotation_review_status": dataset_review_status,
        },
        "source_manifests": [
            {"name": path.name, "sha256": sha256_file(path.resolve())}
            for path in source_manifest_paths
        ],
        "integrity": {
            **structural,
            "missing_record_count": missing_record_count,
            "pending_record_review_count": pending_record_review_count,
            "missing_capture_session_count": missing_capture_session_count,
            "image_identity_mismatch_count": image_identity_mismatch_count,
            "image_dimension_mismatch_count": image_dimension_mismatch_count,
        },
        "development_overlap_audit": {
            "known_source_row_count": len(source_rows),
            "exact_overlap_image_count": len(exact_overlap_image_ids),
            "near_overlap_image_count": len(near_overlap_image_ids),
            "maximum_hamming_distance": maximum_hamming_distance,
            "minimum_cross_set_hamming_distance": minimum_cross_set_hamming_distance,
            "exact_overlap_image_ids": exact_overlap_image_ids,
            "near_overlap_image_ids": near_overlap_image_ids,
            "examples": overlap_examples,
        },
        "checks": checks,
        "eligible_for_independent_lock": eligible,
        "promotion_evidence_allowed": eligible,
        "failure_reasons": failure_reasons,
        "model_inference_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a Bread 1.1 dataset before locking it")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--record-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--maximum-hamming-distance", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_independent_dataset(
        dataset_root=args.dataset_root,
        annotation_path=args.annotation,
        metadata_path=args.metadata,
        record_manifest_path=args.record_manifest,
        source_manifest_paths=args.source_manifest,
        dataset_version=args.dataset_version,
        candidate_id=args.candidate_id,
        candidate_commit=args.candidate_commit,
        maximum_hamming_distance=args.maximum_hamming_distance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
