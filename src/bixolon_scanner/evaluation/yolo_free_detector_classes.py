from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..contracts.catalog import load_store_catalog_package, sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2
from ..runtime.assisted_detector import attach_classifier_assisted_detector
from ..runtime.catalog import build_catalog_classifier
from ..runtime.detector_v2 import build_detector_v2
from ..runtime.imaging import decode_image
from ..runtime.proposal_selection import box_iou


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256_index(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    return {sha256_file(path): path for path in root.iterdir() if path.is_file()}


def _matches(detections, annotations: list[dict]) -> list[tuple[int, int, float]]:
    candidates = []
    for detection_index, detection in enumerate(detections):
        detection_box = [detection.x1, detection.y1, detection.x2, detection.y2]
        for annotation_index, annotation in enumerate(annotations):
            x, y, width, height = annotation.get("bbox_xywh", annotation.get("bbox"))
            overlap = box_iou(detection_box, [x, y, x + width, y + height])
            if overlap >= 0.5:
                candidates.append((overlap, detection_index, annotation_index))
    used_detections: set[int] = set()
    used_annotations: set[int] = set()
    matches = []
    for overlap, detection_index, annotation_index in sorted(candidates, reverse=True):
        if detection_index in used_detections or annotation_index in used_annotations:
            continue
        used_detections.add(detection_index)
        used_annotations.add(annotation_index)
        matches.append((detection_index, annotation_index, overlap))
    return matches


def evaluate(args: argparse.Namespace) -> dict:
    runtime = load_runtime_package_v2(args.runtime)
    catalog = load_store_catalog_package(args.catalog, expected_store_id=args.store_id)
    detector = build_detector_v2(runtime, args.provider, args.cuda_dll_dir)
    classifier, _embedder = build_catalog_classifier(
        runtime, catalog, args.provider, args.cuda_dll_dir
    )
    detector = attach_classifier_assisted_detector(detector, classifier)
    image_sha256_index = _sha256_index(args.image_sha_root)
    support_totals: Counter[int] = Counter()
    support_correct: Counter[int] = Counter()
    class_support_totals: Counter[tuple[int, int]] = Counter()
    class_support_correct: Counter[tuple[int, int]] = Counter()
    vote_rows = []
    failures = []
    matched_count = 0
    try:
        detector.warmup()
        classifier.warmup()
        for position, row in enumerate(_jsonl(args.manifest), start=1):
            image_path = args.dataset_root / row["image_path"]
            if not image_path.is_file() and image_sha256_index:
                image_path = image_sha256_index.get(row["image_sha256"], image_path)
            if sha256_file(image_path) != row["image_sha256"]:
                raise ValueError("detector-class input checksum mismatch")
            image = decode_image(
                image_path.read_bytes(),
                max_bytes=50_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
            )
            try:
                result = detector.detect(image)
            finally:
                image.close()
            matches = _matches(result.detections, row["annotations"])
            matched_count += len(matches)
            for detection_index, annotation_index, overlap in matches:
                predicted = int(result.detector_class_ids[detection_index])
                support = int(result.detector_class_support_counts[detection_index])
                expected = int(row["annotations"][annotation_index]["category_id"]) - 1
                support_totals[support] += 1
                class_support_totals[(predicted, support)] += 1
                vote_rows.append(
                    {
                        "image_id": int(row["image_id"]),
                        "detection_index": detection_index,
                        "expected_class_index": expected,
                        "predicted_class_index": predicted,
                        "support_count": support,
                        "detector_score": result.detections[detection_index].score,
                        "match_iou": overlap,
                    }
                )
                if predicted == expected:
                    support_correct[support] += 1
                    class_support_correct[(predicted, support)] += 1
                else:
                    failures.append(
                        {
                            "image_id": int(row["image_id"]),
                            "detection_index": detection_index,
                            "expected_class_index": expected,
                            "predicted_class_index": predicted,
                            "support_count": support,
                            "match_iou": overlap,
                        }
                    )
            if position % 50 == 0:
                print(f"detector classes: {position}", flush=True)
    finally:
        detector.close()
        classifier.close()
    total = sum(support_totals.values())
    correct = sum(support_correct.values())
    report = {
        "schema_version": "1.0",
        "evaluation": "yolo_free_detector_class_votes",
        "image_count": position,
        "matched_count": matched_count,
        "class_vote_count": total,
        "class_vote_correct_count": correct,
        "class_vote_accuracy": correct / total,
        "support": {
            str(support): {
                "count": support_totals[support],
                "correct_count": support_correct[support],
                "accuracy": support_correct[support] / support_totals[support],
            }
            for support in sorted(support_totals)
        },
        "predicted_class_support": {
            f"{predicted}:{support}": {
                "count": class_support_totals[(predicted, support)],
                "correct_count": class_support_correct[(predicted, support)],
                "accuracy": class_support_correct[(predicted, support)]
                / class_support_totals[(predicted, support)],
            }
            for predicted, support in sorted(class_support_totals)
        },
        "votes": vote_rows,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--store-id", default="bread-dev")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-sha-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
