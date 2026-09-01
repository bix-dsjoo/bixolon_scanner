from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...contracts.catalog import load_store_catalog_package, sha256_file
from ...contracts.runtime_package_v2 import load_runtime_package_v2
from ...pipeline.classification import normalize_classification
from ...pipeline.ports import Detection
from ...runtime.assisted_detector import (
    ClassifierAssistedEnsembleDetector,
    attach_classifier_assisted_detector,
)
from ...runtime.catalog import build_catalog_classifier
from ...runtime.detector_v2 import build_detector_v2
from ...runtime.imaging import decode_image
from ...runtime.proposal_selection import box_iou, count_and_class_assisted_select


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _annotation_bbox_xyxy(annotation: dict[str, Any]) -> list[float]:
    bbox = annotation.get("bbox_xywh", annotation.get("bbox"))
    if bbox is None:
        raise ValueError("annotation is missing bbox_xywh/bbox")
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def _match(
    detections: list[Detection], annotations: list[dict[str, Any]]
) -> dict[int, tuple[int, float]]:
    pairs = sorted(
        (
            (
                box_iou(
                    [detection.x1, detection.y1, detection.x2, detection.y2],
                    _annotation_bbox_xyxy(annotation),
                ),
                detection_index,
                annotation_index,
            )
            for detection_index, detection in enumerate(detections)
            for annotation_index, annotation in enumerate(annotations)
        ),
        reverse=True,
    )
    matches: dict[int, tuple[int, float]] = {}
    used_annotations: set[int] = set()
    for overlap, detection_index, annotation_index in pairs:
        if detection_index in matches or annotation_index in used_annotations:
            continue
        matches[detection_index] = (annotation_index, overlap)
        used_annotations.add(annotation_index)
    return matches


def _top(labels, probabilities: np.ndarray, count: int = 5) -> list[dict[str, Any]]:
    order = np.argsort(-probabilities, kind="stable")[:count]
    return [
        {
            "class_id": labels[int(index)].class_id,
            "score": float(probabilities[int(index)]),
        }
        for index in order
    ]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    runtime = load_runtime_package_v2(args.runtime)
    catalog = load_store_catalog_package(args.catalog, expected_store_id=args.store_id)
    detector = build_detector_v2(
        runtime,
        args.provider,
        args.cuda_dll_dir,
        cpu_intra_op_threads=args.cpu_detector_threads,
        openvino_cache_dir=args.openvino_cache_dir,
    )
    classifier, _embedder = build_catalog_classifier(
        runtime,
        catalog,
        args.provider,
        args.cuda_dll_dir,
        cpu_intra_op_threads=args.cpu_embedder_threads,
        openvino_cache_dir=args.openvino_cache_dir,
    )
    detector = attach_classifier_assisted_detector(detector, classifier)
    requested = set(args.image_ids)
    rows = [row for row in _jsonl(args.manifest) if int(row["image_id"]) in requested]
    if {int(row["image_id"]) for row in rows} != requested:
        raise ValueError("one or more diagnostic image IDs were not found")
    results = []
    try:
        detector.warmup()
        classifier.warmup()
        for row in rows:
            image_path = args.dataset_root / row["image_path"]
            if sha256_file(image_path) != row["image_sha256"]:
                raise ValueError("diagnostic image checksum mismatch")
            image = decode_image(
                image_path.read_bytes(),
                max_bytes=50_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
            )
            try:
                selection_diagnostics = None
                if isinstance(detector, ClassifierAssistedEnsembleDetector):
                    base, raw, _agreement, _uncertain, _saturated = (
                        detector.detector.predict_candidates(
                            image,
                            apply_low_resolution_override=(not args.full_ensemble_selection),
                        )
                    )
                    predicted_count, count_confidence = detector.verifier.verify(image)
                    ambiguous = detector._localization_is_ambiguous(base)
                    entries = (
                        []
                        if args.selection_only
                        else detector._classify_proposals(image, base, raw)
                    )
                    selected = base
                    selector = None
                    if entries and (predicted_count != len(base["boxes_xyxy"]) or ambiguous):
                        selected, selector = count_and_class_assisted_select(
                            base,
                            raw,
                            entries,
                            predicted_count,
                            detector._selection_policy(),
                        )
                    selection_diagnostics = {
                        "predicted_count": predicted_count,
                        "count_confidence": count_confidence,
                        "base": base,
                        "raw": raw if args.selection_only else None,
                        "ambiguous": ambiguous,
                        "proposal_entries": entries,
                        "selector": selector,
                        "selected": selected,
                    }
                if args.selection_only:
                    results.append(
                        {
                            "image_id": int(row["image_id"]),
                            "image_path": row["image_path"],
                            "selection": selection_diagnostics,
                        }
                    )
                    continue
                detection_result = detector.detect(image)
                ordered_rows = sorted(
                    enumerate(detection_result.detections),
                    key=lambda value: (value[1].y1, value[1].x1),
                )
                detections = [value[1] for value in ordered_rows]
                detector_classes = [
                    detection_result.detector_class_ids[value[0]] for value in ordered_rows
                ]
                detector_supports = [
                    detection_result.detector_class_support_counts[value[0]]
                    for value in ordered_rows
                ]
                contextual = normalize_classification(
                    classifier.classify(image, detections),
                    detection_count=len(detections),
                    metadata=classifier.metadata,
                )
                single = normalize_classification(
                    classifier.classify_single_views(image, detections),
                    detection_count=len(detections),
                    metadata=classifier.metadata,
                )
                matches = _match(detections, row["annotations"])
                segments = []
                for index, detection in enumerate(detections):
                    matched = matches.get(index)
                    annotation = None if matched is None else row["annotations"][matched[0]]
                    overlap = 0.0 if matched is None else matched[1]
                    detector_class = detector_classes[index]
                    segments.append(
                        {
                            "bbox_xyxy": [
                                detection.x1,
                                detection.y1,
                                detection.x2,
                                detection.y2,
                            ],
                            "match_iou": overlap,
                            "expected_class_id": (
                                None
                                if annotation is None
                                else f"bread_{int(annotation['category_id']):02d}"
                            ),
                            "detector_class_id": (
                                None
                                if detector_class is None
                                else classifier.metadata.labels[int(detector_class)].class_id
                            ),
                            "detector_class_support": detector_supports[index],
                            "contextual": {
                                "approved": bool(contextual.approved[index]),
                                "approval_score": float(contextual.approval_scores[index]),
                                "segment_recapture_reason": (
                                    None
                                    if contextual.segment_recapture_reasons is None
                                    else contextual.segment_recapture_reasons[index]
                                ),
                                "unknown_reason": (
                                    None
                                    if contextual.unknown_reasons is None
                                    else contextual.unknown_reasons[index]
                                ),
                                "top5": _top(
                                    classifier.metadata.labels,
                                    contextual.ranking_probabilities[index],
                                ),
                            },
                            "single_view": {
                                "approved": bool(single.approved[index]),
                                "approval_score": float(single.approval_scores[index]),
                                "top5": _top(
                                    classifier.metadata.labels,
                                    single.ranking_probabilities[index],
                                ),
                            },
                        }
                    )
                results.append(
                    {
                        "image_id": int(row["image_id"]),
                        "image_path": row["image_path"],
                        "segmentation_count": len(segments),
                        "selection": selection_diagnostics,
                        "segmentations": segments,
                    }
                )
            finally:
                image.close()
    finally:
        detector.close()
        classifier.close()
    report = {
        "schema_version": "1.0",
        "evaluation": "yolo_free_runtime_diagnostics",
        "provider": args.provider,
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=("cpu", "cuda", "openvino", "openvino_gpu"),
        default="cuda",
    )
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--openvino-cache-dir", type=Path)
    parser.add_argument("--cpu-detector-threads", type=int, default=0)
    parser.add_argument("--cpu-embedder-threads", type=int, default=0)
    parser.add_argument("--store-id", default="bread-dev")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--full-ensemble-selection", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
