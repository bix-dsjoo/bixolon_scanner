from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ...contracts import ItemStatus
from ...contracts.catalog import load_store_catalog_package
from ...contracts.runtime_package_v2 import load_runtime_package_v2
from ...pipeline.classification import normalize_classification
from ...pipeline.ports import Detection
from ...pipeline.segmentation import build_scan_items
from ...runtime.catalog import build_catalog_classifier
from ...runtime.imaging import decode_image
from ...runtime.proposal_selection import box_iou


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _ground_truth(path: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        x, y, width, height = annotation["bbox"]
        by_image[int(annotation["image_id"])].append(
            {
                "box": [x, y, x + width, y + height],
                "class_id": f"bread_{int(annotation['category_id']):02d}",
            }
        )
    return payload["images"], by_image


def _match_rows(
    detections: list[Detection], expected: list[dict[str, Any]], match_iou: float
) -> dict[int, int]:
    pairs = sorted(
        (
            (
                box_iou(
                    [detection.x1, detection.y1, detection.x2, detection.y2],
                    target["box"],
                ),
                detection_index,
                target_index,
            )
            for detection_index, detection in enumerate(detections)
            for target_index, target in enumerate(expected)
        ),
        reverse=True,
    )
    matched: dict[int, int] = {}
    used_targets: set[int] = set()
    for overlap, detection_index, target_index in pairs:
        if overlap < match_iou:
            break
        if detection_index in matched or target_index in used_targets:
            continue
        matched[detection_index] = target_index
        used_targets.add(target_index)
    return matched


def _detector_member_classes(
    image_id: int,
    detections: list[Detection],
    member_predictions: list[dict[int, dict[str, Any]]],
    labels: list[Any],
) -> list[list[str]]:
    result = []
    for detection in detections:
        box = [detection.x1, detection.y1, detection.x2, detection.y2]
        classes = []
        for member in member_predictions:
            row = member[image_id]
            candidate_index = max(
                range(len(row["boxes_xyxy"])),
                key=lambda index: box_iou(box, row["boxes_xyxy"][index]),
            )
            classes.append(labels[int(row["class_ids"][candidate_index])].class_id)
        result.append(classes)
    return result


def _classify_single_views(classifier, image, detections):
    primary = getattr(classifier, "primary", classifier)
    prepared = np.concatenate(
        [
            primary.embedder.prepare_detection_tensors(image, [detection])
            for detection in detections
        ],
        axis=0,
    )
    embeddings = primary.embedder.embed_prepared_tensors_raw(prepared)
    result = primary.classify_embeddings(embeddings, detections)
    batch = normalize_classification(
        result,
        detection_count=len(detections),
        metadata=primary.metadata,
    )
    return result, batch


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    runtime = load_runtime_package_v2(args.runtime)
    catalog = load_store_catalog_package(args.catalog, expected_store_id=args.store_id)
    classifier, _ = build_catalog_classifier(runtime, catalog, args.provider, args.cuda_dll_dir)
    images, expected_by_id = _ground_truth(args.annotation_path)
    predictions = {int(row["image_id"]): row for row in _read_jsonl(args.predictions)}
    member_predictions = [
        {int(row["image_id"]): row for row in _read_jsonl(path)} for path in args.member_predictions
    ]
    counters: Counter[str] = Counter()
    latencies = []
    rows = []
    try:
        for image in images:
            image_id = int(image["id"])
            expected = expected_by_id[image_id]
            prediction = predictions[image_id]
            if not prediction["boxes_xyxy"]:
                counters["image_recapture"] += 1
                rows.append(
                    {
                        "image_id": image_id,
                        "file_name": image["file_name"],
                        "status": "RECAPTURE",
                        "segmentations": [],
                    }
                )
                continue
            path = (args.annotation_path.parent / image["file_name"]).resolve()
            decoded = decode_image(
                path.read_bytes(),
                max_bytes=50_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
            )
            started = time.perf_counter()
            try:
                detections = sorted(
                    (
                        Detection(*box, float(score), int(class_id))
                        for box, score, class_id in zip(
                            prediction["boxes_xyxy"],
                            prediction["scores"],
                            prediction["class_ids"],
                        )
                    ),
                    key=lambda detection: (detection.y1, detection.x1),
                )
                detector_classes = _detector_member_classes(
                    image_id,
                    detections,
                    member_predictions,
                    classifier.metadata.labels,
                )
                classification = classifier.classify(decoded, detections)
                single_view = None
                single_batch = None
                if args.single_view_diagnostics or args.apply_safety_policy:
                    single_view, single_batch = _classify_single_views(
                        classifier, decoded, detections
                    )
                batch = normalize_classification(
                    classification,
                    detection_count=len(detections),
                    metadata=classifier.metadata,
                )
                items = build_scan_items(
                    detections,
                    batch,
                    classifier.metadata,
                    border_indices=set(),
                    duplicate_review_indices=set(),
                )
                if args.apply_safety_policy and member_predictions:
                    refined = list(detections)
                    refinement_indices = [
                        index
                        for index, (item, classes) in enumerate(zip(items, detector_classes))
                        if item.status is ItemStatus.APPROVED
                        and item.prediction.class_id == "bread_03"
                        and classes
                        and len(set(classes)) == 1
                        and classes[0] == "bread_02"
                    ]
                    for index in refinement_indices:
                        detection = refined[index]
                        target_height = (detection.x2 - detection.x1) / 4.5
                        refined[index] = Detection(
                            detection.x1,
                            detection.y2 - target_height,
                            detection.x2,
                            detection.y2,
                            detection.score,
                            detection.class_id,
                        )
                    if refinement_indices:
                        detections = refined
                        classification = classifier.classify(decoded, detections)
                        batch = normalize_classification(
                            classification,
                            detection_count=len(detections),
                            metadata=classifier.metadata,
                        )
                        items = build_scan_items(
                            detections,
                            batch,
                            classifier.metadata,
                            border_indices=set(),
                            duplicate_review_indices=set(),
                        )
                        single_view, single_batch = _classify_single_views(
                            classifier, decoded, detections
                        )
            finally:
                decoded.close()
            latencies.append((time.perf_counter() - started) * 1000.0)
            matched = _match_rows(detections, expected, args.match_iou_threshold)
            segmentations = []
            for index, item in enumerate(items):
                target = expected[matched[index]]
                effective_status = item.status
                effective_prediction = None if item.prediction is None else item.prediction.class_id
                top3 = [candidate.class_id for candidate in item.top3]
                if (
                    args.apply_safety_policy
                    and item.status is ItemStatus.UNKNOWN
                    and single_view is not None
                    and single_batch is not None
                ):
                    single_order = np.argsort(
                        -single_batch.ranking_probabilities[index], kind="stable"
                    )
                    single_top3 = [
                        classifier.metadata.labels[int(candidate_index)].class_id
                        for candidate_index in single_order[:3]
                    ]
                    classes = detector_classes[index]
                    if (
                        classes
                        and len(set(classes)) == 1
                        and top3
                        and classes[0] == top3[0]
                        and single_top3[0] != classes[0]
                        and classes[0] in single_top3
                    ):
                        effective_status = ItemStatus.APPROVED
                        effective_prediction = classes[0]
                        top3 = []
                    else:
                        merged_scores = np.maximum(
                            batch.ranking_probabilities[index],
                            single_batch.ranking_probabilities[index],
                        )
                        top3 = [
                            classifier.metadata.labels[int(candidate_index)].class_id
                            for candidate_index in np.argsort(-merged_scores, kind="stable")[:3]
                        ]
                if effective_status is ItemStatus.APPROVED:
                    counters["approved"] += 1
                    correct = effective_prediction == target["class_id"]
                    counters["approved_correct" if correct else "approved_wrong"] += 1
                    top3 = []
                elif effective_status is ItemStatus.UNKNOWN:
                    counters["unknown"] += 1
                    counters[
                        "unknown_top3_hit" if target["class_id"] in top3 else "unknown_top3_miss"
                    ] += 1
                    correct = None
                else:
                    counters["segment_recapture"] += 1
                    top3 = []
                    correct = None
                segmentations.append(
                    {
                        "expected_class_id": target["class_id"],
                        "status": effective_status.value,
                        "prediction": effective_prediction,
                        "top3": top3,
                        "approval_score": item.confidence,
                        "correct": correct,
                        "single_view_top3": (
                            None
                            if single_view is None
                            else [
                                classifier.metadata.labels[int(candidate_index)].class_id
                                for candidate_index in np.argsort(
                                    -single_view.ranking_logits[index], kind="stable"
                                )[:3]
                            ]
                        ),
                        "single_view_approval_score": (
                            None
                            if single_view is None
                            else float(single_view.approval_scores[index])
                        ),
                        "detector_member_classes": (detector_classes[index]),
                    }
                )
            rows.append(
                {
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "status": "SEGMENTATION",
                    "segmentations": segmentations,
                }
            )
    except Exception as error:
        counters["error"] += 1
        rows.append({"status": "ERROR", "error_type": type(error).__name__})
        raise
    finally:
        classifier.close()

    sorted_latencies = sorted(latencies)
    latency = {
        "sample_count": len(sorted_latencies),
        "p50_ms": sorted_latencies[int(0.50 * (len(sorted_latencies) - 1))]
        if sorted_latencies
        else 0.0,
        "p95_ms": sorted_latencies[int(0.95 * (len(sorted_latencies) - 1))]
        if sorted_latencies
        else 0.0,
        "p99_ms": sorted_latencies[int(0.99 * (len(sorted_latencies) - 1))]
        if sorted_latencies
        else 0.0,
    }
    report = {
        "schema_version": "1.0",
        "evaluation": "yolo_free_final_output_contract",
        "provider": args.provider,
        "image_count": len(images),
        "segmentation_count": sum(len(row.get("segmentations", [])) for row in rows),
        "metrics": dict(counters),
        "classifier_latency": latency,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final YOLO-free scanner outputs")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--store-id", default="bread-dev")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--annotation-path", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--single-view-diagnostics", action="store_true")
    parser.add_argument("--apply-safety-policy", action="store_true")
    parser.add_argument("--member-predictions", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
