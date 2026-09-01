from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.catalog import load_store_catalog_package, sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2
from ..pipeline.classification import normalize_classification
from ..pipeline.ports import Detection
from ..runtime.assisted_detector import (
    ClassifierAssistedEnsembleDetector,
    attach_classifier_assisted_detector,
)
from ..runtime.catalog import build_catalog_classifier
from ..runtime.detector_v2 import build_detector_v2
from ..runtime.imaging import decode_image


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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
    if not isinstance(detector, ClassifierAssistedEnsembleDetector):
        raise ValueError("primary-risk evaluation requires the assisted detector")

    rows = _jsonl(args.manifest)
    if args.minimum_image_id is not None:
        rows = [row for row in rows if int(row["image_id"]) >= args.minimum_image_id]
    if args.maximum_image_id is not None:
        rows = [row for row in rows if int(row["image_id"]) <= args.maximum_image_id]
    results = []
    try:
        detector.warmup()
        classifier.warmup()
        for index, row in enumerate(rows, start=1):
            image_path = args.dataset_root / row["image_path"]
            if sha256_file(image_path) != row["image_sha256"]:
                raise ValueError("primary-risk input checksum mismatch")
            image = decode_image(
                image_path.read_bytes(),
                max_bytes=50_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
            )
            try:
                selected, raw, _agreement, _uncertain, _saturated = (
                    detector.detector.predict_candidates(image)
                )
                predicted_count, count_confidence = detector.verifier.verify(image)
                detector_classes, detector_supports = detector._detector_classes(selected, raw)
                detections = [
                    Detection(*box, float(score), int(class_id))
                    for box, score, class_id in zip(
                        selected["boxes_xyxy"],
                        selected["scores"],
                        selected["class_ids"],
                        strict=True,
                    )
                ]
                order = sorted(
                    range(len(detections)),
                    key=lambda value: (
                        detections[value].y1,
                        detections[value].x1,
                    ),
                )
                detections = [detections[value] for value in order]
                detector_classes = [detector_classes[value] for value in order]
                detector_supports = [detector_supports[value] for value in order]
                contextual = normalize_classification(
                    classifier.classify(image, detections),
                    detection_count=len(detections),
                    metadata=classifier.metadata,
                )
                direct_risks = []
                for detection_index in range(len(detections)):
                    class_id = detector_classes[detection_index]
                    support = detector_supports[detection_index]
                    top1 = int(contextual.decision_indices[detection_index, 0])
                    if (
                        contextual.approved[detection_index]
                        and class_id is not None
                        and support >= detector.policy.candidate_minimum_support
                        and top1 != class_id
                    ):
                        direct_risks.append(
                            {
                                "index": detection_index,
                                "approval_score": float(
                                    contextual.approval_scores[detection_index]
                                ),
                                "classifier_class": top1,
                                "detector_class": class_id,
                                "detector_support": support,
                            }
                        )

                promotion_risks = []
                unapproved = np.flatnonzero(~contextual.approved)
                if len(unapproved):
                    single = normalize_classification(
                        classifier.classify_single_views(
                            image,
                            [detections[int(value)] for value in unapproved],
                        ),
                        detection_count=len(unapproved),
                        metadata=classifier.metadata,
                    )
                    for single_index, index_value in enumerate(unapproved):
                        detection_index = int(index_value)
                        class_id = detector_classes[detection_index]
                        if class_id is None:
                            continue
                        single_top3 = single.decision_indices[single_index, :3]
                        if (
                            detector_supports[detection_index]
                            >= detector.policy.candidate_minimum_support
                            and (
                                detector.policy.unknown_promotion_minimum_single_view_approval
                                is None
                                or single.approval_scores[single_index]
                                >= detector.policy.unknown_promotion_minimum_single_view_approval
                            )
                            and int(contextual.decision_indices[detection_index, 0]) == class_id
                            and int(single_top3[0]) != class_id
                            and class_id in single_top3
                        ):
                            promotion_risks.append(
                                {
                                    "index": detection_index,
                                    "contextual_approval_score": float(
                                        contextual.approval_scores[detection_index]
                                    ),
                                    "single_approval_score": float(
                                        single.approval_scores[single_index]
                                    ),
                                    "detector_class": class_id,
                                    "detector_support": detector_supports[detection_index],
                                }
                            )
                results.append(
                    {
                        "image_id": int(row["image_id"]),
                        "image_sha256": row["image_sha256"],
                        "selected_count": len(detections),
                        "predicted_count": predicted_count,
                        "count_confidence": count_confidence,
                        "count_mismatch": predicted_count != len(detections),
                        "direct_risks": direct_risks,
                        "promotion_risks": promotion_risks,
                    }
                )
            finally:
                image.close()
            if index % 50 == 0 or index == len(rows):
                print(f"primary risk: {index}/{len(rows)}", flush=True)
    finally:
        detector.close()
        classifier.close()

    def summary(values: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "image_count": len(values),
            "count_mismatch_image_count": sum(row["count_mismatch"] for row in values),
            "direct_risk_image_count": sum(bool(row["direct_risks"]) for row in values),
            "promotion_risk_image_count": sum(bool(row["promotion_risks"]) for row in values),
            "any_fallback_image_count": sum(
                row["count_mismatch"] or bool(row["direct_risks"]) or bool(row["promotion_risks"])
                for row in values
            ),
        }

    first_100 = [row for row in results if row["image_id"] <= 100]
    return {
        "schema_version": "1.0",
        "evaluation": "yolo_free_primary_risk",
        "provider": args.provider,
        "summary": {
            "all": summary(results),
            "first_100": summary(first_100),
        },
        "rows": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda", "openvino"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--openvino-cache-dir", type=Path)
    parser.add_argument("--cpu-detector-threads", type=int, default=0)
    parser.add_argument("--cpu-embedder-threads", type=int, default=0)
    parser.add_argument("--store-id", default="bread-dev")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--minimum-image-id", type=int)
    parser.add_argument("--maximum-image-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args)
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
