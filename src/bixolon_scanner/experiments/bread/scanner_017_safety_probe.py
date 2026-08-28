from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...contracts.catalog import load_store_catalog_package
from ...contracts.runtime_package_v2 import load_runtime_package_v2
from ...pipeline.ports import ClassificationResult, Detection
from ...runtime.catalog import ConsensusCatalogClassifier, build_catalog_classifier
from ...runtime.detector_v2 import build_detector_v2
from ...runtime.geometry import box_iou
from ...runtime.imaging import decode_image


@dataclass(frozen=True)
class UnknownCase:
    corpus: str
    image_id: int
    image_path: Path
    segment_bbox: tuple[float, float, float, float]
    response_top3: tuple[str, ...]
    expected_class_id: str


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _bbox_detection(payload: dict) -> Detection:
    box = payload["bbox"]
    return Detection(
        float(box["x"]),
        float(box["y"]),
        float(box["x"] + box["width"]),
        float(box["y"] + box["height"]),
        1.0,
    )


def _annotation_detection(annotation: dict) -> Detection:
    x, y, width, height = annotation.get("bbox", annotation.get("bbox_xywh"))
    return Detection(float(x), float(y), float(x + width), float(y + height), 1.0)


def _expected_class_id(segmentation: dict, annotations: list[dict]) -> str:
    target = _bbox_detection(segmentation)
    matched = max(annotations, key=lambda row: box_iou(target, _annotation_detection(row)))
    if box_iou(target, _annotation_detection(matched)) < 0.5:
        raise ValueError("UNKNOWN segmentation does not match a ground-truth annotation")
    return f"bread_{int(matched['category_id']):02d}"


def _target_cases(root: Path, responses_path: Path) -> list[UnknownCase]:
    responses = _read_json(responses_path)
    coco = _read_json(root / "annotations" / "instances.json")
    annotations: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
    cases = []
    for record in responses["records"]:
        for segmentation in record["response"]["segmentations"]:
            if segmentation["status"] != "UNKNOWN":
                continue
            cases.append(
                UnknownCase(
                    corpus="target-20260828",
                    image_id=int(record["index"]),
                    image_path=root / "images" / record["input_file"],
                    segment_bbox=(
                        float(segmentation["bbox"]["x"]),
                        float(segmentation["bbox"]["y"]),
                        float(segmentation["bbox"]["x"] + segmentation["bbox"]["width"]),
                        float(segmentation["bbox"]["y"] + segmentation["bbox"]["height"]),
                    ),
                    response_top3=tuple(item["class_id"] for item in segmentation["top3"]),
                    expected_class_id=_expected_class_id(
                        segmentation, annotations[int(record["index"])]
                    ),
                )
            )
    return cases


def _manifest_cases(
    corpus: str,
    dataset_root: Path,
    manifest_path: Path,
    trace_path: Path,
) -> list[UnknownCase]:
    manifests = {int(row["image_id"]): row for row in _read_jsonl(manifest_path)}
    cases = []
    for trace in _read_jsonl(trace_path):
        image_id = int(trace["image_id"])
        manifest = manifests[image_id]
        for segmentation in trace["response"]["segmentations"]:
            if segmentation["status"] != "UNKNOWN":
                continue
            image_path = Path(manifest["image_path"])
            cases.append(
                UnknownCase(
                    corpus=corpus,
                    image_id=image_id,
                    image_path=dataset_root / image_path,
                    segment_bbox=(
                        float(segmentation["bbox"]["x"]),
                        float(segmentation["bbox"]["y"]),
                        float(segmentation["bbox"]["x"] + segmentation["bbox"]["width"]),
                        float(segmentation["bbox"]["y"] + segmentation["bbox"]["height"]),
                    ),
                    response_top3=tuple(item["class_id"] for item in segmentation["top3"]),
                    expected_class_id=_expected_class_id(segmentation, manifest["annotations"]),
                )
            )
    return cases


def _result_payload(
    classifier,
    result: ClassificationResult,
    row: int,
) -> dict:
    ranking_order = np.argsort(-result.ranking_logits[row], kind="stable")
    retrieval_order = np.argsort(-result.retrieval_logits[row], kind="stable")
    labels = classifier.labels
    return {
        "top1": labels[int(ranking_order[0])].class_id,
        "top3": [labels[int(index)].class_id for index in ranking_order[:3]],
        "retrieval_top1": labels[int(retrieval_order[0])].class_id,
        "retrieval_similarity": float(result.retrieval_logits[row, retrieval_order[0]]),
        "ridge_approval_score": float(result.approval_scores[row]),
        "inverse_entropy_top3_safety_score": (
            None if result.top3_safety_scores is None else float(result.top3_safety_scores[row])
        ),
        "segment_recapture_reason": (
            None
            if result.segment_recapture_reasons is None
            else result.segment_recapture_reasons[row]
        ),
    }


def _evaluate_case(
    case: UnknownCase,
    detector,
    classifier: ConsensusCatalogClassifier,
    *,
    jpeg_draft_size: int | None,
) -> dict:
    image = decode_image(
        case.image_path.read_bytes(),
        max_bytes=50_000_000,
        max_pixels=50_000_000,
        jpeg_draft_size=jpeg_draft_size,
    )
    try:
        detection_result = detector.detect(image)
        detections = sorted(detection_result.detections, key=lambda item: (item.y1, item.x1))
        target = Detection(*case.segment_bbox, 1.0)
        row = int(np.argmax([box_iou(target, item) for item in detections]))
        matched_iou = box_iou(target, detections[row])
        if matched_iou < 0.95:
            raise ValueError(f"detector did not reproduce UNKNOWN bbox: IoU={matched_iou:.6f}")

        prepared = classifier.primary.embedder.prepare_detection_tensors(image, detections)
        primary_raw = classifier.primary.embedder.embed_prepared_tensors_raw(prepared)
        primary = classifier.primary.classify_embeddings(primary_raw, detections)
        rotated = np.ascontiguousarray(prepared[:, :, ::-1, ::-1], dtype=np.float32)
        rotated_raw = classifier.primary.embedder.embed_prepared_tensors_raw(rotated)
        rotation_raw = np.asarray((primary_raw + rotated_raw) * np.float32(0.5), dtype=np.float32)
        rotation = classifier.rotation.classify_embeddings(rotation_raw)
        independent_prepared = classifier.independent.embedder.prepare_detection_tensors(
            image, detections
        )
        independent_raw = classifier.independent.embedder.embed_prepared_tensors_raw(
            independent_prepared
        )
        independent = classifier.independent.classify_embeddings(independent_raw, detections)
        selected = np.asarray([row], dtype=np.int64)
        selected_rotated = np.ascontiguousarray(prepared[selected, :, ::-1, ::-1], dtype=np.float32)
        selected_rotated_raw = classifier.primary.embedder.embed_prepared_tensors_raw(
            selected_rotated
        )
        selected_rotation = classifier.rotation.classify_embeddings(
            np.asarray(
                (primary_raw[selected] + selected_rotated_raw) * np.float32(0.5),
                dtype=np.float32,
            )
        )
        selected_independent_prepared = (
            classifier.independent.embedder.prepare_selected_detection_tensors(
                image, detections, selected
            )
        )
        selected_independent_raw = classifier.independent.embedder.embed_prepared_tensors_raw(
            selected_independent_prepared
        )
        selected_independent = classifier.independent.classify_embeddings(
            selected_independent_raw, [detections[row]]
        )
        captured: dict[str, ClassificationResult] = {}
        rotation_classify = classifier.rotation.classify_embeddings
        independent_classify = classifier.independent.classify_embeddings

        def capture_rotation(*values, **keywords):
            captured["rotation"] = rotation_classify(*values, **keywords)
            return captured["rotation"]

        def capture_independent(*values, **keywords):
            captured["independent"] = independent_classify(*values, **keywords)
            return captured["independent"]

        classifier.rotation.classify_embeddings = capture_rotation
        classifier.independent.classify_embeddings = capture_independent
        try:
            final = classifier.classify(image, detections)
        finally:
            classifier.rotation.classify_embeddings = rotation_classify
            classifier.independent.classify_embeddings = independent_classify
    finally:
        image.close()

    primary_payload = _result_payload(classifier.primary, primary, row)
    rotation_payload = _result_payload(classifier.rotation, rotation, row)
    independent_payload = _result_payload(classifier.independent, independent, row)
    union = sorted(
        set(primary_payload["top3"])
        | set(rotation_payload["top3"])
        | set(independent_payload["top3"])
    )
    return {
        "corpus": case.corpus,
        "image_id": case.image_id,
        "expected_class_id": case.expected_class_id,
        "response_top3": list(case.response_top3),
        "response_top3_hit": case.expected_class_id in case.response_top3,
        "detector_match_iou": matched_iou,
        "primary": primary_payload,
        "rotation": rotation_payload,
        "independent": independent_payload,
        "selected_path": {
            "rotation_segment_recapture_reason": selected_rotation.segment_recapture_reasons[0],
            "independent_segment_recapture_reason": (
                selected_independent.segment_recapture_reasons[0]
            ),
        },
        "final_path": {
            "rotation_segment_recapture_reasons": list(
                captured["rotation"].segment_recapture_reasons
            ),
            "independent_segment_recapture_reasons": list(
                captured["independent"].segment_recapture_reasons
            ),
        },
        "final": {
            "approval_blocked": bool(final.approval_blocked[row]),
            "segment_recapture_reason": final.segment_recapture_reasons[row],
            "unknown_reason": final.unknown_reasons[row],
        },
        "candidate_set_union": union,
        "candidate_set_union_size": len(union),
        "candidate_set_union_hit": case.expected_class_id in union,
    }


def probe(args: argparse.Namespace) -> dict:
    runtime = load_runtime_package_v2(args.runtime)
    catalog = load_store_catalog_package(args.catalog, expected_store_id=args.store_id)
    detector = build_detector_v2(runtime, args.provider, args.cuda_dll_dir)
    built_classifier, _ = build_catalog_classifier(
        runtime, catalog, args.provider, args.cuda_dll_dir
    )
    if not isinstance(built_classifier, ConsensusCatalogClassifier):
        raise ValueError("0.1.7 safety probe requires the packaged consensus classifier")
    built_classifier.unknown_recapture_on_dual_verifier_rejection = (
        args.enable_unknown_dual_recapture
    )
    cases = _target_cases(args.target_root, args.target_responses)
    cases.extend(
        _manifest_cases(
            "detector415",
            args.dataset415_root,
            args.manifest415,
            args.trace415,
        )
    )
    cases.extend(
        _manifest_cases(
            "operational69",
            args.operational69_root,
            args.manifest69,
            args.trace69,
        )
    )
    try:
        rows = [
            _evaluate_case(
                case,
                detector,
                built_classifier,
                jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
            )
            for case in cases
        ]
    finally:
        detector.close()
        built_classifier.close()

    target_miss = next(
        row for row in rows if row["corpus"] == "target-20260828" and not row["response_top3_hit"]
    )
    safe_baseline = [row for row in rows if row["corpus"] != "target-20260828"]
    signals = {
        "union_size_greater_than_3": lambda row: row["candidate_set_union_size"] > 3,
        "independent_top1_outside_primary_top3": lambda row: (
            row["independent"]["top1"] not in row["primary"]["top3"]
        ),
        "rotation_top1_outside_primary_top3": lambda row: (
            row["rotation"]["top1"] not in row["primary"]["top3"]
        ),
        "expected_absent_from_all_verifiers": lambda row: not row["candidate_set_union_hit"],
    }
    signal_summary = {
        name: {
            "target_miss_triggered": bool(rule(target_miss)),
            "baseline_safe_unknown_trigger_count": sum(bool(rule(row)) for row in safe_baseline),
            "baseline_safe_unknown_count": len(safe_baseline),
        }
        for name, rule in signals.items()
    }
    return {
        "schema_version": "1.0",
        "product_version": runtime.metadata.worker_version,
        "provider": args.provider,
        "unknown_case_count": len(rows),
        "target_unknown_count": sum(row["corpus"] == "target-20260828" for row in rows),
        "baseline_safe_unknown_count": len(safe_baseline),
        "signal_summary": signal_summary,
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--store-id", default="bread-dev")
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--target-responses", type=Path, required=True)
    parser.add_argument("--dataset415-root", type=Path, required=True)
    parser.add_argument("--manifest415", type=Path, required=True)
    parser.add_argument("--trace415", type=Path, required=True)
    parser.add_argument("--operational69-root", type=Path, required=True)
    parser.add_argument("--manifest69", type=Path, required=True)
    parser.add_argument("--trace69", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-unknown-dual-recapture", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
