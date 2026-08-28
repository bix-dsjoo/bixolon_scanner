from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ...contracts.runtime_package_v2 import load_runtime_package_v2
from ...pipeline.ports import Detection
from ...runtime.geometry import box_containment, box_iou, nms
from ...runtime.imaging import decode_image, restore_original_resolution
from ...runtime.onnx import OnnxDetector


@dataclass(frozen=True)
class DetectorCase:
    corpus: str
    image_id: int
    image_path: Path
    annotations: tuple[dict, ...]


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _target_cases(root: Path) -> list[DetectorCase]:
    coco = _read_json(root / "annotations" / "instances.json")
    annotations: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
    return [
        DetectorCase(
            corpus="target-20260828",
            image_id=int(image["id"]),
            image_path=root / "images" / Path(image["file_name"]).name,
            annotations=tuple(annotations.get(int(image["id"]), [])),
        )
        for image in coco["images"]
    ]


def _manifest_cases(corpus: str, root: Path, manifest: Path) -> list[DetectorCase]:
    return [
        DetectorCase(
            corpus=corpus,
            image_id=int(row["image_id"]),
            image_path=root / row["image_path"],
            annotations=tuple(row["annotations"]),
        )
        for row in _read_jsonl(manifest)
    ]


def _target(annotation: dict) -> Detection:
    x, y, width, height = annotation.get("bbox", annotation.get("bbox_xywh"))
    return Detection(float(x), float(y), float(x + width), float(y + height), 1.0)


def _match(detections: list[Detection], annotations: tuple[dict, ...]) -> dict:
    targets = [_target(row) for row in annotations]
    pairs = sorted(
        (
            (box_iou(detection, target), detection_index, target_index)
            for detection_index, detection in enumerate(detections)
            for target_index, target in enumerate(targets)
        ),
        reverse=True,
    )
    used_detections: set[int] = set()
    used_targets: set[int] = set()
    for iou, detection_index, target_index in pairs:
        if iou < 0.5:
            break
        if detection_index in used_detections or target_index in used_targets:
            continue
        used_detections.add(detection_index)
        used_targets.add(target_index)
    return {
        "matched_count": len(used_targets),
        "false_negative_count": len(targets) - len(used_targets),
        "false_positive_count": len(detections) - len(used_detections),
        "missed_annotation_indices": [
            index for index in range(len(targets)) if index not in used_targets
        ],
    }


def _map_rotation(
    detector: OnnxDetector,
    detections: list[Detection],
    *,
    degrees: int,
    width: int,
    height: int,
) -> list[Detection]:
    return detector._restore_recovery_coordinates(
        detections,
        degrees=degrees,
        image_width=width,
        image_height=height,
    )


def _tile_pass(detector: OnnxDetector, image: Image.Image) -> list[Detection]:
    width, height = image.size
    tile_width = round(width * 0.65)
    tile_height = round(height * 0.65)
    windows = (
        (0, 0, tile_width, tile_height),
        (width - tile_width, 0, width, tile_height),
        (0, height - tile_height, tile_width, height),
        (width - tile_width, height - tile_height, width, height),
    )
    mapped = []
    for x1, y1, x2, y2 in windows:
        tile = image.crop((x1, y1, x2, y2))
        try:
            result, _, _, _ = detector._detect_once(tile)
        finally:
            tile.close()
        mapped.extend(
            Detection(
                item.x1 + x1,
                item.y1 + y1,
                item.x2 + x1,
                item.y2 + y1,
                item.score,
                item.class_id,
            )
            for item in result.detections
        )
    return nms(
        mapped,
        detector.metadata.nms_iou_threshold,
        detector.metadata.nms_containment_threshold,
        detector.metadata.nms_class_aware_containment,
    )


def _view_adds_independent_object(
    detector: OnnxDetector, base: list[Detection], view: list[Detection]
) -> bool:
    policy = detector.crowding_policy
    threshold = policy.rotation_recovery_agreement_iou_threshold
    return len(view) > len(base) and detector._fully_matches_recovery(base, view, threshold)


def _select_low_candidates(
    detector: OnnxDetector,
    candidates: list[Detection],
    base: list[Detection],
) -> list[Detection]:
    selected = nms(
        candidates,
        detector.metadata.nms_iou_threshold,
        detector.metadata.nms_containment_threshold,
        detector.metadata.nms_class_aware_containment,
    )
    threshold = detector.crowding_policy.rotation_recovery_agreement_iou_threshold
    return [
        candidate
        for candidate in selected
        if candidate.score < detector.metadata.score_threshold
        and not any(box_iou(candidate, accepted) >= threshold for accepted in base)
    ]


def _confirmed_low_candidates(
    detector: OnnxDetector,
    base: list[Detection],
    views: dict[str, list[Detection]],
) -> list[dict]:
    selected = {
        name: _select_low_candidates(detector, candidates, base)
        for name, candidates in views.items()
    }
    threshold = detector.crowding_policy.rotation_recovery_agreement_iou_threshold
    confirmed = []
    names = list(selected)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            for left in selected[left_name]:
                for right in selected[right_name]:
                    iou = box_iou(left, right)
                    if iou < threshold:
                        continue
                    representative = left if left.score >= right.score else right
                    if any(
                        box_iou(representative, row["detection"]) >= threshold for row in confirmed
                    ):
                        continue
                    confirmed.append(
                        {
                            "detection": representative,
                            "views": [left_name, right_name],
                            "agreement_iou": iou,
                            "scores": [left.score, right.score],
                        }
                    )
    return confirmed


def _missed_shadow_details(
    base: list[Detection],
    pre_nms: list[Detection],
    low_score: list[Detection],
    annotations: tuple[dict, ...],
) -> list[dict]:
    base_metrics = _match(base, annotations)
    details = []
    for index in base_metrics["missed_annotation_indices"]:
        target = _target(annotations[index])
        best_low = max(low_score, key=lambda row: box_iou(row, target), default=None)
        best_pre = max(pre_nms, key=lambda row: box_iou(row, target), default=None)
        details.append(
            {
                "annotation_index": index,
                "category_id": int(annotations[index]["category_id"]),
                "best_low_score_iou": 0.0 if best_low is None else box_iou(best_low, target),
                "best_low_score": None if best_low is None else best_low.score,
                "best_pre_nms_iou": 0.0 if best_pre is None else box_iou(best_pre, target),
                "best_pre_nms_score": None if best_pre is None else best_pre.score,
            }
        )
    return details


def _evaluate_case(case: DetectorCase, detector: OnnxDetector, jpeg_draft_size: int) -> dict:
    image = decode_image(
        case.image_path.read_bytes(),
        max_bytes=50_000_000,
        max_pixels=50_000_000,
        jpeg_draft_size=jpeg_draft_size,
    )
    original = image
    try:
        base, crowding_candidates, width, height = detector._detect_once(image)
        score_threshold = detector.metadata.score_threshold
        pre_nms = [row for row in crowding_candidates if row.score >= score_threshold]
        standard_nms = nms(pre_nms, detector.metadata.nms_iou_threshold)

        rotated90 = detector._rotate_for_recovery(image, 90)
        try:
            result90, candidates90, _, _ = detector._detect_once(rotated90)
        finally:
            rotated90.close()
        view90 = _map_rotation(
            detector,
            result90.detections,
            degrees=90,
            width=width,
            height=height,
        )
        low90 = _map_rotation(
            detector,
            candidates90,
            degrees=90,
            width=width,
            height=height,
        )
        tensor, _, _ = detector._prepare_detection_tensor(image)
        result180, candidates180, _, _ = detector._detect_prepared_tensor(
            tensor[:, ::-1, ::-1], original_width=width, original_height=height
        )
        view180 = _map_rotation(
            detector,
            result180.detections,
            degrees=180,
            width=width,
            height=height,
        )
        low180 = _map_rotation(
            detector,
            candidates180,
            degrees=180,
            width=width,
            height=height,
        )

        original = restore_original_resolution(image)
        original_result, original_candidates, _, _ = detector._detect_once(original)
        tiled = _tile_pass(detector, original)
    finally:
        if original is not image:
            original.close()
        image.close()

    views = {
        "rotation90": view90,
        "rotation180": view180,
        "original_resolution": original_result.detections,
        "tile_2x2_065": tiled,
    }
    confirming_views = [
        name
        for name, rows in views.items()
        if _view_adds_independent_object(detector, base.detections, rows)
    ]
    confirmed_low = _confirmed_low_candidates(
        detector,
        base.detections,
        {
            "base": crowding_candidates,
            "rotation90": low90,
            "rotation180": low180,
            "original_resolution": original_candidates,
        },
    )
    confirmed_low_detections = [row["detection"] for row in confirmed_low]
    confirmed_low_metrics = _match(confirmed_low_detections, case.annotations)
    return {
        "corpus": case.corpus,
        "image_id": case.image_id,
        "ground_truth_count": len(case.annotations),
        "base": {
            "selected_count": len(base.detections),
            "pre_nms_count": len(pre_nms),
            "standard_iou_nms_count": len(standard_nms),
            "contained_suppression_count": len(standard_nms) - len(base.detections),
            **_match(base.detections, case.annotations),
        },
        "views": {
            name: {
                "selected_count": len(rows),
                "adds_independent_object": name in confirming_views,
                **_match(rows, case.annotations),
            }
            for name, rows in views.items()
        },
        "standard_iou_nms": _match(standard_nms, case.annotations),
        "confirming_view_count": len(confirming_views),
        "confirming_views": confirming_views,
        "two_view_disagreement_trigger": len(confirming_views) >= 2,
        "confirmed_low_candidate_count": len(confirmed_low),
        "confirmed_low_candidate_trigger": bool(confirmed_low),
        "confirmed_low_candidate_metrics": confirmed_low_metrics,
        "confirmed_low_candidates": [
            {
                "bbox_xyxy": [
                    row["detection"].x1,
                    row["detection"].y1,
                    row["detection"].x2,
                    row["detection"].y2,
                ],
                "views": row["views"],
                "agreement_iou": row["agreement_iou"],
                "scores": row["scores"],
            }
            for row in confirmed_low
        ],
        "missed_shadow_candidates": _missed_shadow_details(
            base.detections, pre_nms, crowding_candidates, case.annotations
        ),
        "contained_pairs": [
            {
                "stronger_index": strong_index,
                "candidate_index": weak_index,
                "containment": box_containment(stronger, weak),
                "iou": box_iou(stronger, weak),
            }
            for weak_index, weak in enumerate(pre_nms)
            for strong_index, stronger in enumerate(pre_nms)
            if stronger.score > weak.score
            and box_containment(stronger, weak) >= detector.metadata.nms_containment_threshold
        ],
    }


def _aggregate(rows: list[dict], path: str) -> dict:
    selected = rows
    if path != "base":
        return {
            "false_negative_count": sum(
                row["views"][path]["false_negative_count"] for row in selected
            ),
            "false_positive_count": sum(
                row["views"][path]["false_positive_count"] for row in selected
            ),
        }
    return {
        "false_negative_count": sum(row["base"]["false_negative_count"] for row in selected),
        "false_positive_count": sum(row["base"]["false_positive_count"] for row in selected),
    }


def probe(args: argparse.Namespace) -> tuple[dict, dict]:
    runtime = load_runtime_package_v2(args.runtime)
    if runtime.metadata.detector.ensemble is not None or runtime.metadata.detector_refinement:
        raise ValueError("0.1.7 safety probe requires the packaged single ONNX detector")
    detector = OnnxDetector(
        runtime.detector_path,
        runtime.metadata.detector,
        args.provider,
        args.cuda_dll_dir,
        crowding_policy=runtime.metadata.detector_crowding,
        enable_cuda_graph=False,
    )
    cases = _target_cases(args.target_root)
    cases.extend(_manifest_cases("detector415", args.dataset415_root, args.manifest415))
    cases.extend(_manifest_cases("operational69", args.operational69_root, args.manifest69))
    try:
        rows = []
        for index, case in enumerate(cases, start=1):
            rows.append(
                _evaluate_case(
                    case,
                    detector,
                    runtime.metadata.input.jpeg_draft_size,
                )
            )
            if index % 25 == 0 or index == len(cases):
                print(f"detector safety probe: {index}/{len(cases)}", flush=True)
    finally:
        detector.runner.close()

    corpora = sorted({row["corpus"] for row in rows})
    summary = {
        corpus: {
            "image_count": sum(row["corpus"] == corpus for row in rows),
            "base": _aggregate([row for row in rows if row["corpus"] == corpus], "base"),
            "rotation90": _aggregate(
                [row for row in rows if row["corpus"] == corpus], "rotation90"
            ),
            "rotation180": _aggregate(
                [row for row in rows if row["corpus"] == corpus], "rotation180"
            ),
            "original_resolution": _aggregate(
                [row for row in rows if row["corpus"] == corpus], "original_resolution"
            ),
            "tile_2x2_065": _aggregate(
                [row for row in rows if row["corpus"] == corpus], "tile_2x2_065"
            ),
            "two_view_disagreement_trigger_count": sum(
                row["corpus"] == corpus and row["two_view_disagreement_trigger"] for row in rows
            ),
            "confirmed_low_candidate_trigger_count": sum(
                row["corpus"] == corpus and row["confirmed_low_candidate_trigger"] for row in rows
            ),
        }
        for corpus in corpora
    }
    recovery = {
        "schema_version": "1.0",
        "product_version": runtime.metadata.worker_version,
        "provider": args.provider,
        "score_threshold_unchanged": runtime.metadata.detector.score_threshold,
        "trigger_definition": (
            "At least two independent recovery views contain all selected base boxes "
            "and at least one additional selected box at the packaged threshold."
        ),
        "summary": summary,
        "rows": rows,
    }
    nms_rows = [
        {
            "corpus": row["corpus"],
            "image_id": row["image_id"],
            "pre_nms_count": row["base"]["pre_nms_count"],
            "current_selected_count": row["base"]["selected_count"],
            "standard_iou_nms_count": row["base"]["standard_iou_nms_count"],
            "current_false_negative_count": row["base"]["false_negative_count"],
            "standard_iou_false_negative_count": row["standard_iou_nms"]["false_negative_count"],
            "standard_iou_false_positive_count": row["standard_iou_nms"]["false_positive_count"],
            "contained_pairs": row["contained_pairs"],
            "missed_shadow_candidates": row["missed_shadow_candidates"],
        }
        for row in rows
    ]
    nms_report = {
        "schema_version": "1.0",
        "product_version": runtime.metadata.worker_version,
        "score_threshold_unchanged": runtime.metadata.detector.score_threshold,
        "current_containment_threshold": runtime.metadata.detector.nms_containment_threshold,
        "current_iou_threshold": runtime.metadata.detector.nms_iou_threshold,
        "summary": {
            corpus: {
                "image_count": sum(row["corpus"] == corpus for row in nms_rows),
                "current_false_negative_count": sum(
                    row["current_false_negative_count"]
                    for row in nms_rows
                    if row["corpus"] == corpus
                ),
                "standard_iou_false_negative_count": sum(
                    row["standard_iou_false_negative_count"]
                    for row in nms_rows
                    if row["corpus"] == corpus
                ),
                "standard_iou_false_positive_count": sum(
                    row["standard_iou_false_positive_count"]
                    for row in nms_rows
                    if row["corpus"] == corpus
                ),
                "images_changed_without_containment": sum(
                    row["standard_iou_nms_count"] != row["current_selected_count"]
                    for row in nms_rows
                    if row["corpus"] == corpus
                ),
            }
            for corpus in corpora
        },
        "rows": nms_rows,
    }
    return recovery, nms_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--dataset415-root", type=Path, required=True)
    parser.add_argument("--manifest415", type=Path, required=True)
    parser.add_argument("--operational69-root", type=Path, required=True)
    parser.add_argument("--manifest69", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nms-output", type=Path, required=True)
    args = parser.parse_args()
    recovery, nms_report = probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(recovery, indent=2) + "\n", encoding="utf-8")
    args.nms_output.write_text(json.dumps(nms_report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
