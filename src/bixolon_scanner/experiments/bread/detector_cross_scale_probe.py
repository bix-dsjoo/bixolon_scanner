from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...contracts.runtime_package_v2 import load_runtime_package_v2
from ...pipeline.ports import Detection
from ...runtime.detector_v2 import CrossScaleOnnxDetector, hierarchical_containment_nms
from ...runtime.imaging import decode_image
from ...runtime.onnx import box_iou


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _match(detections: list[Detection], annotations: list[dict]) -> tuple[int, int]:
    targets = [
        Detection(x, y, x + width, y + height, 1.0)
        for x, y, width, height in (row["bbox_xywh"] for row in annotations)
    ]
    candidates = sorted(
        (
            (box_iou(detection, target), detection_index, target_index)
            for detection_index, detection in enumerate(detections)
            for target_index, target in enumerate(targets)
        ),
        reverse=True,
    )
    used_detections = set()
    used_targets = set()
    for iou, detection_index, target_index in candidates:
        if iou < 0.5:
            break
        if detection_index not in used_detections and target_index not in used_targets:
            used_detections.add(detection_index)
            used_targets.add(target_index)
    return len(targets) - len(used_targets), len(detections) - len(used_detections)


def _average(
    primary: list[Detection], recovery: list[Detection], threshold: float
) -> list[Detection]:
    candidates = sorted(
        (
            (box_iou(left, right), left_index, right_index)
            for left_index, left in enumerate(primary)
            for right_index, right in enumerate(recovery)
        ),
        reverse=True,
    )
    left_used = set()
    right_used = set()
    output = []
    for iou, left_index, right_index in candidates:
        if iou < threshold:
            break
        if left_index in left_used or right_index in right_used:
            continue
        left = primary[left_index]
        right = recovery[right_index]
        total = left.score + right.score
        output.append(
            Detection(
                (left.x1 * left.score + right.x1 * right.score) / total,
                (left.y1 * left.score + right.y1 * right.score) / total,
                (left.x2 * left.score + right.x2 * right.score) / total,
                (left.y2 * left.score + right.y2 * right.score) / total,
                max(left.score, right.score),
                left.class_id,
            )
        )
        left_used.add(left_index)
        right_used.add(right_index)
    output.extend(row for index, row in enumerate(primary) if index not in left_used)
    output.extend(row for index, row in enumerate(recovery) if index not in right_used)
    return output


def _summarize(rows: list[dict], strategy: str) -> dict:
    details = [
        {
            "image_id": row["image_id"],
            "false_negative_count": _match(row[strategy], row["annotations"])[0],
            "false_positive_count": _match(row[strategy], row["annotations"])[1],
        }
        for row in rows
    ]
    return {
        "image_count": len(rows),
        "false_negative_count": sum(row["false_negative_count"] for row in details),
        "false_positive_count": sum(row["false_positive_count"] for row in details),
        "false_negative_image_count": sum(row["false_negative_count"] > 0 for row in details),
        "false_positive_image_count": sum(row["false_positive_count"] > 0 for row in details),
        "error_images": [
            row for row in details if row["false_negative_count"] or row["false_positive_count"]
        ],
    }


def run(args: argparse.Namespace) -> dict:
    runtime = load_runtime_package_v2(args.runtime)
    detector = CrossScaleOnnxDetector(runtime, args.provider, args.cuda_dll_dir)
    refinement = detector.refinement_metadata
    rows = []
    for record in _jsonl(args.manifest):
        image = decode_image(
            (args.dataset_root / record["image_path"]).read_bytes(),
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
        )
        try:
            primary, _ = detector._run(
                image,
                detector.primary,
                input_size=detector.primary_metadata.input_size,
                score_threshold=detector.primary_metadata.score_threshold,
                containment_threshold=float(detector.primary_metadata.nms_containment_threshold),
                group_minimum=2,
            )
            recovery, _ = detector._run(
                image,
                detector.refinement,
                input_size=refinement.input_size,
                score_threshold=refinement.score_threshold,
                containment_threshold=refinement.containment_threshold,
                group_minimum=refinement.group_minimum,
            )
        finally:
            image.close()
        row = {
            "image_id": int(record["image_id"]),
            "annotations": record["annotations"],
            "primary": primary,
            "recovery": recovery,
            "current_disagreement": not detector._fully_agree(
                primary, recovery, refinement.agreement_iou_threshold
            ),
        }
        for threshold in args.average_match_thresholds:
            averaged = _average(primary, recovery, threshold)
            for nms_threshold in args.nms_iou_thresholds:
                name = f"average_{threshold:.2f}_nms_{nms_threshold:.2f}"
                row[name] = hierarchical_containment_nms(
                    averaged,
                    iou_threshold=nms_threshold,
                    containment_threshold=refinement.containment_threshold,
                    group_minimum=refinement.group_minimum,
                )
        rows.append(row)
    strategy_names = ["primary", "recovery"] + sorted(
        key for key in rows[0] if key.startswith("average_")
    )
    strategies = {name: _summarize(rows, name) for name in strategy_names}
    selected = min(
        strategies.items(),
        key=lambda item: (
            item[1]["false_negative_count"],
            item[1]["false_positive_count"],
            len(item[1]["error_images"]),
        ),
    )
    disagreement_ids = [row["image_id"] for row in rows if row["current_disagreement"]]
    report = {
        "schema_version": "2.0",
        "candidate_id": "detector-cross-scale-policy-probe",
        "evidence_role": "development_probe",
        "promotion_evidence": False,
        "current_disagreement_image_count": len(disagreement_ids),
        "current_disagreement_image_ids": disagreement_ids,
        "selected": {"strategy": selected[0], **selected[1]},
        "strategies": strategies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["selected"], indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe cross-scale detector fusion policies")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument(
        "--average-match-thresholds", type=float, nargs="+", default=(0.3, 0.4, 0.5, 0.6)
    )
    parser.add_argument(
        "--nms-iou-thresholds", type=float, nargs="+", default=(0.3, 0.4, 0.5, 0.6, 0.7)
    )
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
