from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...pipeline.ports import Detection
from ...runtime.onnx import (
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)
from ...training.classifier_allowlist import sha256_file
from ...training.models import require_torch
from .bix_classifier_cascade import (
    MEAN,
    STD,
    _build_vit,
    _cascade_metrics,
    _classification_metrics,
    route_predictions,
)
from .bix_classifier_cascade_calibration import _load_convnext


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _difficulty(image_path: str) -> str:
    value = Path(image_path).parent.name
    if value not in {"easy", "medium", "hard"}:
        raise ValueError(f"unsupported difficulty in image path: {image_path}")
    return value


def _xywh_to_xyxy(box: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    return x, y, x + width, y + height


def _load_gt_entries(annotation_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload["images"]}
    category_names = {int(row["id"]): str(row["name"]) for row in payload["categories"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)
    entries: list[dict[str, Any]] = []
    for image_id in sorted(images):
        image = images[image_id]
        image_path = str(Path(str(image["file_name"])).as_posix()).removeprefix("../")
        annotations = sorted(by_image[image_id], key=lambda row: int(row["id"]))
        boxes = [_xywh_to_xyxy(row["bbox"]) for row in annotations]
        for index, annotation in enumerate(annotations):
            category_id = int(annotation["category_id"])
            entries.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "difficulty": _difficulty(image_path),
                    "annotation_id": int(annotation["id"]),
                    "target": category_id - 1,
                    "target_class_id": f"bread_{category_id:02d}",
                    "target_class_name": category_names[category_id],
                    "box_index": index,
                    "boxes": boxes,
                    "bbox_xyxy": boxes[index],
                    "box_source": "ground_truth",
                }
            )
    return entries


def _load_detector_entries(trace_path: Path, image_paths: dict[int, str]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if int(row["image_id"]) in image_paths]
    if len(rows) != len(image_paths):
        raise ValueError(
            f"detector trace covers {len(rows)} of {len(image_paths)} evaluation images"
        )
    entries: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: int(value["image_id"])):
        image_id = int(row["image_id"])
        if row["status"] != "SEGMENTATION":
            continue
        segmentations = row["decision"]["segmentations"]
        diagnostics = row["matched_classifier_diagnostics"]
        if len(segmentations) != len(diagnostics):
            raise ValueError(f"trace segmentation/diagnostic mismatch for image {image_id}")
        boxes = []
        for segmentation in segmentations:
            box = segmentation["bbox"]
            boxes.append(
                (
                    float(box["x"]),
                    float(box["y"]),
                    float(box["x"] + box["width"]),
                    float(box["y"] + box["height"]),
                )
            )
        image_path = image_paths[image_id]
        for index, (segmentation, diagnostic) in enumerate(zip(segmentations, diagnostics)):
            class_id = str(diagnostic["target_class_id"])
            category_id = int(class_id.removeprefix("bread_"))
            entries.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "difficulty": _difficulty(image_path),
                    "annotation_id": None,
                    "target": category_id - 1,
                    "target_class_id": class_id,
                    "target_class_name": None,
                    "box_index": index,
                    "boxes": boxes,
                    "bbox_xyxy": boxes[index],
                    "box_source": "detector_trace",
                    "detector_segmentation_id": segmentation["segmentation_id"],
                }
            )
    return entries


def _tensor(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    box_index: int,
    *,
    size: int,
    margin_ratio: float,
    distance_bias: float,
) -> np.ndarray:
    detections = [Detection(*box, score=1.0) for box in boxes]
    crop_box = classifier_crop_box(
        detections[box_index],
        image.width,
        image.height,
        margin_ratio=margin_ratio,
        crop_mode="box_resize",
    )
    value = prepare_rgb(
        image.crop(crop_box),
        (size, size),
        MEAN,
        STD,
        reducing_gap=1.0,
    )
    mask = classifier_neighbor_ownership_mask(
        detections,
        box_index,
        image_width=image.width,
        image_height=image.height,
        output_size=size,
        margin_ratio=margin_ratio,
        distance_bias=distance_bias,
        shared_scale=False,
    )
    return apply_classifier_background_masks(value[None], mask[None])[0]


def _predict_scope(
    *,
    entries: list[dict[str, Any]],
    image_root: Path,
    convnext: Any,
    vit: Any,
    vit_weight: np.ndarray,
    vit_bias: np.ndarray,
    batch_size: int,
    margin_ratio: float,
    distance_bias: float,
    cpu: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    torch = require_torch()
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[int(entry["image_id"])].append(entry)
    outputs_192: list[np.ndarray] = []
    outputs_224: list[np.ndarray] = []
    outputs_vit: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    tensors_192: list[np.ndarray] = []
    tensors_224: list[np.ndarray] = []
    tensors_160: list[np.ndarray] = []
    pending_labels: list[int] = []
    started = time.perf_counter()

    vit_weight_tensor = torch.from_numpy(vit_weight).to(device)
    vit_bias_tensor = torch.from_numpy(vit_bias).to(device)

    def flush() -> None:
        if not pending_labels:
            return
        batch_192 = torch.from_numpy(np.stack(tensors_192)).to(device)
        batch_224 = torch.from_numpy(np.stack(tensors_224)).to(device)
        batch_160 = torch.from_numpy(np.stack(tensors_160)).to(device)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs_192.append(convnext(batch_192).float().cpu().numpy())
            outputs_224.append(convnext(batch_224).float().cpu().numpy())
            feature_vit = vit.extract_features(batch_160).float()
            outputs_vit.append(
                (feature_vit @ vit_weight_tensor + vit_bias_tensor).float().cpu().numpy()
            )
        labels.append(np.asarray(pending_labels, dtype=np.int64))
        tensors_192.clear()
        tensors_224.clear()
        tensors_160.clear()
        pending_labels.clear()

    for image_id in sorted(grouped):
        image_entries = grouped[image_id]
        with Image.open(image_root / str(image_entries[0]["image_path"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            for entry in image_entries:
                boxes = entry["boxes"]
                index = int(entry["box_index"])
                tensors_192.append(
                    _tensor(
                        image,
                        boxes,
                        index,
                        size=192,
                        margin_ratio=margin_ratio,
                        distance_bias=distance_bias,
                    )
                )
                tensors_224.append(
                    _tensor(
                        image,
                        boxes,
                        index,
                        size=224,
                        margin_ratio=margin_ratio,
                        distance_bias=distance_bias,
                    )
                )
                tensors_160.append(
                    _tensor(
                        image,
                        boxes,
                        index,
                        size=160,
                        margin_ratio=margin_ratio,
                        distance_bias=distance_bias,
                    )
                )
                pending_labels.append(int(entry["target"]))
                if len(pending_labels) >= batch_size:
                    flush()
        print(
            json.dumps(
                {
                    "image_id": image_id,
                    "scope_objects_complete": sum(
                        len(value) for key, value in grouped.items() if key <= image_id
                    ),
                }
            ),
            flush=True,
        )
    flush()
    return (
        np.concatenate(outputs_192),
        np.concatenate(outputs_224),
        np.concatenate(outputs_vit),
        np.concatenate(labels),
        time.perf_counter() - started,
    )


def _slice_metrics(
    result: dict[str, np.ndarray],
    labels: np.ndarray,
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for difficulty in ("easy", "medium", "hard"):
        mask = np.asarray([entry["difficulty"] == difficulty for entry in entries])
        output[difficulty] = _cascade_metrics(
            {key: value[mask] for key, value in result.items()}, labels[mask]
        )
    return output


def _prediction_rows(
    *,
    name: str,
    entries: list[dict[str, Any]],
    labels: np.ndarray,
    logits_192: np.ndarray,
    logits_224: np.ndarray,
    logits_vit: np.ndarray,
    result: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    ranking = np.argsort(-result["scores"], axis=1, kind="stable")
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        top3 = [f"bread_{int(value) + 1:02d}" for value in ranking[index, :3]]
        rows.append(
            {
                "scope": name,
                "image_id": int(entry["image_id"]),
                "image_path": entry["image_path"],
                "difficulty": entry["difficulty"],
                "annotation_id": entry["annotation_id"],
                "bbox_xyxy": [float(value) for value in entry["bbox_xyxy"]],
                "box_source": entry["box_source"],
                "target_class_id": entry["target_class_id"],
                "predicted_class_id": f"bread_{int(result['predictions'][index]) + 1:02d}",
                "top3": top3,
                "top1_correct": bool(result["predictions"][index] == labels[index]),
                "top3_hit": entry["target_class_id"] in top3,
                "approved": bool(result["approved"][index]),
                "fallback_224": bool(result["fallback"][index]),
                "verifier_160": bool(result["verifier"][index]),
                "logits_192_top1": f"bread_{int(logits_192[index].argmax()) + 1:02d}",
                "logits_224_top1": f"bread_{int(logits_224[index].argmax()) + 1:02d}",
                "logits_vit_top1": f"bread_{int(logits_vit[index].argmax()) + 1:02d}",
            }
        )
    return rows


def _evaluate_scope(
    *,
    name: str,
    entries: list[dict[str, Any]],
    image_root: Path,
    convnext: Any,
    vit: Any,
    vit_weight: np.ndarray,
    vit_bias: np.ndarray,
    policy: dict[str, float],
    batch_size: int,
    margin_ratio: float,
    distance_bias: float,
    cpu: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logits_192, logits_224, logits_vit, labels, duration = _predict_scope(
        entries=entries,
        image_root=image_root,
        convnext=convnext,
        vit=vit,
        vit_weight=vit_weight,
        vit_bias=vit_bias,
        batch_size=batch_size,
        margin_ratio=margin_ratio,
        distance_bias=distance_bias,
        cpu=cpu,
    )
    result = route_predictions(logits_192, logits_224, logits_vit, **policy)
    report = {
        "name": name,
        "box_source": entries[0]["box_source"],
        "preprocessing": {
            "crop_mode": "box_resize",
            "crop_margin_ratio": margin_ratio,
            "neighbor_mask": True,
            "neighbor_distance_bias": distance_bias,
            "neighbor_shared_scale": False,
        },
        "image_count": len({int(entry["image_id"]) for entry in entries}),
        "object_count": len(entries),
        "duration_seconds": duration,
        "convnext_192": _classification_metrics(logits_192, labels),
        "convnext_224": _classification_metrics(logits_224, labels),
        "vit_160": _classification_metrics(logits_vit, labels),
        "selective_cascade": _cascade_metrics(result, labels),
        "selective_cascade_by_difficulty": _slice_metrics(result, labels, entries),
    }
    rows = _prediction_rows(
        name=name,
        entries=entries,
        labels=labels,
        logits_192=logits_192,
        logits_224=logits_224,
        logits_vit=logits_vit,
        result=result,
    )
    return report, rows


def run_evaluation(
    *,
    image_root: Path,
    annotation_path: Path,
    detector_trace: Path,
    convnext_weights: Path,
    convnext_checkpoint: Path,
    vit_weights: Path,
    vit_head: Path,
    package_metadata: Path,
    output_dir: Path,
    batch_size: int,
    cpu: bool,
) -> dict[str, Any]:
    image_root = image_root.resolve()
    if image_root.name != "bread_dataset":
        raise ValueError("image_root must be the bread_dataset directory")
    gt_entries = _load_gt_entries(annotation_path)
    image_paths = {int(entry["image_id"]): str(entry["image_path"]) for entry in gt_entries}
    detector_entries = _load_detector_entries(detector_trace, image_paths)
    metadata = json.loads(package_metadata.read_text(encoding="utf-8"))
    policy = {key: float(value) for key, value in metadata["calibrated_routing"].items()}
    head = np.load(vit_head)
    convnext = _load_convnext(convnext_weights, convnext_checkpoint, cpu=cpu)
    vit = _build_vit(vit_weights)
    torch = require_torch()
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    vit = vit.to(device).eval()
    scopes = (
        ("gt_boxes_training_preprocess", gt_entries, 0.05, 0.0),
        ("gt_boxes_runtime_preprocess", gt_entries, 0.0, -0.1),
        ("detector_boxes_runtime_preprocess", detector_entries, 0.0, -0.1),
    )
    scope_reports: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for name, entries, margin_ratio, distance_bias in scopes:
        print(json.dumps({"scope_started": name, "objects": len(entries)}), flush=True)
        scope_report, rows = _evaluate_scope(
            name=name,
            entries=entries,
            image_root=image_root,
            convnext=convnext,
            vit=vit,
            vit_weight=head["weight"],
            vit_bias=head["bias"],
            policy=policy,
            batch_size=batch_size,
            margin_ratio=margin_ratio,
            distance_bias=distance_bias,
            cpu=cpu,
        )
        scope_reports[name] = scope_report
        prediction_rows.extend(rows)
    trace_rows = [
        json.loads(line) for line in detector_trace.read_text(encoding="utf-8").splitlines()
    ]
    trace_rows = [row for row in trace_rows if int(row["image_id"]) in image_paths]
    detector_counts = {
        "image_count": len(trace_rows),
        "segmentation_image_count": sum(row["status"] == "SEGMENTATION" for row in trace_rows),
        "image_recapture_count": sum(row["status"] == "IMAGE_RECAPTURE" for row in trace_rows),
        "ground_truth_count": sum(int(row["ground_truth_count"]) for row in trace_rows),
        "prediction_count": sum(int(row["prediction_count"]) for row in trace_rows),
        "matched_count": sum(int(row["matched_count"]) for row in trace_rows),
        "false_negative_count": sum(int(row["false_negative_count"]) for row in trace_rows),
        "false_positive_count": sum(int(row["false_positive_count"]) for row in trace_rows),
    }
    report = {
        "schema_version": "1.0",
        "evaluation": "bix_classifier_cascade_on_multi_object_scenes",
        "evaluation_role": "development_diagnostic_not_threshold_selection",
        "dataset": {
            "image_root": str(image_root),
            "annotation_path": str(annotation_path.resolve()),
            "annotation_sha256": sha256_file(annotation_path),
            "image_count": len(image_paths),
            "object_count": len(gt_entries),
            "difficulty_image_counts": dict(
                Counter(_difficulty(path) for path in image_paths.values())
            ),
            "independence_limit": (
                "Annotations identify bread_xx:original_item physical items; this is a held-out "
                "multi-scene composition/view diagnostic, not an independent product/store test."
            ),
        },
        "model": {
            "convnext_checkpoint": str(convnext_checkpoint.resolve()),
            "convnext_checkpoint_sha256": sha256_file(convnext_checkpoint),
            "vit_head": str(vit_head.resolve()),
            "vit_head_sha256": sha256_file(vit_head),
            "calibrated_routing": policy,
            "runtime_activated": False,
        },
        "detector_trace": {
            "path": str(detector_trace.resolve()),
            "sha256": sha256_file(detector_trace),
            "counts": detector_counts,
            "note": "Reused fixed 0.1.13 detector trace; detector was not rerun.",
        },
        "scopes": scope_reports,
        "interpretation_guardrail": (
            "GT-box results isolate classifier behavior. Detector-box results combine the fixed "
            "detector ROI geometry with the new PyTorch classifier, but are not ONNX Runtime parity."
        ),
    }
    _write_json(output_dir / "report.json", report)
    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the BIX classifier on multi-object scenes"
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--detector-trace", type=Path, required=True)
    parser.add_argument("--convnext-weights", type=Path, required=True)
    parser.add_argument("--convnext-checkpoint", type=Path, required=True)
    parser.add_argument("--vit-weights", type=Path, required=True)
    parser.add_argument("--vit-head", type=Path, required=True)
    parser.add_argument("--package-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    report = run_evaluation(
        image_root=args.image_root,
        annotation_path=args.annotations,
        detector_trace=args.detector_trace,
        convnext_weights=args.convnext_weights,
        convnext_checkpoint=args.convnext_checkpoint,
        vit_weights=args.vit_weights,
        vit_head=args.vit_head,
        package_metadata=args.package_metadata,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        cpu=args.cpu,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
