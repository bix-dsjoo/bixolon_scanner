from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..contracts.catalog import sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2
from ..runtime.detector_v2 import FixedEnsembleOnnxDetector
from ..runtime.imaging import decode_image
from ..runtime.proposal_selection import box_iou


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256_index(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    return {sha256_file(path): path for path in root.iterdir() if path.is_file()}


def _matches(boxes: list[list[float]], annotations: list[dict]) -> list[tuple[int, int]]:
    candidates = []
    for prediction_index, prediction in enumerate(boxes):
        for annotation_index, annotation in enumerate(annotations):
            x, y, width, height = annotation.get("bbox_xywh", annotation.get("bbox"))
            overlap = box_iou(prediction, [x, y, x + width, y + height])
            if overlap >= 0.5:
                candidates.append((overlap, prediction_index, annotation_index))
    selected = []
    used_predictions: set[int] = set()
    used_annotations: set[int] = set()
    for _overlap, prediction_index, annotation_index in sorted(candidates, reverse=True):
        if prediction_index in used_predictions or annotation_index in used_annotations:
            continue
        used_predictions.add(prediction_index)
        used_annotations.add(annotation_index)
        selected.append((prediction_index, annotation_index))
    return selected


def extract(args: argparse.Namespace) -> dict:
    runtime = load_runtime_package_v2(args.runtime)
    detector = FixedEnsembleOnnxDetector(runtime, args.provider, args.cuda_dll_dir)
    sha_index = _sha256_index(args.image_sha_root)
    feature_rows = []
    labels = []
    folds = []
    image_ids = []
    matched_count = 0
    target_count = 0
    rows = _jsonl(args.manifest)
    try:
        detector.warmup()
        for position, row in enumerate(rows, start=1):
            image_path = args.dataset_root / row["image_path"]
            if not image_path.is_file() and sha_index:
                image_path = sha_index.get(row["image_sha256"], image_path)
            if sha256_file(image_path) != row["image_sha256"]:
                raise ValueError("class-fusion input checksum mismatch")
            image = decode_image(
                image_path.read_bytes(),
                max_bytes=50_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
            )
            try:
                selected, features = detector.predict_member_class_features(image)
            finally:
                image.close()
            annotations = row.get("annotations", [])
            matches = _matches(selected["boxes_xyxy"], annotations)
            target_count += len(annotations)
            matched_count += len(matches)
            for prediction_index, annotation_index in matches:
                feature_rows.append(features[prediction_index])
                labels.append(int(annotations[annotation_index]["category_id"]) - 1)
                folds.append(int(row.get("fold", int(row["image_id"]) % 3)))
                image_ids.append(int(row["image_id"]))
            if position % 50 == 0 or position == len(rows):
                print(f"class fusion extract: {position}/{len(rows)}", flush=True)
    finally:
        detector.close()
    matrix = np.asarray(feature_rows, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=matrix,
        labels=np.asarray(labels, dtype=np.int64),
        folds=np.asarray(folds, dtype=np.int64),
        image_ids=np.asarray(image_ids, dtype=np.int64),
    )
    return {
        "schema_version": "1.0",
        "evaluation": "yolo_free_member_logit_feature_extraction",
        "image_count": len(rows),
        "target_count": target_count,
        "matched_count": matched_count,
        "feature_count": len(matrix),
        "feature_dimension": matrix.shape[1],
        "output": args.output.resolve().as_posix(),
        "output_sha256": sha256_file(args.output),
    }


def _fit(features: np.ndarray, labels: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000, random_state=20260831),
    )
    model.fit(features, labels)
    return model


def evaluate(args: argparse.Namespace) -> dict:
    development = np.load(args.development)
    operational = np.load(args.operational)
    features = development["features"]
    labels = development["labels"]
    folds = development["folds"]
    probabilities = np.zeros((len(labels), 20), dtype=np.float64)
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        model = _fit(features[train], labels[train])
        values = model.predict_proba(features[test])
        probabilities[np.ix_(test, model.classes_.astype(int))] = values
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    wrong = predicted != labels
    threshold = math.nextafter(float(np.max(confidence[wrong])), math.inf) if np.any(wrong) else 0.0
    accepted = confidence >= threshold

    final_model = _fit(features, labels)
    operational_probabilities = final_model.predict_proba(operational["features"])
    operational_predicted = final_model.classes_[np.argmax(operational_probabilities, axis=1)]
    operational_confidence = np.max(operational_probabilities, axis=1)
    operational_accepted = operational_confidence >= threshold
    operational_labels = operational["labels"]
    operational_wrong = operational_predicted != operational_labels
    return {
        "schema_version": "1.0",
        "evaluation": "yolo_free_selective_member_logit_fusion",
        "model": "standard_scaler_plus_multinomial_logistic_regression",
        "selection": "three-fold out-of-fold zero-error global confidence threshold",
        "development": {
            "count": len(labels),
            "top1_correct_count": int(np.sum(~wrong)),
            "top1_accuracy": float(np.mean(~wrong)),
            "selected_confidence_threshold": threshold,
            "accepted_count": int(np.sum(accepted)),
            "accepted_wrong_count": int(np.sum(wrong & accepted)),
            "accepted_coverage": float(np.mean(accepted)),
        },
        "operational": {
            "count": len(operational_labels),
            "top1_correct_count": int(np.sum(~operational_wrong)),
            "top1_accuracy": float(np.mean(~operational_wrong)),
            "accepted_count": int(np.sum(operational_accepted)),
            "accepted_wrong_count": int(np.sum(operational_wrong & operational_accepted)),
            "accepted_coverage": float(np.mean(operational_accepted)),
            "accepted_failures": [
                {
                    "image_id": int(operational["image_ids"][index]),
                    "expected_class_index": int(operational_labels[index]),
                    "predicted_class_index": int(operational_predicted[index]),
                    "confidence": float(operational_confidence[index]),
                }
                for index in np.flatnonzero(operational_wrong & operational_accepted)
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--runtime", type=Path, required=True)
    extract_parser.add_argument("--cuda-dll-dir", type=Path)
    extract_parser.add_argument("--provider", choices=("cpu", "cuda", "openvino"), default="cuda")
    extract_parser.add_argument("--dataset-root", type=Path, required=True)
    extract_parser.add_argument("--image-sha-root", type=Path)
    extract_parser.add_argument("--manifest", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--report", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--development", type=Path, required=True)
    evaluate_parser.add_argument("--operational", type=Path, required=True)
    evaluate_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = extract(args) if args.command == "extract" else evaluate(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
