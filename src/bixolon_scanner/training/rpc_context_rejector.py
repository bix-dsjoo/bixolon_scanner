from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .calibration import softmax
from .rpc_data_scale import LEVELS, evaluate_worker_taxonomy
from .rpc_worker_gate import _iou, postprocess_worker_gate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _intersection(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _detector_features(
    records: list[dict[str, Any]],
    raw_predictions: dict[str, dict[str, Any]],
    options: dict[str, Any],
) -> dict[str, list[float]]:
    by_sample: dict[str, list[float]] = {}
    for record in records:
        result = postprocess_worker_gate(
            record,
            raw_predictions[f"{record['source']}:{record['image_id']}"],
            options,
        )
        detections = result["detections"]
        for index in range(len(detections)):
            by_sample[f"val:{record['image_id']}:det{index}"] = _geometry_features(
                detections, float(record["width"]), float(record["height"]), index
            )
    return by_sample


def _geometry_features(
    detections: list[dict[str, Any]], width: float, height: float, index: int
) -> list[float]:
    image_area = width * height
    detection = detections[index]
    box = [float(value) for value in detection["bbox_xyxy"]]
    box_width = max(box[2] - box[0], 1e-6)
    box_height = max(box[3] - box[1], 1e-6)
    area = box_width * box_height
    other_features: list[tuple[float, float, float, float, float]] = []
    for other_index, other_detection in enumerate(detections):
        if other_index == index:
            continue
        other_box = [float(value) for value in other_detection["bbox_xyxy"]]
        other_area = max(
            (other_box[2] - other_box[0]) * (other_box[3] - other_box[1]), 1e-6
        )
        intersection = _intersection(box, other_box)
        other_features.append(
            (
                _iou(box, other_box),
                intersection / min(area, other_area),
                intersection / area,
                intersection / other_area,
                float(other_detection["score"]),
            )
        )
    nearest = max(
        other_features, key=lambda value: value[0], default=(0, 0, 0, 0, 0)
    )
    scores = sorted((float(item["score"]) for item in detections), reverse=True)
    score = float(detection["score"])
    rank = scores.index(score) / max(len(scores) - 1, 1)
    return [
        score,
        area / image_area,
        math.log(box_width / box_height),
        ((box[0] + box[2]) * 0.5) / width,
        ((box[1] + box[3]) * 0.5) / height,
        min(
            box[0] / width,
            box[1] / height,
            (width - box[2]) / width,
            (height - box[3]) / height,
        ),
        float(len(detections)),
        rank,
        nearest[0],
        nearest[1],
        nearest[2],
        nearest[3],
        score - nearest[4],
        float(sum(value[0] >= 0.1 for value in other_features)),
        float(sum(value[0] >= 0.3 for value in other_features)),
        float(sum(value[0] >= 0.5 for value in other_features)),
    ]


def _feature_matrix(
    archive: dict[str, np.ndarray],
    detector_features: dict[str, list[float]],
    temperature: float,
) -> np.ndarray:
    probabilities = softmax(archive["logits"], temperature)
    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    top3_mass = ordered[:, -3:].sum(axis=1)
    sorted_logits = np.sort(archive["logits"], axis=1)
    classifier = np.column_stack(
        [
            confidence,
            margin,
            entropy / math.log(probabilities.shape[1]),
            top3_mass,
            sorted_logits[:, -1],
            sorted_logits[:, -1] - sorted_logits[:, -2],
        ]
    )
    context = np.asarray(
        [detector_features[str(sample_id)] for sample_id in archive["sample_ids"]],
        dtype=np.float64,
    )
    return np.column_stack([context, classifier])


def runtime_context_features(
    detections: list[Any],
    logits: np.ndarray,
    width: int,
    height: int,
    temperature: float,
) -> np.ndarray:
    """Build the exact validation feature contract from live Worker outputs."""
    rows = [
        {
            "bbox_xyxy": [
                float(detection.x1),
                float(detection.y1),
                float(detection.x2),
                float(detection.y2),
            ],
            "score": float(detection.score),
        }
        for detection in detections
    ]
    values = np.asarray(logits, dtype=np.float32)
    if values.shape[0] != len(rows):
        raise ValueError("live detector and classifier item counts differ")
    sample_ids = np.asarray([str(index) for index in range(len(rows))])
    geometry = {
        str(index): _geometry_features(rows, float(width), float(height), index)
        for index in range(len(rows))
    }
    return _feature_matrix(
        {"logits": values, "sample_ids": sample_ids}, geometry, temperature
    ).astype(np.float32)


def _models(seed: int) -> dict[str, Any]:
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def _fit_predict_oof(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(labels), dtype=np.float64)
    for train_index, validation_index in GroupKFold(n_splits=5).split(
        features, labels, groups
    ):
        model.fit(features[train_index], labels[train_index])
        result[validation_index] = model.predict_proba(features[validation_index])[:, 1]
    return result


def _static_masks(
    archive: dict[str, np.ndarray],
    detector_report: dict[str, Any],
    role: str,
) -> dict[str, dict[str, Any]]:
    image_ids = archive["image_ids"].astype(np.int64)
    available_image_ids = set(image_ids.tolist())
    outcomes = [
        row
        for row in detector_report["validation_image_outcomes"]
        if row["role"] == role
        and int(row["image_id"]) in available_image_ids
    ]
    result: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        level_outcomes = [row for row in outcomes if row["level"] == level]
        level_ids = {int(row["image_id"]) for row in level_outcomes}
        recapture_ids = {
            int(row["image_id"])
            for row in level_outcomes
            if row["recapture_reasons"]
        }
        result[level] = {
            "level": np.asarray([int(value) in level_ids for value in image_ids]),
            "normal": np.asarray([int(value) not in recapture_ids for value in image_ids]),
            "ground_truth_count": sum(
                int(row["ground_truth_count"]) for row in level_outcomes
            ),
        }
    return result


def _export_logistic_onnx(model: Any, path: Path, feature_count: int) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    coefficient = classifier.coef_.astype(np.float32).reshape(feature_count, 1)
    scale = scaler.scale_.astype(np.float32).reshape(feature_count, 1)
    mean = scaler.mean_.astype(np.float32).reshape(feature_count, 1)
    weight = coefficient / scale
    bias = classifier.intercept_.astype(np.float32) - (weight * mean).sum(axis=0)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "weight"], ["linear"]),
            helper.make_node("Add", ["linear", "bias"], ["biased"]),
            helper.make_node("Sigmoid", ["biased"], ["quality_score"]),
        ],
        "rpc-context-validator-logistic-v1",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, feature_count])],
        [helper.make_tensor_value_info("quality_score", TensorProto.FLOAT, [None, 1])],
        [
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    exported = helper.make_model(
        graph,
        producer_name="bixolon-rpc-context-validator",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    exported.ir_version = min(exported.ir_version, 10)
    onnx.checker.check_model(exported)
    onnx.save(exported, path)


def _onnx_parity(
    path: Path, features: np.ndarray, expected: np.ndarray, threshold: float
) -> dict[str, float | int]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    observed = session.run(
        ["quality_score"], {"features": features.astype(np.float32)}
    )[0].reshape(-1)
    return {
        "max_abs_error": float(np.max(np.abs(observed - expected))),
        "decision_mismatch_count": int(
            np.count_nonzero((observed >= threshold) != (expected >= threshold))
        ),
    }


def _policy_metrics(
    archive: dict[str, np.ndarray],
    probabilities: np.ndarray,
    quality: np.ndarray,
    masks: dict[str, dict[str, Any]],
    classifier_threshold: float,
    quality_threshold: float,
) -> dict[str, dict[str, float]]:
    targets = archive["targets"].astype(np.int64)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    border = archive["touches_border"].astype(bool)
    matched = targets >= 0
    correct = matched & (predicted == targets)
    report: dict[str, dict[str, float]] = {}
    for level in LEVELS:
        level_mask = masks[level]["level"]
        normal = masks[level]["normal"]
        recapture = level_mask & normal & (
            (border & (confidence < classifier_threshold))
            | (quality < quality_threshold)
        )
        recognition_target = level_mask & normal & ~recapture & matched
        approved = (
            level_mask
            & normal
            & ~recapture
            & (confidence >= classifier_threshold)
        )
        correct_approved = approved & correct
        approved_count = int(approved.sum())
        target_count = int(recognition_target.sum())
        report[level] = {
            "recognition_rate": (
                int(correct_approved.sum()) / target_count if target_count else 0.0
            ),
            "misrecognition_rate": (
                int((approved & ~correct).sum()) / approved_count
                if approved_count
                else 0.0
            ),
            "end_to_end_success_rate": int(correct_approved.sum())
            / int(masks[level]["ground_truth_count"]),
            "segment_recapture_count": int(recapture.sum()),
        }
    return report


def _select_policy(
    archive: dict[str, np.ndarray],
    probabilities: np.ndarray,
    quality: np.ndarray,
    masks: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    confidence = probabilities.max(axis=1)
    classifier_candidates = np.unique(
        np.concatenate(
            [np.quantile(confidence, np.linspace(0, 1, 81)), [0.0, 0.9, 0.99, 0.999]]
        )
    )
    negative_quality = quality[archive["targets"] < 0]
    quality_candidates = np.unique(
        np.concatenate(
            [[0.0], np.quantile(negative_quality, np.linspace(0, 1, 41))]
        )
    )
    feasible: list[dict[str, Any]] = []
    for quality_threshold in quality_candidates:
        for classifier_threshold in classifier_candidates:
            report = _policy_metrics(
                archive,
                probabilities,
                quality,
                masks,
                float(classifier_threshold),
                float(quality_threshold),
            )
            if all(
                report[level]["recognition_rate"] >= 0.99
                and report[level]["misrecognition_rate"] <= 0.005
                for level in LEVELS
            ):
                feasible.append(
                    {
                        "classifier_threshold": float(classifier_threshold),
                        "quality_threshold": float(quality_threshold),
                        "difficulty": report,
                    }
                )
    return max(
        feasible,
        key=lambda row: (
            min(
                row["difficulty"][level]["end_to_end_success_rate"]
                for level in LEVELS
            ),
            -sum(
                row["difficulty"][level]["segment_recapture_count"]
                for level in LEVELS
            ),
        ),
        default=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = args.output_dir / "runs" / "full" / f"seed{args.seed}"
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    temperature = float(calibration["temperature"])
    archives: dict[str, dict[str, np.ndarray]] = {}
    for name, filename in (
        ("calibration", "partial_calibration_predictions.npz"),
        ("selection", "selection_predictions.npz"),
    ):
        loaded = np.load(run_dir / filename)
        archives[name] = {key: loaded[key] for key in loaded.files}
    detector_dir = args.output_dir / "detector"
    records = _read_jsonl(detector_dir / "manifest" / "manifest.jsonl")
    raw_predictions = {
        str(row["sample_key"]): row
        for row in _read_jsonl(detector_dir / "predictions" / "val_oof.jsonl")
    }
    threshold = json.loads(
        (detector_dir / "threshold.json").read_text(encoding="utf-8")
    )["selected_score_threshold"]
    detector_options = dict(config["detector"], score_threshold=threshold)
    detector_features = _detector_features(records, raw_predictions, detector_options)
    features = {
        name: _feature_matrix(archive, detector_features, temperature)
        for name, archive in archives.items()
    }
    calibration_labels = (archives["calibration"]["targets"] >= 0).astype(np.int64)
    groups = archives["calibration"]["groups"].astype(str)
    detector_report = json.loads(
        (args.output_dir / "prepared" / "worker_gate_report.json").read_text(
            encoding="utf-8"
        )
    )
    calibration_probabilities = softmax(
        archives["calibration"]["logits"], temperature
    )
    calibration_masks = _static_masks(
        archives["calibration"], detector_report, "calibration"
    )
    output_dir = run_dir / "context-rejector"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for name, model in _models(args.seed).items():
        quality_oof = _fit_predict_oof(
            model, features["calibration"], calibration_labels, groups
        )
        policy = _select_policy(
            archives["calibration"],
            calibration_probabilities,
            quality_oof,
            calibration_masks,
        )
        model.fit(features["calibration"], calibration_labels)
        selection_quality = model.predict_proba(features["selection"])[:, 1]
        model_report: dict[str, Any] = {
            "oof_roc_auc": roc_auc_score(calibration_labels, quality_oof),
            "oof_average_precision": average_precision_score(
                calibration_labels, quality_oof
            ),
            "policy": policy,
        }
        if policy is not None:
            policy_calibration = dict(
                calibration,
                approval_threshold=policy["classifier_threshold"],
                risk_control_satisfied=True,
            )
            model_report["selection"] = evaluate_worker_taxonomy(
                archives["selection"],
                policy_calibration,
                detector_report,
                role="selection",
                segment_quality_scores=selection_quality,
                segment_quality_threshold=policy["quality_threshold"],
            )
        reports[name] = model_report
        joblib.dump(model, output_dir / f"{name}.joblib")
        np.savez_compressed(
            output_dir / f"{name}_scores.npz",
            calibration_oof=quality_oof,
            selection=selection_quality,
        )
        if name == "logistic":
            onnx_path = output_dir / "logistic.onnx"
            _export_logistic_onnx(model, onnx_path, features["calibration"].shape[1])
            model_report["onnx_cpu_parity"] = _onnx_parity(
                onnx_path,
                features["selection"],
                selection_quality,
                float(policy["quality_threshold"]),
            )
    result = {"feature_count": int(features["calibration"].shape[1]), "models": reports}
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    logistic_policy = reports["logistic"]["policy"]
    if logistic_policy is None:
        raise RuntimeError("logistic context validator has no feasible calibration policy")
    candidate_inputs = output_dir / "validation-candidate-inputs"
    candidate_inputs.mkdir(parents=True, exist_ok=True)
    detector_threshold = json.loads(
        (detector_dir / "threshold.json").read_text(encoding="utf-8")
    )
    (candidate_inputs / "detector-evaluation.json").write_text(
        json.dumps(
            {
                "selected_score_threshold": detector_threshold[
                    "selected_score_threshold"
                ],
                "nms_iou_threshold": detector_options["nms_iou_threshold"],
                "target_recall": detector_threshold["target_recall"],
                "target_recall_satisfied": detector_threshold[
                    "target_recall_satisfied"
                ],
                "metrics": detector_threshold["calibration_metrics"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    candidate_calibration = dict(
        calibration,
        approval_threshold=logistic_policy["classifier_threshold"],
        risk_control_satisfied=True,
        sample_count=int(calibration["matched_count"])
        + int(calibration["unmatched_detector_count"]),
    )
    (candidate_inputs / "classifier-calibration.json").write_text(
        json.dumps(candidate_calibration, indent=2), encoding="utf-8"
    )
    experiment = json.loads(
        (args.output_dir / "prepared" / "experiment.json").read_text(
            encoding="utf-8"
        )
    )
    labels = [
        {"class_id": str(row["id"]), "class_name": str(row["name"])}
        for row in sorted(experiment["categories"], key=lambda row: int(row["id"]))
    ]
    (candidate_inputs / "manifest-metadata.json").write_text(
        json.dumps(
            {
                "dataset_version": "rpc2019-validation-context-logistic-v1",
                "labels": labels,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (candidate_inputs / "context-policy.json").write_text(
        json.dumps(
            {
                "contract": "rpc-context-validator-logistic-v1",
                "feature_count": int(features["calibration"].shape[1]),
                "classifier_threshold": logistic_policy["classifier_threshold"],
                "quality_threshold": logistic_policy["quality_threshold"],
                "onnx_cpu_parity": reports["logistic"]["onnx_cpu_parity"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
