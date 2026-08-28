from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...contracts.catalog import sha256_file
from ...runtime.onnx import prepare_rgb

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

_CUDA_DLL_DIRECTORY: object | None = None
_CUDA_DLLS: list[object] = []


def configure_cuda_runtime(provider: str, cuda_dll_dir: Path | None) -> None:
    """Load packaged CUDA dependencies and keep their handles alive for the experiment."""

    global _CUDA_DLL_DIRECTORY
    if provider != "cuda":
        return
    if cuda_dll_dir is None:
        raise ValueError("--cuda-dll-dir is required for CUDA experiments")
    resolved = cuda_dll_dir.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if hasattr(os, "add_dll_directory"):
        _CUDA_DLL_DIRECTORY = os.add_dll_directory(str(resolved))
    if os.name != "nt":
        import onnxruntime as ort

        ort.preload_dlls(directory=str(resolved))
        return
    dependency_order = (
        "cudart64_13.dll",
        "cublasLt64_13.dll",
        "cublas64_13.dll",
        "cufft64_12.dll",
        "nvJitLink_130_0.dll",
        "nvrtc-builtins64_130.dll",
        "nvrtc64_130_0.dll",
        "zlibwapi.dll",
        "cudnn64_9.dll",
        "cudnn_ops64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn_heuristic64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
    )
    for filename in dependency_order:
        path = resolved / filename
        if path.is_file():
            _CUDA_DLLS.append(ctypes.WinDLL(str(path)))


def create_ort_session(model: str | bytes, provider: str):
    """Create an ORT session and reject an explicit provider downgrade."""

    import onnxruntime as ort

    provider_name = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }[provider]
    if provider_name not in ort.get_available_providers():
        raise RuntimeError(f"ONNX Runtime provider is unavailable: {provider_name}")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers: list[Any] = (
        [(provider_name, {"use_tf32": "0"})] if provider == "cuda" else [provider_name]
    )
    session = ort.InferenceSession(model, sess_options=options, providers=providers)
    if session.get_providers()[0] != provider_name:
        raise RuntimeError(f"ONNX Runtime downgraded from {provider_name}")
    return session


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _resolve_inside(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    path.relative_to(resolved_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_manifest_records(
    manifest_path: Path,
    dataset_root: Path,
    *,
    require_folds: bool,
) -> list[dict[str, Any]]:
    records = _jsonl(manifest_path)
    seen_ids: set[int] = set()
    for record in records:
        image_id = int(record["image_id"])
        if image_id in seen_ids:
            raise ValueError(f"duplicate count image id: {image_id}")
        seen_ids.add(image_id)
        annotations = record.get("annotations", [])
        record["count_label"] = len(annotations)
        record["resolved_path"] = _resolve_inside(dataset_root, str(record["image_path"]))
        if require_folds and int(record.get("fold", -1)) not in {0, 1, 2}:
            raise ValueError("exact-count training requires folds 0, 1, and 2")
    if not records:
        raise ValueError("exact-count records are empty")
    return records


def load_coco_records(annotations_path: Path) -> list[dict[str, Any]]:
    payload = _json(annotations_path)
    annotation_counts: dict[int, int] = {}
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        annotation_counts[image_id] = annotation_counts.get(image_id, 0) + 1
    records = []
    for image in payload["images"]:
        image_id = int(image["id"])
        path = (annotations_path.parent / str(image["file_name"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "image_id": image_id,
                "image_path": str(image["file_name"]),
                "resolved_path": path,
                "count_label": annotation_counts.get(image_id, 0),
            }
        )
    if not records:
        raise ValueError("exact-count COCO records are empty")
    return sorted(records, key=lambda record: int(record["image_id"]))


def _feature_output_name(model, feature_mode: str = "cls") -> str:
    if feature_mode == "patch_mean":
        name = "/norm/LayerNormalization_output_0"
        if any(name in node.output for node in model.graph.node):
            return name
        raise ValueError("DINOv3 normalized token output was not found")
    initializer_names = {value.name for value in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Sub" and len(node.input) == 2 and node.input[1] == "feature_mean":
            if "feature_mean" not in initializer_names:
                break
            return str(node.input[0])
    raise ValueError("presence verifier feature output was not found")


def create_feature_session(model_path: Path, provider: str, *, feature_mode: str = "cls"):
    import onnx

    model = onnx.load(model_path)
    feature_name = _feature_output_name(model, feature_mode)
    del model.graph.output[:]
    shape = [1, 149, 384] if feature_mode == "patch_mean" else [1, 384]
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            feature_name,
            onnx.TensorProto.FLOAT,
            shape,
        )
    )
    session = create_ort_session(model.SerializeToString(), provider)
    return session, feature_name


def _image_tensor(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        tensor = prepare_rgb(
            image,
            (image_size, image_size),
            MEAN,
            STD,
            reducing_gap=1.0,
        )
        image.close()
    return np.expand_dims(tensor, axis=0)


def extract_features(
    records: list[dict[str, Any]],
    model_path: Path,
    *,
    image_size: int,
    provider: str,
    feature_mode: str,
) -> tuple[np.ndarray, float]:
    session, feature_name = create_feature_session(
        model_path,
        provider,
        feature_mode=feature_mode,
    )
    input_name = session.get_inputs()[0].name
    features = []
    started = time.perf_counter()
    for record in records:
        values = session.run(
            [feature_name],
            {input_name: _image_tensor(record["resolved_path"], image_size)},
        )[0]
        feature = values[0, 5:].mean(axis=0) if feature_mode == "patch_mean" else values[0]
        features.append(np.asarray(feature, dtype=np.float32))
    elapsed = time.perf_counter() - started
    return np.stack(features), elapsed


def fit_count_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    regularization_c: float,
    seed: int,
) -> dict[str, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    scale[scale == 0] = 1.0
    classifier = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    ).fit((features - scaler.mean_) / scale, labels)
    coefficient = np.asarray(classifier.coef_, dtype=np.float32)
    intercept = np.asarray(classifier.intercept_, dtype=np.float32)
    if len(classifier.classes_) == 2 and coefficient.shape[0] == 1:
        coefficient = np.concatenate([-coefficient / 2.0, coefficient / 2.0], axis=0)
        intercept = np.concatenate([-intercept / 2.0, intercept / 2.0], axis=0)
    return {
        "classes": np.asarray(classifier.classes_, dtype=np.int64),
        "mean": np.asarray(scaler.mean_, dtype=np.float32),
        "scale": scale,
        "coefficient": coefficient,
        "intercept": intercept,
    }


def count_logits(features: np.ndarray, head: dict[str, np.ndarray]) -> np.ndarray:
    scaled = (features - head["mean"]) / head["scale"]
    return scaled @ head["coefficient"].T + head["intercept"]


def softmax(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    shifted = logits / temperature
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def count_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    classes: np.ndarray,
    *,
    temperature: float = 1.0,
) -> dict[str, Any]:
    probabilities = softmax(logits, temperature=temperature)
    predictions = classes[np.argmax(probabilities, axis=1)]
    error = np.abs(predictions - labels)
    confusion: dict[str, dict[str, int]] = {}
    for expected, predicted in zip(labels, predictions, strict=True):
        expected_key = str(int(expected))
        predicted_key = str(int(predicted))
        confusion.setdefault(expected_key, {})[predicted_key] = (
            confusion.setdefault(expected_key, {}).get(predicted_key, 0) + 1
        )
    return {
        "image_count": int(len(labels)),
        "exact_correct_count": int(np.count_nonzero(error == 0)),
        "exact_accuracy": float(np.mean(error == 0)),
        "within_one_count": int(np.count_nonzero(error <= 1)),
        "within_one_accuracy": float(np.mean(error <= 1)),
        "mean_absolute_error": float(np.mean(error)),
        "maximum_absolute_error": int(error.max()),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "minimum_confidence": float(probabilities.max(axis=1).min()),
        "confusion": confusion,
    }


def _negative_log_likelihood(
    labels: np.ndarray,
    logits: np.ndarray,
    classes: np.ndarray,
    temperature: float,
) -> float:
    probabilities = softmax(logits, temperature=temperature)
    indices = {int(value): index for index, value in enumerate(classes)}
    selected = np.asarray(
        [probabilities[index, indices[int(label)]] for index, label in enumerate(labels)]
    )
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def select_temperature(
    labels: np.ndarray,
    logits: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, float]:
    candidates = np.geomspace(0.25, 4.0, num=81)
    scored = [
        (
            _negative_log_likelihood(labels, logits, classes, float(temperature)),
            float(temperature),
        )
        for temperature in candidates
    ]
    nll, temperature = min(scored)
    return temperature, nll


def cross_validated_count_probe(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    regularization_values: list[float],
    seed: int,
    auxiliary_features: np.ndarray | None = None,
    auxiliary_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    if sorted(np.unique(folds).tolist()) != [0, 1, 2]:
        raise ValueError("exact-count cross-validation requires folds 0, 1, and 2")
    if (auxiliary_features is None) != (auxiliary_labels is None):
        raise ValueError("auxiliary features and labels must be provided together")
    classes = np.unique(labels)
    if auxiliary_labels is not None and not np.all(np.isin(np.unique(auxiliary_labels), classes)):
        raise ValueError("auxiliary data contains a count class absent from development data")
    candidates = []
    for regularization_c in regularization_values:
        logits = np.zeros((len(labels), len(classes)), dtype=np.float64)
        for fold in range(3):
            training = folds != fold
            validation = ~training
            fitting_features = features[training]
            fitting_labels = labels[training]
            if auxiliary_features is not None and auxiliary_labels is not None:
                fitting_features = np.concatenate([fitting_features, auxiliary_features], axis=0)
                fitting_labels = np.concatenate([fitting_labels, auxiliary_labels], axis=0)
            head = fit_count_head(
                fitting_features,
                fitting_labels,
                regularization_c=regularization_c,
                seed=seed,
            )
            if not np.array_equal(head["classes"], classes):
                raise ValueError("count class coverage differs across folds")
            logits[validation] = count_logits(features[validation], head)
        temperature, nll = select_temperature(labels, logits, classes)
        metrics = count_metrics(
            labels,
            logits,
            classes,
            temperature=temperature,
        )
        candidates.append(
            {
                "regularization_c": regularization_c,
                "temperature": temperature,
                "negative_log_likelihood": nll,
                "metrics": metrics,
                "logits": logits,
            }
        )
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["metrics"]["exact_accuracy"],
            -candidate["metrics"]["mean_absolute_error"],
            -candidate["negative_log_likelihood"],
            -candidate["regularization_c"],
        ),
    )
    return {
        "classes": classes,
        "selected_regularization_c": selected["regularization_c"],
        "selected_temperature": selected["temperature"],
        "selected_metrics": selected["metrics"],
        "selected_logits": selected["logits"],
        "candidates": [
            {key: value for key, value in candidate.items() if key != "logits"}
            for candidate in candidates
        ],
    }


def export_exact_count_onnx(
    source_path: Path,
    output_path: Path,
    head: dict[str, np.ndarray],
    *,
    feature_mode: str = "cls",
) -> str:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(source_path)
    backbone_output_name = _feature_output_name(model, feature_mode)
    producer_index = next(
        index for index, node in enumerate(model.graph.node) if backbone_output_name in node.output
    )
    retained = list(model.graph.node[: producer_index + 1])
    del model.graph.node[:]
    model.graph.node.extend(retained)
    del model.graph.output[:]

    feature_name = backbone_output_name
    if feature_mode == "patch_mean":
        feature_name = "exact_count_patch_mean"
        for name, values in {
            "exact_count_patch_start": np.asarray([5], dtype=np.int64),
            "exact_count_patch_end": np.asarray([np.iinfo(np.int64).max], dtype=np.int64),
            "exact_count_patch_axis": np.asarray([1], dtype=np.int64),
            "exact_count_patch_step": np.asarray([1], dtype=np.int64),
        }.items():
            model.graph.initializer.append(numpy_helper.from_array(values, name=name))
        model.graph.node.extend(
            [
                onnx.helper.make_node(
                    "Slice",
                    [
                        backbone_output_name,
                        "exact_count_patch_start",
                        "exact_count_patch_end",
                        "exact_count_patch_axis",
                        "exact_count_patch_step",
                    ],
                    ["exact_count_patch_tokens"],
                    name="ExactCountPatchSlice",
                ),
                onnx.helper.make_node(
                    "ReduceMean",
                    ["exact_count_patch_tokens", "exact_count_patch_axis"],
                    [feature_name],
                    keepdims=0,
                    name="ExactCountPatchMean",
                ),
            ]
        )

    initializers = {
        "exact_count_feature_mean": head["mean"],
        "exact_count_feature_scale": head["scale"],
        "exact_count_coefficient": head["coefficient"].T,
        "exact_count_intercept": head["intercept"],
    }
    for name, values in initializers.items():
        model.graph.initializer.append(numpy_helper.from_array(values, name=name))
    model.graph.node.extend(
        [
            onnx.helper.make_node(
                "Sub",
                [feature_name, "exact_count_feature_mean"],
                ["exact_count_centered"],
                name="ExactCountCenter",
            ),
            onnx.helper.make_node(
                "Div",
                ["exact_count_centered", "exact_count_feature_scale"],
                ["exact_count_scaled"],
                name="ExactCountScale",
            ),
            onnx.helper.make_node(
                "MatMul",
                ["exact_count_scaled", "exact_count_coefficient"],
                ["exact_count_linear"],
                name="ExactCountMatMul",
            ),
            onnx.helper.make_node(
                "Add",
                ["exact_count_linear", "exact_count_intercept"],
                ["logits"],
                name="ExactCountAdd",
            ),
        ]
    )
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            "logits",
            onnx.TensorProto.FLOAT,
            [1, len(head["classes"])],
        )
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return feature_name


def _detector_counts_from_trace(path: Path) -> dict[int, dict[str, Any]]:
    values = {}
    for row in _jsonl(path):
        response = row["response"]
        values[int(row["image_id"])] = {
            "status": response["status"],
            "detector_count": len(response["segmentations"]),
            "segment_recapture": any(
                item["status"] == "SEGMENT_RECAPTURE" for item in response["segmentations"]
            ),
        }
    return values


def _detector_counts_from_responses(path: Path) -> dict[int, dict[str, Any]]:
    values = {}
    for row in _json(path)["records"]:
        response = row["response"]
        values[int(row["index"])] = {
            "status": response["status"],
            "detector_count": len(response["segmentations"]),
            "segment_recapture": any(
                item["status"] == "SEGMENT_RECAPTURE" for item in response["segmentations"]
            ),
        }
    return values


def safety_sweep(
    records: list[dict[str, Any]],
    logits: np.ndarray,
    classes: np.ndarray,
    detector_values: dict[int, dict[str, Any]],
    *,
    temperature: float,
) -> list[dict[str, Any]]:
    probabilities = softmax(logits, temperature=temperature)
    predictions = classes[np.argmax(probabilities, axis=1)]
    confidences = probabilities.max(axis=1)
    results = []
    for threshold in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        caught = 0
        dangerous = 0
        unnecessary = 0
        trigger_count = 0
        for record, predicted, confidence in zip(records, predictions, confidences, strict=True):
            image_id = int(record["image_id"])
            detector = detector_values[image_id]
            expected = int(record["count_label"])
            currently_safe = (
                detector["status"] == "IMAGE_RECAPTURE" or detector["segment_recapture"]
            )
            unsafe_miss = (
                detector["status"] == "SEGMENTATION"
                and int(detector["detector_count"]) < expected
                and not currently_safe
            )
            trigger = confidence < threshold or int(predicted) != int(detector["detector_count"])
            dangerous += int(unsafe_miss)
            caught += int(unsafe_miss and trigger)
            trigger_count += int(trigger)
            unnecessary += int(trigger and not unsafe_miss and not currently_safe)
        results.append(
            {
                "confidence_threshold": threshold,
                "trigger_count": trigger_count,
                "dangerous_miss_image_count": dangerous,
                "caught_dangerous_miss_count": caught,
                "unnecessary_new_recapture_count": unnecessary,
            }
        )
    return results


def _prediction_rows(
    records: list[dict[str, Any]],
    logits: np.ndarray,
    classes: np.ndarray,
    detector_values: dict[int, dict[str, Any]],
    *,
    temperature: float,
) -> list[dict[str, Any]]:
    probabilities = softmax(logits, temperature=temperature)
    predictions = classes[np.argmax(probabilities, axis=1)]
    return [
        {
            "image_id": int(record["image_id"]),
            "image_path": str(record["image_path"]),
            "expected_count": int(record["count_label"]),
            "predicted_count": int(predicted),
            "confidence": float(probability.max()),
            "detector_count": int(detector_values[int(record["image_id"])]["detector_count"]),
            "current_status": detector_values[int(record["image_id"])]["status"],
        }
        for record, predicted, probability in zip(records, predictions, probabilities, strict=True)
    ]


def onnx_parity(
    records: list[dict[str, Any]],
    model_path: Path,
    expected_logits: np.ndarray,
    *,
    image_size: int,
    provider: str,
) -> dict[str, Any]:
    session = create_ort_session(str(model_path), provider)
    input_name = session.get_inputs()[0].name
    actual = np.concatenate(
        [
            session.run(
                ["logits"],
                {input_name: _image_tensor(record["resolved_path"], image_size)},
            )[0]
            for record in records
        ],
        axis=0,
    )
    delta = np.abs(actual - expected_logits)
    return {
        "row_count": len(records),
        "maximum_logit_delta": float(delta.max()),
        "mean_logit_delta": float(delta.mean()),
        "argmax_diff_count": int(
            np.count_nonzero(np.argmax(actual, axis=1) != np.argmax(expected_logits, axis=1))
        ),
    }


def latency_benchmark(
    model_path: Path,
    image_path: Path,
    *,
    image_size: int,
    provider: str,
    warmup_count: int,
    repeat_count: int,
) -> dict[str, Any]:
    session = create_ort_session(str(model_path), provider)
    input_name = session.get_inputs()[0].name
    tensor = _image_tensor(image_path, image_size)
    for _ in range(warmup_count):
        session.run(None, {input_name: tensor})
    values = []
    for _ in range(repeat_count):
        started = time.perf_counter()
        session.run(None, {input_name: tensor})
        values.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(values)
    return {
        "provider": provider,
        "warmup_count": warmup_count,
        "repeat_count": repeat_count,
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def _distribution(labels: np.ndarray) -> dict[str, int]:
    return {str(int(label)): int(np.count_nonzero(labels == label)) for label in np.unique(labels)}


def _sha256_rows(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["image_id"]).encode("ascii"))
        digest.update(b":")
        digest.update(str(record["count_label"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_cuda_runtime(args.provider, args.cuda_dll_dir)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train_records = load_manifest_records(
        args.training_manifest,
        args.training_dataset_root,
        require_folds=True,
    )
    operational_records = load_manifest_records(
        args.operational69_manifest,
        args.operational69_root,
        require_folds=False,
    )
    target_records = load_coco_records(args.target_annotations)

    train_features, train_seconds = extract_features(
        train_records,
        args.source_model,
        image_size=args.image_size,
        provider=args.provider,
        feature_mode=args.feature_mode,
    )
    operational_features, operational_seconds = extract_features(
        operational_records,
        args.source_model,
        image_size=args.image_size,
        provider=args.provider,
        feature_mode=args.feature_mode,
    )
    target_features, target_seconds = extract_features(
        target_records,
        args.source_model,
        image_size=args.image_size,
        provider=args.provider,
        feature_mode=args.feature_mode,
    )
    train_labels = np.asarray([record["count_label"] for record in train_records], dtype=np.int64)
    operational_labels = np.asarray(
        [record["count_label"] for record in operational_records], dtype=np.int64
    )
    target_labels = np.asarray([record["count_label"] for record in target_records], dtype=np.int64)
    folds = np.asarray([record["fold"] for record in train_records], dtype=np.int64)
    cross_validation = cross_validated_count_probe(
        train_features,
        train_labels,
        folds,
        regularization_values=args.regularization_c,
        seed=args.seed,
    )
    auxiliary_cross_validation = cross_validated_count_probe(
        train_features,
        train_labels,
        folds,
        regularization_values=args.regularization_c,
        seed=args.seed,
        auxiliary_features=operational_features,
        auxiliary_labels=operational_labels,
    )
    classes = cross_validation["classes"]
    temperature = float(cross_validation["selected_temperature"])
    head = fit_count_head(
        train_features,
        train_labels,
        regularization_c=float(cross_validation["selected_regularization_c"]),
        seed=args.seed,
    )
    operational_logits = count_logits(operational_features, head)
    target_logits = count_logits(target_features, head)
    train_logits = count_logits(train_features, head)
    auxiliary_head = fit_count_head(
        np.concatenate([train_features, operational_features], axis=0),
        np.concatenate([train_labels, operational_labels], axis=0),
        regularization_c=float(auxiliary_cross_validation["selected_regularization_c"]),
        seed=args.seed,
    )
    auxiliary_temperature = float(auxiliary_cross_validation["selected_temperature"])
    auxiliary_operational_logits = count_logits(operational_features, auxiliary_head)
    auxiliary_target_logits = count_logits(target_features, auxiliary_head)

    model_path = output / "count-verifier-exact-count.onnx"
    export_exact_count_onnx(
        args.source_model,
        model_path,
        head,
        feature_mode=args.feature_mode,
    )
    detector415 = _detector_counts_from_trace(args.detector415_trace)
    detector69 = _detector_counts_from_trace(args.operational69_trace)
    detector28 = _detector_counts_from_responses(args.target_responses)

    predictions = {
        "detector415": _prediction_rows(
            train_records,
            train_logits,
            classes,
            detector415,
            temperature=temperature,
        ),
        "operational69": _prediction_rows(
            operational_records,
            operational_logits,
            classes,
            detector69,
            temperature=temperature,
        ),
        "target20260828": _prediction_rows(
            target_records,
            target_logits,
            classes,
            detector28,
            temperature=temperature,
        ),
    }
    for name, rows in predictions.items():
        path = output / f"{name}-predictions.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )

    parity_records = target_records + operational_records[:10] + train_records[:10]
    parity_features = np.concatenate(
        [target_features, operational_features[:10], train_features[:10]], axis=0
    )
    parity = onnx_parity(
        parity_records,
        model_path,
        count_logits(parity_features, head),
        image_size=args.image_size,
        provider=args.provider,
    )
    latency = {
        "source_object_presence": latency_benchmark(
            args.source_model,
            target_records[0]["resolved_path"],
            image_size=args.image_size,
            provider=args.provider,
            warmup_count=args.latency_warmup,
            repeat_count=args.latency_repeats,
        ),
        "exact_count_candidate": latency_benchmark(
            model_path,
            target_records[0]["resolved_path"],
            image_size=args.image_size,
            provider=args.provider,
            warmup_count=args.latency_warmup,
            repeat_count=args.latency_repeats,
        ),
    }
    latency["p95_delta_percent"] = (
        latency["exact_count_candidate"]["p95_ms"] / latency["source_object_presence"]["p95_ms"]
        - 1.0
    ) * 100.0

    report = {
        "schema_version": "1.0",
        "experiment": "dinov3_frozen_backbone_exact_count_probe",
        "feature_mode": args.feature_mode,
        "active_runtime_modified": False,
        "execution": {
            "provider": args.provider,
            "cuda_dll_dir": (
                args.cuda_dll_dir.resolve().as_posix() if args.cuda_dll_dir is not None else None
            ),
            "explicit_provider_downgrade_rejected": True,
        },
        "source_model": {
            "path": args.source_model.resolve().as_posix(),
            "sha256": sha256_file(args.source_model),
            "size_bytes": args.source_model.stat().st_size,
            "role": "0.1.7 object_presence verifier backbone source",
        },
        "candidate_model": {
            "path": model_path.resolve().as_posix(),
            "sha256": sha256_file(model_path),
            "size_bytes": model_path.stat().st_size,
            "count_labels": classes.tolist(),
            "comparison_mode": "exact_count",
            "temperature": temperature,
            "confidence_threshold": None,
        },
        "data": {
            "detector415": {
                "image_count": len(train_records),
                "count_distribution": _distribution(train_labels),
                "identity_sha256": _sha256_rows(train_records),
                "role": "development group-aware 3-fold CV and final head fitting",
            },
            "operational69": {
                "image_count": len(operational_records),
                "count_distribution": _distribution(operational_labels),
                "identity_sha256": _sha256_rows(operational_records),
                "role": "previously used operational diagnostic; not independent test",
            },
            "target20260828": {
                "image_count": len(target_records),
                "count_distribution": _distribution(target_labels),
                "identity_sha256": _sha256_rows(target_records),
                "role": "recapture challenge diagnostic; not used for fitting",
            },
        },
        "feature_extraction_seconds": {
            "detector415": train_seconds,
            "operational69": operational_seconds,
            "target20260828": target_seconds,
        },
        "cross_validation": {
            key: value
            for key, value in cross_validation.items()
            if key not in {"classes", "selected_logits"}
        },
        "operational69_as_auxiliary_training_comparison": {
            "purpose": (
                "Measure domain adaptation without fitting the 2026-08-28 challenge images"
            ),
            "independent_test": False,
            "cross_validation_on_detector415": {
                key: value
                for key, value in auxiliary_cross_validation.items()
                if key not in {"classes", "selected_logits"}
            },
            "final_head_metrics": {
                "operational69_in_sample": count_metrics(
                    operational_labels,
                    auxiliary_operational_logits,
                    classes,
                    temperature=auxiliary_temperature,
                ),
                "target20260828": count_metrics(
                    target_labels,
                    auxiliary_target_logits,
                    classes,
                    temperature=auxiliary_temperature,
                ),
            },
            "safety_sweep": {
                "detector415_oof": safety_sweep(
                    train_records,
                    auxiliary_cross_validation["selected_logits"],
                    classes,
                    detector415,
                    temperature=auxiliary_temperature,
                ),
                "target20260828": safety_sweep(
                    target_records,
                    auxiliary_target_logits,
                    classes,
                    detector28,
                    temperature=auxiliary_temperature,
                ),
            },
        },
        "final_head_metrics": {
            "detector415_in_sample": count_metrics(
                train_labels,
                train_logits,
                classes,
                temperature=temperature,
            ),
            "operational69": count_metrics(
                operational_labels,
                operational_logits,
                classes,
                temperature=temperature,
            ),
            "target20260828": count_metrics(
                target_labels,
                target_logits,
                classes,
                temperature=temperature,
            ),
        },
        "safety_sweep": {
            "detector415_oof": safety_sweep(
                train_records,
                cross_validation["selected_logits"],
                classes,
                detector415,
                temperature=temperature,
            ),
            "operational69": safety_sweep(
                operational_records,
                operational_logits,
                classes,
                detector69,
                temperature=temperature,
            ),
            "target20260828": safety_sweep(
                target_records,
                target_logits,
                classes,
                detector28,
                temperature=temperature,
            ),
        },
        "onnx_parity": parity,
        "latency": latency,
        "limitations": [
            "count=2 training images are absent",
            "only four empty training images are available",
            "detector415 is development data and operational69 was used in earlier diagnostics",
            "target20260828 contains only two closely related crowded scene families",
            "no confidence threshold is selected without an independent calibration session",
        ],
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Probe an exact-count linear head on the shipped DINOv3 presence backbone"
    )
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-dataset-root", type=Path, required=True)
    parser.add_argument("--operational69-manifest", type=Path, required=True)
    parser.add_argument("--operational69-root", type=Path, required=True)
    parser.add_argument("--target-annotations", type=Path, required=True)
    parser.add_argument("--detector415-trace", type=Path, required=True)
    parser.add_argument("--operational69-trace", type=Path, required=True)
    parser.add_argument("--target-responses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--feature-mode", choices=("cls", "patch_mean"), default="cls")
    parser.add_argument(
        "--regularization-c",
        type=float,
        nargs="+",
        default=[0.0001, 0.001, 0.01, 0.1, 1.0],
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=100)
    args = parser.parse_args(argv)
    if not args.source_model.is_file():
        raise FileNotFoundError(args.source_model)
    if any(value <= 0 or not math.isfinite(value) for value in args.regularization_c):
        raise ValueError("regularization values must be finite and positive")
    run(args)


if __name__ == "__main__":
    main()
