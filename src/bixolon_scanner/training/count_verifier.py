from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..contracts.catalog import sha256_file
from ..runtime.onnx import prepare_rgb
from .dinov3_objectness_detector import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_dinov3_convnext_tiny,
)
from .models import DINO_V3_HUB_REPOSITORY, require_torch


def load_source_only_count_records(
    manifest_path: Path,
    dataset_root: Path,
    *,
    expected_dataset_version: str,
) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("count verifier manifest is empty")
    resolved_root = dataset_root.resolve()
    seen_hashes: set[str] = set()
    for record in records:
        if record.get("training_allowed") is False:
            raise ValueError("training-prohibited image cannot train the count verifier")
        if record.get("source_dataset") != expected_dataset_version:
            raise ValueError("count verifier record is outside the locked source dataset")
        digest = str(record["image_sha256"])
        if digest in seen_hashes:
            raise ValueError("duplicate count verifier image")
        seen_hashes.add(digest)
        path = (resolved_root / str(record["image_path"])).resolve()
        path.relative_to(resolved_root)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"count verifier image binding mismatch: {path.name}")
        record["resolved_path"] = path
        record["count_label"] = len(record.get("annotations", []))
    return records


def _image_tensor(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return prepare_rgb(
            image,
            (image_size, image_size),
            IMAGENET_MEAN,
            IMAGENET_STD,
            reducing_gap=1.0,
        )


def extract_convnext_features(
    records: list[dict[str, Any]],
    weights_path: Path,
    *,
    image_size: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, float]:
    torch = require_torch()
    model = load_dinov3_convnext_tiny(weights_path, device=device).eval()
    features = []
    started = time.perf_counter()
    for offset in range(0, len(records), batch_size):
        tensor = np.stack(
            [
                _image_tensor(record["resolved_path"], image_size)
                for record in records[offset : offset + batch_size]
            ]
        )
        with torch.inference_mode():
            values = model(torch.from_numpy(tensor).to(device))
        features.append(values.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(features), time.perf_counter() - started


def _fit_head(
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
    return {
        "classes": np.asarray(classifier.classes_, dtype=np.int64),
        "mean": np.asarray(scaler.mean_, dtype=np.float32),
        "scale": scale,
        "coefficient": np.asarray(classifier.coef_, dtype=np.float32),
        "intercept": np.asarray(classifier.intercept_, dtype=np.float32),
    }


def count_logits(features: np.ndarray, head: dict[str, np.ndarray]) -> np.ndarray:
    return ((features - head["mean"]) / head["scale"]) @ head["coefficient"].T + head["intercept"]


def probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    shifted = logits.astype(np.float64) / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    classes: np.ndarray,
    *,
    temperature: float = 1.0,
) -> dict[str, Any]:
    values = probabilities(logits, temperature)
    predictions = classes[np.argmax(values, axis=1)]
    error = np.abs(predictions - labels)
    return {
        "image_count": int(len(labels)),
        "exact_correct_count": int(np.count_nonzero(error == 0)),
        "exact_accuracy": float(np.mean(error == 0)),
        "within_one_accuracy": float(np.mean(error <= 1)),
        "mean_absolute_error": float(np.mean(error)),
        "maximum_absolute_error": int(error.max()),
        "mean_confidence": float(values.max(axis=1).mean()),
        "minimum_confidence": float(values.max(axis=1).min()),
    }


def _negative_log_likelihood(
    labels: np.ndarray,
    logits: np.ndarray,
    classes: np.ndarray,
    temperature: float,
) -> float:
    values = probabilities(logits, temperature)
    indices = {int(value): index for index, value in enumerate(classes)}
    selected = np.asarray(
        [values[index, indices[int(label)]] for index, label in enumerate(labels)]
    )
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def select_count_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    regularization_values: list[float],
    seed: int,
) -> tuple[dict[str, np.ndarray], float, dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    fitted = []
    for regularization_c in regularization_values:
        head = _fit_head(
            train_features,
            train_labels,
            regularization_c=regularization_c,
            seed=seed,
        )
        logits = count_logits(validation_features, head)
        temperatures = np.geomspace(0.25, 4.0, 81)
        temperature = min(
            temperatures,
            key=lambda value: _negative_log_likelihood(
                validation_labels, logits, head["classes"], float(value)
            ),
        )
        candidate_metrics = metrics(
            validation_labels,
            logits,
            head["classes"],
            temperature=float(temperature),
        )
        candidates.append(
            {
                "regularization_c": regularization_c,
                "temperature": float(temperature),
                "negative_log_likelihood": _negative_log_likelihood(
                    validation_labels, logits, head["classes"], float(temperature)
                ),
                "metrics": candidate_metrics,
            }
        )
        fitted.append(head)
    selected_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["metrics"]["exact_accuracy"],
            -candidates[index]["metrics"]["mean_absolute_error"],
            -candidates[index]["negative_log_likelihood"],
        ),
    )
    selected = candidates[selected_index]
    return fitted[selected_index], selected["temperature"], selected["metrics"], candidates


def export_count_verifier(
    weights_path: Path,
    output_path: Path,
    head: dict[str, np.ndarray],
    *,
    image_size: int,
    opset: int,
) -> None:
    torch = require_torch()
    backbone = load_dinov3_convnext_tiny(weights_path, device="cpu").eval()
    weight = head["coefficient"] / head["scale"][None, :]
    bias = head["intercept"] - (head["mean"] / head["scale"]) @ head["coefficient"].T

    class CountVerifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.register_buffer("coefficient", torch.from_numpy(weight.astype(np.float32)))
            self.register_buffer("intercept", torch.from_numpy(bias.astype(np.float32)))

        def forward(self, pixel_values):
            features = self.backbone(pixel_values)
            return features @ self.coefficient.T + self.intercept

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        CountVerifier().eval(),
        (torch.zeros(1, 3, image_size, image_size, dtype=torch.float32),),
        output_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(output_path))


def source_revision() -> str:
    return DINO_V3_HUB_REPOSITORY.rsplit(":", 1)[-1]


def finite_number(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("count verifier metric is not finite")
    return value
