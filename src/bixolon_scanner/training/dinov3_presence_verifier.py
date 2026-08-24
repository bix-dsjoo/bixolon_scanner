from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..contracts.catalog import sha256_file
from ..runtime.onnx import prepare_rgb
from .models import DINO_V3_HUB_REPOSITORY, require_torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_presence_records(
    manifest_path: Path,
    dataset_root: Path,
    *,
    expected_image_count: int,
    expected_multi_object_count: int,
    expected_operational_count: int,
) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(records) != expected_image_count:
        raise ValueError(
            f"presence training requires {expected_image_count} images; observed {len(records)}"
        )
    expected_sources = {
        "multi_object_scenes": expected_multi_object_count,
        "operational_collections/2026-08-18": expected_operational_count,
    }
    observed_sources = {
        name: sum(record.get("evaluation_set") == name for record in records)
        for name in expected_sources
    }
    if observed_sources != expected_sources or sum(observed_sources.values()) != len(records):
        raise ValueError(
            f"presence training source mismatch: expected {expected_sources}, observed {observed_sources}"
        )
    resolved_root = dataset_root.resolve()
    seen_ids: set[int] = set()
    for record in records:
        image_id = int(record["image_id"])
        if image_id in seen_ids:
            raise ValueError(f"duplicate presence training image id: {image_id}")
        seen_ids.add(image_id)
        if int(record["fold"]) not in {0, 1, 2}:
            raise ValueError("presence training requires folds 0, 1, and 2")
        path = (resolved_root / str(record["image_path"])).resolve()
        path.relative_to(resolved_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        record["resolved_path"] = path
        record["presence_label"] = int(bool(record["annotations"]))
    if {int(record["presence_label"]) for record in records} != {0, 1}:
        raise ValueError("presence training requires both empty and non-empty images")
    return records


def fit_presence_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    regularization_c: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    ).fit(scaler.transform(features), labels)
    return (
        np.asarray(scaler.mean_, dtype=np.float32),
        np.asarray(scaler.scale_, dtype=np.float32),
        np.asarray(classifier.coef_[0], dtype=np.float32),
        float(classifier.intercept_[0]),
    )


def presence_probabilities(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
) -> np.ndarray:
    logits = ((features - mean) / scale) @ coefficient + intercept
    positive = logits >= 0
    probabilities = np.empty_like(logits, dtype=np.float64)
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    probabilities[~positive] = exponential / (1.0 + exponential)
    return probabilities


def cross_validated_presence_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    regularization_c: float,
    seed: int,
    decision_threshold: float = 0.5,
) -> tuple[dict[str, Any], np.ndarray]:
    if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(folds):
        raise ValueError("presence features, labels, and folds must have aligned rows")
    if sorted(np.unique(folds).tolist()) != [0, 1, 2]:
        raise ValueError("presence cross-validation requires folds 0, 1, and 2")
    probabilities = np.zeros(len(labels), dtype=np.float64)
    for fold in range(3):
        training = folds != fold
        validation = ~training
        mean, scale, coefficient, intercept = fit_presence_head(
            features[training],
            labels[training],
            regularization_c=regularization_c,
            seed=seed,
        )
        probabilities[validation] = presence_probabilities(
            features[validation], mean, scale, coefficient, intercept
        )
    predictions = (probabilities >= decision_threshold).astype(np.int64)
    empty = labels == 0
    nonempty = ~empty
    metrics = {
        "image_count": int(len(labels)),
        "empty_image_count": int(empty.sum()),
        "nonempty_image_count": int(nonempty.sum()),
        "empty_correct_count": int(np.count_nonzero(predictions[empty] == 0)),
        "empty_missed_count": int(np.count_nonzero(predictions[empty] != 0)),
        "nonempty_correct_count": int(np.count_nonzero(predictions[nonempty] == 1)),
        "nonempty_false_empty_count": int(np.count_nonzero(predictions[nonempty] != 1)),
        "maximum_empty_presence_probability": float(probabilities[empty].max()),
        "minimum_nonempty_presence_probability": float(probabilities[nonempty].min()),
        "probability_separation_margin": float(
            probabilities[nonempty].min() - probabilities[empty].max()
        ),
        "decision_threshold": float(decision_threshold),
    }
    return metrics, probabilities


def _load_backbone(weights_path: Path, device: str):
    torch = require_torch()
    backbone = torch.hub.load(
        DINO_V3_HUB_REPOSITORY,
        "dinov3_vits16",
        source="github",
        trust_repo=True,
        verbose=False,
        pretrained=False,
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    return backbone.eval().to(device)


def extract_features(
    records: list[dict[str, Any]],
    weights_path: Path,
    *,
    image_size: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    torch = require_torch()
    backbone = _load_backbone(weights_path, device)
    batches: list[np.ndarray] = []
    for offset in range(0, len(records), batch_size):
        tensors = []
        for record in records[offset : offset + batch_size]:
            with Image.open(record["resolved_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                tensors.append(
                    prepare_rgb(
                        image,
                        (image_size, image_size),
                        MEAN,
                        STD,
                        reducing_gap=1.0,
                    )
                )
                image.close()
        batch = torch.from_numpy(np.stack(tensors)).to(device)
        with torch.inference_mode():
            output = backbone.forward_features(batch, masks=None)["x_norm_clstoken"]
        batches.append(output.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(batches)


def export_presence_verifier(
    weights_path: Path,
    output_path: Path,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
    image_size: int,
    opset: int,
) -> None:
    torch = require_torch()
    backbone = _load_backbone(weights_path, "cpu")

    class PresenceVerifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.register_buffer("feature_mean", torch.from_numpy(mean))
            self.register_buffer("feature_scale", torch.from_numpy(scale))
            self.register_buffer("coefficient", torch.from_numpy(coefficient))
            self.register_buffer("intercept", torch.tensor(intercept, dtype=torch.float32))

        def forward(self, pixel_values):
            features = self.backbone.forward_features(pixel_values, masks=None)["x_norm_clstoken"]
            decision = (
                ((features - self.feature_mean) / self.feature_scale) * self.coefficient
            ).sum(dim=-1) + self.intercept
            return torch.stack((-decision * 0.5, decision * 0.5), dim=-1)

    model = PresenceVerifier().eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a generic DINOv3 object-presence verifier from detector images"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--expected-image-count", type=int, default=415)
    parser.add_argument("--expected-multi-object-count", type=int, default=300)
    parser.add_argument("--expected-operational-count", type=int, default=115)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--regularization-c", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    records = load_presence_records(
        args.manifest,
        args.dataset_root,
        expected_image_count=args.expected_image_count,
        expected_multi_object_count=args.expected_multi_object_count,
        expected_operational_count=args.expected_operational_count,
    )
    image_ids = np.asarray([int(record["image_id"]) for record in records], dtype=np.int64)
    labels = np.asarray([int(record["presence_label"]) for record in records], dtype=np.int64)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    started = time.perf_counter()
    features = extract_features(
        records,
        args.weights,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    if args.feature_cache is not None:
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.feature_cache,
            features=features,
            labels=labels,
            folds=folds,
            image_ids=image_ids,
        )
    cross_validation, probabilities = cross_validated_presence_metrics(
        features,
        labels,
        folds,
        regularization_c=args.regularization_c,
        seed=args.seed,
    )
    mean, scale, coefficient, intercept = fit_presence_head(
        features,
        labels,
        regularization_c=args.regularization_c,
        seed=args.seed,
    )
    export_presence_verifier(
        args.weights,
        args.output,
        mean=mean,
        scale=scale,
        coefficient=coefficient,
        intercept=intercept,
        image_size=args.image_size,
        opset=args.opset,
    )
    revision = DINO_V3_HUB_REPOSITORY.rsplit(":", 1)[-1]
    report = {
        "schema_version": "2.0",
        "model_role": "generic_object_presence_count_verifier",
        "comparison_mode": "object_presence",
        "backbone_kind": "dinov3_vits16",
        "source_revision": revision,
        "source_weight_filename": args.weights.name,
        "source_weight_sha256": sha256_file(args.weights),
        "onnx_sha256": sha256_file(args.output),
        "image_size": args.image_size,
        "embedding_dimension": int(features.shape[1]),
        "regularization_c": args.regularization_c,
        "seed": args.seed,
        "confidence_threshold": 0.5,
        "temperature": 1.0,
        "training_dataset_version": args.dataset_version,
        "training_manifest_sha256": sha256_file(args.manifest),
        "training_image_count": len(records),
        "training_empty_image_count": int(np.count_nonzero(labels == 0)),
        "training_nonempty_image_count": int(np.count_nonzero(labels == 1)),
        "source_image_counts": {
            "multi_object_scenes": args.expected_multi_object_count,
            "operational_collections/2026-08-18": args.expected_operational_count,
        },
        "cross_validation": cross_validation,
        "empty_oof_presence_probabilities": [
            {
                "image_id": int(image_ids[index]),
                "fold": int(folds[index]),
                "presence_probability": float(probabilities[index]),
            }
            for index in np.flatnonzero(labels == 0)
        ],
        "evidence_role": "development_group_aware_cross_validation",
        "selection_scope": "detector415_development_only",
        "independent_test_claimed": False,
        "feature_extraction_seconds": time.perf_counter() - started,
        "opset": args.opset,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
