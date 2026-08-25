from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..runtime.onnx import prepare_rgb
from .dfine_export import checkpoint_model_state, compatible_checkpoint_state
from .dfine_objectness_checkpoint import collapse_dfine_state_to_objectness


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(manifest: Path, maximum_images: int) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if maximum_images < 2:
        raise ValueError("objectness distillation requires at least two images")
    if len(rows) <= maximum_images:
        return rows
    indices = np.linspace(0, len(rows) - 1, maximum_images, dtype=np.int64)
    return [rows[int(index)] for index in indices]


def _fit_linear_max(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score

    if features.ndim != 2 or targets.shape != (len(features),):
        raise ValueError("objectness distillation feature contract mismatch")
    split = max(1, min(len(features) - 1, int(len(features) * 0.8)))
    audit_model = Ridge(alpha=alpha).fit(features[:split], targets[:split])
    audit_predictions = audit_model.predict(features[split:])
    final_model = Ridge(alpha=alpha).fit(features, targets)
    report = {
        "sample_count": float(len(features)),
        "validation_sample_count": float(len(features) - split),
        "validation_r2": float(r2_score(targets[split:], audit_predictions)),
        "validation_mean_absolute_error": float(
            mean_absolute_error(targets[split:], audit_predictions)
        ),
    }
    return (
        np.asarray(final_model.coef_, dtype=np.float32),
        float(final_model.intercept_),
        report,
    )


def _fit_linear_objectness(
    features_by_image: list[np.ndarray],
    iou_by_image: list[np.ndarray],
    *,
    positive_iou: float,
    regularization: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    split_image = max(1, min(len(features_by_image) - 1, int(len(features_by_image) * 0.8)))
    train_features = np.concatenate(features_by_image[:split_image])
    train_targets = np.concatenate(iou_by_image[:split_image]) >= positive_iou
    validation_features = np.concatenate(features_by_image[split_image:])
    validation_targets = np.concatenate(iou_by_image[split_image:]) >= positive_iou
    if len(np.unique(train_targets)) != 2 or len(np.unique(validation_targets)) != 2:
        raise ValueError("objectness supervision requires positive and negative box candidates")

    scaler = StandardScaler().fit(train_features)
    audit_model = LogisticRegression(
        C=regularization,
        class_weight="balanced",
        max_iter=300,
        solver="lbfgs",
    ).fit(scaler.transform(train_features), train_targets)
    validation_score = audit_model.decision_function(scaler.transform(validation_features))

    all_features = np.concatenate(features_by_image)
    all_targets = np.concatenate(iou_by_image) >= positive_iou
    final_scaler = StandardScaler().fit(all_features)
    final_model = LogisticRegression(
        C=regularization,
        class_weight="balanced",
        max_iter=300,
        solver="lbfgs",
    ).fit(final_scaler.transform(all_features), all_targets)
    scaled_weight = np.asarray(final_model.coef_[0], dtype=np.float64)
    scale = np.asarray(final_scaler.scale_, dtype=np.float64)
    mean = np.asarray(final_scaler.mean_, dtype=np.float64)
    weight = scaled_weight / scale
    bias = float(final_model.intercept_[0] - np.dot(scaled_weight, mean / scale))
    return (
        np.asarray(weight, dtype=np.float32),
        bias,
        {
            "sample_count": float(len(all_features)),
            "validation_sample_count": float(len(validation_features)),
            "positive_iou": positive_iou,
            "positive_rate": float(all_targets.mean()),
            "validation_roc_auc": float(roc_auc_score(validation_targets, validation_score)),
            "validation_balanced_accuracy": float(
                balanced_accuracy_score(validation_targets, validation_score >= 0.0)
            ),
        },
    )


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = np.empty_like(boxes, dtype=np.float32)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    result[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    result[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    result[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    return result


def _maximum_iou(candidate_cxcywh: np.ndarray, ground_truth_xyxy: np.ndarray) -> np.ndarray:
    if len(ground_truth_xyxy) == 0:
        return np.zeros(len(candidate_cxcywh), dtype=np.float32)
    candidates = _cxcywh_to_xyxy(candidate_cxcywh)
    intersection_x1 = np.maximum(candidates[:, None, 0], ground_truth_xyxy[None, :, 0])
    intersection_y1 = np.maximum(candidates[:, None, 1], ground_truth_xyxy[None, :, 1])
    intersection_x2 = np.minimum(candidates[:, None, 2], ground_truth_xyxy[None, :, 2])
    intersection_y2 = np.minimum(candidates[:, None, 3], ground_truth_xyxy[None, :, 3])
    intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(
        0.0, intersection_y2 - intersection_y1
    )
    candidate_area = np.maximum(0.0, candidates[:, 2] - candidates[:, 0]) * np.maximum(
        0.0, candidates[:, 3] - candidates[:, 1]
    )
    target_area = np.maximum(0.0, ground_truth_xyxy[:, 2] - ground_truth_xyxy[:, 0]) * np.maximum(
        0.0, ground_truth_xyxy[:, 3] - ground_truth_xyxy[:, 1]
    )
    union = candidate_area[:, None] + target_area[None, :] - intersection
    return np.max(intersection / np.maximum(union, 1e-8), axis=1).astype(np.float32)


def _normalized_ground_truth(row: dict[str, Any]) -> np.ndarray:
    width = float(row["width"])
    height = float(row["height"])
    boxes = []
    for annotation in row["annotations"]:
        x, y, box_width, box_height = [float(value) for value in annotation["bbox_xywh"]]
        boxes.append([x / width, y / height, (x + box_width) / width, (y + box_height) / height])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def _ground_truth_recall_at_topk(
    features_by_image: list[np.ndarray],
    boxes_by_image: list[np.ndarray],
    ground_truth_by_image: list[np.ndarray],
    weight: np.ndarray,
    bias: float,
    *,
    topk: int,
    match_iou: float = 0.5,
) -> float:
    matched = 0
    total = 0
    for features, boxes, ground_truth in zip(
        features_by_image, boxes_by_image, ground_truth_by_image
    ):
        if not len(ground_truth):
            continue
        scores = features @ weight + bias
        count = min(topk, len(scores))
        selected = boxes[np.argpartition(scores, -count)[-count:]]
        candidates = _cxcywh_to_xyxy(selected)
        for target in ground_truth:
            matched += int(
                _maximum_iou(
                    np.column_stack(
                        (
                            (candidates[:, 0] + candidates[:, 2]) * 0.5,
                            (candidates[:, 1] + candidates[:, 3]) * 0.5,
                            candidates[:, 2] - candidates[:, 0],
                            candidates[:, 3] - candidates[:, 1],
                        )
                    ),
                    target[None],
                ).max()
                >= match_iou
            )
            total += 1
    return matched / total if total else 0.0


def _topk_overlap(
    features_by_image: list[np.ndarray],
    targets_by_image: list[np.ndarray],
    weight: np.ndarray,
    bias: float,
    *,
    topk: int,
) -> float:
    overlaps = []
    for features, targets in zip(features_by_image, targets_by_image):
        count = min(topk, len(targets))
        teacher = set(np.argpartition(targets, -count)[-count:].tolist())
        predicted = features @ weight + bias
        student = set(np.argpartition(predicted, -count)[-count:].tolist())
        overlaps.append(len(teacher & student) / count)
    return float(np.mean(overlaps))


def distill_dfine_objectness_checkpoint(
    *,
    repository: Path,
    config: Path,
    checkpoint: Path,
    manifest: Path,
    dataset_root: Path,
    output: Path,
    maximum_images: int = 64,
    ridge_alpha: float = 1.0,
    device: str = "cuda",
    target_mode: str = "teacher_max",
) -> dict[str, Any]:
    import torch

    repository = repository.resolve()
    config = config.resolve()
    checkpoint = checkpoint.resolve()
    manifest = manifest.resolve()
    dataset_root = dataset_root.resolve()
    rows = _records(manifest, maximum_images)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested distillation device")

    sys.path.insert(0, str(repository))
    try:
        from src.core import YAMLConfig

        cfg = YAMLConfig(str(config), resume=str(checkpoint))
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        teacher_state = checkpoint_model_state(payload)
        state = compatible_checkpoint_state(cfg.model.state_dict(), teacher_state)
        missing, unexpected = cfg.model.load_state_dict(state, strict=False)
        allowed_missing = {"decoder.anchors", "decoder.valid_mask"}
        if set(missing) - allowed_missing or unexpected:
            raise ValueError(
                f"incompatible D-FINE checkpoint: missing={missing}, unexpected={unexpected}"
            )
        model = cfg.model.eval().to(device)
        decoder = model.decoder

        encoder_features: list[np.ndarray] = []
        encoder_targets: list[np.ndarray] = []
        encoder_boxes: list[np.ndarray] = []
        decoder0_features: list[np.ndarray] = []
        decoder0_targets: list[np.ndarray] = []
        decoder2_features: list[np.ndarray] = []
        decoder2_targets: list[np.ndarray] = []
        decoder_boxes: list[np.ndarray] = []
        ground_truth: list[np.ndarray] = []

        def capture_encoder(_module, _inputs, output_value):
            output_value = output_value.detach()
            targets = decoder.enc_score_head(output_value).max(dim=-1).values
            encoder_features.append(output_value[0].float().cpu().numpy())
            encoder_targets.append(targets[0].float().cpu().numpy())
            anchors = decoder.anchors.to(device=output_value.device, dtype=output_value.dtype)
            boxes = torch.sigmoid(decoder.enc_bbox_head(output_value) + anchors)
            encoder_boxes.append(boxes[0].float().cpu().numpy())

        def decoder_hook(feature_store, target_store):
            def capture(_module, inputs, output_value):
                feature_store.append(inputs[0][0].detach().float().cpu().numpy())
                target_store.append(
                    output_value[0].detach().max(dim=-1).values.float().cpu().numpy()
                )

            return capture

        handles = [
            decoder.enc_output.register_forward_hook(capture_encoder),
            decoder.dec_score_head[0].register_forward_hook(
                decoder_hook(decoder0_features, decoder0_targets)
            ),
            decoder.dec_score_head[decoder.eval_idx].register_forward_hook(
                decoder_hook(decoder2_features, decoder2_targets)
            ),
        ]
        try:
            with torch.inference_mode():
                for row in rows:
                    image_path = (dataset_root / str(row["image_path"])).resolve()
                    image_path.relative_to(dataset_root)
                    with Image.open(image_path) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                        tensor = prepare_rgb(
                            image,
                            (640, 640),
                            (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0),
                            reducing_gap=1.0,
                        )
                    result = model(torch.from_numpy(tensor[None]).to(device))
                    decoder_boxes.append(result["pred_boxes"][0].float().cpu().numpy())
                    targets = _normalized_ground_truth(row)
                    ground_truth.append(targets)
                    if target_mode == "box_iou":
                        encoder_targets[-1] = _maximum_iou(encoder_boxes[-1], targets)
                        decoder2_targets[-1] = _maximum_iou(decoder_boxes[-1], targets)
                        decoder0_targets[-1] = decoder2_targets[-1]
        finally:
            for handle in handles:
                handle.remove()
            model.to("cpu")

        if target_mode == "teacher_max":
            encoder_weight, encoder_bias, encoder_report = _fit_linear_max(
                np.concatenate(encoder_features),
                np.concatenate(encoder_targets),
                alpha=ridge_alpha,
            )
            decoder0_weight, decoder0_bias, decoder0_report = _fit_linear_max(
                np.concatenate(decoder0_features),
                np.concatenate(decoder0_targets),
                alpha=ridge_alpha,
            )
            decoder2_weight, decoder2_bias, decoder2_report = _fit_linear_max(
                np.concatenate(decoder2_features),
                np.concatenate(decoder2_targets),
                alpha=ridge_alpha,
            )
            encoder_report["teacher_topk_overlap"] = _topk_overlap(
                encoder_features,
                encoder_targets,
                encoder_weight,
                encoder_bias,
                topk=decoder.num_queries,
            )
        elif target_mode == "box_iou":
            encoder_weight, encoder_bias, encoder_report = _fit_linear_objectness(
                encoder_features,
                encoder_targets,
                positive_iou=0.3,
                regularization=ridge_alpha,
            )
            decoder0_weight, decoder0_bias, decoder0_report = _fit_linear_objectness(
                decoder0_features,
                decoder0_targets,
                positive_iou=0.5,
                regularization=ridge_alpha,
            )
            decoder2_weight, decoder2_bias, decoder2_report = _fit_linear_objectness(
                decoder2_features,
                decoder2_targets,
                positive_iou=0.5,
                regularization=ridge_alpha,
            )
            encoder_report["ground_truth_recall_at_top300_iou_0_5"] = _ground_truth_recall_at_topk(
                encoder_features,
                encoder_boxes,
                ground_truth,
                encoder_weight,
                encoder_bias,
                topk=decoder.num_queries,
            )
        else:
            raise ValueError(f"unsupported objectness distillation target mode: {target_mode}")

        converted, conversion = collapse_dfine_state_to_objectness(teacher_state)
        tensor_specs = {
            "decoder.enc_score_head": (encoder_weight, encoder_bias),
            "decoder.dec_score_head.0": (decoder0_weight, decoder0_bias),
            "decoder.dec_score_head.1": (decoder2_weight, decoder2_bias),
            "decoder.dec_score_head.2": (decoder2_weight, decoder2_bias),
        }
        for prefix, (weight, bias) in tensor_specs.items():
            target_weight = converted[f"{prefix}.weight"]
            converted[f"{prefix}.weight"] = torch.from_numpy(weight[None]).to(
                dtype=target_weight.dtype
            )
            converted[f"{prefix}.bias"] = torch.tensor(
                [bias], dtype=converted[f"{prefix}.bias"].dtype
            )

        report = {
            "recipe": (
                "box_only_linear_objectness_supervision"
                if target_mode == "box_iou"
                else "label_blind_linear_distillation_of_teacher_max_class_logits"
            ),
            "target_mode": target_mode,
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": _sha256(checkpoint),
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "image_count": len(rows),
            "image_selection": "deterministic_even_spacing",
            "ridge_alpha": ridge_alpha,
            "encoder": encoder_report,
            "decoder_layer_0": decoder0_report,
            "decoder_layer_2": decoder2_report,
            "conversion": conversion,
            "product_labels_read": False,
            "bounding_boxes_read": target_mode == "box_iou",
            "requires_one_class_tuning": True,
            "release_model": False,
            "independent_test_claimed": False,
        }
        result = {
            "date": datetime.now(timezone.utc).isoformat(),
            "last_epoch": -1,
            "model": converted,
            "ema": {"module": converted, "updates": 0},
            "detector_objectness_distillation": report,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output)
        return report
    finally:
        if sys.path and sys.path[0] == str(repository):
            sys.path.pop(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill D-FINE max class logits into a one-class objectness checkpoint"
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-images", type=int, default=64)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target-mode", choices=("teacher_max", "box_iou"), default="teacher_max")
    args = parser.parse_args()
    report = distill_dfine_objectness_checkpoint(
        repository=args.repository,
        config=args.config,
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        dataset_root=args.dataset_root,
        output=args.output,
        maximum_images=args.maximum_images,
        ridge_alpha=args.ridge_alpha,
        device=args.device,
        target_mode=args.target_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
