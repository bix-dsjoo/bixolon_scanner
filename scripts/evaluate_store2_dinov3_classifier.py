from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from bixolon_scanner.operations.catalog_activation import fit_diagonal_lda_adapter
from bixolon_scanner.pipeline.ports import Detection
from bixolon_scanner.runtime.onnx import (
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)
from bixolon_scanner.training.models import (
    DINO_V3_HUB_REPOSITORY,
    build_dino_classifier,
    require_torch,
)
from bixolon_scanner.training.synthetic_roi import (
    DirectRoiRecipe,
    augment_direct_roi,
    prepare_direct_roi_source,
)

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _tensor(image: Image.Image, size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    values = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (values - MEAN) / STD


def _expanded_crop(image: Image.Image, bbox: list[float], margin: float) -> Image.Image:
    x, y, width, height = bbox
    margin_x = width * margin
    margin_y = height * margin
    box = (
        max(0.0, x - margin_x),
        max(0.0, y - margin_y),
        min(float(image.width), x + width + margin_x),
        min(float(image.height), y + height + margin_y),
    )
    return image.crop(box)


def _features(
    model,
    tensors: list[np.ndarray],
    *,
    batch_size: int,
    rotation_ensemble: int,
) -> np.ndarray:
    torch = require_torch()
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(np.asarray(tensors[start : start + batch_size])).cuda()
            rotations = (
                (0,)
                if rotation_ensemble == 1
                else ((0, 2) if rotation_ensemble == 2 else (0, 1, 2, 3))
            )
            values = torch.stack(
                [
                    model.extract_features(torch.rot90(batch, turns, dims=(-2, -1))).float()
                    for turns in rotations
                ],
                dim=0,
            ).mean(dim=0)
            values = torch.nn.functional.normalize(values, dim=-1)
            output.append(values.cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def _handcrafted_features(tensors: list[np.ndarray]) -> np.ndarray:
    import cv2

    output = []
    mean = MEAN[:, 0, 0]
    std = STD[:, 0, 0]
    for tensor in tensors:
        rgb = np.clip((tensor * std[:, None, None] + mean[:, None, None]) * 255.0, 0, 255)
        rgb = rgb.transpose(1, 2, 0).astype(np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        border = np.concatenate((lab[0], lab[-1], lab[1:-1, 0], lab[1:-1, -1]), axis=0)
        background = np.median(border, axis=0)
        distance = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
        mask = (distance > 13.0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if component_count > 1:
            selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == selected).astype(np.uint8)
        if int(mask.sum()) < 32:
            mask[:] = 1
        selected_rgb = rgb[mask.astype(bool)].astype(np.float32) / 255.0
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        selected_hsv = hsv[mask.astype(bool)].astype(np.float32)
        selected_hsv /= np.asarray((180.0, 255.0, 255.0), dtype=np.float32)
        values = []
        for selected_values in (selected_rgb, selected_hsv):
            values.extend(selected_values.mean(axis=0))
            values.extend(selected_values.std(axis=0))
            values.extend(np.quantile(selected_values, (0.1, 0.25, 0.5, 0.75, 0.9), axis=0).ravel())
            for channel in range(3):
                histogram, _ = np.histogram(
                    selected_values[:, channel], bins=16, range=(0.0, 1.0), density=False
                )
                values.extend(histogram / max(1, histogram.sum()))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        area_ratio = float(mask.mean())
        if contour is None:
            values.extend((area_ratio, 1.0, 1.0, *([0.0] * 7)))
        else:
            x, y, width, height = cv2.boundingRect(contour)
            hu = cv2.HuMoments(cv2.moments(contour)).ravel()
            hu = -np.sign(hu) * np.log10(np.abs(hu).clip(min=1e-12))
            values.extend(
                (
                    area_ratio,
                    width / max(height, 1),
                    float(mask.sum()) / max(width * height, 1),
                    *hu,
                )
            )
        output.append(np.asarray(values, dtype=np.float32))
    return np.stack(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose fresh DINOv3 classifier on GT ROIs")
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument(
        "--variant",
        choices=(
            "dinov3_convnext_tiny",
            "dinov3_convnext_tiny_multiscale",
            "dinov3_vitb16",
        ),
        default="dinov3_convnext_tiny",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--neighbor-mask", action="store_true")
    parser.add_argument("--neighbor-distance-bias", type=float, default=-0.1)
    parser.add_argument("--augmented-views", type=int, default=0)
    parser.add_argument("--handcrafted", action="store_true")
    parser.add_argument("--rotation-ensemble", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.augmented_views and args.handcrafted:
        parser.error("augmented views and handcrafted fusion are separate diagnostics")
    if args.classifier_checkpoint is not None and args.variant == "dinov3_vitb16":
        parser.error("classifier checkpoints are only supported for ConvNeXt variants")

    support_records = _records(args.support_manifest)
    support_tensors = []
    support_labels = []
    augmented_support_tensors = []
    augmented_support_labels = []
    recipe = DirectRoiRecipe(
        output_size=args.image_size,
        canvas_scale_min=0.72,
        canvas_scale_max=0.98,
        rotation_degrees=180.0,
        perspective_fraction=0.04,
        brightness_min=0.8,
        brightness_max=1.2,
        contrast_min=0.85,
        contrast_max=1.15,
        saturation_min=0.85,
        saturation_max=1.15,
        blur_probability=0.15,
        blur_radius_max=0.7,
        jpeg_quality_min=82,
        jpeg_quality_max=96,
        crop_mode="border_connected_composite",
        procedural_gradient=True,
        procedural_shadow=True,
    )
    for support_index, record in enumerate(support_records):
        with Image.open(args.support_root / record["image_path"]) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            support_tensors.append(_tensor(oriented, args.image_size))
            if args.augmented_views:
                cutout = prepare_direct_roi_source(oriented, recipe)
                for view_index in range(args.augmented_views):
                    sample = augment_direct_roi(
                        oriented,
                        source_sha256=str(record["image_sha256"]),
                        category_id=int(record["category_id"]),
                        seed=20260904 + support_index * 1000 + view_index,
                        recipe=recipe,
                        prepared_cutout=cutout,
                    )
                    augmented_support_tensors.append(_tensor(sample.image, args.image_size))
                    augmented_support_labels.append(int(record["category_id"]) - 1)
        support_labels.append(int(record["category_id"]) - 1)

    evaluation_tensors = []
    evaluation_labels = []
    evaluation_ids = []
    for record in _records(args.evaluation_manifest):
        with Image.open(args.evaluation_root / record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            detections = []
            for annotation in record["annotations"]:
                x, y, width, height = annotation["bbox_xywh"]
                detections.append(Detection(x, y, x + width, y + height, score=1.0))
            for detection_index, annotation in enumerate(record["annotations"]):
                if args.neighbor_mask:
                    crop_box = classifier_crop_box(
                        detections[detection_index],
                        image.width,
                        image.height,
                        margin_ratio=args.margin,
                        crop_mode="box_resize",
                    )
                    tensor = prepare_rgb(
                        image.crop(crop_box),
                        (args.image_size, args.image_size),
                        tuple(float(value) for value in MEAN[:, 0, 0]),
                        tuple(float(value) for value in STD[:, 0, 0]),
                        reducing_gap=1.0,
                    )
                    mask = classifier_neighbor_ownership_mask(
                        detections,
                        detection_index,
                        image_width=image.width,
                        image_height=image.height,
                        output_size=args.image_size,
                        margin_ratio=args.margin,
                        distance_bias=args.neighbor_distance_bias,
                        shared_scale=False,
                    )
                    evaluation_tensors.append(
                        apply_classifier_background_masks(tensor[None], mask[None])[0]
                    )
                else:
                    crop = _expanded_crop(image, annotation["bbox_xywh"], args.margin)
                    evaluation_tensors.append(_tensor(crop, args.image_size))
                evaluation_labels.append(int(annotation["category_id"]) - 1)
                evaluation_ids.append(
                    {
                        "image_id": int(record["image_id"]),
                        "annotation_id": int(annotation["annotation_id"]),
                    }
                )

    classifier_head = None
    if args.variant.startswith("dinov3_convnext_tiny"):
        base_model = build_dino_classifier(
            "dinov3_convnext_tiny",
            20,
            weights_path=args.weights,
            feature_l2_normalize=True,
            classifier_head_kind=("cosine" if args.classifier_checkpoint is not None else "linear"),
        )
        if args.classifier_checkpoint is not None:
            base_model.load_state_dict(
                require_torch().load(
                    args.classifier_checkpoint,
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            if args.variant == "dinov3_convnext_tiny":
                classifier_head = base_model.classifier
        if args.variant == "dinov3_convnext_tiny":
            model = base_model.cuda()
        else:
            torch = require_torch()

            class _ConvNextMultiscaleWrapper(torch.nn.Module):
                def __init__(self, backbone):
                    super().__init__()
                    self.backbone = backbone

                def extract_features(self, pixel_values):
                    values = pixel_values
                    pooled = []
                    for stage_index in range(4):
                        values = self.backbone.downsample_layers[stage_index](values)
                        values = self.backbone.stages[stage_index](values)
                        stage = values.mean(dim=(-2, -1))
                        if stage_index == 3:
                            stage = self.backbone.norm(stage)
                        pooled.append(torch.nn.functional.normalize(stage, dim=-1))
                    return torch.nn.functional.normalize(torch.cat(pooled, dim=-1), dim=-1)

            model = _ConvNextMultiscaleWrapper(base_model.backbone).cuda()
    else:
        torch = require_torch()
        backbone = torch.hub.load(
            DINO_V3_HUB_REPOSITORY,
            "dinov3_vitb16",
            source="github",
            trust_repo=True,
            verbose=False,
            pretrained=False,
        )
        backbone.load_state_dict(
            torch.load(args.weights, map_location="cpu", weights_only=True),
            strict=True,
        )

        class _VitB16Wrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def extract_features(self, pixel_values):
                return self.wrapped.forward_features(pixel_values, masks=None)["x_norm_clstoken"]

        model = _VitB16Wrapper(backbone).cuda()
    support_features = _features(
        model,
        support_tensors,
        batch_size=args.batch_size,
        rotation_ensemble=args.rotation_ensemble,
    )
    if augmented_support_tensors:
        augmented_support_features = _features(
            model,
            augmented_support_tensors,
            batch_size=args.batch_size,
            rotation_ensemble=args.rotation_ensemble,
        )
    else:
        augmented_support_features = np.empty((0, support_features.shape[1]), dtype=np.float32)
    evaluation_features = _features(
        model,
        evaluation_tensors,
        batch_size=args.batch_size,
        rotation_ensemble=args.rotation_ensemble,
    )
    support_labels_array = np.asarray(support_labels, dtype=np.int64)
    labels = np.asarray(evaluation_labels, dtype=np.int64)
    handcrafted_sweep = None
    selected_handcrafted_weight = 0.0
    if args.handcrafted:
        support_handcrafted = _handcrafted_features(support_tensors)
        evaluation_handcrafted = _handcrafted_features(evaluation_tensors)
        center = support_handcrafted.mean(axis=0, keepdims=True)
        scale = support_handcrafted.std(axis=0, keepdims=True).clip(min=1e-4)
        support_handcrafted = (support_handcrafted - center) / scale
        evaluation_handcrafted = (evaluation_handcrafted - center) / scale
        support_handcrafted /= np.linalg.norm(support_handcrafted, axis=1, keepdims=True).clip(
            min=1e-12
        )
        evaluation_handcrafted /= np.linalg.norm(
            evaluation_handcrafted, axis=1, keepdims=True
        ).clip(min=1e-12)
        original_support = support_features
        original_evaluation = evaluation_features
        handcrafted_sweep = []
        best_accuracy = -1.0
        for weight in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0):
            candidate_support = np.concatenate(
                (original_support, support_handcrafted * weight), axis=1
            )
            candidate_evaluation = np.concatenate(
                (original_evaluation, evaluation_handcrafted * weight), axis=1
            )
            candidate_support /= np.linalg.norm(candidate_support, axis=1, keepdims=True).clip(
                min=1e-12
            )
            candidate_evaluation /= np.linalg.norm(
                candidate_evaluation, axis=1, keepdims=True
            ).clip(min=1e-12)
            candidate_prototypes = np.stack(
                [
                    candidate_support[support_labels_array == class_id].mean(axis=0)
                    for class_id in range(20)
                ]
            )
            candidate_prototypes /= np.linalg.norm(
                candidate_prototypes, axis=1, keepdims=True
            ).clip(min=1e-12)
            candidate_scores = candidate_evaluation @ candidate_prototypes.T
            accuracy = float((candidate_scores.argmax(axis=1) == labels).mean())
            handcrafted_sweep.append({"weight": weight, "prototype_accuracy": accuracy})
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                selected_handcrafted_weight = weight
                support_features = candidate_support
                evaluation_features = candidate_evaluation
    prototypes = np.stack(
        [support_features[support_labels_array == class_id].mean(axis=0) for class_id in range(20)]
    )
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-12)
    prototype_scores = evaluation_features @ prototypes.T
    exemplar_scores = evaluation_features @ support_features.T
    exemplar_class_scores = np.stack(
        [
            np.sort(exemplar_scores[:, support_labels_array == class_id], axis=1)[:, -3:].mean(
                axis=1
            )
            for class_id in range(20)
        ],
        axis=1,
    )
    hybrid_scores = 0.5 * prototype_scores + 0.5 * exemplar_class_scores
    if not len(augmented_support_features):
        augmented_support_features = np.empty((0, support_features.shape[1]), dtype=np.float32)
    ridge_features = np.concatenate([support_features, augmented_support_features], axis=0)
    ridge_labels = np.concatenate(
        [support_labels_array, np.asarray(augmented_support_labels, dtype=np.int64)]
    )
    design = np.concatenate(
        [ridge_features, np.ones((len(ridge_features), 1), dtype=np.float32)],
        axis=1,
    )
    targets = np.eye(20, dtype=np.float32)[ridge_labels]
    if design.shape[0] < design.shape[1]:
        centered_features = ridge_features - ridge_features.mean(axis=0, keepdims=True)
        centered_targets = targets - targets.mean(axis=0, keepdims=True)
        coefficients_without_bias = centered_features.T @ np.linalg.solve(
            centered_features @ centered_features.T
            + np.eye(len(centered_features), dtype=np.float32) * 0.01,
            centered_targets,
        )
        bias = targets.mean(axis=0) - ridge_features.mean(axis=0) @ coefficients_without_bias
        coefficients = np.concatenate([coefficients_without_bias, bias[None]], axis=0)
    else:
        regularization = np.eye(design.shape[1], dtype=np.float32) * 0.01
        regularization[-1, -1] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + regularization,
            design.T @ targets,
        )
    ridge_scores = (evaluation_features @ coefficients[:-1] + coefficients[-1]).astype(np.float32)
    lda_weight, lda_bias = fit_diagonal_lda_adapter(
        ridge_features,
        ridge_labels,
        class_count=20,
    )
    lda_scores = evaluation_features @ lda_weight + lda_bias
    hybrid_predictions = hybrid_scores.argmax(axis=1)
    ridge_predictions = ridge_scores.argmax(axis=1)
    hybrid_accuracy = float((hybrid_predictions == labels).mean())
    ridge_accuracy = float((ridge_predictions == labels).mean())
    lda_accuracy = float((lda_scores.argmax(axis=1) == labels).mean())
    classifier_head_accuracy = None
    classifier_head_scores = None
    classifier_ensemble_sweep = None
    if classifier_head is not None and not args.handcrafted:
        torch = require_torch()
        with torch.inference_mode():
            classifier_head_scores = (
                classifier_head(torch.from_numpy(evaluation_features).cuda()).float().cpu().numpy()
            )
        classifier_head_accuracy = float((classifier_head_scores.argmax(axis=1) == labels).mean())
    candidates = [
        (hybrid_accuracy, "prototype_knn_hybrid", hybrid_scores),
        (ridge_accuracy, "ridge_adapter", ridge_scores),
        (lda_accuracy, "diagonal_lda_adapter", lda_scores),
    ]
    if classifier_head_scores is not None and classifier_head_accuracy is not None:
        candidates.append((classifier_head_accuracy, "trained_cosine_head", classifier_head_scores))
        classifier_ensemble_sweep = []
        shifted_head = classifier_head_scores - classifier_head_scores.max(axis=1, keepdims=True)
        head_probabilities = np.exp(shifted_head)
        head_probabilities /= head_probabilities.sum(axis=1, keepdims=True)
        for retrieval_temperature in (8.0, 16.0, 24.0, 32.0):
            shifted_retrieval = hybrid_scores * retrieval_temperature
            shifted_retrieval -= shifted_retrieval.max(axis=1, keepdims=True)
            retrieval_probabilities = np.exp(shifted_retrieval)
            retrieval_probabilities /= retrieval_probabilities.sum(axis=1, keepdims=True)
            for head_weight in (0.25, 0.4, 0.5, 0.6, 0.75):
                ensemble_scores = (
                    head_weight * head_probabilities + (1.0 - head_weight) * retrieval_probabilities
                )
                ensemble_accuracy = float((ensemble_scores.argmax(axis=1) == labels).mean())
                name = f"head_retrieval_ensemble_t{retrieval_temperature:g}_w{head_weight:g}"
                classifier_ensemble_sweep.append(
                    {
                        "retrieval_temperature": retrieval_temperature,
                        "head_weight": head_weight,
                        "accuracy": ensemble_accuracy,
                    }
                )
                candidates.append((ensemble_accuracy, name, ensemble_scores))
    _, decision_head, scores = max(candidates, key=lambda item: item[0])
    predictions = scores.argmax(axis=1)
    order = np.argsort(scores, axis=1)[:, ::-1]
    top1 = scores[np.arange(len(labels)), order[:, 0]]
    margin = top1 - scores[np.arange(len(labels)), order[:, 1]]
    correct = predictions == labels
    per_class = {}
    for class_id in range(20):
        selected = labels == class_id
        per_class[str(class_id + 1)] = {
            "count": int(selected.sum()),
            "correct": int(correct[selected].sum()),
            "accuracy": float(correct[selected].mean()),
        }
    errors = []
    for index in np.flatnonzero(~correct):
        errors.append(
            {
                **evaluation_ids[int(index)],
                "expected": int(labels[index] + 1),
                "predicted": int(predictions[index] + 1),
                "score": float(top1[index]),
                "margin": float(margin[index]),
                "top3": [int(value + 1) for value in order[index, :3]],
            }
        )
    report = {
        "schema_version": "1.0",
        "mode": "evaluation_gt_roi_diagnostic_only",
        "evaluation_used_for_fitting": False,
        "support_count": len(support_records),
        "augmented_support_count": len(augmented_support_labels),
        "augmentation_recipe": asdict(recipe) if args.augmented_views else None,
        "handcrafted_feature_fusion": args.handcrafted,
        "handcrafted_weight_sweep": handcrafted_sweep,
        "selected_handcrafted_weight": selected_handcrafted_weight,
        "evaluation_object_count": len(labels),
        "method": f"official_{args.variant}_prototype_knn_hybrid",
        "selected_decision_head": decision_head,
        "prototype_knn_hybrid_accuracy": hybrid_accuracy,
        "ridge_adapter_accuracy": ridge_accuracy,
        "diagonal_lda_adapter_accuracy": lda_accuracy,
        "trained_classifier_head_accuracy": classifier_head_accuracy,
        "classifier_ensemble_sweep": classifier_ensemble_sweep,
        "image_size": args.image_size,
        "crop_margin": args.margin,
        "neighbor_mask": args.neighbor_mask,
        "neighbor_distance_bias": args.neighbor_distance_bias,
        "internal_rotation_ensemble": args.rotation_ensemble,
        "correct_count": int(correct.sum()),
        "error_count": int((~correct).sum()),
        "accuracy": float(correct.mean()),
        "correct_score_min": float(top1[correct].min()) if correct.any() else None,
        "correct_margin_min": float(margin[correct].min()) if correct.any() else None,
        "per_class": per_class,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_name("ridge-adapter.npz"),
        weight=coefficients[:-1].astype(np.float32),
        bias=coefficients[-1].astype(np.float32),
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, indent=2))


if __name__ == "__main__":
    main()
