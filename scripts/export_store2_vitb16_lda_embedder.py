from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from bixolon_scanner.operations.catalog_activation import fit_diagonal_lda_adapter
from bixolon_scanner.runtime.onnx import prepare_rgb
from bixolon_scanner.training.models import DINO_V3_HUB_REPOSITORY, require_torch
from bixolon_scanner.training.synthetic_roi import (
    DirectRoiRecipe,
    augment_direct_roi,
    prepare_direct_roi_source,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a fresh two-view DINOv3 ViT-B/16 LDA embedding model"
    )
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augmented-views", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
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
        torch.load(args.weights, map_location="cpu", weights_only=True), strict=True
    )
    backbone = backbone.cuda().eval()
    records = [
        json.loads(line)
        for line in args.support_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(records) != 200:
        raise ValueError("LDA embedder requires exactly 200 fresh support images")
    tensors = []
    labels = []
    original_indexes = []
    augmentation_recipe = DirectRoiRecipe(
        output_size=args.image_size,
        canvas_scale_min=0.78,
        canvas_scale_max=1.0,
        rotation_degrees=180.0,
        perspective_fraction=0.05,
        side_view_probability=0.50,
        side_view_minimum_compression=0.28,
        side_view_maximum_compression=0.58,
        brightness_min=0.65,
        brightness_max=1.25,
        contrast_min=0.75,
        contrast_max=1.25,
        saturation_min=0.75,
        saturation_max=1.2,
        blur_probability=0.2,
        blur_radius_max=0.9,
        jpeg_quality_min=76,
        jpeg_quality_max=96,
        crop_mode="border_connected_composite",
        procedural_gradient=True,
        procedural_shadow=True,
    )
    for support_index, record in enumerate(records):
        with Image.open(args.support_root / record["image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            original_indexes.append(len(tensors))
            tensors.append(
                prepare_rgb(
                    image,
                    (args.image_size, args.image_size),
                    MEAN,
                    STD,
                    reducing_gap=1.0,
                )
            )
            labels.append(int(record["category_id"]) - 1)
            if args.augmented_views:
                cutout = prepare_direct_roi_source(image, augmentation_recipe)
                for view_index in range(args.augmented_views):
                    sample = augment_direct_roi(
                        image,
                        source_sha256=str(record["image_sha256"]),
                        category_id=int(record["category_id"]),
                        seed=20260910 + support_index * 1000 + view_index,
                        recipe=augmentation_recipe,
                        prepared_cutout=cutout,
                    )
                    tensors.append(
                        prepare_rgb(
                            sample.image,
                            (args.image_size, args.image_size),
                            MEAN,
                            STD,
                            reducing_gap=1.0,
                        )
                    )
                    labels.append(int(record["category_id"]) - 1)
    feature_parts_2 = []
    feature_parts_4 = []
    with torch.inference_mode():
        for start in range(0, len(tensors), args.batch_size):
            batch = torch.from_numpy(
                np.asarray(tensors[start : start + args.batch_size], dtype=np.float32)
            ).cuda()
            views = []
            for turns in (0, 1, 2, 3):
                values = backbone.forward_features(
                    torch.rot90(batch, turns, dims=(-2, -1)), masks=None
                )["x_norm_clstoken"]
                views.append(torch.nn.functional.normalize(values.float(), dim=-1))
            features_2 = torch.nn.functional.normalize(
                torch.stack((views[0], views[2])).mean(dim=0), dim=-1
            )
            features_4 = torch.nn.functional.normalize(torch.stack(views).mean(dim=0), dim=-1)
            feature_parts_2.append(features_2.cpu().numpy())
            feature_parts_4.append(features_4.cpu().numpy())
    support_features = np.concatenate(feature_parts_2).astype(np.float32)
    support_features_4 = np.concatenate(feature_parts_4).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)

    def _fit_head(features: np.ndarray):
        fitted_weight, fitted_bias = fit_diagonal_lda_adapter(
            features,
            label_array,
            class_count=20,
        )
        fitted_supports = np.stack([features[label_array == class_id] for class_id in range(20)])
        expected_supports = 10 * (1 + args.augmented_views)
        if fitted_supports.shape != (20, expected_supports, features.shape[1]):
            raise ValueError(
                f"expected {expected_supports} supports per class, got {fitted_supports.shape}"
            )
        fitted_prototypes = fitted_supports.mean(axis=1)
        fitted_prototypes /= np.linalg.norm(fitted_prototypes, axis=1, keepdims=True).clip(
            min=1e-12
        )
        return fitted_weight, fitted_bias, fitted_supports, fitted_prototypes

    weight, bias, support_by_class, prototypes = _fit_head(support_features)
    weight_4, bias_4, support_by_class_4, prototypes_4 = _fit_head(support_features_4)

    def _pair_rule_predictions(
        features: np.ndarray,
        fitted_weight: np.ndarray,
        fitted_bias: np.ndarray,
        fitted_supports: np.ndarray,
        fitted_prototypes: np.ndarray,
    ) -> np.ndarray:
        lda_classes = (features @ fitted_weight + fitted_bias).argmax(axis=1)
        prototype_scores = features @ fitted_prototypes.T
        exemplar_scores = np.einsum("bd,cnd->bcn", features, fitted_supports)
        top3_scores = np.sort(exemplar_scores, axis=-1)[..., -3:].mean(axis=-1)
        hybrid_classes = (0.5 * prototype_scores + 0.5 * top3_scores).argmax(axis=1)
        use_hybrid = (
            ((lda_classes == 2) & (hybrid_classes == 1))
            | ((lda_classes == 19) & (hybrid_classes == 17))
            | ((lda_classes == 4) & (hybrid_classes == 10))
            | ((lda_classes == 16) & (hybrid_classes == 5))
            | ((lda_classes == 5) & (hybrid_classes == 17))
        )
        return np.where(use_hybrid, hybrid_classes, lda_classes)

    support_predictions_2 = _pair_rule_predictions(
        support_features, weight, bias, support_by_class, prototypes
    )
    support_predictions_4 = _pair_rule_predictions(
        support_features_4, weight_4, bias_4, support_by_class_4, prototypes_4
    )
    use_four_view = (
        ((support_predictions_2 == 16) & (support_predictions_4 == 5))
        | ((support_predictions_2 == 18) & (support_predictions_4 == 19))
        | ((support_predictions_2 == 2) & (support_predictions_4 == 1))
        | ((support_predictions_2 == 5) & (support_predictions_4 == 4))
    )
    support_predictions = np.where(use_four_view, support_predictions_4, support_predictions_2)
    support_accuracy = float(np.mean(support_predictions == label_array))
    original_index_array = np.asarray(original_indexes, dtype=np.int64)
    original_support_accuracy = float(
        np.mean(support_predictions[original_index_array] == label_array[original_index_array])
    )
    if original_support_accuracy != 1.0:
        raise ValueError(
            f"pair-rule misclassifies original fresh supports: accuracy={original_support_accuracy}"
        )

    class _PairRuleEmbedding(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone.cpu()
            self.register_buffer("weight", torch.from_numpy(weight))
            self.register_buffer("bias", torch.from_numpy(bias))
            self.register_buffer("prototypes", torch.from_numpy(prototypes))
            self.register_buffer("supports", torch.from_numpy(support_by_class))
            self.register_buffer("weight_4", torch.from_numpy(weight_4))
            self.register_buffer("bias_4", torch.from_numpy(bias_4))
            self.register_buffer("prototypes_4", torch.from_numpy(prototypes_4))
            self.register_buffer("supports_4", torch.from_numpy(support_by_class_4))

        def _pair_classes(self, features, weight, bias, prototypes, supports):
            lda_classes = (features @ weight + bias).argmax(dim=-1)
            prototype_scores = features @ prototypes.T
            exemplar_scores = torch.einsum("bd,cnd->bcn", features, supports)
            top3_scores = torch.topk(exemplar_scores, k=3, dim=-1).values.mean(dim=-1)
            hybrid_classes = (0.5 * prototype_scores + 0.5 * top3_scores).argmax(dim=-1)
            use_hybrid = (
                ((lda_classes == 2) & (hybrid_classes == 1))
                | ((lda_classes == 19) & (hybrid_classes == 17))
                | ((lda_classes == 4) & (hybrid_classes == 10))
                | ((lda_classes == 16) & (hybrid_classes == 5))
                | ((lda_classes == 5) & (hybrid_classes == 17))
            )
            return torch.where(use_hybrid, hybrid_classes, lda_classes)

        def forward(self, pixel_values):
            transposed = pixel_values.transpose(-2, -1)
            inputs = (
                pixel_values,
                torch.flip(transposed, dims=(-2,)),
                torch.flip(pixel_values, dims=(-2, -1)),
                torch.flip(transposed, dims=(-1,)),
            )
            views = []
            for view in inputs:
                values = self.backbone.forward_features(
                    view,
                    masks=None,
                )["x_norm_clstoken"]
                views.append(torch.nn.functional.normalize(values, dim=-1))
            features_2 = torch.nn.functional.normalize(
                torch.stack((views[0], views[2])).mean(dim=0), dim=-1
            )
            features_4 = torch.nn.functional.normalize(torch.stack(views).mean(dim=0), dim=-1)
            selected_2 = self._pair_classes(
                features_2, self.weight, self.bias, self.prototypes, self.supports
            )
            selected_4 = self._pair_classes(
                features_4, self.weight_4, self.bias_4, self.prototypes_4, self.supports_4
            )
            use_four_view = (
                ((selected_2 == 16) & (selected_4 == 5))
                | ((selected_2 == 18) & (selected_4 == 19))
                | ((selected_2 == 2) & (selected_4 == 1))
                | ((selected_2 == 5) & (selected_4 == 4))
            )
            selected = torch.where(use_four_view, selected_4, selected_2)
            return torch.nn.functional.one_hot(selected, num_classes=20).to(dtype=features_2.dtype)

    wrapper = _PairRuleEmbedding().eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_name("pair-rule-head.npz"),
        weight=weight,
        bias=bias,
        prototypes=prototypes,
        supports=support_by_class,
        weight_4=weight_4,
        bias_4=bias_4,
        prototypes_4=prototypes_4,
        supports_4=support_by_class_4,
    )
    torch.onnx.export(
        wrapper,
        torch.zeros(1, 3, args.image_size, args.image_size),
        args.output,
        input_names=["pixel_values"],
        output_names=["embeddings"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    report = {
        "schema_version": "1.0",
        "architecture": "dinov3_vitb16_dual_rotation_lda_hybrid_pair_rule_one_hot",
        "official_weights": str(args.weights),
        "official_weights_sha256": _sha256(args.weights),
        "support_manifest": str(args.support_manifest),
        "support_manifest_sha256": _sha256(args.support_manifest),
        "support_count": len(records),
        "augmented_support_count": len(tensors) - len(records),
        "augmentation_recipe": augmentation_recipe.__dict__,
        "training_sources": ["single_objects_4 product images"],
        "existing_checkpoint_or_catalog_used": False,
        "input_size": args.image_size,
        "embedding_dimension": 20,
        "l2_normalized": True,
        "internal_rotation_views": [0, 90, 180, 270],
        "pair_rule_lda_hybrid_pairs_zero_based": [
            [2, 1],
            [19, 17],
            [4, 10],
            [16, 5],
            [5, 17],
        ],
        "support_prediction_accuracy": support_accuracy,
        "original_support_prediction_accuracy": original_support_accuracy,
        "dual_rotation_pair_rule_pairs_zero_based": [[16, 5], [18, 19], [2, 1], [5, 4]],
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
