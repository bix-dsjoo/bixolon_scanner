from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from ..contracts.catalog import StoreCatalogPackage
from ..contracts.model_package import (
    ClassifierMetadata,
    ClassLabel,
    NeighborMaskClassifierMetadata,
    NeighborMaskClassifierView,
)
from ..contracts.runtime_package_v2 import RuntimePackageV2
from ..pipeline.ports import ClassificationResult, Detection
from .imaging import image_original_size
from .onnx import (
    OrtRunner,
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(~np.isfinite(array)) or np.any(norms <= 1e-12):
        raise ValueError("embeddings must be finite and have non-zero norm")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


@dataclass(frozen=True)
class MetricTransform:
    input_dimension: int
    residual_weight: float
    projection_weight: float
    mean: np.ndarray | None = None
    matrix: np.ndarray | None = None

    @property
    def output_dimension(self) -> int:
        projected = 0 if self.matrix is None else int(self.matrix.shape[1])
        residual = self.input_dimension if self.residual_weight > 0.0 else 0
        return residual + projected

    def apply(self, values: np.ndarray) -> np.ndarray:
        raw = l2_normalize(values)
        if raw.shape[1] != self.input_dimension:
            raise ValueError("embedding dimension does not match metric policy")
        branches = []
        if self.residual_weight > 0.0:
            branches.append(raw * np.float32(self.residual_weight))
        if self.projection_weight > 0.0:
            if self.mean is None or self.matrix is None:
                raise ValueError("metric projection arrays are missing")
            projected = l2_normalize((raw - self.mean) @ self.matrix)
            branches.append(projected * np.float32(self.projection_weight))
        return l2_normalize(np.concatenate(branches, axis=1))


def load_metric_transform(package: RuntimePackageV2) -> MetricTransform:
    metadata = package.metadata.metric_projection
    mean = None
    matrix = None
    if package.metric_projection_path is not None:
        with package.metric_projection_path.open("rb") as stream:
            with np.load(stream, allow_pickle=False) as payload:
                if set(payload.files) != {"mean", "matrix"}:
                    raise ValueError("metric projection must contain mean and matrix arrays")
                mean = np.asarray(payload["mean"], dtype=np.float32).copy()
                matrix = np.asarray(payload["matrix"], dtype=np.float32).copy()
        if mean.shape != (1, metadata.input_dimension):
            raise ValueError("metric projection mean shape is invalid")
        if matrix.shape != (metadata.input_dimension, metadata.output_dimension):
            raise ValueError("metric projection matrix shape is invalid")
    return MetricTransform(
        input_dimension=metadata.input_dimension,
        residual_weight=metadata.residual_weight,
        projection_weight=metadata.projection_weight,
        mean=mean,
        matrix=matrix,
    )


class OnnxEmbedder:
    def __init__(
        self,
        package: RuntimePackageV2,
        provider: Literal["cuda", "cpu"],
        cuda_dll_dir: Path | None = None,
        *,
        cpu_intra_op_threads: int = 0,
    ):
        self.metadata = package.metadata.embedder
        self.runner = OrtRunner(
            package.embedder_path,
            provider,
            cuda_dll_dir,
            cpu_intra_op_threads=cpu_intra_op_threads,
        )
        self.transform = load_metric_transform(package)
        self.version = self.metadata.version

    def warmup(self) -> None:
        height, width = self.metadata.input_size
        batch_sizes = self.metadata.warmup_batch_sizes if self.runner.cuda else [1]
        for batch_size in batch_sizes:
            self.runner.run(
                [self.metadata.output_name],
                self.metadata.input_name,
                np.zeros((batch_size, 3, height, width), dtype=np.float32),
            )

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        return self.transform.apply(self.embed_images_raw(images))

    def embed_images_raw(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.metadata.embedding_dimension), dtype=np.float32)
        batch = np.stack(
            [
                prepare_rgb(
                    image,
                    self.metadata.input_size,
                    self.metadata.mean,
                    self.metadata.std,
                    reducing_gap=self.metadata.resize_reducing_gap,
                )
                for image in images
            ]
        )
        return self._run_raw_tensors(batch)

    def _run_raw_tensors(self, batch: np.ndarray) -> np.ndarray:
        (raw,) = self.runner.run(
            [self.metadata.output_name],
            self.metadata.input_name,
            batch.astype(np.float32),
        )
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (len(batch), self.metadata.embedding_dimension):
            raise ValueError("embedder output shape does not match runtime metadata")
        return raw

    def _embed_tensors(self, batch: np.ndarray) -> np.ndarray:
        return self.transform.apply(self._run_raw_tensors(batch))

    def embed_detections(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> np.ndarray:
        return self.transform.apply(self.embed_detections_raw(image, detections))

    def embed_detections_raw(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> np.ndarray:
        if isinstance(image, Image.Image):
            source = image
            original_width, original_height = image_original_size(image)
            scale_x = source.width / original_width
            scale_y = source.height / original_height
        else:
            source = Image.fromarray(image, mode="RGB")
            original_width, original_height = source.size
            scale_x = scale_y = 1.0
        crops: list[np.ndarray] = []
        for detection in detections:
            box = classifier_crop_box(
                detection,
                original_width,
                original_height,
                margin_ratio=self.metadata.crop_margin_ratio,
                crop_mode=self.metadata.crop_mode,
            )
            scaled_box = (
                int(np.floor(box[0] * scale_x)),
                int(np.floor(box[1] * scale_y)),
                int(np.ceil(box[2] * scale_x)),
                int(np.ceil(box[3] * scale_y)),
            )
            crops.append(
                prepare_rgb(
                    source.crop(scaled_box),
                    self.metadata.input_size,
                    self.metadata.mean,
                    self.metadata.std,
                    reducing_gap=self.metadata.resize_reducing_gap,
                )
            )
        batch = np.stack(crops).astype(np.float32, copy=False)
        if self.metadata.neighbor_mask:
            masks = np.stack(
                [
                    classifier_neighbor_ownership_mask(
                        detections,
                        index,
                        image_width=original_width,
                        image_height=original_height,
                        output_size=batch.shape[-1],
                        margin_ratio=self.metadata.crop_margin_ratio,
                        distance_bias=self.metadata.neighbor_distance_bias,
                        shared_scale=self.metadata.neighbor_shared_scale,
                    )
                    for index in range(len(detections))
                ]
            )
            batch = apply_classifier_background_masks(batch, masks)
        return self._run_raw_tensors(batch)


def _load_array(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        return np.asarray(np.load(stream, allow_pickle=False), dtype=np.float32)


class OnnxCatalogClassifier:
    def __init__(
        self, runtime: RuntimePackageV2, catalog: StoreCatalogPackage, embedder: OnnxEmbedder
    ):
        if (
            catalog.metadata.embedder_id != runtime.metadata.embedder.embedder_id
            or catalog.metadata.embedder_version != runtime.metadata.embedder.version
            or catalog.metadata.classifier_policy_version
            != runtime.metadata.classifier_policy.version
        ):
            raise ValueError("Catalog is not compatible with the selected runtime package")
        self.runtime = runtime
        self.catalog = catalog
        self.embedder = embedder
        self.policy = runtime.metadata.classifier_policy
        self.version = self.policy.version
        self.labels = catalog.metadata.labels
        self.supports = _load_array(catalog.supports_path)
        self.prototypes = _load_array(catalog.prototypes_path)
        expected = (catalog.metadata.support_count, catalog.metadata.embedding_dimension)
        if self.supports.shape != expected:
            raise ValueError("catalog support array shape is invalid")
        if self.prototypes.shape != (len(self.labels), expected[1]):
            raise ValueError("catalog prototype array shape is invalid")
        if expected[1] != embedder.transform.output_dimension:
            raise ValueError("catalog and runtime metric dimensions do not match")
        self.restricted_ids = set(catalog.activation.restricted_class_ids)
        self.restricted_pairs = {pair.class_ids for pair in catalog.activation.restricted_pairs}
        self.adapter_weight = None
        self.adapter_bias = None
        if catalog.adapter_path is not None:
            with catalog.adapter_path.open("rb") as stream:
                with np.load(stream, allow_pickle=False) as payload:
                    if set(payload.files) != {"weight", "bias"}:
                        raise ValueError("Catalog adapter must contain weight and bias")
                    self.adapter_weight = np.asarray(payload["weight"], dtype=np.float32).copy()
                    self.adapter_bias = np.asarray(payload["bias"], dtype=np.float32).copy()
            if self.adapter_weight.shape != (expected[1], len(self.labels)):
                raise ValueError("Catalog adapter weight shape is invalid")
            if self.adapter_bias.shape != (len(self.labels),):
                raise ValueError("Catalog adapter bias shape is invalid")
        self.metadata = ClassifierMetadata(
            filename=runtime.metadata.embedder.filename,
            version=self.policy.version,
            input_name=runtime.metadata.embedder.input_name,
            input_size=runtime.metadata.embedder.input_size,
            mean=runtime.metadata.embedder.mean,
            std=runtime.metadata.embedder.std,
            crop_margin_ratio=runtime.metadata.embedder.crop_margin_ratio,
            crop_mode=runtime.metadata.embedder.crop_mode,
            approval_threshold=(
                1.0 if self.adapter_weight is None else self._ridge_approval_threshold()
            ),
            temperature=1.0,
            labels=[
                ClassLabel(class_id=label.class_id, class_name=label.class_name)
                for label in self.labels
            ],
            resize_reducing_gap=runtime.metadata.embedder.resize_reducing_gap,
            warmup_batch_sizes=runtime.metadata.embedder.warmup_batch_sizes,
            neighbor_mask_inference=(
                None
                if self.adapter_weight is None
                else NeighborMaskClassifierMetadata(
                    views=[
                        NeighborMaskClassifierView(
                            name="catalog_adapter",
                            distance_bias=runtime.metadata.embedder.neighbor_distance_bias,
                            weight=1.0,
                            shared_scale=runtime.metadata.embedder.neighbor_shared_scale,
                        )
                    ],
                    approval_metric="l2_normalized_logit_margin",
                    top3_safety_threshold=float(
                        runtime.metadata.classifier_policy.ridge_top3_minimum_inverse_entropy
                    ),
                )
            ),
        )

    def _ridge_approval_threshold(self) -> float:
        if self.policy.ridge_approval_metric == "top2_pair_probability":
            value = self.policy.ridge_approval_minimum_pair_probability
        else:
            value = self.policy.ridge_approval_minimum_margin
        if value is None:
            raise ValueError("ridge Catalog approval threshold is missing")
        return float(value)

    def _class_scores(self, embeddings: np.ndarray) -> np.ndarray:
        support_similarity = embeddings @ self.supports.T
        prototype_similarity = embeddings @ self.prototypes.T
        scores = np.empty_like(prototype_similarity)
        for class_index, label in enumerate(self.labels):
            start = label.support_offset
            end = start + label.support_count
            values = support_similarity[:, start:end]
            top_k = min(self.policy.support_top_k, values.shape[1])
            nearest = np.partition(values, values.shape[1] - top_k, axis=1)[:, -top_k:].mean(axis=1)
            scores[:, class_index] = (
                self.policy.prototype_weight * prototype_similarity[:, class_index]
                + (1.0 - self.policy.prototype_weight) * nearest
            )
        return scores

    def classify(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> ClassificationResult:
        raw_embeddings = self.embedder.embed_detections_raw(image, detections)
        embeddings = self.embedder.transform.apply(raw_embeddings)
        cosine_scores = self._class_scores(embeddings)
        if self.adapter_weight is not None:
            return self._classify_adapter(raw_embeddings, cosine_scores)
        order = np.argsort(-cosine_scores, axis=1, kind="stable")
        rows = np.arange(len(detections))
        top1 = cosine_scores[rows, order[:, 0]]
        top2 = (
            cosine_scores[rows, order[:, 1]]
            if cosine_scores.shape[1] > 1
            else np.full_like(top1, -1.0)
        )
        margin = top1 - top2
        similarity_denominator = max(
            self.policy.approval_minimum_similarity - self.policy.ood_maximum_similarity,
            1e-6,
        )
        similarity_safety = np.clip(
            (top1 - self.policy.ood_maximum_similarity) / similarity_denominator, 0.0, 1.0
        )
        margin_safety = np.clip(margin / max(self.policy.approval_minimum_margin, 1e-6), 0.0, 1.0)
        approval_scores = np.minimum(similarity_safety, margin_safety).astype(np.float32)
        recapture_reasons: list[str | None] = []
        unknown_reasons: list[str | None] = []
        approval_blocked = np.zeros(len(detections), dtype=bool)
        for row, indices in enumerate(order):
            top_ids = tuple(self.labels[int(index)].class_id for index in indices[:2])
            pair = tuple(sorted(top_ids)) if len(top_ids) == 2 else None
            restricted = top_ids[0] in self.restricted_ids or pair in self.restricted_pairs
            approval_blocked[row] = restricted
            third = cosine_scores[row, indices[2]] if len(indices) >= 3 else -1.0
            if top1[row] < self.policy.ood_maximum_similarity:
                recapture_reasons.append("CLASSIFIER_OUT_OF_CATALOG")
            elif third < self.policy.top3_minimum_similarity:
                recapture_reasons.append("CLASSIFIER_TOP3_UNSAFE")
            else:
                recapture_reasons.append(None)
            if restricted:
                unknown_reasons.append("CLASSIFIER_CATALOG_CONFLICT")
            elif margin[row] < self.policy.approval_minimum_margin:
                unknown_reasons.append("CLASSIFIER_AMBIGUOUS_TOP2")
            else:
                unknown_reasons.append("BELOW_APPROVAL_THRESHOLD")
        ranking_scores = np.clip((cosine_scores + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
        return ClassificationResult(
            logits=cosine_scores,
            ranking_logits=cosine_scores,
            retrieval_logits=cosine_scores,
            approval_scores=approval_scores,
            ranking_scores=ranking_scores,
            segment_recapture_reasons=tuple(recapture_reasons),
            unknown_reasons=tuple(unknown_reasons),
            approval_blocked=approval_blocked,
        )

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values.astype(np.float64) - values.max(axis=1, keepdims=True)
        exponential = np.exp(shifted)
        return (exponential / exponential.sum(axis=1, keepdims=True)).astype(np.float32)

    def _classify_adapter(
        self, embeddings: np.ndarray, retrieval_scores: np.ndarray
    ) -> ClassificationResult:
        if self.adapter_weight is None or self.adapter_bias is None:
            raise ValueError("Catalog adapter is not loaded")
        approval_threshold = self._ridge_approval_threshold()
        minimum_entropy = self.policy.ridge_top3_minimum_inverse_entropy
        if minimum_entropy is None:
            raise ValueError("ridge Catalog policy is incomplete")
        logits = l2_normalize(embeddings) @ self.adapter_weight + self.adapter_bias
        logit_order = np.argsort(-logits, axis=1, kind="stable")
        sorted_logits = np.take_along_axis(logits, logit_order, axis=1)
        logit_gap = sorted_logits[:, 0] - sorted_logits[:, 1]
        normalized_margin = logit_gap / np.linalg.norm(logits, axis=1).clip(min=1e-12)
        if self.policy.ridge_approval_metric == "top2_pair_probability":
            scaled_gap = logit_gap / np.float32(self.policy.ridge_pair_temperature)
            approval_scores = (1.0 / (1.0 + np.exp(-scaled_gap))).astype(np.float32)
        else:
            approval_scores = np.clip(normalized_margin, 0.0, 1.0).astype(np.float32)
        probabilities = self._softmax(logits)
        ranks = np.empty_like(logit_order)
        np.put_along_axis(
            ranks,
            logit_order,
            np.arange(logits.shape[1], dtype=logit_order.dtype)[None],
            axis=1,
        )
        ranking_logits = 1.0 / (ranks + 1.0) + probabilities * 1e-3
        ranking_scores = self._softmax(ranking_logits)
        inverse_entropy = np.sum(ranking_scores * np.log(ranking_scores.clip(1e-12)), axis=1)
        retrieval_order = np.argsort(-retrieval_scores, axis=1, kind="stable")
        rows = np.arange(len(embeddings))
        retrieval_top1 = retrieval_scores[rows, retrieval_order[:, 0]]
        retrieval_minimum = self.policy.ridge_retrieval_minimum_similarity
        recapture_reasons: list[str | None] = []
        unknown_reasons: list[str | None] = []
        approval_blocked = np.zeros(len(embeddings), dtype=bool)
        for row, indices in enumerate(logit_order):
            top_ids = tuple(self.labels[int(index)].class_id for index in indices[:2])
            pair = tuple(sorted(top_ids)) if len(top_ids) == 2 else None
            restricted = top_ids[0] in self.restricted_ids or pair in self.restricted_pairs
            heads_disagree = int(indices[0]) != int(retrieval_order[row, 0])
            disagreement_threshold = self.policy.ridge_disagreement_minimum_pair_probability
            disagreement_ambiguous = (
                heads_disagree
                and disagreement_threshold is not None
                and approval_scores[row] < disagreement_threshold
            )
            agreement_blocked = self.policy.ridge_require_retrieval_agreement and heads_disagree
            retrieval_too_low = (
                retrieval_minimum is not None and retrieval_top1[row] < retrieval_minimum
            )
            approval_blocked[row] = (
                restricted or disagreement_ambiguous or agreement_blocked or retrieval_too_low
            )
            if retrieval_too_low or retrieval_top1[row] < self.policy.ood_maximum_similarity:
                recapture_reasons.append("CLASSIFIER_OUT_OF_CATALOG")
            else:
                recapture_reasons.append(None)
            if restricted:
                unknown_reasons.append("CLASSIFIER_CATALOG_CONFLICT")
            elif (
                approval_scores[row] < approval_threshold
                or disagreement_ambiguous
                or agreement_blocked
            ):
                unknown_reasons.append("CLASSIFIER_AMBIGUOUS_TOP2")
            else:
                unknown_reasons.append("BELOW_APPROVAL_THRESHOLD")
        return ClassificationResult(
            logits=logits.astype(np.float32),
            ranking_logits=ranking_logits.astype(np.float32),
            retrieval_logits=retrieval_scores.astype(np.float32),
            approval_scores=approval_scores,
            ranking_scores=ranking_scores,
            top3_safety_scores=inverse_entropy.astype(np.float32),
            segment_recapture_reasons=tuple(recapture_reasons),
            unknown_reasons=tuple(unknown_reasons),
            approval_blocked=approval_blocked,
        )
