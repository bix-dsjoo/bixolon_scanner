from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts.catalog import StoreCatalogPackage, load_store_catalog_package
from ..contracts.errors import ModelExecutionError
from ..contracts.model_package import (
    ClassifierMetadata,
    ClassLabel,
    NeighborMaskClassifierMetadata,
    NeighborMaskClassifierView,
)
from ..contracts.runtime_package_v2 import RuntimePackageV2, RuntimePackageV2Metadata
from ..pipeline.ports import ClassificationResult, Detection
from .imaging import image_original_size
from .onnx import (
    OrtRunner,
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)
from .onnx_session import ExecutionProvider


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
        provider: ExecutionProvider,
        cuda_dll_dir: Path | None = None,
        *,
        cpu_intra_op_threads: int = 0,
        openvino_cache_dir: Path | None = None,
    ):
        self.metadata = package.metadata.embedder
        self.runner = OrtRunner(
            package.embedder_path,
            provider,
            cuda_dll_dir,
            cpu_intra_op_threads=cpu_intra_op_threads,
            openvino_cache_dir=openvino_cache_dir,
        )
        self.transform = load_metric_transform(package)
        self.version = self.metadata.version

    def warmup(self) -> None:
        height, width = self.metadata.input_size
        fixed_batch_size = getattr(self.metadata, "fixed_batch_size", None)
        batch_sizes = (
            [fixed_batch_size]
            if fixed_batch_size is not None
            else self.metadata.warmup_batch_sizes
            if self.runner.accelerated
            else [1]
        )
        for batch_size in batch_sizes:
            inference_batch_size = batch_size * self._view_count
            self.runner.run(
                [self.metadata.output_name],
                self.metadata.input_name,
                np.zeros((inference_batch_size, 3, height, width), dtype=np.float32),
            )

    def close(self) -> None:
        self.runner.close()

    @property
    def _horizontal_flip_tta(self) -> bool:
        return bool(getattr(self.metadata, "horizontal_flip_tta", False))

    @property
    def _rotation_180_tta(self) -> bool:
        return bool(getattr(self.metadata, "rotation_180_tta", False))

    @property
    def _view_count(self) -> int:
        return 1 + int(self._horizontal_flip_tta) + int(self._rotation_180_tta)

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
        return self.embed_prepared_tensors_raw(batch)

    def embed_prepared_tensors_raw(self, batch: np.ndarray) -> np.ndarray:
        """Run already normalized NCHW classifier tensors through the embedder."""
        values = np.asarray(batch, dtype=np.float32)
        height, width = self.metadata.input_size
        if values.ndim != 4 or values.shape[1:] != (3, height, width):
            raise ValueError("prepared embedder tensors do not match runtime metadata")
        return self._run_view_averaged_tensors(np.ascontiguousarray(values))

    def _run_raw_tensors(self, batch: np.ndarray) -> np.ndarray:
        values = batch.astype(np.float32, copy=False)
        fixed_batch_size = getattr(self.metadata, "fixed_batch_size", None)
        if fixed_batch_size is None:
            (raw,) = self.runner.run(
                [self.metadata.output_name],
                self.metadata.input_name,
                values,
            )
            raw = np.asarray(raw, dtype=np.float32)
        else:
            chunks: list[np.ndarray] = []
            for start in range(0, len(values), fixed_batch_size):
                chunk = values[start : start + fixed_batch_size]
                valid_count = len(chunk)
                if valid_count < fixed_batch_size:
                    padding = np.zeros(
                        (fixed_batch_size - valid_count, *values.shape[1:]),
                        dtype=np.float32,
                    )
                    chunk = np.concatenate((chunk, padding), axis=0)
                (chunk_raw,) = self.runner.run(
                    [self.metadata.output_name],
                    self.metadata.input_name,
                    np.ascontiguousarray(chunk),
                )
                chunks.append(np.asarray(chunk_raw, dtype=np.float32)[:valid_count])
            raw = np.concatenate(chunks, axis=0)
        if raw.shape != (len(batch), self.metadata.embedding_dimension):
            raise ValueError("embedder output shape does not match runtime metadata")
        if not np.isfinite(raw).all():
            raise ModelExecutionError
        return raw

    def _run_view_averaged_tensors(self, batch: np.ndarray) -> np.ndarray:
        if self._view_count == 1:
            return self._run_raw_tensors(batch)
        views = [batch]
        if self._horizontal_flip_tta:
            views.append(np.ascontiguousarray(batch[:, :, :, ::-1]))
        if self._rotation_180_tta:
            views.append(np.ascontiguousarray(batch[:, :, ::-1, ::-1]))
        raw = self._run_raw_tensors(np.concatenate(views, axis=0))
        return np.asarray(
            raw.reshape(self._view_count, len(batch), -1).mean(axis=0), dtype=np.float32
        )

    def _embed_tensors(self, batch: np.ndarray) -> np.ndarray:
        return self.transform.apply(self._run_raw_tensors(batch))

    def embed_detections(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> np.ndarray:
        return self.transform.apply(self.embed_detections_raw(image, detections))

    def embed_detections_raw(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> np.ndarray:
        return self.embed_prepared_tensors_raw(self.prepare_detection_tensors(image, detections))

    def prepare_detection_tensors(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> np.ndarray:
        """Apply the runtime crop and neighbor-mask policy without inference."""
        return self.prepare_selected_detection_tensors(
            image,
            detections,
            np.arange(len(detections), dtype=np.int64),
        )

    def prepare_selected_detection_tensors(
        self,
        image: np.ndarray | Image.Image,
        detections: list[Detection],
        detection_indices: np.ndarray,
    ) -> np.ndarray:
        """Prepare selected ROIs while retaining every detection as mask context."""
        indices = np.asarray(detection_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("selected detection indices must be one-dimensional")
        if np.any(indices < 0) or np.any(indices >= len(detections)):
            raise ValueError("selected detection index is outside the detection list")
        if not len(indices):
            height, width = self.metadata.input_size
            return np.empty((0, 3, height, width), dtype=np.float32)
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
        for detection_index in indices:
            detection = detections[int(detection_index)]
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
                        int(detection_index),
                        image_width=original_width,
                        image_height=original_height,
                        output_size=batch.shape[-1],
                        margin_ratio=self.metadata.crop_margin_ratio,
                        distance_bias=self.metadata.neighbor_distance_bias,
                        shared_scale=self.metadata.neighbor_shared_scale,
                        crop_mode=self.metadata.crop_mode,
                    )
                    for detection_index in indices
                ]
            )
            batch = apply_classifier_background_masks(batch, masks)
        return np.ascontiguousarray(batch, dtype=np.float32)


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
        self.append_only_base_class_count = getattr(
            catalog.metadata, "append_only_base_class_count", None
        )
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
            approval_thresholds=self.policy.ridge_approval_thresholds,
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

    def close(self) -> None:
        self.embedder.close()

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
        return self.classify_embeddings(raw_embeddings, detections)

    def classify_selected(
        self,
        image: np.ndarray | Image.Image,
        detections: list[Detection],
        detection_indices: np.ndarray,
    ) -> ClassificationResult:
        indices = np.asarray(detection_indices, dtype=np.int64)
        prepared = self.embedder.prepare_selected_detection_tensors(image, detections, indices)
        raw_embeddings = self.embedder.embed_prepared_tensors_raw(prepared)
        selected_detections = [detections[int(index)] for index in indices]
        return self.classify_embeddings(raw_embeddings, selected_detections)

    def classify_single_views(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> ClassificationResult:
        prepared = np.concatenate(
            [
                self.embedder.prepare_detection_tensors(image, [detection])
                for detection in detections
            ],
            axis=0,
        )
        raw_embeddings = self.embedder.embed_prepared_tensors_raw(prepared)
        return self.classify_embeddings(raw_embeddings, detections)

    def classify_embeddings(
        self,
        raw_embeddings: np.ndarray,
        detections: list[Detection] | None = None,
        *,
        class_limit: int | None = None,
    ) -> ClassificationResult:
        """Classify raw embeddings produced by the compatible runtime embedder."""
        raw_embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        if (
            raw_embeddings.ndim != 2
            or raw_embeddings.shape[1] != self.embedder.metadata.embedding_dimension
        ):
            raise ValueError("raw embeddings do not match the Catalog embedder dimension")
        if detections is not None and len(detections) != len(raw_embeddings):
            raise ValueError("detections and raw embeddings are not aligned")
        embeddings = self.embedder.transform.apply(raw_embeddings)
        cosine_scores = self._class_scores(embeddings)
        if class_limit is not None:
            if not 1 <= class_limit <= len(self.labels):
                raise ValueError("Catalog class limit is invalid")
            cosine_scores = cosine_scores[:, :class_limit]
        if self.adapter_weight is not None:
            return self._classify_adapter(
                raw_embeddings,
                cosine_scores,
                detections,
                class_limit=class_limit,
            )
        labels = self.labels if class_limit is None else self.labels[:class_limit]
        order = np.argsort(-cosine_scores, axis=1, kind="stable")
        rows = np.arange(len(raw_embeddings))
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
        approval_blocked = np.zeros(len(raw_embeddings), dtype=bool)
        for row, indices in enumerate(order):
            top_ids = tuple(labels[int(index)].class_id for index in indices[:2])
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
        self,
        embeddings: np.ndarray,
        retrieval_scores: np.ndarray,
        detections: list[Detection] | None = None,
        *,
        class_limit: int | None = None,
    ) -> ClassificationResult:
        if self.adapter_weight is None or self.adapter_bias is None:
            raise ValueError("Catalog adapter is not loaded")
        approval_threshold = self._ridge_approval_threshold()
        minimum_entropy = self.policy.ridge_top3_minimum_inverse_entropy
        if minimum_entropy is None:
            raise ValueError("ridge Catalog policy is incomplete")
        labels = self.labels if class_limit is None else self.labels[:class_limit]
        weight = (
            self.adapter_weight if class_limit is None else self.adapter_weight[:, :class_limit]
        )
        bias = self.adapter_bias if class_limit is None else self.adapter_bias[:class_limit]
        logits = l2_normalize(embeddings) @ weight + bias
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
            top_ids = tuple(labels[int(index)].class_id for index in indices[:2])
            pair = tuple(sorted(top_ids)) if len(top_ids) == 2 else None
            restricted = top_ids[0] in self.restricted_ids or pair in self.restricted_pairs
            heads_disagree = int(indices[0]) != int(retrieval_order[row, 0])
            disagreement_threshold = (
                self.policy.ridge_disagreement_minimum_pair_probability
                if self.policy.ridge_approval_metric == "top2_pair_probability"
                else self.policy.ridge_disagreement_minimum_margin
            )
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
        corroboration_minimum_score = self.policy.detector_corroboration_minimum_score
        corroboration_maximum_approval = self.policy.detector_corroboration_maximum_approval_score
        low_similarity_minimum_score = (
            self.policy.detector_corroboration_low_similarity_minimum_score
        )
        low_similarity_maximum_retrieval = (
            self.policy.detector_corroboration_low_similarity_maximum_retrieval
        )
        low_similarity_minimum_approval = (
            self.policy.detector_corroboration_low_similarity_minimum_approval_score
        )
        if detections is not None and (
            (corroboration_minimum_score is not None and corroboration_maximum_approval is not None)
            or (
                low_similarity_minimum_score is not None
                and low_similarity_maximum_retrieval is not None
                and low_similarity_minimum_approval is not None
            )
        ):
            if len(detections) != len(embeddings):
                raise ValueError("detector corroboration inputs do not match embeddings")
            for row, detection in enumerate(detections):
                detector_index = detection.class_id
                low_approval_corroboration = (
                    corroboration_minimum_score is not None
                    and corroboration_maximum_approval is not None
                    and detection.score >= corroboration_minimum_score
                    and approval_scores[row] <= corroboration_maximum_approval
                )
                low_similarity_corroboration = (
                    low_similarity_minimum_score is not None
                    and low_similarity_maximum_retrieval is not None
                    and low_similarity_minimum_approval is not None
                    and detection.score >= low_similarity_minimum_score
                    and retrieval_top1[row] <= low_similarity_maximum_retrieval
                    and approval_scores[row] >= low_similarity_minimum_approval
                )
                if (
                    detector_index is None
                    or not 0 <= detector_index < logits.shape[1]
                    or not (low_approval_corroboration or low_similarity_corroboration)
                    or detector_index != int(logit_order[row, 1])
                ):
                    continue
                top1_index = int(logit_order[row, 0])
                ranking_logits[row, [top1_index, detector_index]] = ranking_logits[
                    row, [detector_index, top1_index]
                ]
                ranking_scores[row, [top1_index, detector_index]] = ranking_scores[
                    row, [detector_index, top1_index]
                ]
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


def verification_runtime_package(package: RuntimePackageV2) -> RuntimePackageV2:
    """Create the checksum-validated independent-embedder view of a Runtime package."""
    verification = package.metadata.classifier_verification
    if verification is None or package.verification_embedder_path is None:
        raise ValueError("runtime does not contain an independent classifier verifier")
    payload = package.metadata.model_dump(mode="json")
    payload["embedder"] = verification.independent_embedder.model_dump(mode="json")
    payload["metric_projection"] = verification.independent_metric_projection.model_dump(
        mode="json"
    )
    payload["classifier_verification"] = None
    payload["classifier_resolution_fallback"] = None
    metadata = RuntimePackageV2Metadata.model_validate(payload)
    return RuntimePackageV2(
        root=package.root,
        metadata=metadata,
        detector_path=package.detector_path,
        count_verifier_path=package.count_verifier_path,
        embedder_path=package.verification_embedder_path,
        classifier_fallback_embedder_path=None,
        metric_projection_path=package.verification_metric_projection_path,
        verification_embedder_path=None,
        verification_metric_projection_path=None,
    )


def classifier_fallback_runtime_package(package: RuntimePackageV2) -> RuntimePackageV2:
    """Create the checksum-validated higher-resolution classifier view."""
    fallback = package.metadata.classifier_resolution_fallback
    if fallback is None or package.classifier_fallback_embedder_path is None:
        raise ValueError("runtime does not contain a classifier resolution fallback")
    payload = package.metadata.model_dump(mode="json")
    payload["embedder"] = fallback.embedder.model_dump(mode="json")
    payload["classifier_resolution_fallback"] = None
    metadata = RuntimePackageV2Metadata.model_validate(payload)
    return RuntimePackageV2(
        root=package.root,
        metadata=metadata,
        detector_path=package.detector_path,
        count_verifier_path=package.count_verifier_path,
        embedder_path=package.classifier_fallback_embedder_path,
        classifier_fallback_embedder_path=None,
        metric_projection_path=package.metric_projection_path,
        verification_embedder_path=package.verification_embedder_path,
        verification_metric_projection_path=package.verification_metric_projection_path,
    )


class ConsensusCatalogClassifier:
    """Selectively require geometric and independent-backbone agreement."""

    def __init__(
        self,
        primary: OnnxCatalogClassifier,
        rotation: OnnxCatalogClassifier,
        independent: OnnxCatalogClassifier,
        *,
        ambiguity_maximum_approval_score: float,
        verify_all_approved_candidates: bool = False,
        unknown_recapture_on_dual_verifier_rejection: bool = False,
        unknown_recapture_on_any_verifier_rejection: bool = False,
    ):
        if not 0.0 <= ambiguity_maximum_approval_score <= 1.0:
            raise ValueError("classifier verification ambiguity score must be in [0, 1]")
        label_ids = tuple(label.class_id for label in primary.labels)
        if tuple(label.class_id for label in rotation.labels) != label_ids:
            raise ValueError("rotation verifier labels differ from the primary Catalog")
        if tuple(label.class_id for label in independent.labels) != label_ids:
            raise ValueError("independent verifier labels differ from the primary Catalog")
        if primary.embedder._view_count != 1:
            raise ValueError("selective rotation verification requires a single-view primary")
        if independent.embedder.metadata.fixed_batch_size is None:
            raise ValueError("independent verifier must declare its fixed ONNX batch size")
        append_only_counts = {
            primary.append_only_base_class_count,
            rotation.append_only_base_class_count,
            independent.append_only_base_class_count,
        }
        if len(append_only_counts) != 1:
            raise ValueError("consensus Catalogs differ in append-only base class count")
        self.primary = primary
        self.rotation = rotation
        self.independent = independent
        self.ambiguity_maximum_approval_score = ambiguity_maximum_approval_score
        self.verify_all_approved_candidates = verify_all_approved_candidates
        self.unknown_recapture_on_dual_verifier_rejection = (
            unknown_recapture_on_dual_verifier_rejection
        )
        self.unknown_recapture_on_any_verifier_rejection = (
            unknown_recapture_on_any_verifier_rejection
        )
        self.append_only_base_class_count = primary.append_only_base_class_count
        self.version = primary.version
        self.metadata = primary.metadata

    def warmup(self) -> None:
        self.independent.embedder.warmup()

    def close(self) -> None:
        closed: set[int] = set()
        for classifier in (self.primary, self.rotation, self.independent):
            embedder = classifier.embedder
            if id(embedder) in closed:
                continue
            closed.add(id(embedder))
            embedder.close()

    @staticmethod
    def _top1(result: ClassificationResult) -> np.ndarray:
        return np.argsort(-result.ranking_logits, axis=1, kind="stable")[:, 0]

    @staticmethod
    def _merge_class_matrix(
        base: np.ndarray | None,
        extended: np.ndarray | None,
        use_extended: np.ndarray,
        *,
        fill_value: float,
    ) -> np.ndarray | None:
        if base is None or extended is None:
            if base is not extended:
                raise ValueError("base and extended classifier result fields differ")
            return None
        merged = np.full(extended.shape, fill_value, dtype=extended.dtype)
        merged[:, : base.shape[1]] = base
        merged[use_extended] = extended[use_extended]
        return merged

    @staticmethod
    def _merge_row_array(
        base: np.ndarray | None,
        extended: np.ndarray | None,
        use_extended: np.ndarray,
    ) -> np.ndarray | None:
        if base is None or extended is None:
            if base is not extended:
                raise ValueError("base and extended classifier result fields differ")
            return None
        merged = np.asarray(base).copy()
        merged[use_extended] = np.asarray(extended)[use_extended]
        return merged

    @staticmethod
    def _merge_row_tuple(
        base: tuple[str | None, ...] | None,
        extended: tuple[str | None, ...] | None,
        use_extended: np.ndarray,
    ) -> tuple[str | None, ...] | None:
        if base is None or extended is None:
            if base is not extended:
                raise ValueError("base and extended classifier result fields differ")
            return None
        merged = list(base)
        for index in np.flatnonzero(use_extended):
            merged[int(index)] = extended[int(index)]
        return tuple(merged)

    @classmethod
    def _merge_append_only_results(
        cls,
        base: ClassificationResult,
        extended: ClassificationResult,
        use_extended: np.ndarray,
    ) -> ClassificationResult:
        return ClassificationResult(
            logits=cls._merge_class_matrix(
                base.logits, extended.logits, use_extended, fill_value=-np.inf
            ),
            ranking_logits=cls._merge_class_matrix(
                base.ranking_logits,
                extended.ranking_logits,
                use_extended,
                fill_value=-np.inf,
            ),
            retrieval_logits=cls._merge_class_matrix(
                base.retrieval_logits,
                extended.retrieval_logits,
                use_extended,
                fill_value=-np.inf,
            ),
            approval_scores=cls._merge_row_array(
                base.approval_scores, extended.approval_scores, use_extended
            ),
            top3_safety_scores=cls._merge_row_array(
                base.top3_safety_scores, extended.top3_safety_scores, use_extended
            ),
            ranking_scores=cls._merge_class_matrix(
                base.ranking_scores,
                extended.ranking_scores,
                use_extended,
                fill_value=0.0,
            ),
            segment_recapture_reasons=cls._merge_row_tuple(
                base.segment_recapture_reasons,
                extended.segment_recapture_reasons,
                use_extended,
            ),
            unknown_reasons=cls._merge_row_tuple(
                base.unknown_reasons, extended.unknown_reasons, use_extended
            ),
            approval_blocked=cls._merge_row_array(
                base.approval_blocked, extended.approval_blocked, use_extended
            ),
        )

    def _apply_append_only_consensus(
        self,
        image: np.ndarray | Image.Image,
        detections: list[Detection],
        prepared: np.ndarray,
        primary_raw: np.ndarray,
        extended_result: ClassificationResult,
        *,
        context_detections: list[Detection] | None = None,
        context_indices: np.ndarray | None = None,
    ) -> ClassificationResult:
        base_count = self.append_only_base_class_count
        if base_count is None:
            return extended_result
        base_result = self.primary.classify_embeddings(
            primary_raw,
            detections,
            class_limit=base_count,
        )
        base_result = self._apply_selective_verification(
            image,
            detections,
            prepared,
            primary_raw,
            base_result,
            class_limit=base_count,
            context_detections=context_detections,
            context_indices=context_indices,
        )
        primary_top1 = self._top1(extended_result)
        candidate_indices = np.flatnonzero(primary_top1 >= base_count)
        use_extended = np.zeros(len(detections), dtype=bool)
        if not len(candidate_indices):
            return self._merge_append_only_results(base_result, extended_result, use_extended)

        rotated = np.ascontiguousarray(prepared[candidate_indices, :, ::-1, ::-1], dtype=np.float32)
        rotated_raw = self.primary.embedder.embed_prepared_tensors_raw(rotated)
        rotation_raw = np.asarray(
            (primary_raw[candidate_indices] + rotated_raw) * np.float32(0.5),
            dtype=np.float32,
        )
        rotation_result = self.rotation.classify_embeddings(rotation_raw)
        selected_detections = [detections[int(index)] for index in candidate_indices]
        if context_detections is None:
            verification_detections = detections
            verification_indices = candidate_indices
        else:
            if context_indices is None or len(context_indices) != len(detections):
                raise ValueError("selected classifier context indices do not match detections")
            verification_detections = context_detections
            verification_indices = np.asarray(context_indices, dtype=np.int64)[candidate_indices]
        independent_prepared = self.independent.embedder.prepare_selected_detection_tensors(
            image,
            verification_detections,
            verification_indices,
        )
        independent_raw = self.independent.embedder.embed_prepared_tensors_raw(independent_prepared)
        independent_result = self.independent.classify_embeddings(
            independent_raw, selected_detections
        )
        extension_ids = primary_top1[candidate_indices]
        accepted = (self._top1(rotation_result) == extension_ids) & (
            self._top1(independent_result) == extension_ids
        )
        use_extended[candidate_indices[accepted]] = True
        return self._merge_append_only_results(base_result, extended_result, use_extended)

    def _apply_selective_verification(
        self,
        image: np.ndarray | Image.Image,
        detections: list[Detection],
        prepared: np.ndarray,
        primary_raw: np.ndarray,
        result: ClassificationResult,
        *,
        class_limit: int | None = None,
        context_detections: list[Detection] | None = None,
        context_indices: np.ndarray | None = None,
    ) -> ClassificationResult:
        if result.approval_scores is None:
            raise ValueError("selective verification requires explicit approval scores")
        approval_blocked = (
            np.zeros(len(detections), dtype=bool)
            if result.approval_blocked is None
            else np.asarray(result.approval_blocked, dtype=bool).copy()
        )
        recapture_reasons = result.segment_recapture_reasons or (None,) * len(detections)
        configured_thresholds = self.metadata.approval_thresholds
        primary_top1_all = self._top1(result)
        decision_thresholds = (
            np.full(
                len(detections),
                self.metadata.approval_threshold,
                dtype=np.float32,
            )
            if configured_thresholds is None
            else np.asarray(
                [
                    self.metadata.approval_threshold
                    if configured_thresholds[int(index)] is None
                    else configured_thresholds[int(index)]
                    for index in primary_top1_all
                ],
                dtype=np.float32,
            )
        )
        primary_unknown = (result.approval_scores < decision_thresholds) | approval_blocked
        approved_candidates = ~primary_unknown
        verify_unknown_recapture = (
            self.unknown_recapture_on_dual_verifier_rejection
            or self.unknown_recapture_on_any_verifier_rejection
        )
        unknown_candidates = verify_unknown_recapture & primary_unknown
        within_ambiguity_band = result.approval_scores < self.ambiguity_maximum_approval_score
        verification_candidates = (
            approved_candidates
            if self.verify_all_approved_candidates
            else approved_candidates & within_ambiguity_band
        ) | (unknown_candidates & within_ambiguity_band)
        candidate_indices = np.flatnonzero(
            verification_candidates
            & np.asarray([reason is None for reason in recapture_reasons], dtype=bool)
        )
        if not len(candidate_indices):
            return result

        rotated = np.ascontiguousarray(prepared[candidate_indices, :, ::-1, ::-1], dtype=np.float32)
        rotated_raw = self.primary.embedder.embed_prepared_tensors_raw(rotated)
        rotation_raw = np.asarray(
            (primary_raw[candidate_indices] + rotated_raw) * np.float32(0.5),
            dtype=np.float32,
        )
        rotation_result = self.rotation.classify_embeddings(rotation_raw, class_limit=class_limit)
        selected_detections = [detections[int(index)] for index in candidate_indices]
        if context_detections is None:
            verification_detections = detections
            verification_indices = candidate_indices
        else:
            if context_indices is None or len(context_indices) != len(detections):
                raise ValueError("selected classifier context indices do not match detections")
            verification_detections = context_detections
            verification_indices = np.asarray(context_indices, dtype=np.int64)[candidate_indices]
        independent_prepared = self.independent.embedder.prepare_selected_detection_tensors(
            image,
            verification_detections,
            verification_indices,
        )
        independent_raw = self.independent.embedder.embed_prepared_tensors_raw(independent_prepared)
        independent_result = self.independent.classify_embeddings(
            independent_raw,
            selected_detections,
            class_limit=class_limit,
        )
        if (
            independent_result.approval_scores is None
            or independent_result.retrieval_logits is None
        ):
            raise ValueError("independent verifier must expose approval and retrieval scores")

        unsafe_verifier_rejection = np.zeros(len(candidate_indices), dtype=bool)
        if verify_unknown_recapture:
            rotation_recapture = rotation_result.segment_recapture_reasons or (None,) * len(
                candidate_indices
            )
            independent_recapture = independent_result.segment_recapture_reasons or (None,) * len(
                candidate_indices
            )
            if self.unknown_recapture_on_any_verifier_rejection:
                unsafe_verifier_rejection = np.asarray(
                    [
                        left is not None or right is not None
                        for left, right in zip(
                            rotation_recapture,
                            independent_recapture,
                            strict=True,
                        )
                    ],
                    dtype=bool,
                )
            else:
                unsafe_verifier_rejection = np.asarray(
                    [
                        left is not None and right is not None
                        for left, right in zip(
                            rotation_recapture,
                            independent_recapture,
                            strict=True,
                        )
                    ],
                    dtype=bool,
                )

        primary_top1 = self._top1(result)[candidate_indices]
        rotation_top1 = self._top1(rotation_result)
        independent_top1 = self._top1(independent_result)
        independent_retrieval_top1 = np.argmax(independent_result.retrieval_logits, axis=1)
        verifier_threshold = self.independent.metadata.approval_threshold
        rejected = np.zeros(len(candidate_indices), dtype=bool)
        rotation_disagreement = rotation_top1 != primary_top1
        independently_corroborated = (independent_top1 == primary_top1) & (
            independent_retrieval_top1 == primary_top1
        )
        rejected[rotation_disagreement] = ~independently_corroborated[rotation_disagreement]
        rotation_agreement = ~rotation_disagreement
        independent_disagreement = independent_top1 != primary_top1
        strong_independent_disagreement = (
            independent_result.approval_scores >= verifier_threshold
        ) | (independent_retrieval_top1 != primary_top1)
        rejected[rotation_agreement] = (independent_disagreement & strong_independent_disagreement)[
            rotation_agreement
        ]

        approved_verification_candidate = approved_candidates[candidate_indices]
        unsafe_recapture = unsafe_verifier_rejection & (
            primary_unknown[candidate_indices]
            | rejected
            | (self.verify_all_approved_candidates & approved_verification_candidate)
        )
        if np.any(unsafe_recapture):
            updated_recapture_reasons = list(recapture_reasons)
            for index in candidate_indices[unsafe_recapture]:
                updated_recapture_reasons[int(index)] = "CLASSIFIER_TOP3_UNSAFE"
            result = replace(
                result,
                segment_recapture_reasons=tuple(updated_recapture_reasons),
            )

        rejected_indices = candidate_indices[rejected]
        if not len(rejected_indices):
            return result
        approval_blocked[rejected_indices] = True
        unknown_reasons = list(result.unknown_reasons or (None,) * len(detections))
        for index in rejected_indices:
            unknown_reasons[int(index)] = "CLASSIFIER_AMBIGUOUS_TOP2"
        return replace(
            result,
            approval_blocked=approval_blocked,
            unknown_reasons=tuple(unknown_reasons),
        )

    def classify(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> ClassificationResult:
        prepared = self.primary.embedder.prepare_detection_tensors(image, detections)
        primary_raw = self.primary.embedder.embed_prepared_tensors_raw(prepared)
        result = self.primary.classify_embeddings(primary_raw, detections)
        if self.append_only_base_class_count is not None:
            return self._apply_append_only_consensus(
                image,
                detections,
                prepared,
                primary_raw,
                result,
            )
        return self._apply_selective_verification(
            image,
            detections,
            prepared,
            primary_raw,
            result,
        )

    def classify_selected(
        self,
        image: np.ndarray | Image.Image,
        detections: list[Detection],
        detection_indices: np.ndarray,
    ) -> ClassificationResult:
        indices = np.asarray(detection_indices, dtype=np.int64)
        prepared = self.primary.embedder.prepare_selected_detection_tensors(
            image,
            detections,
            indices,
        )
        primary_raw = self.primary.embedder.embed_prepared_tensors_raw(prepared)
        selected_detections = [detections[int(index)] for index in indices]
        result = self.primary.classify_embeddings(primary_raw, selected_detections)
        if self.append_only_base_class_count is not None:
            return self._apply_append_only_consensus(
                image,
                selected_detections,
                prepared,
                primary_raw,
                result,
                context_detections=detections,
                context_indices=indices,
            )
        return self._apply_selective_verification(
            image,
            selected_detections,
            prepared,
            primary_raw,
            result,
            context_detections=detections,
            context_indices=indices,
        )

    def classify_single_views(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> ClassificationResult:
        return self.primary.classify_single_views(image, detections)


class ResolutionFallbackCatalogClassifier:
    """Expose fast and higher-resolution classifier paths to the decision policy."""

    def __init__(
        self,
        primary_classifier: OnnxCatalogClassifier | ConsensusCatalogClassifier,
        fallback_classifier: OnnxCatalogClassifier | ConsensusCatalogClassifier,
        *,
        owned_embedders: tuple[OnnxEmbedder, ...],
        resolution_fallback_metadata,
    ) -> None:
        self.primary_classifier = primary_classifier
        self.fallback_classifier = fallback_classifier
        # The assisted detector needs direct access to the fast primary embedder.
        self.primary = getattr(primary_classifier, "primary", primary_classifier)
        self.metadata = primary_classifier.metadata
        self.version = primary_classifier.version
        self.resolution_fallback_metadata = resolution_fallback_metadata
        self._owned_embedders = owned_embedders

    def warmup(self) -> None:
        fallback_primary = getattr(
            self.fallback_classifier,
            "primary",
            self.fallback_classifier,
        )
        fallback_primary.embedder.warmup()
        warmup = getattr(self.primary_classifier, "warmup", None)
        if callable(warmup):
            warmup()

    def close(self) -> None:
        closed: set[int] = set()
        for embedder in self._owned_embedders:
            if id(embedder) in closed:
                continue
            closed.add(id(embedder))
            embedder.close()

    def classify(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> ClassificationResult:
        return self.primary_classifier.classify(image, detections)

    def classify_selected(
        self,
        image: np.ndarray | Image.Image,
        detections: list[Detection],
        detection_indices: np.ndarray,
    ) -> ClassificationResult:
        classify_selected = getattr(self.primary_classifier, "classify_selected", None)
        if not callable(classify_selected):
            raise ValueError("primary classifier does not support selected ROIs")
        return classify_selected(image, detections, detection_indices)

    def classify_fallback(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> ClassificationResult:
        return self.fallback_classifier.classify(image, detections)

    def classify_fallback_selected(
        self,
        image: np.ndarray | Image.Image,
        detections: list[Detection],
        detection_indices: np.ndarray,
    ) -> ClassificationResult:
        classify_selected = getattr(self.fallback_classifier, "classify_selected", None)
        if not callable(classify_selected):
            raise ValueError("classifier resolution fallback does not support selected ROIs")
        return classify_selected(image, detections, detection_indices)

    def classify_single_views(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> ClassificationResult:
        return self.primary_classifier.classify_single_views(image, detections)

    def classify_fallback_single_views(
        self, image: np.ndarray | Image.Image, detections: list[Detection]
    ) -> ClassificationResult:
        return self.fallback_classifier.classify_single_views(image, detections)


def build_catalog_classifier(
    runtime: RuntimePackageV2,
    catalog: StoreCatalogPackage,
    provider: ExecutionProvider,
    cuda_dll_dir: Path | None = None,
    *,
    cpu_intra_op_threads: int = 0,
    openvino_cache_dir: Path | None = None,
) -> tuple[
    OnnxCatalogClassifier | ConsensusCatalogClassifier | ResolutionFallbackCatalogClassifier,
    OnnxEmbedder,
]:
    """Build the primary Catalog classifier and its optional selective verifier."""
    primary_embedder = OnnxEmbedder(
        runtime,
        provider,
        cuda_dll_dir,
        cpu_intra_op_threads=cpu_intra_op_threads,
        openvino_cache_dir=openvino_cache_dir,
    )
    primary = OnnxCatalogClassifier(runtime, catalog, primary_embedder)
    verification = runtime.metadata.classifier_verification
    catalog_has_verification = catalog.metadata.verification is not None
    if (verification is None) != (not catalog_has_verification):
        raise ValueError("Runtime and Catalog classifier verification contracts differ")
    fallback_policy = runtime.metadata.classifier_resolution_fallback
    fallback_runtime = (
        None if fallback_policy is None else classifier_fallback_runtime_package(runtime)
    )
    fallback_embedder = (
        None
        if fallback_runtime is None
        else OnnxEmbedder(
            fallback_runtime,
            provider,
            cuda_dll_dir,
            cpu_intra_op_threads=cpu_intra_op_threads,
            openvino_cache_dir=openvino_cache_dir,
        )
    )
    fallback_primary = (
        None
        if fallback_runtime is None or fallback_embedder is None
        else OnnxCatalogClassifier(fallback_runtime, catalog, fallback_embedder)
    )
    if verification is None:
        if fallback_primary is None or fallback_embedder is None or fallback_policy is None:
            return primary, primary_embedder
        return (
            ResolutionFallbackCatalogClassifier(
                primary,
                fallback_primary,
                owned_embedders=(primary_embedder, fallback_embedder),
                resolution_fallback_metadata=fallback_policy,
            ),
            primary_embedder,
        )
    if catalog.rotation_catalog_root is None or catalog.independent_catalog_root is None:
        raise ValueError("Catalog verification payloads are missing")
    rotation_catalog = load_store_catalog_package(
        catalog.rotation_catalog_root,
        expected_store_id=catalog.metadata.store_id,
    )
    independent_catalog = load_store_catalog_package(
        catalog.independent_catalog_root,
        expected_store_id=catalog.metadata.store_id,
    )
    independent_runtime = verification_runtime_package(runtime)
    independent_embedder = OnnxEmbedder(
        independent_runtime,
        provider,
        cuda_dll_dir,
        cpu_intra_op_threads=cpu_intra_op_threads,
        openvino_cache_dir=openvino_cache_dir,
    )
    independent_classifier = OnnxCatalogClassifier(
        independent_runtime,
        independent_catalog,
        independent_embedder,
    )
    classifier = ConsensusCatalogClassifier(
        primary,
        OnnxCatalogClassifier(runtime, rotation_catalog, primary_embedder),
        independent_classifier,
        ambiguity_maximum_approval_score=verification.ambiguity_maximum_approval_score,
        verify_all_approved_candidates=verification.verify_all_approved_candidates,
        unknown_recapture_on_dual_verifier_rejection=(
            verification.unknown_recapture_on_dual_verifier_rejection
        ),
        unknown_recapture_on_any_verifier_rejection=(
            verification.unknown_recapture_on_any_verifier_rejection
        ),
    )
    if (
        fallback_primary is not None
        and fallback_embedder is not None
        and fallback_policy is not None
    ):
        if fallback_runtime is None:
            raise ValueError("classifier fallback runtime is missing")
        fallback_classifier = ConsensusCatalogClassifier(
            fallback_primary,
            OnnxCatalogClassifier(
                fallback_runtime,
                rotation_catalog,
                fallback_embedder,
            ),
            independent_classifier,
            ambiguity_maximum_approval_score=verification.ambiguity_maximum_approval_score,
            verify_all_approved_candidates=verification.verify_all_approved_candidates,
            unknown_recapture_on_dual_verifier_rejection=(
                verification.unknown_recapture_on_dual_verifier_rejection
            ),
            unknown_recapture_on_any_verifier_rejection=(
                verification.unknown_recapture_on_any_verifier_rejection
            ),
        )
        classifier = ResolutionFallbackCatalogClassifier(
            classifier,
            fallback_classifier,
            owned_embedders=(
                primary_embedder,
                fallback_embedder,
                independent_embedder,
            ),
            resolution_fallback_metadata=fallback_policy,
        )
    return classifier, primary_embedder
