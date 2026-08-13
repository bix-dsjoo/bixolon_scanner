from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from .errors import ModelExecutionError, ProviderInitializationError
from .imaging import image_original_size
from .package import ClassifierMetadata, CountVerifierMetadata, DetectorMetadata, ModelPackage


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass(frozen=True)
class DetectionResult:
    detections: list[Detection]
    capacity_saturated: bool = False
    verified_count: int | None = None
    count_confidence: float | None = None
    uncertain_candidate_count: int = 0
    uncertain_candidate_scores: tuple[float, ...] = ()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _nms(detections: list[Detection], threshold: float) -> list[Detection]:
    ordered = sorted(detections, key=lambda detection: detection.score, reverse=True)
    kept: list[Detection] = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        remaining: list[Detection] = []
        current_area = max(0.0, current.x2 - current.x1) * max(0.0, current.y2 - current.y1)
        for candidate in ordered:
            ix1 = max(current.x1, candidate.x1)
            iy1 = max(current.y1, candidate.y1)
            ix2 = min(current.x2, candidate.x2)
            iy2 = min(current.y2, candidate.y2)
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            candidate_area = max(0.0, candidate.x2 - candidate.x1) * max(
                0.0, candidate.y2 - candidate.y1
            )
            union = current_area + candidate_area - intersection
            if union <= 0.0 or intersection / union <= threshold:
                remaining.append(candidate)
        ordered = remaining
    return kept


def _box_iou(left: Detection, right: Detection) -> float:
    ix1 = max(left.x1, right.x1)
    iy1 = max(left.y1, right.y1)
    ix2 = min(left.x2, right.x2)
    iy2 = min(left.y2, right.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


class OrtRunner:
    def __init__(
        self,
        model_path: Path,
        provider: Literal["cuda", "cpu"],
        cuda_dll_dir: Path | None = None,
    ):
        try:
            import onnxruntime as ort

            if provider == "cuda" and cuda_dll_dir is not None:
                if not cuda_dll_dir.is_dir():
                    raise ProviderInitializationError
                ort.preload_dlls(directory=str(cuda_dll_dir))
            available = ort.get_available_providers()
            provider_name = (
                "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
            )
            if provider_name not in available:
                raise ProviderInitializationError
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=[provider_name]
            )
            if self.session.get_providers()[0] != provider_name:
                raise ProviderInitializationError
            self.cuda = provider == "cuda"
        except ProviderInitializationError:
            raise
        except Exception as exc:
            raise ProviderInitializationError from exc

    def run(self, output_names: list[str], input_name: str, tensor: np.ndarray) -> list[np.ndarray]:
        try:
            if not self.cuda:
                return self.session.run(output_names, {input_name: tensor})
            binding = self.session.io_binding()
            binding.bind_cpu_input(input_name, tensor)
            for output_name in output_names:
                binding.bind_output(output_name, "cuda")
            self.session.run_with_iobinding(binding)
            return binding.copy_outputs_to_cpu()
        except Exception as exc:
            raise ModelExecutionError from exc


def select_provider(mode: Literal["auto", "cuda", "cpu"]) -> Literal["cuda", "cpu"]:
    if mode == "cpu":
        return "cpu"
    try:
        import onnxruntime as ort

        has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception as exc:
        raise ProviderInitializationError from exc
    if mode == "cuda" and not has_cuda:
        raise ProviderInitializationError
    return "cuda" if has_cuda else "cpu"


def _prepare_rgb(
    image: np.ndarray | Image.Image,
    size: tuple[int, int],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    *,
    reducing_gap: float | None = None,
) -> np.ndarray:
    source = image if isinstance(image, Image.Image) else Image.fromarray(image, mode="RGB")
    pil = source.resize(
        (size[1], size[0]),
        Image.Resampling.BILINEAR,
        reducing_gap=reducing_gap,
    )
    tensor = np.asarray(pil, dtype=np.float32) / 255.0
    tensor = (tensor - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.transpose(tensor, (2, 0, 1))


class OnnxDetector:
    def __init__(
        self,
        model_path: Path,
        metadata: DetectorMetadata,
        provider: Literal["cuda", "cpu"],
        cuda_dll_dir: Path | None = None,
    ):
        self.metadata = metadata
        self.runner = OrtRunner(model_path, provider, cuda_dll_dir)
        self.version = metadata.version

    def warmup(self) -> None:
        height, width = self.metadata.input_size
        dummy = np.zeros((1, 3, height, width), dtype=np.float32)
        self.runner.run(
            [self.metadata.logits_output, self.metadata.boxes_output],
            self.metadata.input_name,
            dummy,
        )

    def detect(self, image: np.ndarray | Image.Image) -> DetectionResult:
        if isinstance(image, Image.Image):
            original_width, original_height = image_original_size(image)
        else:
            original_height, original_width = image.shape[:2]
        tensor = _prepare_rgb(
            image,
            self.metadata.input_size,
            self.metadata.mean,
            self.metadata.std,
            reducing_gap=self.metadata.resize_reducing_gap,
        )[None]
        logits, boxes = self.runner.run(
            [self.metadata.logits_output, self.metadata.boxes_output],
            self.metadata.input_name,
            tensor,
        )
        logits = np.asarray(logits)[0]
        boxes = np.asarray(boxes)[0]
        if logits.ndim == 1:
            scores = _sigmoid(logits)
        else:
            scores = _sigmoid(logits).max(axis=-1)
        selected_indices = np.flatnonzero(scores >= self.metadata.score_threshold)
        raw_saturated = len(selected_indices) >= self.metadata.max_queries

        def convert(indices) -> list[Detection]:
            converted: list[Detection] = []
            for index in indices:
                cx, cy, width, height = [float(value) for value in boxes[index]]
                x1 = max(0.0, (cx - width / 2.0) * original_width)
                y1 = max(0.0, (cy - height / 2.0) * original_height)
                x2 = min(float(original_width), (cx + width / 2.0) * original_width)
                y2 = min(float(original_height), (cy + height / 2.0) * original_height)
                if x2 > x1 and y2 > y1:
                    converted.append(Detection(x1, y1, x2, y2, float(scores[index])))
            return converted

        detections = _nms(convert(selected_indices), self.metadata.nms_iou_threshold)
        uncertain_candidate_count = 0
        uncertain_candidate_scores: list[float] = []
        if self.metadata.uncertainty_score_threshold is not None:
            shadow_indices = np.flatnonzero(scores >= self.metadata.uncertainty_score_threshold)
            shadow = _nms(convert(shadow_indices), self.metadata.nms_iou_threshold)
            for candidate in shadow:
                if candidate.score >= self.metadata.score_threshold:
                    continue
                candidate_area_ratio = (
                    (candidate.x2 - candidate.x1)
                    * (candidate.y2 - candidate.y1)
                    / float(original_width * original_height)
                )
                if candidate_area_ratio < self.metadata.uncertainty_min_area_ratio:
                    continue
                overlaps = [_box_iou(candidate, accepted) for accepted in detections]
                if not overlaps or max(overlaps) < self.metadata.uncertainty_match_iou_threshold:
                    uncertain_candidate_count += 1
                    uncertain_candidate_scores.append(candidate.score)
        return DetectionResult(
            detections,
            raw_saturated,
            uncertain_candidate_count=uncertain_candidate_count,
            uncertain_candidate_scores=tuple(sorted(uncertain_candidate_scores, reverse=True)),
        )


class OnnxClassifier:
    def __init__(
        self,
        model_path: Path,
        metadata: ClassifierMetadata,
        provider: Literal["cuda", "cpu"],
        cuda_dll_dir: Path | None = None,
    ):
        self.metadata = metadata
        self.runner = OrtRunner(model_path, provider, cuda_dll_dir)
        self.version = metadata.version

    def warmup(self) -> None:
        height, width = self.metadata.input_size
        for batch_size in self.metadata.warmup_batch_sizes:
            dummy = np.zeros((batch_size, 3, height, width), dtype=np.float32)
            self.runner.run([self.metadata.logits_output], self.metadata.input_name, dummy)

    def classify(self, image: np.ndarray | Image.Image, detections: list[Detection]) -> np.ndarray:
        if isinstance(image, Image.Image):
            pil_image = image
            pixel_width, pixel_height = image.size
            image_width, image_height = image_original_size(image)
            scale_x = pixel_width / image_width
            scale_y = pixel_height / image_height
        else:
            pil_image = Image.fromarray(image, mode="RGB")
            image_height, image_width = image.shape[:2]
            scale_x = scale_y = 1.0
        crops: list[np.ndarray] = []
        for detection in detections:
            margin_x = (detection.x2 - detection.x1) * self.metadata.crop_margin_ratio
            margin_y = (detection.y2 - detection.y1) * self.metadata.crop_margin_ratio
            x1 = max(0, int(np.floor(detection.x1 - margin_x)))
            y1 = max(0, int(np.floor(detection.y1 - margin_y)))
            x2 = min(image_width, int(np.ceil(detection.x2 + margin_x)))
            y2 = min(image_height, int(np.ceil(detection.y2 + margin_y)))
            crop = pil_image.crop(
                (
                    int(np.floor(x1 * scale_x)),
                    int(np.floor(y1 * scale_y)),
                    int(np.ceil(x2 * scale_x)),
                    int(np.ceil(y2 * scale_y)),
                )
            )
            if crop.width == 0 or crop.height == 0:
                raise ModelExecutionError
            crops.append(
                _prepare_rgb(
                    crop,
                    self.metadata.input_size,
                    self.metadata.mean,
                    self.metadata.std,
                    reducing_gap=self.metadata.resize_reducing_gap,
                )
            )
        batch = np.stack(crops).astype(np.float32, copy=False)
        (logits,) = self.runner.run([self.metadata.logits_output], self.metadata.input_name, batch)
        return np.asarray(logits, dtype=np.float32)


class OnnxCountVerifier:
    def __init__(
        self,
        model_path: Path,
        metadata: CountVerifierMetadata,
        provider: Literal["cuda", "cpu"],
        cuda_dll_dir: Path | None = None,
    ):
        self.metadata = metadata
        self.runner = OrtRunner(model_path, provider, cuda_dll_dir)
        self.version = metadata.version

    def warmup(self) -> None:
        height, width = self.metadata.input_size
        dummy = np.zeros((1, 3, height, width), dtype=np.float32)
        self.runner.run([self.metadata.logits_output], self.metadata.input_name, dummy)

    def verify(self, image: np.ndarray | Image.Image) -> tuple[int, float]:
        tensor = _prepare_rgb(
            image,
            self.metadata.input_size,
            self.metadata.mean,
            self.metadata.std,
            reducing_gap=self.metadata.resize_reducing_gap,
        )[None]
        (logits,) = self.runner.run([self.metadata.logits_output], self.metadata.input_name, tensor)
        values = np.asarray(logits, dtype=np.float32)
        if values.shape != (1, len(self.metadata.count_labels)):
            raise ModelExecutionError
        scaled = values[0].astype(np.float64) / self.metadata.temperature
        scaled -= scaled.max()
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        return self.metadata.count_labels[index], float(probabilities[index])


class CountVerifiedDetector:
    def __init__(self, detector: OnnxDetector, verifier: OnnxCountVerifier):
        self.detector = detector
        self.verifier = verifier
        self.version = detector.version

    def detect(self, image: np.ndarray | Image.Image) -> DetectionResult:
        result = self.detector.detect(image)
        verified_count, confidence = self.verifier.verify(image)
        return DetectionResult(
            detections=result.detections,
            capacity_saturated=result.capacity_saturated,
            verified_count=verified_count,
            count_confidence=confidence,
            uncertain_candidate_count=result.uncertain_candidate_count,
            uncertain_candidate_scores=result.uncertain_candidate_scores,
        )


def build_onnx_adapters(
    model_package: ModelPackage,
    provider_mode: Literal["auto", "cuda", "cpu"],
    *,
    cuda_dll_dir: Path | None = None,
):
    provider = select_provider(provider_mode)

    def create(selected_provider: Literal["cuda", "cpu"]):
        detector = OnnxDetector(
            model_package.detector_path,
            model_package.metadata.detector,
            selected_provider,
            cuda_dll_dir,
        )
        classifier = OnnxClassifier(
            model_package.classifier_path,
            model_package.metadata.classifier,
            selected_provider,
            cuda_dll_dir,
        )
        detector.warmup()
        classifier.warmup()
        count_metadata = getattr(model_package.metadata, "count_verifier", None)
        count_path = getattr(model_package, "count_verifier_path", None)
        if count_metadata is not None:
            if count_path is None:
                raise ProviderInitializationError
            count_verifier = OnnxCountVerifier(
                count_path,
                count_metadata,
                selected_provider,
                cuda_dll_dir,
            )
            count_verifier.warmup()
            detector = CountVerifiedDetector(detector, count_verifier)
        return detector, classifier, selected_provider

    try:
        return create(provider)
    except (ProviderInitializationError, ModelExecutionError):
        if provider_mode != "auto" or provider != "cuda":
            raise
        return create("cpu")
