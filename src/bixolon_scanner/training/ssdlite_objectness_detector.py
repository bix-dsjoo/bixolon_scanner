from __future__ import annotations

import json
import random
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from .models import require_torch

SSDLITE_MEAN = (0.5, 0.5, 0.5)
SSDLITE_STD = (0.5, 0.5, 0.5)


def build_ssdlite_objectness(
    *,
    image_size: int = 320,
    score_threshold: float = 0.001,
    nms_threshold: float = 0.8,
    detections_per_image: int = 300,
    topk_candidates: int = 1000,
    foreground_class_count: int = 1,
    pretrained_backbone: bool = False,
    pretrained_detector_transfer: bool = False,
    device: str = "cpu",
):
    if image_size != 320:
        raise ValueError("the torchvision SSDLite architecture requires a 320px input")
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=(
            MobileNet_V3_Large_Weights.IMAGENET1K_V2
            if pretrained_backbone and not pretrained_detector_transfer
            else None
        ),
        num_classes=foreground_class_count + 1,
        score_thresh=score_threshold,
        nms_thresh=nms_threshold,
        detections_per_img=detections_per_image,
        topk_candidates=topk_candidates,
        trainable_backbone_layers=6,
        image_mean=list(SSDLITE_MEAN),
        image_std=list(SSDLITE_STD),
    )
    if pretrained_detector_transfer:
        source = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        current = model.state_dict()
        transferable = {
            name: value
            for name, value in source.state_dict().items()
            if name in current and current[name].shape == value.shape
        }
        model.load_state_dict(transferable, strict=False)
    return model.to(device)


def enable_empty_image_hard_negative_loss(model, *, minimum_negatives: int) -> None:
    """Teach SSD from empty images, which torchvision otherwise gives zero loss."""
    if minimum_negatives <= 0:
        raise ValueError("minimum_negatives must be positive")

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        torch = require_torch()
        from torch.nn import functional as F

        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (
            target,
            regression,
            logits,
            anchors_per_image,
            matched_per_image,
        ) in zip(
            targets,
            bbox_regression,
            cls_logits,
            anchors,
            matched_idxs,
            strict=True,
        ):
            foreground = torch.where(matched_per_image >= 0)[0]
            foreground_matches = matched_per_image[foreground]
            num_foreground += foreground_matches.numel()
            matched_boxes = target["boxes"][foreground_matches]
            regression_targets = self.box_coder.encode_single(
                matched_boxes,
                anchors_per_image[foreground],
            )
            bbox_loss.append(
                F.smooth_l1_loss(
                    regression[foreground],
                    regression_targets,
                    reduction="sum",
                )
            )
            classes = torch.zeros(
                logits.size(0),
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )
            classes[foreground] = target["labels"][foreground_matches]
            cls_targets.append(classes)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)
        class_count = cls_logits.size(-1)
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, class_count),
            cls_targets.view(-1),
            reduction="none",
        ).view(cls_targets.size())
        foreground = cls_targets > 0
        foreground_counts = foreground.sum(1, keepdim=True)
        negative_counts = self.neg_to_pos_ratio * foreground_counts
        negative_counts = torch.where(
            foreground_counts == 0,
            torch.full_like(negative_counts, minimum_negatives),
            negative_counts,
        ).clamp_max(cls_targets.size(1))
        negative_loss = cls_loss.clone()
        negative_loss[foreground] = -float("inf")
        _values, indexes = negative_loss.sort(1, descending=True)
        background = indexes.sort(1)[1] < negative_counts
        normalizer = max(1, num_foreground)
        return {
            "bbox_regression": bbox_loss.sum() / normalizer,
            "classification": (cls_loss[foreground].sum() + cls_loss[background].sum())
            / normalizer,
        }

    model.compute_loss = MethodType(compute_loss, model)


class CachedObjectnessDataset:
    def __init__(
        self,
        manifest: Path,
        dataset_root: Path,
        cache_dir: Path,
        *,
        training: bool,
        class_aware: bool = False,
        quarter_turn_augmentation: bool = False,
    ) -> None:
        self.records = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type", "detection") == "detection":
                self.records.append(record)
        self.dataset_root = dataset_root.resolve()
        self.training = training
        self.class_aware = class_aware
        self.quarter_turn_augmentation = quarter_turn_augmentation
        metadata = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
        self.cache_index = {str(key): int(value) for key, value in metadata["index"].items()}
        self.cache_array_path = (cache_dir / metadata["array_filename"]).resolve()
        self.cache_images = np.load(self.cache_array_path, mmap_mode="r")
        self.image_size = int(metadata["image_size"])
        self.source_shapes = {
            str(key): (int(value[0]), int(value[1]))
            for key, value in metadata.get("source_shapes", {}).items()
        }
        missing = [
            int(row["image_id"])
            for row in self.records
            if str(row["image_id"]) not in self.cache_index
        ]
        if missing:
            raise ValueError(f"SSDLite image cache is incomplete: {missing[:3]}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["cache_images"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cache_images = np.load(self.cache_array_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        torch = require_torch()
        from torchvision.transforms.functional import pil_to_tensor

        record = self.records[index]
        cache_row = self.cache_index[str(record["image_id"])]
        image = Image.fromarray(np.asarray(self.cache_images[cache_row]), mode="RGB")
        source_width = record.get("width")
        source_height = record.get("height")
        if source_width is None or source_height is None:
            source_width, source_height = self.source_shapes[str(record["image_id"])]
        scale_x = self.image_size / float(source_width)
        scale_y = self.image_size / float(source_height)
        boxes = []
        for annotation in record["annotations"]:
            x, y, width, height = annotation.get("bbox_xywh", annotation.get("bbox"))
            boxes.append(
                [
                    x * scale_x,
                    y * scale_y,
                    (x + width) * scale_x,
                    (y + height) * scale_y,
                ]
            )
        boxes_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if self.training and self.quarter_turn_augmentation:
            turns = random.randrange(4)
            if turns:
                image = image.rotate(90 * turns)
                for _ in range(turns):
                    if len(boxes_array):
                        previous = boxes_array.copy()
                        boxes_array[:, 0] = previous[:, 1]
                        boxes_array[:, 1] = self.image_size - previous[:, 2]
                        boxes_array[:, 2] = previous[:, 3]
                        boxes_array[:, 3] = self.image_size - previous[:, 0]
        if self.training and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(boxes_array):
                left = self.image_size - boxes_array[:, 2].copy()
                right = self.image_size - boxes_array[:, 0].copy()
                boxes_array[:, 0] = left
                boxes_array[:, 2] = right
        if self.training:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        tensor = pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)
        labels = [
            int(annotation["category_id"]) if self.class_aware else 1
            for annotation in record["annotations"]
        ]
        target: dict[str, Any] = {
            "boxes": torch.as_tensor(boxes_array, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(int(record["image_id"]), dtype=torch.int64),
        }
        return tensor, target


def collate_detection_batch(batch):
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def build_export_wrapper(
    model,
    *,
    input_size: int = 320,
    maximum_queries: int = 3234,
    detector_class_count: int = 20,
):
    torch = require_torch()
    from torchvision.models.detection.image_list import ImageList

    model.eval()
    with torch.inference_mode():
        sample = torch.zeros((1, 3, input_size, input_size), device=next(model.parameters()).device)
        features = list(model.backbone(sample).values())
        anchors = model.anchor_generator(ImageList(sample, [(input_size, input_size)]), features)[0]

    class Wrapper(torch.nn.Module):
        """Expose SSDLite as fixed D-FINE-compatible logits and cxcywh boxes."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = model.backbone
            self.head = model.head
            self.box_coder = model.box_coder
            self.register_buffer("anchors", anchors)
            self.input_size = float(input_size)
            self.maximum_queries = maximum_queries
            self.detector_class_count = detector_class_count

        def forward(self, pixel_values):
            features = list(self.backbone(pixel_values).values())
            outputs = self.head(features)
            class_logits = outputs["cls_logits"]
            regression = outputs["bbox_regression"]
            foreground_logits = class_logits[..., 1:] - class_logits[..., :1]
            objectness = foreground_logits.max(dim=-1).values
            values, indices = torch.topk(objectness, self.maximum_queries, dim=1)
            selected_regression = torch.gather(
                regression,
                1,
                indices.unsqueeze(-1).expand(-1, -1, 4),
            )
            selected_anchors = self.anchors[indices]
            decoded_batches = []
            for batch_index in range(pixel_values.shape[0]):
                decoded_batches.append(
                    self.box_coder.decode_single(
                        selected_regression[batch_index],
                        selected_anchors[batch_index],
                    )
                )
            xyxy = torch.stack(decoded_batches).clamp(0.0, self.input_size)
            center_x = (xyxy[..., 0] + xyxy[..., 2]) * 0.5 / self.input_size
            center_y = (xyxy[..., 1] + xyxy[..., 3]) * 0.5 / self.input_size
            width = (xyxy[..., 2] - xyxy[..., 0]) / self.input_size
            height = (xyxy[..., 3] - xyxy[..., 1]) / self.input_size
            boxes = torch.stack((center_x, center_y, width, height), dim=-1)
            if foreground_logits.shape[-1] == 1:
                background = torch.full(
                    (*values.shape, self.detector_class_count - 1),
                    -20.0,
                    dtype=values.dtype,
                    device=values.device,
                )
                logits = torch.cat((values.unsqueeze(-1), background), dim=-1)
            else:
                logits = torch.gather(
                    foreground_logits,
                    1,
                    indices.unsqueeze(-1).expand(-1, -1, foreground_logits.shape[-1]),
                )
            return logits, boxes

    return Wrapper()


def export_ssdlite_onnx(
    model,
    output: Path,
    *,
    input_size: int = 320,
    detector_class_count: int = 20,
) -> None:
    torch = require_torch()
    wrapper = build_export_wrapper(
        model,
        input_size=input_size,
        detector_class_count=detector_class_count,
    )
    sample = torch.zeros((1, 3, input_size, input_size), device=next(model.parameters()).device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        sample,
        output,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
