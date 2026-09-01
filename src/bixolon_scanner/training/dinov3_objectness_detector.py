from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image

from .models import DINO_V3_HUB_REPOSITORY, require_torch

DINO_V3_CONVNEXT_TINY = "dinov3_convnext_tiny"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dinov3_convnext_tiny(weights_path: Path, *, device: str = "cpu"):
    """Load the pinned official DINOv3 backbone without a detector dependency."""
    torch = require_torch()
    backbone = torch.hub.load(
        DINO_V3_HUB_REPOSITORY,
        DINO_V3_CONVNEXT_TINY,
        source="github",
        trust_repo=True,
        verbose=False,
        pretrained=True,
        weights=str(weights_path),
    )
    return backbone.to(device)


def build_dinov3_fcos(
    weights_path: Path,
    *,
    image_size: int,
    score_threshold: float,
    nms_threshold: float,
    detections_per_image: int,
    topk_candidates: int,
    fpn_channels: int = 256,
    unfreeze_last_stages: int = 0,
    device: str = "cpu",
):
    """Build a one-class, anchor-free detector over official DINOv3 features.

    This deliberately initializes a new FPN and FCOS objectness/box head. No
    weights, predictions, proposals, or labels from the shipped detector or a
    YOLO-family model are consumed.
    """
    torch = require_torch()
    from torchvision.models.detection.fcos import FCOS
    from torchvision.ops.feature_pyramid_network import (
        FeaturePyramidNetwork,
        LastLevelP6P7,
    )

    class DinoV3FeaturePyramid(torch.nn.Module):
        out_channels = fpn_channels

        def __init__(self) -> None:
            super().__init__()
            self.dinov3 = load_dinov3_convnext_tiny(weights_path, device=device)
            if not 0 <= unfreeze_last_stages <= len(self.dinov3.stages):
                raise ValueError("invalid DINOv3 stage-unfreezing count")
            self.unfreeze_last_stages = unfreeze_last_stages
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=[192, 384, 768],
                out_channels=fpn_channels,
                extra_blocks=LastLevelP6P7(768, fpn_channels),
            )
            for parameter in self.dinov3.parameters():
                parameter.requires_grad = False
            if unfreeze_last_stages:
                for stage in self.dinov3.stages[-unfreeze_last_stages:]:
                    for parameter in stage.parameters():
                        parameter.requires_grad = True
                for parameter in self.dinov3.norm.parameters():
                    parameter.requires_grad = True

        def train(self, mode: bool = True):
            super().train(mode)
            self.dinov3.eval()
            if self.unfreeze_last_stages:
                for stage in self.dinov3.stages[-self.unfreeze_last_stages :]:
                    stage.train(mode)
                self.dinov3.norm.train(mode)
            return self

        def forward(self, images):
            if not self.unfreeze_last_stages:
                self.dinov3.eval()
                with torch.no_grad():
                    features = self.dinov3.get_intermediate_layers(
                        images,
                        n=[1, 2, 3],
                        reshape=True,
                        norm=True,
                    )
            else:
                features = self.dinov3.get_intermediate_layers(
                    images,
                    n=[1, 2, 3],
                    reshape=True,
                    norm=True,
                )
            named = OrderedDict((str(index), value) for index, value in enumerate(features))
            return self.fpn(named)

    backbone = DinoV3FeaturePyramid()
    model = FCOS(
        backbone,
        num_classes=1,
        min_size=image_size,
        max_size=image_size,
        image_mean=list(IMAGENET_MEAN),
        image_std=list(IMAGENET_STD),
        center_sampling_radius=1.5,
        score_thresh=score_threshold,
        nms_thresh=nms_threshold,
        detections_per_img=detections_per_image,
        topk_candidates=topk_candidates,
    )
    return model.to(device)


def image_to_tensor(image: Image.Image):
    from torchvision.transforms.functional import pil_to_tensor

    return pil_to_tensor(image).to(dtype=require_torch().float32).div_(255.0)


def target_to_tensors(target: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    boxes = []
    for annotation in target["annotations"]:
        x, y, width, height = annotation["bbox"]
        boxes.append([x, y, x + width, y + height])
    return {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.zeros(len(boxes), dtype=torch.int64),
        "image_id": torch.tensor(int(target["image_id"]), dtype=torch.int64),
    }


def collate_detection_batch(batch):
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)
