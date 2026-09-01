from __future__ import annotations

import numpy as np


def build_dinov3_center_localizer(*, hidden_dimension: int = 128):
    from .models import require_torch

    torch = require_torch()

    class DINOv3CenterLocalizer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stride8 = torch.nn.Conv2d(192, hidden_dimension, 1)
            self.stride16 = torch.nn.Conv2d(384, hidden_dimension, 1)
            self.stride32 = torch.nn.Conv2d(768, hidden_dimension, 1)
            self.fusion = torch.nn.Sequential(
                torch.nn.Conv2d(hidden_dimension, hidden_dimension, 3, padding=1),
                torch.nn.GroupNorm(16, hidden_dimension),
                torch.nn.GELU(),
                torch.nn.Conv2d(hidden_dimension, hidden_dimension, 3, padding=1),
                torch.nn.GroupNorm(16, hidden_dimension),
                torch.nn.GELU(),
            )
            self.heatmap = torch.nn.Conv2d(hidden_dimension, 1, 1)
            self.offset = torch.nn.Conv2d(hidden_dimension, 2, 1)
            self.size = torch.nn.Conv2d(hidden_dimension, 2, 1)
            torch.nn.init.constant_(self.heatmap.bias, -2.19)

        def forward(self, feature8, feature16, feature32):
            target_size = feature8.shape[-2:]
            fused = self.stride8(feature8)
            fused = fused + torch.nn.functional.interpolate(
                self.stride16(feature16), size=target_size, mode="bilinear", align_corners=False
            )
            fused = fused + torch.nn.functional.interpolate(
                self.stride32(feature32), size=target_size, mode="bilinear", align_corners=False
            )
            fused = self.fusion(fused)
            return self.heatmap(fused), self.offset(fused).sigmoid(), self.size(fused).sigmoid()

    return DINOv3CenterLocalizer()


def center_targets(records: list[dict], *, height: int, width: int, device):
    import torch

    heatmap = torch.zeros((len(records), 1, height, width), device=device)
    offset = torch.zeros((len(records), 2, height, width), device=device)
    size = torch.zeros((len(records), 2, height, width), device=device)
    mask = torch.zeros((len(records), 1, height, width), device=device)
    for batch_index, record in enumerate(records):
        for annotation in record["annotations"]:
            x, y, box_width, box_height = annotation["bbox_xywh"]
            center_x = (x + box_width * 0.5) / record["width"] * width
            center_y = (y + box_height * 0.5) / record["height"] * height
            grid_x = min(max(int(center_x), 0), width - 1)
            grid_y = min(max(int(center_y), 0), height - 1)
            radius = 2
            y0, y1 = max(0, grid_y - radius), min(height, grid_y + radius + 1)
            x0, x1 = max(0, grid_x - radius), min(width, grid_x + radius + 1)
            yy, xx = torch.meshgrid(
                torch.arange(y0, y1, device=device),
                torch.arange(x0, x1, device=device),
                indexing="ij",
            )
            gaussian = torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / 2.0)
            heatmap[batch_index, 0, y0:y1, x0:x1] = torch.maximum(
                heatmap[batch_index, 0, y0:y1, x0:x1], gaussian
            )
            heatmap[batch_index, 0, grid_y, grid_x] = 1.0
            offset[batch_index, :, grid_y, grid_x] = torch.tensor(
                [center_x - grid_x, center_y - grid_y], device=device
            )
            size[batch_index, :, grid_y, grid_x] = torch.tensor(
                [box_width / record["width"], box_height / record["height"]], device=device
            )
            mask[batch_index, 0, grid_y, grid_x] = 1.0
    return heatmap, offset, size, mask


def center_localizer_loss(prediction, targets):
    import torch
    import torch.nn.functional as functional

    heatmap_logits, offset_prediction, size_prediction = prediction
    heatmap_target, offset_target, size_target, mask = targets
    probability = heatmap_logits.sigmoid().clamp(1e-5, 1.0 - 1e-5)
    positive = heatmap_target.eq(1.0)
    negative = ~positive
    positive_loss = -torch.log(probability) * (1.0 - probability).pow(2) * positive
    negative_loss = (
        -torch.log(1.0 - probability)
        * probability.pow(2)
        * (1.0 - heatmap_target).pow(4)
        * negative
    )
    positive_count = positive.sum().clamp(min=1)
    heatmap_loss = (positive_loss.sum() + negative_loss.sum()) / positive_count
    coordinate_count = mask.sum().clamp(min=1)
    offset_loss = (
        functional.l1_loss(offset_prediction * mask, offset_target * mask, reduction="sum")
        / coordinate_count
    )
    size_loss = (
        functional.l1_loss(size_prediction * mask, size_target * mask, reduction="sum")
        / coordinate_count
    )
    total = heatmap_loss + offset_loss + 2.0 * size_loss
    return total, {
        "heatmap": heatmap_loss.detach(),
        "offset": offset_loss.detach(),
        "size": size_loss.detach(),
    }


def decode_center_predictions(prediction, *, maximum_objects: int):
    import torch

    heatmap_logits, offsets, sizes = prediction
    heatmap = heatmap_logits.sigmoid()
    local_maximum = torch.nn.functional.max_pool2d(heatmap, 3, stride=1, padding=1)
    heatmap = heatmap * heatmap.eq(local_maximum)
    batch, _, height, width = heatmap.shape
    scores, indices = torch.topk(heatmap.reshape(batch, -1), maximum_objects, dim=1)
    y = torch.div(indices, width, rounding_mode="floor")
    x = indices % width
    batch_indices = torch.arange(batch, device=heatmap.device)[:, None]
    selected_offsets = offsets.permute(0, 2, 3, 1)[batch_indices, y, x]
    selected_sizes = sizes.permute(0, 2, 3, 1)[batch_indices, y, x]
    centers = torch.stack(
        ((x + selected_offsets[..., 0]) / width, (y + selected_offsets[..., 1]) / height),
        dim=-1,
    )
    return scores, torch.cat((centers, selected_sizes), dim=-1)


def flip_detection_records(records: list[dict], flips: np.ndarray) -> list[dict]:
    outputs = []
    for record, flip in zip(records, flips, strict=True):
        if not flip:
            outputs.append(record)
            continue
        transformed = {**record, "annotations": []}
        for annotation in record["annotations"]:
            copied = dict(annotation)
            x, y, width, height = annotation["bbox_xywh"]
            copied["bbox_xywh"] = [record["width"] - x - width, y, width, height]
            transformed["annotations"].append(copied)
        outputs.append(transformed)
    return outputs
