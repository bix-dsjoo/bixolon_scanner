from __future__ import annotations

import numpy as np


def build_dinov3_set_localizer(
    *,
    maximum_objects: int = 7,
    hidden_dimension: int = 128,
    decoder_layers: int = 3,
):
    from .models import require_torch

    torch = require_torch()

    class DINOv3SetLocalizer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.maximum_objects = maximum_objects
            self.project_stride16 = torch.nn.Conv2d(384, hidden_dimension, 1)
            self.project_stride32 = torch.nn.Conv2d(768, hidden_dimension, 1)
            layer = torch.nn.TransformerDecoderLayer(
                d_model=hidden_dimension,
                nhead=8,
                dim_feedforward=hidden_dimension * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = torch.nn.TransformerDecoder(layer, num_layers=decoder_layers)
            self.query = torch.nn.Parameter(torch.randn(maximum_objects, hidden_dimension) * 0.02)
            self.objectness = torch.nn.Linear(hidden_dimension, 1)
            self.box = torch.nn.Sequential(
                torch.nn.Linear(hidden_dimension, hidden_dimension),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dimension, 4),
            )

        @staticmethod
        def _position(height: int, width: int, channels: int, *, device, dtype):
            y, x = torch.meshgrid(
                torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
                torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
                indexing="ij",
            )
            frequencies = torch.arange(channels // 4, device=device, dtype=dtype)
            frequencies = 2.0 ** (frequencies / max(channels // 4 - 1, 1) * 8.0)
            values = torch.cat(
                (
                    torch.sin(x[..., None] * frequencies),
                    torch.cos(x[..., None] * frequencies),
                    torch.sin(y[..., None] * frequencies),
                    torch.cos(y[..., None] * frequencies),
                ),
                dim=-1,
            )
            if values.shape[-1] < channels:
                values = torch.nn.functional.pad(values, (0, channels - values.shape[-1]))
            return values[..., :channels].reshape(1, height * width, channels)

        def _tokens(self, feature, projection):
            values = projection(feature)
            batch, channels, height, width = values.shape
            values = values.flatten(2).transpose(1, 2)
            return values + self._position(
                height,
                width,
                channels,
                device=values.device,
                dtype=values.dtype,
            )

        def forward(self, stride16, stride32):
            memory = torch.cat(
                (
                    self._tokens(stride16, self.project_stride16),
                    self._tokens(stride32, self.project_stride32),
                ),
                dim=1,
            )
            queries = self.query[None].expand(len(memory), -1, -1)
            decoded = self.decoder(queries, memory)
            return self.objectness(decoded).squeeze(-1), self.box(decoded).sigmoid()

    return DINOv3SetLocalizer()


def cxcywh_to_xyxy(boxes):
    center_x, center_y, width, height = boxes.unbind(-1)
    return boxes.new_tensor([-0.5, -0.5, 0.5, 0.5]) * torch_stack(
        (width, height, width, height), dim=-1
    ) + torch_stack((center_x, center_y, center_x, center_y), dim=-1)


def torch_stack(values, *, dim: int):
    import torch

    return torch.stack(values, dim=dim)


def generalized_box_iou_aligned(left, right):
    left_xyxy = cxcywh_to_xyxy(left)
    right_xyxy = cxcywh_to_xyxy(right)
    intersection_top_left = torch_stack(
        (
            torch_maximum(left_xyxy[:, 0], right_xyxy[:, 0]),
            torch_maximum(left_xyxy[:, 1], right_xyxy[:, 1]),
        ),
        dim=1,
    )
    intersection_bottom_right = torch_stack(
        (
            torch_minimum(left_xyxy[:, 2], right_xyxy[:, 2]),
            torch_minimum(left_xyxy[:, 3], right_xyxy[:, 3]),
        ),
        dim=1,
    )
    intersection = (intersection_bottom_right - intersection_top_left).clamp(min=0).prod(1)
    left_area = (left_xyxy[:, 2:] - left_xyxy[:, :2]).clamp(min=0).prod(1)
    right_area = (right_xyxy[:, 2:] - right_xyxy[:, :2]).clamp(min=0).prod(1)
    union = left_area + right_area - intersection
    iou = intersection / union.clamp(min=1e-8)
    enclosing_top_left = torch_stack(
        (
            torch_minimum(left_xyxy[:, 0], right_xyxy[:, 0]),
            torch_minimum(left_xyxy[:, 1], right_xyxy[:, 1]),
        ),
        dim=1,
    )
    enclosing_bottom_right = torch_stack(
        (
            torch_maximum(left_xyxy[:, 2], right_xyxy[:, 2]),
            torch_maximum(left_xyxy[:, 3], right_xyxy[:, 3]),
        ),
        dim=1,
    )
    enclosing = (enclosing_bottom_right - enclosing_top_left).clamp(min=0).prod(1)
    return iou - (enclosing - union) / enclosing.clamp(min=1e-8)


def torch_maximum(left, right):
    import torch

    return torch.maximum(left, right)


def torch_minimum(left, right):
    import torch

    return torch.minimum(left, right)


def hungarian_set_loss(
    object_logits,
    predicted_boxes,
    targets: list,
    *,
    objectness_weight: float = 1.0,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
):
    import torch
    import torch.nn.functional as functional
    from scipy.optimize import linear_sum_assignment

    object_targets = torch.zeros_like(object_logits)
    l1_losses = []
    giou_losses = []
    for batch_index, target in enumerate(targets):
        prediction = predicted_boxes[batch_index]
        with torch.no_grad():
            l1_cost = torch.cdist(prediction, target, p=1)
            prediction_xyxy = cxcywh_to_xyxy(prediction)
            target_xyxy = cxcywh_to_xyxy(target)
            top_left = torch.maximum(prediction_xyxy[:, None, :2], target_xyxy[None, :, :2])
            bottom_right = torch.minimum(prediction_xyxy[:, None, 2:], target_xyxy[None, :, 2:])
            intersection = (bottom_right - top_left).clamp(min=0).prod(2)
            prediction_area = (prediction_xyxy[:, 2:] - prediction_xyxy[:, :2]).clamp(min=0).prod(1)
            target_area = (target_xyxy[:, 2:] - target_xyxy[:, :2]).clamp(min=0).prod(1)
            union = prediction_area[:, None] + target_area[None] - intersection
            iou = intersection / union.clamp(min=1e-8)
            query_indices, target_indices = linear_sum_assignment(
                (2.0 * l1_cost + 2.0 * (1.0 - iou)).detach().cpu().numpy()
            )
        query_indices = torch.as_tensor(query_indices, device=prediction.device)
        target_indices = torch.as_tensor(target_indices, device=prediction.device)
        object_targets[batch_index, query_indices] = 1.0
        matched_prediction = prediction[query_indices]
        matched_target = target[target_indices]
        l1_losses.append(functional.l1_loss(matched_prediction, matched_target))
        giou_losses.append(
            (1.0 - generalized_box_iou_aligned(matched_prediction, matched_target)).mean()
        )
    object_weight = torch.where(object_targets > 0.0, 1.0, 0.2)
    object_loss = functional.binary_cross_entropy_with_logits(
        object_logits, object_targets, weight=object_weight
    )
    l1_loss = torch.stack(l1_losses).mean()
    giou_loss = torch.stack(giou_losses).mean()
    total = objectness_weight * object_loss + l1_weight * l1_loss + giou_weight * giou_loss
    return total, {
        "objectness": object_loss.detach(),
        "l1": l1_loss.detach(),
        "giou": giou_loss.detach(),
    }


def normalized_target_boxes(record: dict) -> np.ndarray:
    rows = []
    for annotation in record["annotations"]:
        x, y, width, height = annotation["bbox_xywh"]
        rows.append(
            [
                (x + width * 0.5) / record["width"],
                (y + height * 0.5) / record["height"],
                width / record["width"],
                height / record["height"],
            ]
        )
    return np.asarray(rows, dtype=np.float32)
