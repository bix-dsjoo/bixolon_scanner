from __future__ import annotations

from pathlib import Path


DINO_V3_CONVNEXT_TINY = "dinov3_convnext_tiny"
DINO_V3_HUB_REPOSITORY = "facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751"


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("install the 'training' extra to use training commands") from exc
    return torch


def build_dino_classifier(
    backbone_kind: str,
    num_classes: int,
    *,
    pretrained_name: str = "facebook/dinov2-small",
    weights_path: Path | None = None,
    hub_repository: str = DINO_V3_HUB_REPOSITORY,
):
    torch = require_torch()

    class DinoClassifier(torch.nn.Module):
        def __init__(self, backbone, hidden_size: int, kind: str):
            super().__init__()
            self.backbone = backbone
            self.backbone_kind = kind
            self.classifier = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, pixel_values):
            if self.backbone_kind == "dinov2":
                features = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
            else:
                features = self.backbone(pixel_values)
            return self.classifier(features)

    if backbone_kind == "dinov2":
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(pretrained_name)
        return DinoClassifier(backbone, int(backbone.config.hidden_size), backbone_kind)
    if backbone_kind == "dinov3_convnext_tiny":
        load_options = (
            {"pretrained": True, "weights": str(weights_path)}
            if weights_path
            else {"pretrained": False}
        )
        backbone = torch.hub.load(
            hub_repository,
            DINO_V3_CONVNEXT_TINY,
            source="github",
            trust_repo=True,
            verbose=False,
            **load_options,
        )
        # The official pretrained ConvNeXt head is Identity and returns a
        # normalized 768-dimensional representation.
        return DinoClassifier(backbone, 768, backbone_kind)
    raise ValueError(f"unsupported classifier backbone: {backbone_kind}")


def set_frozen_backbone(model, *, unfreeze_last_blocks: int = 0) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    if unfreeze_last_blocks:
        if model.backbone_kind == "dinov2":
            blocks = model.backbone.encoder.layer
            normalization = model.backbone.layernorm
        elif model.backbone_kind == "dinov3_convnext_tiny":
            blocks = model.backbone.stages[-1]
            normalization = model.backbone.norm
        else:  # pragma: no cover - guarded by the model builder
            raise ValueError(f"unsupported classifier backbone: {model.backbone_kind}")
        for layer in blocks[-unfreeze_last_blocks:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in normalization.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
