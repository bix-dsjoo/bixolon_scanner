from __future__ import annotations

from pathlib import Path

DINO_V3_CONVNEXT_TINY = "dinov3_convnext_tiny"
DINO_V3_VIT_BASE_16 = "dinov3_vitb16"
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
    feature_l2_normalize: bool = False,
    classifier_head_kind: str = "linear",
    support_per_class: int | None = None,
    hybrid_knn_k: int = 3,
    hybrid_prototype_weight: float = 0.5,
    cosine_scale: float = 16.0,
):
    torch = require_torch()

    class PrototypeKnnHybridHead(torch.nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            if support_per_class is None or support_per_class < 1:
                raise ValueError("hybrid classifier requires support_per_class")
            if not 1 <= hybrid_knn_k <= support_per_class:
                raise ValueError("hybrid k must be between 1 and support_per_class")
            if not 0.0 <= hybrid_prototype_weight <= 1.0:
                raise ValueError("hybrid prototype weight must be between 0 and 1")
            self.register_buffer("prototypes", torch.zeros(num_classes, hidden_size))
            self.register_buffer(
                "exemplars",
                torch.zeros(num_classes, support_per_class, hidden_size),
            )
            self.knn_k = int(hybrid_knn_k)
            self.prototype_weight = float(hybrid_prototype_weight)

        def forward(self, features):
            prototype_scores = features @ self.prototypes.transpose(0, 1)
            flat_exemplars = self.exemplars.flatten(0, 1)
            exemplar_scores = features @ flat_exemplars.transpose(0, 1)
            exemplar_scores = exemplar_scores.reshape(
                features.shape[0], num_classes, int(support_per_class)
            )
            knn_scores = torch.topk(
                exemplar_scores, self.knn_k, dim=-1, largest=True, sorted=False
            ).values.mean(dim=-1)
            return (
                self.prototype_weight * prototype_scores
                + (1.0 - self.prototype_weight) * knn_scores
            )

    class CosineClassifierHead(torch.nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            if cosine_scale <= 0.0:
                raise ValueError("cosine classifier scale must be positive")
            self.weight = torch.nn.Parameter(torch.empty(num_classes, hidden_size))
            torch.nn.init.trunc_normal_(self.weight, std=0.02)
            self.scale = float(cosine_scale)

        def forward(self, features):
            normalized_features = torch.nn.functional.normalize(features, p=2.0, dim=-1, eps=1e-12)
            normalized_weight = torch.nn.functional.normalize(self.weight, p=2.0, dim=-1, eps=1e-12)
            return self.scale * normalized_features @ normalized_weight.transpose(0, 1)

    class DinoClassifier(torch.nn.Module):
        def __init__(self, backbone, hidden_size: int, kind: str):
            super().__init__()
            self.backbone = backbone
            self.backbone_kind = kind
            self.feature_l2_normalize = feature_l2_normalize
            if classifier_head_kind == "linear":
                self.classifier = torch.nn.Linear(hidden_size, num_classes)
            elif classifier_head_kind == "cosine":
                self.classifier = CosineClassifierHead(hidden_size)
            elif classifier_head_kind == "prototype_knn_hybrid":
                self.classifier = PrototypeKnnHybridHead(hidden_size)
            else:
                raise ValueError(f"unsupported classifier head: {classifier_head_kind}")

        def extract_features(self, pixel_values):
            if self.backbone_kind == "dinov2":
                features = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
            else:
                features = self.backbone(pixel_values)
            if self.feature_l2_normalize:
                features = torch.nn.functional.normalize(features, p=2.0, dim=-1, eps=1e-12)
            return features

        def forward(self, pixel_values):
            features = self.extract_features(pixel_values)
            return self.classifier(features)

    if backbone_kind == "dinov2":
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(pretrained_name)
        return DinoClassifier(backbone, int(backbone.config.hidden_size), backbone_kind)
    if backbone_kind in {DINO_V3_CONVNEXT_TINY, DINO_V3_VIT_BASE_16}:
        load_options = (
            {"pretrained": True, "weights": str(weights_path)}
            if weights_path
            else {"pretrained": False}
        )
        backbone = torch.hub.load(
            hub_repository,
            backbone_kind,
            source="github",
            trust_repo=True,
            verbose=False,
            **load_options,
        )
        # The official pretrained heads are Identity and both backbones return
        # a 768-dimensional global representation.
        return DinoClassifier(backbone, 768, backbone_kind)
    raise ValueError(f"unsupported classifier backbone: {backbone_kind}")


def set_frozen_backbone(
    model,
    *,
    unfreeze_last_blocks: int = 0,
    unfreeze_last_stages: int = 0,
    unfreeze_all: bool = False,
) -> None:
    if sum(bool(value) for value in (unfreeze_last_blocks, unfreeze_last_stages, unfreeze_all)) > 1:
        raise ValueError("choose one backbone unfreezing policy")
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    if unfreeze_all:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
    elif unfreeze_last_stages:
        if model.backbone_kind != "dinov3_convnext_tiny":
            raise ValueError("stage unfreezing is only supported for ConvNeXt backbones")
        stages = model.backbone.stages
        if not 1 <= unfreeze_last_stages <= len(stages):
            raise ValueError("unfreeze_last_stages exceeds the ConvNeXt stage count")
        for stage in stages[-unfreeze_last_stages:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True
        for parameter in model.backbone.norm.parameters():
            parameter.requires_grad = True
    elif unfreeze_last_blocks:
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
