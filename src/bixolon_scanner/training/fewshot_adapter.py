from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import build_dino_classifier, require_torch


@dataclass(frozen=True)
class AdapterSpec:
    """Deployable 10-shot head. Support examples are never runtime inputs."""

    hidden_size: int
    bottleneck_size: int
    num_classes: int
    cosine_scale: float = 30.0
    cosine_margin: float = 0.2
    proxies_per_class: int = 1
    proxy_temperature: float = 10.0
    use_local_features: bool = False

    def validate(self) -> None:
        if self.hidden_size < 1 or self.bottleneck_size < 1:
            raise ValueError("adapter dimensions must be positive")
        if self.num_classes < 2:
            raise ValueError("adapter requires at least two classes")
        if self.cosine_scale <= 0:
            raise ValueError("cosine scale must be positive")
        if not 0.0 <= self.cosine_margin < 1.0:
            raise ValueError("cosine_margin must be in [0, 1)")
        if self.proxies_per_class < 1:
            raise ValueError("proxies_per_class must be positive")
        if self.proxy_temperature <= 0:
            raise ValueError("proxy_temperature must be positive")


def build_residual_cosine_head(spec: AdapterSpec):
    spec.validate()
    torch = require_torch()

    class ResidualCosineHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter_down = torch.nn.Linear(spec.hidden_size, spec.bottleneck_size)
            self.adapter_up = torch.nn.Linear(spec.bottleneck_size, spec.hidden_size)
            self.activation = torch.nn.GELU()
            if spec.use_local_features:
                self.local_query = torch.nn.Parameter(torch.empty(spec.hidden_size))
                self.local_projection = torch.nn.Linear(spec.hidden_size, spec.hidden_size)
                self.fusion_norm = torch.nn.LayerNorm(spec.hidden_size)
                torch.nn.init.normal_(self.local_query, std=0.02)
                torch.nn.init.zeros_(self.local_projection.weight)
                torch.nn.init.zeros_(self.local_projection.bias)
            self.class_weights = torch.nn.Parameter(
                torch.empty(spec.num_classes, spec.proxies_per_class, spec.hidden_size)
            )
            self.register_buffer("cosine_scale", torch.tensor(spec.cosine_scale))
            self.cosine_margin = spec.cosine_margin
            torch.nn.init.normal_(self.class_weights, std=0.02)
            torch.nn.init.zeros_(self.adapter_up.weight)
            torch.nn.init.zeros_(self.adapter_up.bias)

        def fuse(self, features, patch_features=None):
            if not spec.use_local_features:
                return features
            if patch_features is None or patch_features.ndim != 3:
                raise ValueError("local head requires [batch, patches, hidden] features")
            if (
                patch_features.shape[0] != features.shape[0]
                or patch_features.shape[2] != spec.hidden_size
            ):
                raise ValueError("global and local feature shapes are not aligned")
            attention = torch.softmax(
                patch_features @ self.local_query / (spec.hidden_size**0.5), dim=1
            )
            local = (patch_features * attention.unsqueeze(-1)).sum(dim=1)
            return self.fusion_norm(features + self.local_projection(local))

        def adapt(self, features, patch_features=None):
            features = self.fuse(features, patch_features)
            residual = self.adapter_up(self.activation(self.adapter_down(features)))
            return torch.nn.functional.normalize(features + residual, p=2.0, dim=-1, eps=1e-12)

        def _class_scores(self, adapted):
            weights = torch.nn.functional.normalize(self.class_weights, p=2.0, dim=-1, eps=1e-12)
            proxy_scores = torch.einsum("bh,ckh->bck", adapted, weights)
            if spec.proxies_per_class == 1:
                return proxy_scores[..., 0]
            return (
                torch.logsumexp(proxy_scores * spec.proxy_temperature, dim=-1)
                / spec.proxy_temperature
            )

        def forward(self, features, patch_features=None):
            adapted = self.adapt(features, patch_features)
            return self._class_scores(adapted) * self.cosine_scale

        def training_logits(self, features, labels, patch_features=None):
            adapted = self.adapt(features, patch_features)
            scores = self._class_scores(adapted)
            if self.cosine_margin:
                margin = torch.zeros_like(scores)
                margin.scatter_(1, labels.reshape(-1, 1), self.cosine_margin)
                scores = scores - margin
            return scores * self.cosine_scale, adapted

        def initialize_class_weights_from_support(self, features, labels, proxy_ids=None):
            values = torch.as_tensor(
                features, dtype=self.class_weights.dtype, device=self.class_weights.device
            )
            targets = torch.as_tensor(labels, dtype=torch.long, device=values.device)
            proxies = (
                torch.zeros_like(targets)
                if proxy_ids is None
                else torch.as_tensor(proxy_ids, dtype=torch.long, device=values.device)
            )
            if values.ndim != 2 or values.shape[1] != spec.hidden_size:
                raise ValueError("support features have the wrong shape")
            if (
                proxies.shape != targets.shape
                or bool((proxies < 0).any())
                or bool((proxies >= spec.proxies_per_class).any())
            ):
                raise ValueError("support proxy ids are invalid")
            prototypes = []
            for class_index in range(spec.num_classes):
                selected = values[targets == class_index]
                if selected.shape[0] == 0:
                    raise ValueError(f"class {class_index} has no support features")
                class_prototypes = []
                for proxy_index in range(spec.proxies_per_class):
                    proxy_values = values[(targets == class_index) & (proxies == proxy_index)]
                    class_prototypes.append(
                        proxy_values.mean(dim=0) if proxy_values.shape[0] else selected.mean(dim=0)
                    )
                prototypes.append(torch.stack(class_prototypes))
            with torch.no_grad():
                self.class_weights.copy_(
                    torch.nn.functional.normalize(torch.stack(prototypes), p=2.0, dim=-1, eps=1e-12)
                )

    return ResidualCosineHead()


def supervised_contrastive_loss(features, labels, *, temperature: float = 0.1):
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    torch = require_torch()
    values = torch.nn.functional.normalize(features, p=2.0, dim=-1, eps=1e-12)
    targets = labels.reshape(-1)
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError("contrastive features and labels are not aligned")
    if len(values) < 2:
        return values.sum() * 0.0
    logits = values @ values.transpose(0, 1) / temperature
    identity = torch.eye(len(values), device=values.device, dtype=torch.bool)
    positives = targets[:, None].eq(targets[None, :]) & ~identity
    valid = positives.any(dim=1)
    if not bool(valid.any()):
        return values.sum() * 0.0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~identity)
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive_log_probability = (log_probability * positives).sum(dim=1) / positives.sum(
        dim=1
    ).clamp_min(1)
    return -mean_positive_log_probability[valid].mean()


def compatible_proxy_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Upgrade pre-0.2.1 single-proxy class weights without changing checkpoints."""
    upgraded = dict(state_dict)
    for key in ("class_weights", "classifier.class_weights"):
        value = upgraded.get(key)
        if value is not None and getattr(value, "ndim", None) == 2:
            upgraded[key] = value.unsqueeze(1)
    return upgraded


def wrap_inference_classifier(
    classifier,
    *,
    input_size: int,
    crop_scale: float | None,
    num_classes: int,
    logit_quantum: float | None = None,
    logit_phase: float = 0.0,
    tie_break_bias_span: float = 0.0,
    logit_divisor: float = 1.0,
):
    """Embed the selected crop and deterministic logit policy in exported ONNX."""
    if input_size < 1:
        raise ValueError("input_size must be positive")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if crop_scale is not None and not 0.5 <= crop_scale < 1.0:
        raise ValueError("center crop scale must be in [0.5, 1.0)")
    if logit_quantum is not None and logit_quantum <= 0:
        raise ValueError("logit quantum must be positive")
    if tie_break_bias_span < 0:
        raise ValueError("tie-break bias span must be non-negative")
    if logit_divisor <= 0:
        raise ValueError("logit divisor must be positive")
    identity = (
        crop_scale is None
        and logit_quantum is None
        and tie_break_bias_span == 0.0
        and logit_divisor == 1.0
    )
    if identity:
        return classifier
    torch = require_torch()
    crop_size = input_size if crop_scale is None else max(1, round(input_size * crop_scale))
    offset = (input_size - crop_size) // 2

    class InferenceClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = classifier
            self.register_buffer(
                "tie_break_bias",
                torch.linspace(0.0, -tie_break_bias_span, num_classes),
            )

        def forward(self, pixel_values):
            if crop_scale is None:
                classifier_input = pixel_values
            else:
                cropped = pixel_values[
                    ...,
                    offset : offset + crop_size,
                    offset : offset + crop_size,
                ]
                classifier_input = torch.nn.functional.interpolate(
                    cropped,
                    size=(input_size, input_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
            logits = self.classifier(classifier_input)
            if logit_quantum is not None:
                logits = (
                    torch.round((logits + logit_phase) / logit_quantum) * logit_quantum
                    - logit_phase
                )
            if tie_break_bias_span:
                logits = logits + self.tie_break_bias
            return logits / logit_divisor

    return InferenceClassifier()


def build_ten_shot_classifier(
    *, backbone_kind: str, weights_path: Path | None, hub_repository: str, spec: AdapterSpec
):
    torch = require_torch()
    base = build_dino_classifier(
        backbone_kind,
        spec.num_classes,
        weights_path=weights_path,
        hub_repository=hub_repository,
    )

    class TenShotClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = base.backbone
            self.backbone_kind = backbone_kind
            self.classifier = build_residual_cosine_head(spec)

        def extract_features(self, pixel_values):
            if self.backbone_kind == "dinov2":
                return self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
            values = self.backbone.forward_features(pixel_values)
            if spec.use_local_features:
                return values["x_norm_clstoken"], values["x_norm_patchtokens"]
            return values["x_norm_clstoken"]

        def forward(self, pixel_values):
            features = self.extract_features(pixel_values)
            if isinstance(features, tuple):
                return self.classifier(*features)
            return self.classifier(features)

    return TenShotClassifier()


def adapter_spec_from_dict(value: dict[str, Any]) -> AdapterSpec:
    spec = AdapterSpec(
        hidden_size=int(value["hidden_size"]),
        bottleneck_size=int(value["bottleneck_size"]),
        num_classes=int(value["num_classes"]),
        cosine_scale=float(value.get("cosine_scale", 30.0)),
        cosine_margin=float(value.get("cosine_margin", 0.2)),
        proxies_per_class=int(value.get("proxies_per_class", 1)),
        proxy_temperature=float(value.get("proxy_temperature", 10.0)),
        use_local_features=bool(value.get("use_local_features", False)),
    )
    spec.validate()
    return spec
