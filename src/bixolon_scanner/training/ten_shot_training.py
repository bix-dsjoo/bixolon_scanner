from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fewshot_adapter import (
    AdapterSpec,
    build_residual_cosine_head,
    supervised_contrastive_loss,
)
from .models import require_torch


@dataclass(frozen=True)
class HeadTrainingConfig:
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.1
    seed: int = 20260812

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("training epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("training optimizer values are invalid")
        if self.contrastive_weight < 0 or self.contrastive_temperature <= 0:
            raise ValueError("contrastive training values are invalid")


def _seed_everything(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_cache_fingerprint(
    *,
    manifest_sha256: str,
    backbone_sha256: str,
    synthetic_recipe_sha256: str,
    background_sha256: list[str] | None = None,
) -> str:
    value = {
        "manifest_sha256": manifest_sha256,
        "backbone_sha256": backbone_sha256,
        "synthetic_recipe_sha256": synthetic_recipe_sha256,
        # Kept in the fingerprint schema only for rejecting legacy caches that
        # used operating backgrounds. Strict 10-shot caches always store [].
        "background_sha256": sorted(background_sha256 or []),
    }
    body = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def validate_feature_cache(
    features: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray,
    *,
    spec: AdapterSpec,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    patch_features: np.ndarray | None = None,
    support_patch_features: np.ndarray | None = None,
    support_proxy_ids: np.ndarray | None = None,
) -> None:
    values = np.asarray(features)
    targets = np.asarray(labels)
    sources = np.asarray(source_indices)
    support = np.asarray(support_features)
    support_targets = np.asarray(support_labels)
    if values.ndim != 2 or values.shape[1] != spec.hidden_size:
        raise ValueError("training features have the wrong shape")
    if targets.shape != (len(values),) or sources.shape != (len(values),):
        raise ValueError("feature labels and source indices are not aligned")
    if not np.isfinite(values).all() or not np.isfinite(support).all():
        raise ValueError("feature cache contains non-finite values")
    if support.ndim != 2 or support.shape[1] != spec.hidden_size:
        raise ValueError("support feature cache has the wrong shape")
    if support_targets.shape != (len(support),):
        raise ValueError("support labels have the wrong shape")
    expected_classes = np.arange(spec.num_classes)
    if not np.array_equal(np.unique(targets), expected_classes):
        raise ValueError("training feature cache does not cover every class")
    counts = np.bincount(support_targets.astype(np.int64), minlength=spec.num_classes)
    if len(counts) != spec.num_classes or np.any(counts == 0) or len(set(counts)) != 1:
        raise ValueError("support feature cache is not balanced")
    if sources.min(initial=0) < 0 or sources.max(initial=0) >= len(support):
        raise ValueError("training source index is outside the 10-shot support set")
    if spec.use_local_features:
        patches = np.asarray(patch_features)
        support_patches = np.asarray(support_patch_features)
        if patches.ndim != 3 or patches.shape[0] != len(values) or patches.shape[2] != spec.hidden_size:
            raise ValueError("training patch features have the wrong shape")
        if support_patches.ndim != 3 or support_patches.shape[0] != len(support) or support_patches.shape[2] != spec.hidden_size:
            raise ValueError("support patch features have the wrong shape")
    if support_proxy_ids is not None:
        proxy_ids = np.asarray(support_proxy_ids)
        if proxy_ids.shape != support_targets.shape:
            raise ValueError("support proxy ids are not aligned")
        if proxy_ids.min(initial=0) < 0 or proxy_ids.max(initial=0) >= spec.proxies_per_class:
            raise ValueError("support proxy ids are invalid")


def train_adapter_head(
    features: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray,
    *,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    patch_features: np.ndarray | None = None,
    support_patch_features: np.ndarray | None = None,
    support_proxy_ids: np.ndarray | None = None,
    spec: AdapterSpec,
    config: HeadTrainingConfig,
    device: str = "cpu",
):
    spec.validate()
    config.validate()
    validate_feature_cache(
        features,
        labels,
        source_indices,
        spec=spec,
        support_features=support_features,
        support_labels=support_labels,
        patch_features=patch_features,
        support_patch_features=support_patch_features,
        support_proxy_ids=support_proxy_ids,
    )
    torch = require_torch()
    _seed_everything(config.seed)
    head = build_residual_cosine_head(spec).to(device)
    head.initialize_class_weights_from_support(
        support_features, support_labels, support_proxy_ids
    )
    values = torch.as_tensor(features, dtype=torch.float32)
    targets = torch.as_tensor(labels, dtype=torch.long)
    sources = torch.as_tensor(source_indices, dtype=torch.long)
    patches = None
    if spec.use_local_features:
        patches = torch.from_numpy(
            np.array(patch_features, dtype=np.float32, copy=True)
        )
        dataset = torch.utils.data.TensorDataset(values, targets, sources, patches)
    else:
        dataset = torch.utils.data.TensorDataset(values, targets, sources)
    generator = torch.Generator().manual_seed(config.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        head.train()
        total_loss = total_cross_entropy = total_contrastive = 0.0
        sample_count = 0
        source_loss_sums: dict[int, float] = {}
        source_loss_counts: dict[int, int] = {}
        for batch in loader:
            batch_features, batch_labels, batch_sources = batch[:3]
            batch_patches = batch[3].to(device) if len(batch) == 4 else None
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_sources = batch_sources.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, adapted = head.training_logits(
                batch_features, batch_labels, batch_patches
            )
            per_sample = torch.nn.functional.cross_entropy(
                logits, batch_labels, reduction="none"
            )
            # Each original SHA receives equal total weight even when a future
            # cache contains a different number of accepted synthetic views.
            unique_sources, inverse, counts = torch.unique(
                batch_sources, return_inverse=True, return_counts=True
            )
            weights = counts[inverse].to(per_sample.dtype).reciprocal()
            cross_entropy = (per_sample * weights).sum() / weights.sum()
            contrastive = supervised_contrastive_loss(
                adapted,
                batch_labels,
                temperature=config.contrastive_temperature,
            )
            loss = cross_entropy + config.contrastive_weight * contrastive
            loss.backward()
            optimizer.step()
            count = len(batch_features)
            sample_count += count
            total_loss += float(loss.detach()) * count
            total_cross_entropy += float(cross_entropy.detach()) * count
            total_contrastive += float(contrastive.detach()) * count
            for source, value in zip(
                batch_sources.detach().cpu().tolist(), per_sample.detach().cpu().tolist()
            ):
                source_loss_sums[source] = source_loss_sums.get(source, 0.0) + float(value)
                source_loss_counts[source] = source_loss_counts.get(source, 0) + 1
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / sample_count,
                "cross_entropy": total_cross_entropy / sample_count,
                "contrastive": total_contrastive / sample_count,
                "source_balanced_cross_entropy": float(
                    np.mean(
                        [
                            source_loss_sums[key] / source_loss_counts[key]
                            for key in sorted(source_loss_sums)
                        ]
                    )
                ),
            }
        )
    head.eval()
    return head, history


def save_head_checkpoint(
    path: Path,
    *,
    head,
    spec: AdapterSpec,
    training_config: HeadTrainingConfig,
    history: list[dict[str, Any]],
    dataset_version: str,
    manifest_sha256: str,
    feature_cache_sha256: str,
    backbone_kind: str,
    backbone_revision: str,
    backbone_weight_sha256: str,
    backbone_weight_filename: str,
    image_size: int,
) -> None:
    torch = require_torch()
    checkpoint = {
        "schema_version": "1.0",
        "architecture": "ten_shot_residual_cosine",
        "adapter_spec": asdict(spec),
        "training_config": asdict(training_config),
        "head_state_dict": head.state_dict(),
        "history": history,
        "dataset_version": dataset_version,
        "manifest_sha256": manifest_sha256,
        "feature_cache_sha256": feature_cache_sha256,
        "backbone_kind": backbone_kind,
        "source_revision": backbone_revision,
        "source_weight_sha256": backbone_weight_sha256,
        "source_weight_filename": backbone_weight_filename,
        "image_size": image_size,
        "num_classes": spec.num_classes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
