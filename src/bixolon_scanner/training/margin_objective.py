"""Align classifier training with the Runtime's normalized logit-gap evidence."""

from __future__ import annotations

from .models import require_torch


def normalized_margin_loss(logits, labels, *, target_margin: float):
    torch = require_torch()
    if not 0 < target_margin <= 1:
        raise ValueError("target margin must be inside (0, 1]")
    if logits.ndim != 2 or logits.shape[1] < 2 or labels.shape != (len(logits),):
        raise ValueError("margin loss requires aligned class logits and labels")
    values = logits.float()
    correct = values.gather(1, labels[:, None]).squeeze(1)
    others = values.scatter(1, labels[:, None], float("-inf")).max(dim=1).values
    # Use a signed GT margin: a confident wrong answer must receive a penalty.
    margin = (correct - others) / values.norm(dim=1).clamp_min(1e-12)
    return torch.relu(target_margin - margin).square().mean()
