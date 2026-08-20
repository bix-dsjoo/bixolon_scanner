from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ...configuration import load_json_config
from ...training.models import require_torch
from .data_scale import (
    RpcCachedDataset,
    _classifier_domain_split,
    _load_checkpoint_model,
    _write_json,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _features(model, records, cache_path: Path, image_size: int, batch_size: int, workers: int):
    torch = require_torch()
    from torch.utils.data import DataLoader

    dataset = RpcCachedDataset(
        records,
        cache_path,
        image_size=image_size,
        training=False,
        include_metadata=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    feature_rows = []
    logit_rows = []
    targets = []
    image_ids = []
    levels: list[str] = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, labels, batch_levels, _groups, _ids, batch_image_ids, _border in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = model.extract_features(images)
                logits = model.classifier(features)
            feature_rows.append(features.float().cpu().numpy())
            logit_rows.append(logits.float().cpu().numpy())
            targets.append(labels.numpy())
            image_ids.append(batch_image_ids.numpy())
            levels.extend(batch_levels)
    return {
        "features": np.concatenate(feature_rows),
        "logits": np.concatenate(logit_rows),
        "targets": np.concatenate(targets),
        "image_ids": np.concatenate(image_ids),
        "levels": np.asarray(levels),
    }


def _fit_rejector(features: np.ndarray, labels: np.ndarray, *, seed: int, epochs: int):
    torch = require_torch()
    torch.manual_seed(seed)
    device = torch.device("cuda")
    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 1),
    ).to(device)
    x = torch.from_numpy(features).float()
    y = torch.from_numpy(labels.astype(np.float32))
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("segment rejector needs both normal and recapture samples")
    sample_weights = torch.where(
        y > 0.5,
        torch.full_like(y, 0.5 / positive_count),
        torch.full_like(y, 0.5 / negative_count),
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights.double(), len(labels), replacement=True, generator=generator
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=512,
        sampler=sampler,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(batch_x).squeeze(1), batch_y
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(json.dumps({"rejector_epoch": epoch + 1, "loss": float(np.mean(losses))}), flush=True)
    return model.eval()


def _scores(model, features: np.ndarray) -> np.ndarray:
    torch = require_torch()
    device = next(model.parameters()).device
    rows = []
    with torch.no_grad():
        for start in range(0, len(features), 2048):
            batch = torch.from_numpy(features[start : start + 2048]).float().to(device)
            # Calibrate on unbounded float32 logits. Sigmoid can round extreme
            # scores to exactly 1.0 and make a zero-error threshold unusable.
            rows.append(model(batch).squeeze(1).cpu().numpy())
    return np.concatenate(rows)


def _gate_report(scores: np.ndarray, logits: np.ndarray, targets: np.ndarray, threshold: float):
    predicted = logits.argmax(axis=1)
    correct = (targets >= 0) & (predicted == targets)
    accepted = scores >= threshold
    wrong = accepted & ~correct
    return {
        "sample_count": len(targets),
        "matched_count": int((targets >= 0).sum()),
        "accepted_count": int(accepted.sum()),
        "correct_approved_count": int((accepted & correct).sum()),
        "wrong_approved_count": int(wrong.sum()),
        "approved_precision": float((accepted & correct).sum() / accepted.sum())
        if accepted.any()
        else 1.0,
        "matched_end_to_end_rate": float((accepted & correct).sum() / (targets >= 0).sum()),
        "recognition_rate": float((accepted & correct).sum() / accepted.sum())
        if accepted.any()
        else None,
        "segment_recapture_rate": float((~accepted).mean()),
        "accepted_unmatched_count": int((accepted & (targets < 0)).sum()),
        "accepted_misclassification_count": int((accepted & (targets >= 0) & ~correct).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    torch = require_torch()
    config = load_json_config(args.config)
    training = config["training"]
    run_dir = args.output_dir / "runs" / "full" / f"seed{args.seed}"
    records = _read_jsonl(args.output_dir / "prepared" / "cache" / "records.jsonl")
    calibration = [row for row in records if row["role"] == "calibration"]
    adaptation, risk = _classifier_domain_split(
        calibration,
        fraction=float(training["checkout_adaptation_group_fraction"]),
        seed=int(training["checkout_adaptation_seed"]),
    )
    selection = [row for row in records if row["role"] == "selection"]
    model, _checkpoint = _load_checkpoint_model(run_dir / "best.pt", config, torch.device("cuda"))
    cache_path = args.output_dir / "prepared" / "cache" / "images.npy"
    extracted = {}
    for name, split in (("adaptation", adaptation), ("risk", risk), ("selection", selection)):
        extracted[name] = _features(
            model,
            split,
            cache_path,
            int(training["image_size"]),
            int(training["batch_size"]),
            int(training["workers"]),
        )
    adapt = extracted["adaptation"]
    adapt_labels = (adapt["targets"] >= 0) & (adapt["logits"].argmax(axis=1) == adapt["targets"])
    rejector = _fit_rejector(adapt["features"], adapt_labels, seed=args.seed, epochs=args.epochs)
    risk = extracted["risk"]
    risk_scores = _scores(rejector, risk["features"])
    risk_correct = (risk["targets"] >= 0) & (risk["logits"].argmax(axis=1) == risk["targets"])
    bad_scores = risk_scores[~risk_correct]
    threshold = float(np.nextafter(bad_scores.max(), math.inf))
    selection = extracted["selection"]
    selection_scores = _scores(rejector, selection["features"])
    report = {
        "policy": "checkout-group-trained-segment-rejector-risk-calibrated-zero-error",
        "threshold": threshold,
        "adaptation_normal_count": int(adapt_labels.sum()),
        "adaptation_recapture_count": int((~adapt_labels).sum()),
        "risk": _gate_report(risk_scores, risk["logits"], risk["targets"], threshold),
        "selection": _gate_report(
            selection_scores, selection["logits"], selection["targets"], threshold
        ),
        "selection_by_level": {
            level: _gate_report(
                selection_scores[selection["levels"] == level],
                selection["logits"][selection["levels"] == level],
                selection["targets"][selection["levels"] == level],
                threshold,
            )
            for level in ("easy", "medium", "hard")
        },
    }
    artifact_dir = run_dir / "segment-rejector"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": rejector.state_dict(), "input_size": adapt["features"].shape[1]},
        artifact_dir / "rejector.pt",
    )
    np.savez_compressed(
        artifact_dir / "selection_scores.npz",
        scores=selection_scores,
        targets=selection["targets"],
        image_ids=selection["image_ids"],
        levels=selection["levels"],
    )
    _write_json(artifact_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
