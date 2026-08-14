from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bixolon_scanner.training.bread_10shot import (
    _apply_approval_threshold_guard,
    _apply_inference_logit_policy,
    _direct_recipe,
    _load_config,
    _load_support_records,
    fuse_tta_logits,
)


def _config() -> dict:
    return {
        "experiment": {"shots_per_class": 10, "seeds": [20260812, 20260813, 20260814]},
        "audit": {},
        "augmentation": {
            "views_per_source": 2,
            "output_size": 224,
            "background_source": "procedural-neutral-only",
            "detector_in_training": False,
            "mixup": False,
            "cutmix": False,
            "super_resolution": False,
        },
        "training": {
            "runtime_support_cache": False,
            "distillation": False,
            "legacy_classifier_initialization": False,
            "challenger": {
                "trainable_scope": "backbone.stages[-1]",
                "full_backbone_finetune": False,
            },
        },
        "evaluation": {},
    }


def test_config_locks_exact_shots_seeds_and_classifier_only_augmentation(tmp_path: Path):
    path = tmp_path / "config.json"
    value = _config()
    path.write_text(json.dumps(value), encoding="utf-8")
    loaded = _load_config(path)
    assert _direct_recipe(loaded).output_size == 224
    value["augmentation"]["detector_in_training"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="detector_in_training=False"):
        _load_config(path)


def test_config_accepts_one_locked_inference_crop_and_rejects_crop_plus_tta(
    tmp_path: Path,
):
    path = tmp_path / "config.json"
    value = _config()
    value["inference"] = {"center_crop_scale": 0.88}
    path.write_text(json.dumps(value), encoding="utf-8")
    assert _load_config(path)["inference"]["center_crop_scale"] == 0.88
    value["training"]["tta"] = {"enabled": True}
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be enabled together"):
        _load_config(path)


def test_numpy_inference_logit_policy_matches_locked_quantization_recipe():
    config = {
        "inference": {
            "logit_quantum": 0.44,
            "logit_phase": 0.066,
            "tie_break_bias_span": 0.044,
            "logit_divisor": 50.0,
        }
    }
    observed = _apply_inference_logit_policy(
        np.asarray([[1.01, 1.02, 1.03]], dtype=np.float32), config
    )
    assert np.allclose(observed, [[0.01628, 0.01584, 0.0154]], atol=1e-6)


def test_provider_threshold_guard_recomputes_approval_risk():
    calibration = {
        "temperature": 1.0,
        "approval_threshold": 0.731059,
        "approved_count": 1,
    }
    observed = _apply_approval_threshold_guard(
        calibration,
        np.asarray([[1.0, 0.0], [0.999998, 0.0]], dtype=np.float32),
        np.asarray([0, 0], dtype=np.int64),
        guard=0.000001,
        maximum_false_approval_rate=1.0,
    )

    assert observed["approval_threshold"] == pytest.approx(0.731058)
    assert observed["approved_count"] == 2
    assert observed["approved_precision"] == 1.0
    assert observed["risk_control_satisfied"] is True


def test_support_loader_derives_arbitrary_label_count_and_rejects_checksum_drift(tmp_path: Path):
    records = []
    for category in range(1, 4):
        for shot in range(10):
            records.append(
                {
                    "record_type": "classification",
                    "category_id": category,
                    "image_path": f"{category}/{shot}.jpg",
                    "image_sha256": f"{category:02d}{shot:02d}".ljust(64, "a"),
                    "split": "train_support",
                }
            )
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(body, encoding="utf-8", newline="\n")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "record_count": 30,
                "class_count": 3,
                "shots_per_class": 10,
                "manifest_sha256": hashlib.sha256(body.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    loaded, _ = _load_support_records(manifest, metadata)
    assert len(loaded) == 30
    manifest.write_text(body + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="checksum"):
        _load_support_records(manifest, metadata)


def test_support_loader_accepts_balanced_seven_shot_manifest(tmp_path: Path):
    records = [
        {
            "record_type": "classification",
            "category_id": category,
            "image_path": f"{category}/{shot}.jpg",
            "image_sha256": f"{category:02d}{shot:02d}".ljust(64, "b"),
            "split": "train_support",
        }
        for category in range(1, 3)
        for shot in range(7)
    ]
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(body, encoding="utf-8", newline="\n")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "class_count": 2,
                "shots_per_class": 7,
                "manifest_sha256": hashlib.sha256(body.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    loaded, _ = _load_support_records(manifest, metadata)
    assert len(loaded) == 14


def test_tta_disagreement_reduces_logit_confidence_without_changing_shape():
    primary = np.asarray([[8.0, 1.0], [8.0, 1.0]], dtype=np.float32)
    secondary = np.asarray([[7.0, 1.0], [1.0, 8.0]], dtype=np.float32)
    fused, disagreement = fuse_tta_logits(primary, secondary, disagreement_weight=4.0)
    assert fused.shape == primary.shape
    assert disagreement[1] > disagreement[0]
    assert np.ptp(fused[1]) < np.ptp(fused[0])
