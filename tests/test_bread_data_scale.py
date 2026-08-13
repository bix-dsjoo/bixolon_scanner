from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from bixolon_scanner.training import bread_data_scale
from bixolon_scanner.training.bread_data_scale import (
    _evaluate_crossfit,
    _known_source_family,
    _load_config,
    _phash_bits,
    _records_and_counts,
    cross_fold_calibrations,
    diverse_order,
    validate_nested_orders,
)
from bixolon_scanner.training.small_data import (
    brightness_c2_frofa,
    build_frofa_training_set,
    fit_cosine_prototype_head,
    fit_linear_svm_head,
    fit_logistic_head,
    l2_normalize,
)


def _config(*, classes: int = 2) -> dict:
    return {
        "experiment": {
            "sample_sizes": [5, 10, 15, 20],
            "seed": 20260810,
            "expected_num_classes": classes,
            "expected_aux_images_per_class": 84,
            "fold_count": 3,
            "bootstrap_repetitions": 50,
            "max_false_approval_rate": 0.005,
            "confidence_level": 0.95,
        },
        "sampling": {},
        "training": {},
        "evaluation": {},
    }


def test_config_locks_sizes_seed_and_folds(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    assert _load_config(path)["experiment"]["sample_sizes"] == [5, 10, 15, 20]
    invalid = _config()
    invalid["experiment"]["sample_sizes"] = [5, 10, 20]
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="sample_sizes"):
        _load_config(path)


def test_hybrid_config_locks_knn_and_mixture_ranges(tmp_path: Path):
    config = _config()
    config["training"] = {
        "strategy": "frozen_prototype_knn_hybrid",
        "feature_l2_normalize": True,
        "hybrid_knn_k": 3,
        "hybrid_prototype_weight": 0.5,
    }
    path = tmp_path / "hybrid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert _load_config(path)["training"]["hybrid_knn_k"] == 3
    config["training"]["hybrid_knn_k"] = 6
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hybrid k"):
        _load_config(path)


def test_diverse_order_is_deterministic_and_uses_one_per_perceptual_group():
    records = [{"image_path": f"image-{index:02}.jpg"} for index in range(6)]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    groups = ["duplicate", "duplicate", "north", "west", "south", "center"]
    first = diverse_order(records, embeddings, groups, 5)
    second = diverse_order(records, embeddings, groups, 5)
    assert first == second
    assert len(first) == len(set(first)) == 5
    assert len({groups[index] for index in first}) == 5


def test_brightness_c2_frofa_is_seeded_and_channel_bounded():
    patches = np.asarray([[[0.0, 10.0], [1.0, 12.0], [2.0, 14.0]]], dtype=np.float32)
    first = brightness_c2_frofa(patches, magnitude=1.0, rng=np.random.default_rng(20260810))
    second = brightness_c2_frofa(patches, magnitude=1.0, rng=np.random.default_rng(20260810))
    assert np.array_equal(first, second)
    assert np.all(first >= patches.min(axis=1, keepdims=True))
    assert np.all(first <= patches.max(axis=1, keepdims=True))


def test_frofa_training_set_is_nested_deterministic_and_l2_normalized():
    patches = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    labels = np.asarray([0, 1])
    kwargs = {
        "layer_norm_weight": np.ones(3, dtype=np.float32),
        "layer_norm_bias": np.zeros(3, dtype=np.float32),
        "layer_norm_epsilon": 1e-6,
        "magnitude": 0.5,
        "views": 2,
        "seed": 20260810,
    }
    first_features, first_labels = build_frofa_training_set(patches, labels, **kwargs)
    second_features, second_labels = build_frofa_training_set(patches, labels, **kwargs)
    assert first_features.shape == (6, 3)
    assert np.array_equal(first_features, second_features)
    assert np.array_equal(first_labels, second_labels)
    assert np.allclose(np.linalg.norm(first_features, axis=1), 1.0)
    assert np.array_equal(first_features[:2], l2_normalize(first_features[:2]))


def test_regularized_logistic_head_requires_and_orders_every_class():
    features = np.asarray(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1])
    head = fit_logistic_head(
        features,
        labels,
        num_classes=2,
        regularization_c=10.0,
        max_iterations=100,
        seed=20260810,
    )
    assert head.classes.tolist() == [0, 1]
    assert head.weights.shape == (2, 2)
    with pytest.raises(ValueError, match="requires classes"):
        fit_logistic_head(
            features[:2],
            labels[:2],
            num_classes=2,
            regularization_c=10.0,
            max_iterations=100,
            seed=20260810,
        )


def test_linear_svm_head_is_exportable_as_multiclass_weights():
    features = np.asarray(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1])
    head = fit_linear_svm_head(
        features,
        labels,
        num_classes=2,
        regularization_c=1.0,
        max_iterations=1000,
        seed=20260810,
    )
    assert head.classes.tolist() == [0, 1]
    assert head.weights.shape == (2, 2)
    assert head.iterations < 1000


def test_cosine_prototype_head_is_normalized_and_uses_class_means():
    features = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1])
    head = fit_cosine_prototype_head(features, labels, num_classes=2)
    assert head.classes.tolist() == [0, 1]
    assert head.counts.tolist() == [2, 2]
    assert np.allclose(np.linalg.norm(head.weights, axis=1), 1.0)
    assert np.array_equal(head.bias, np.zeros(2, dtype=np.float32))
    normalized = l2_normalize(features)
    logits = normalized @ head.weights.T
    assert np.array_equal(logits.argmax(axis=1), labels)


def test_cosine_prototype_head_rejects_missing_class():
    with pytest.raises(ValueError, match="requires classes"):
        fit_cosine_prototype_head(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([0]),
            num_classes=2,
        )


def test_nested_order_validation_rejects_missing_or_duplicate_entries():
    orders = {
        str(category): [f"{category}-{index}" for index in range(20)] for category in range(1, 21)
    }
    validate_nested_orders(orders, category_count=20, sample_sizes=[5, 10, 15, 20])
    orders["3"][-1] = orders["3"][0]
    with pytest.raises(ValueError, match="unique"):
        validate_nested_orders(orders, category_count=20, sample_sizes=[5, 10, 15, 20])


def test_rotation_invariant_phash_groups_rotated_source(tmp_path: Path):
    image = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 7, 45, 22), fill="black")
    draw.ellipse((54, 25, 88, 58), fill="gray")
    original = tmp_path / "original.png"
    rotated = tmp_path / "rotated.png"
    image.save(original)
    image.rotate(90, expand=True).save(rotated)
    assert np.array_equal(_phash_bits(original, 8), _phash_bits(rotated, 8))


def test_bread19_rotation_family_is_prioritized_before_reusing_source():
    records = [
        {"category_id": 19, "image_path": f"train/19.{angle} ({source}).jpg"}
        for source in range(1, 7)
        for angle in (0, 90)
    ]
    groups = [_known_source_family(record, f"p{index}") for index, record in enumerate(records)]
    embeddings = np.eye(len(records), dtype=np.float32)
    order = diverse_order(records, embeddings, groups, 10)
    assert len({groups[index] for index in order[:5]}) == 5
    assert len(order) == 10


def _write_manifest_fixture(tmp_path: Path, *, overlap: bool = False):
    records = []
    for category in range(1, 21):
        for index in range(2):
            records.append(
                {
                    "record_type": "classification",
                    "category_id": category,
                    "image_path": f"train/{category}-{index}.jpg",
                    "image_sha256": f"aux-{category}-{index}",
                }
            )
    for fold in range(3):
        records.append(
            {
                "record_type": "detection",
                "split": "development",
                "fold": fold,
                "image_sha256": f"dev-{fold}",
                "annotations": [
                    {"category_id": category, "bbox_xywh": [0, 0, 10, 10]}
                    for category in range(1, 21)
                ],
            }
        )
    records.append(
        {
            "record_type": "detection",
            "split": "test",
            "fold": None,
            "image_sha256": "aux-1-0" if overlap else "test-0",
            "annotations": [],
        }
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "dataset_version": "bread-test",
                "labels": [
                    {
                        "category_id": category,
                        "class_id": f"bread_{category:02}",
                        "class_name": f"Bread {category}",
                    }
                    for category in range(1, 21)
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, metadata


def test_current_counts_and_sha_leakage_guard(tmp_path: Path):
    manifest, metadata = _write_manifest_fixture(tmp_path)
    aux, development, _, counts = _records_and_counts(manifest, metadata, 20, 2)
    assert len(aux) == 40
    assert len(development) == 3
    assert counts[0]["auxiliary_images"] == 2
    assert counts[0]["development_rois"] == 3
    assert counts[0]["current_final_train_total"] == 5

    manifest, metadata = _write_manifest_fixture(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="SHA overlap"):
        _records_and_counts(manifest, metadata, 20, 2)


def test_cross_fold_calibration_never_uses_held_out_fold(monkeypatch):
    logits = np.arange(36, dtype=np.float32).reshape(18, 2)
    targets = np.asarray([0, 1] * 9)
    folds = np.repeat(np.arange(3), 6)
    seen: list[set[int]] = []

    def fake_fit(selected_logits, selected_targets, config):
        del selected_targets, config
        selected_ids = {int(value // 2) for value in selected_logits[:, 0]}
        selected_folds = {index // 6 for index in selected_ids}
        seen.append(selected_folds)
        return {
            "temperature": 1.0,
            "approval_threshold": 0.9,
            "risk_control_satisfied": True,
        }

    monkeypatch.setattr(bread_data_scale, "_fit_calibration", fake_fit)
    result = cross_fold_calibrations(logits, targets, folds, _config())
    assert result[0]["calibration_folds"] == [1, 2]
    assert result[1]["calibration_folds"] == [0, 2]
    assert result[2]["calibration_folds"] == [0, 1]
    assert seen == [{1, 2}, {0, 2}, {0, 1}]


def test_worker_metrics_include_unknown_and_classifier_border_recapture():
    rows = [
        {
            "target": 0,
            "fold": 0,
            "image_id": 1,
            "group_id": "a",
            "touches_border": False,
        },
        {
            "target": 1,
            "fold": 1,
            "image_id": 2,
            "group_id": "b",
            "touches_border": False,
        },
        {
            "target": 0,
            "fold": 2,
            "image_id": 3,
            "group_id": "c",
            "touches_border": True,
        },
    ]
    logits = np.asarray([[4.0, 0.0], [0.2, 0.3], [0.1, 0.0]], dtype=np.float32)
    detector = {
        "image_count": 3,
        "ground_truth_count": 3,
        "prediction_count": 3,
        "matched_count": 3,
        "recall": 1.0,
        "precision": 1.0,
        "count_accuracy": 1.0,
        "recapture_image_count": 0,
        "recapture_reasons": {},
        "outcomes": [
            {
                "image_id": image_id,
                "ground_truth_count": 1,
                "detection_count": 1,
                "matched_count": 1,
                "recapture_reasons": [],
            }
            for image_id in (1, 2, 3)
        ],
    }
    calibrations = {
        fold: {
            "temperature": 1.0,
            "approval_threshold": 0.7,
            "risk_control_satisfied": True,
        }
        for fold in range(3)
    }
    report = _evaluate_crossfit(logits, rows, detector, calibrations, _config())
    assert report["frame_policy"]["status_counts"] == {
        "APPROVED": 1,
        "RECAPTURE": 1,
        "UNKNOWN": 1,
    }
    assert report["classifier_border_recapture_images"] == 1
    assert report["unknown_count"] == 1
    assert report["approved_count"] == 1
    assert report["approved_point_precision_gate_satisfied"] is True
    assert report["approved_precision_gate_satisfied"] is False
