from __future__ import annotations

import hashlib
import json
import random
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bixolon_scanner.package import sha256_file
from bixolon_scanner.training.models import require_torch, set_frozen_backbone
from bixolon_scanner.training.rpc_data_scale import (
    _balanced_training_order,
    _classifier_domain_split,
    _classifier_stage_rank,
    _classifier_training_loss,
    _crop,
    _difficulty_worker_metrics,
    _ground_truth_worker_outcomes,
    _load_stage_progress,
    _operational_gate,
    _run_complete,
    _save_stage_progress,
    _test_operational_gate,
    _validation_partition,
    _visual_farthest_order,
    _write_experiment_metadata,
    evaluate_logits,
    evaluate_worker_taxonomy,
    prepare,
    summarize,
    train_all,
)
from bixolon_scanner.training.rpc_data_scale import (
    main as rpc_main,
)
from bixolon_scanner.training.rpc_data_scale import (
    test_selected as run_final_test,
)
from bixolon_scanner.training.rpc_worker_gate import (
    _adaptation_progressive_fold_gate,
    _baseline_detector_complete,
    _best_detector_epoch,
    _canonical_json,
    _checkpoint_complete,
    _detector_namespace,
    _detector_phase_complete,
    _domain_adaptation_namespace,
    _domain_adaptation_source_replay,
    _domain_adaptation_train_subset,
    _hard_negative_rows,
    _prediction_artifact_valid,
    _prediction_identity,
    _select_hard_negative_rows,
    _target_oof_gate_report,
    _train_gate_complete,
    _train_product_row,
    _train_records,
    _update_unique_reason_counts,
    _write_prediction_artifact,
    assign_oof_folds,
    evaluate_frozen_detector_threshold_selection,
    postprocess_worker_gate,
    prepare_detector_domain_adaptation,
    prepare_detector_phase,
    prepare_final_test_records,
    train_final_detector,
)
from bixolon_scanner.training.train_detector import detector_optimizer_recipe


def _write_coco(
    root: Path, split: str, categories: list[dict], images: list[dict], annotations: list[dict]
):
    (root / f"{split}2019").mkdir(parents=True, exist_ok=True)
    (root / f"instances_{split}2019.json").write_text(
        json.dumps({"categories": categories, "images": images, "annotations": annotations}),
        encoding="utf-8",
    )
    for image in images:
        canvas = Image.new("RGB", (image["width"], image["height"]), "white")
        canvas.save(root / f"{split}2019" / image["file_name"])


def _write_detector_phase_completion(output_dir: Path) -> None:
    detector_dir = output_dir / "detector"
    predictions_dir = detector_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "threshold_sha256": detector_dir / "threshold.json",
        "val_predictions_sha256": predictions_dir / "val_oof.jsonl",
        "val_predictions_metadata_sha256": predictions_dir / "val_oof.jsonl.metadata.json",
        "train_predictions_sha256": predictions_dir / "train_assigned.jsonl",
        "train_predictions_metadata_sha256": predictions_dir / "train_assigned.jsonl.metadata.json",
    }
    for path in artifacts.values():
        path.write_text("{}\n", encoding="utf-8")
    (detector_dir / "complete.json").write_text(
        json.dumps({key: sha256_file(path) for key, path in artifacts.items()}),
        encoding="utf-8",
    )


def test_classifier_stage_rank_uses_accuracy_when_both_stages_have_zero_coverage():
    frozen = {
        "risk_control_satisfied": False,
        "approval_coverage": 0.0,
        "top3_accuracy": 0.76,
        "accuracy": 0.60,
    }
    partial = {
        "risk_control_satisfied": False,
        "approval_coverage": 0.0,
        "top3_accuracy": 0.88,
        "accuracy": 0.78,
    }

    assert _classifier_stage_rank(partial) > _classifier_stage_rank(frozen)


def test_classifier_stage_rank_prefers_risk_controlled_stage():
    uncontrolled = {
        "risk_control_satisfied": False,
        "approval_coverage": 0.8,
        "top3_accuracy": 0.99,
        "accuracy": 0.99,
    }
    controlled = {
        "risk_control_satisfied": True,
        "approval_coverage": 0.01,
        "top3_accuracy": 0.80,
        "accuracy": 0.70,
    }

    assert _classifier_stage_rank(controlled) > _classifier_stage_rank(uncontrolled)


def test_classifier_training_loss_uses_uniform_outlier_exposure():
    torch = require_torch()
    logits = torch.tensor(
        [[4.0, 0.0, -1.0], [8.0, -2.0, -3.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, -1])

    loss, positive, negative = _classifier_training_loss(
        logits, labels, hard_negative_loss_weight=0.5
    )
    loss.backward()

    assert positive is not None and negative is not None
    assert float(negative.detach()) > 3.0
    assert logits.grad is not None
    assert float(logits.grad[1, 0]) > 0.0
    assert float(logits.grad[1, 1]) < 0.0


def test_classifier_domain_split_is_group_safe_reproducible_and_keeps_negatives():
    records = [
        {
            "sample_id": f"{group}-{target}",
            "group_id": group,
            "target": target,
            "role": "calibration",
        }
        for group in ("a", "b", "c", "d", "e", "f")
        for target in (0, 1, -1)
    ]

    first_train, first_risk = _classifier_domain_split(records, fraction=2 / 3, seed=7)
    second_train, second_risk = _classifier_domain_split(records, fraction=2 / 3, seed=7)

    assert first_train == second_train
    assert first_risk == second_risk
    assert {row["group_id"] for row in first_train}.isdisjoint(
        {row["group_id"] for row in first_risk}
    )
    assert {row["target"] for row in first_train} == {-1, 0, 1}
    assert {row["target"] for row in first_risk} == {-1, 0, 1}
    assert all(
        row["role"] == ("checkout_hard_negative" if row["target"] < 0 else "checkout_adaptation")
        for row in first_train
    )


def test_train_hard_negatives_require_low_gt_overlap_and_are_view_diverse():
    training = {
        "hard_negative_enabled": True,
        "hard_negative_max_gt_iou": 0.1,
        "hard_negative_min_score": 0.1,
        "hard_negative_min_area_ratio": 0.005,
        "hard_negative_max_per_image": 1,
        "hard_negative_views_per_surface_camera": 2,
        "hard_negative_max_ratio": 1.0,
        "hard_negative_seed": 20260810,
    }
    record = {
        "image_id": 7,
        "image_path": "train2019/product_camera1-1.jpg",
        "width": 100,
        "height": 100,
        "physical_group": "product",
        "prediction_fold": 0,
        "annotations": [{"bbox_xywh": [40.0, 40.0, 20.0, 20.0]}],
    }
    result = {
        "detections": [
            {"bbox_xyxy": [0.0, 0.0, 20.0, 20.0], "score": 0.9},
            {"bbox_xyxy": [35.0, 35.0, 65.0, 65.0], "score": 0.99},
        ],
        "unmatched_detection_indices": [0, 1],
    }

    rows = _hard_negative_rows(record, result, training)

    assert [row["sample_id"] for row in rows] == ["train-hard-negative:7:det0"]
    assert rows[0]["target"] == -1
    candidates = []
    for view_id, score in ((1, 0.9), (2, 0.95), (10, 0.8)):
        row = dict(rows[0])
        row["sample_id"] = f"negative-{view_id}"
        row["view_id"] = view_id
        row["detector_score"] = score
        candidates.append(row)
    selected = _select_hard_negative_rows(candidates, positive_count=10, training=training)

    assert {row["view_id"] for row in selected} == {2, 10}


def test_recapture_positive_uses_matched_prediction_without_restoring_image_status():
    record = {
        "image_id": 7,
        "image_path": "train2019/product_camera1-8.jpg",
        "width": 100,
        "height": 100,
        "prediction_fold": 2,
        "annotations": [{"annotation_id": 11, "category_id": 3}],
    }
    result = {
        "detections": [
            {"bbox_xyxy": [10.0, 20.0, 60.0, 80.0], "score": 0.9},
            {"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "score": 0.8},
        ],
        "matches": {"0": [0, 0.75]},
        "missed_annotation_indices": [],
    }

    row = _train_product_row(
        record,
        result,
        reasons=["DATA_ALIGNMENT_REJECT"],
        training={
            "recapture_positive_enabled": True,
            "recapture_positive_min_iou": 0.5,
        },
    )

    assert row is not None
    assert row["bbox_xyxy"] == result["detections"][0]["bbox_xyxy"]
    assert row["bbox_xywh"] == [10.0, 20.0, 50.0, 60.0]
    assert row["target"] == 2
    assert row["worker_gate_role"] == "recapture_positive"
    assert row["worker_recapture_reasons"] == ["DATA_ALIGNMENT_REJECT"]


def test_convnext_partial_finetune_unfreezes_last_two_stages():
    torch = require_torch()

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(2, 2)) for _ in range(4)]
            )
            self.norm = torch.nn.LayerNorm(2)

    class FakeClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = FakeBackbone()
            self.backbone_kind = "dinov3_convnext_tiny"
            self.classifier = torch.nn.Linear(2, 2)

    model = FakeClassifier()
    set_frozen_backbone(model, unfreeze_last_stages=2)

    assert all(
        not parameter.requires_grad
        for stage in model.backbone.stages[:2]
        for parameter in stage.parameters()
    )
    assert all(
        parameter.requires_grad
        for stage in model.backbone.stages[2:]
        for parameter in stage.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.backbone.norm.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_convnext_full_finetune_unfreezes_stem_and_every_stage():
    torch = require_torch()

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Linear(2, 2)
            self.stages = torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(2, 2)) for _ in range(4)]
            )
            self.norm = torch.nn.LayerNorm(2)

    class FakeClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = FakeBackbone()
            self.backbone_kind = "dinov3_convnext_tiny"
            self.classifier = torch.nn.Linear(2, 2)

    model = FakeClassifier()
    set_frozen_backbone(model, unfreeze_all=True)

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_backbone_unfreeze_policies_are_mutually_exclusive():
    torch = require_torch()

    class FakeClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.backbone_kind = "dinov3_convnext_tiny"
            self.classifier = torch.nn.Linear(2, 2)

    with pytest.raises(ValueError, match="choose one backbone unfreezing policy"):
        set_frozen_backbone(FakeClassifier(), unfreeze_all=True, unfreeze_last_blocks=2)


def test_oof_detector_uses_identical_initialization_seed_for_all_folds(tmp_path):
    options = {
        "pretrained_name": "detector",
        "image_size": 640,
        "batch_size": 8,
        "workers": 0,
        "epochs": 100,
        "patience": 20,
        "learning_rate": 1e-5,
        "head_lr_multiplier": 1.0,
        "class_head_prior_probability": 0.5,
        "warmup_epochs": 0,
        "weight_decay": 1e-4,
        "min_score_threshold": 0.05,
        "max_score_threshold": 0.95,
        "threshold_steps": 91,
        "nms_iou_threshold": 0.7,
        "match_iou_threshold": 0.5,
        "target_recall": 0.99,
        "max_queries": 300,
        "seed": 20260810,
    }

    namespaces = [
        _detector_namespace(
            options,
            tmp_path / "manifest.jsonl",
            tmp_path,
            tmp_path / f"fold{fold}",
            fold,
        )
        for fold in range(3)
    ]
    seeds = {namespace.seed for namespace in namespaces}

    assert seeds == {20260810}
    assert {namespace.head_lr_multiplier for namespace in namespaces} == {1.0}
    assert {namespace.class_head_prior_probability for namespace in namespaces} == {0.5}
    assert {namespace.warmup_epochs for namespace in namespaces} == {0}
    assert {namespace.weight_decay for namespace in namespaces} == {1e-4}
    assert {namespace.min_score_threshold for namespace in namespaces} == {0.05}
    assert {namespace.max_score_threshold for namespace in namespaces} == {0.95}
    assert {namespace.threshold_steps for namespace in namespaces} == {91}
    assert {namespace.nms_iou_threshold for namespace in namespaces} == {0.7}
    assert {namespace.match_iou_threshold for namespace in namespaces} == {0.5}
    assert {namespace.target_recall for namespace in namespaces} == {0.99}
    assert {namespace.max_queries for namespace in namespaces} == {300}


def test_detector_completion_rejects_a_different_initialization_seed(tmp_path):
    checkpoint = tmp_path / "fold"
    best = checkpoint / "best"
    best.mkdir(parents=True)
    (best / "config.json").write_text("{}", encoding="utf-8")
    weights = best / "model.safetensors"
    weights.write_bytes(b"weights")
    history = [{"epoch": 1, "validation_loss": 1.0}]
    (checkpoint / "history.json").write_text(json.dumps(history), encoding="utf-8")
    recipe = {"optimizer": "AdamW", "head_lr_multiplier": 10.0}
    (checkpoint / "run.json").write_text(
        json.dumps({"arguments": {"seed": 20260810}, "optimizer_recipe": recipe}),
        encoding="utf-8",
    )
    (checkpoint / "complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "history_epochs": 1,
                "weights_sha256": sha256_file(weights),
                "optimizer_recipe": recipe,
            }
        ),
        encoding="utf-8",
    )

    assert _checkpoint_complete(checkpoint, expected_seed=20260810)
    assert not _checkpoint_complete(checkpoint, expected_seed=20260811)
    assert _checkpoint_complete(checkpoint, expected_optimizer_recipe=recipe)
    assert not _checkpoint_complete(
        checkpoint,
        expected_optimizer_recipe={"optimizer": "AdamW", "head_lr_multiplier": 20.0},
    )


def test_detector_prediction_resume_requires_inputs_weights_config_and_checksum(tmp_path):
    checkpoint = tmp_path / "fold0" / "best"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights-v1")
    records = [{"source": "rpc_val2019", "image_id": 1, "image_path": "val2019/a.jpg"}]
    predictions_path = tmp_path / "predictions" / "val_oof.jsonl"
    identity = _prediction_identity(
        records,
        [checkpoint],
        source_sha256="annotation-v1",
        inference_config={"batch_size": 8, "minimum_score": 0.05},
    )
    _write_prediction_artifact(
        predictions_path,
        [{"sample_key": "rpc_val2019:1", "boxes_xyxy": [], "scores": []}],
        identity,
    )
    assert _prediction_artifact_valid(predictions_path, identity)

    changed_manifest = _prediction_identity(
        [dict(records[0], image_path="val2019/changed.jpg")],
        [checkpoint],
        source_sha256="annotation-v1",
        inference_config={"batch_size": 8, "minimum_score": 0.05},
    )
    assert not _prediction_artifact_valid(predictions_path, changed_manifest)
    changed_annotation = _prediction_identity(
        records,
        [checkpoint],
        source_sha256="annotation-v2",
        inference_config={"batch_size": 8, "minimum_score": 0.05},
    )
    assert not _prediction_artifact_valid(predictions_path, changed_annotation)
    changed_config = _prediction_identity(
        records,
        [checkpoint],
        source_sha256="annotation-v1",
        inference_config={"batch_size": 8, "minimum_score": 0.10},
    )
    assert not _prediction_artifact_valid(predictions_path, changed_config)

    (checkpoint / "model.safetensors").write_bytes(b"weights-v2")
    changed_weights = _prediction_identity(
        records,
        [checkpoint],
        source_sha256="annotation-v1",
        inference_config={"batch_size": 8, "minimum_score": 0.05},
    )
    assert not _prediction_artifact_valid(predictions_path, changed_weights)

    (checkpoint / "model.safetensors").write_bytes(b"weights-v1")
    predictions_path.write_text("{}\n", encoding="utf-8")
    assert not _prediction_artifact_valid(predictions_path, identity)


def test_detector_complete_marker_validates_every_internal_artifact_hash(tmp_path):
    detector_dir = tmp_path / "detector"
    predictions_dir = detector_dir / "predictions"
    predictions_dir.mkdir(parents=True)
    artifacts = {
        "threshold_sha256": detector_dir / "threshold.json",
        "val_predictions_sha256": predictions_dir / "val_oof.jsonl",
        "val_predictions_metadata_sha256": predictions_dir / "val_oof.jsonl.metadata.json",
        "train_predictions_sha256": predictions_dir / "train_assigned.jsonl",
        "train_predictions_metadata_sha256": predictions_dir / "train_assigned.jsonl.metadata.json",
    }
    for path in artifacts.values():
        path.write_text("{}\n", encoding="utf-8")
    marker = {key: sha256_file(path) for key, path in artifacts.items()}
    (detector_dir / "complete.json").write_text(json.dumps(marker), encoding="utf-8")
    assert _detector_phase_complete(detector_dir)

    (predictions_dir / "train_assigned.jsonl").write_text("tampered\n", encoding="utf-8")
    assert not _detector_phase_complete(detector_dir)


def test_prepare_rejects_a_tampered_detector_completion_before_loading_data(tmp_path, monkeypatch):
    _write_detector_phase_completion(tmp_path)
    (tmp_path / "detector" / "threshold.json").write_text("tampered\n", encoding="utf-8")
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._load_coco",
        lambda *_args, **_kwargs: pytest.fail("dataset must not be loaded"),
    )
    config = {
        "experiment": {"expected_num_classes": 1},
        "training": {},
    }
    with pytest.raises(ValueError, match="artifact checksum"):
        prepare(
            Namespace(dataset_root=tmp_path, output_dir=tmp_path, resume=True),
            config,
        )


def test_final_detector_epoch_policy_uses_metric_quality_not_lowest_loss(tmp_path):
    detector_dir = tmp_path / "detector"
    histories = [
        [
            {"epoch": 1, "validation_loss": 1.0, "detector_quality_key": [0.0, 0.98, 0.5, 0.5]},
            {"epoch": 2, "validation_loss": 1.2, "detector_quality_key": [1.0, 0.4, 0.6, 0.2]},
        ],
        [
            {"epoch": 3, "validation_loss": 0.9, "detector_quality_key": [1.0, 0.5, 0.7, 0.3]},
        ],
        [
            {"epoch": 4, "validation_loss": 0.8, "detector_quality_key": [1.0, 0.6, 0.8, 0.4]},
        ],
    ]
    for fold, history in enumerate(histories):
        path = detector_dir / "folds" / f"fold{fold}"
        path.mkdir(parents=True)
        (path / "history.json").write_text(json.dumps(history), encoding="utf-8")

    assert _best_detector_epoch(detector_dir, 3) == 3


def test_final_detector_is_baseline_val_all_and_ignores_adaptation_epochs(tmp_path, monkeypatch):
    detector_dir = tmp_path / "output" / "detector"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "instances_val2019.json").write_text("{}", encoding="utf-8")
    (dataset_root / "instances_train2019.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config = {
        "experiment": {"mode": "full_dataset", "seeds": [7]},
        "detector": {
            "pretrained_name": "immutable-pretrained",
            "fold_count": 3,
            "image_size": 640,
            "batch_size": 8,
            "epochs": 100,
            "patience": 20,
            "learning_rate": 1e-5,
            "head_lr_multiplier": 1.0,
            "class_head_prior_probability": 0.5,
            "warmup_epochs": 0,
            "weight_decay": 1e-4,
            "seed": 7,
            "min_score_threshold": 0.05,
            "max_score_threshold": 0.95,
            "threshold_steps": 91,
            "nms_iou_threshold": 0.7,
            "match_iou_threshold": 0.5,
            "target_recall": 0.99,
            "max_queries": 300,
            "inference_batch_size": 8,
            "domain_adaptation": {
                "enabled": True,
                "samples_per_surface_camera": 2,
                "epochs": 1,
                "patience": 1,
                "learning_rate": 2.5e-7,
                "skip_epoch_validation": True,
                "seed": 7,
            },
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    baseline_manifest = detector_dir / "manifest" / "manifest.jsonl"
    baseline_manifest.parent.mkdir(parents=True)
    baseline_manifest.write_text('{"source":"val"}\n', encoding="utf-8")
    (baseline_manifest.parent / "metadata.json").write_text("{}", encoding="utf-8")
    (detector_dir / "baseline").mkdir()
    (detector_dir / "baseline" / "complete.json").write_text(
        '{"checkpoint_set":"baseline"}', encoding="utf-8"
    )
    (detector_dir / "train-gate").mkdir()
    (detector_dir / "train-gate" / "complete.json").write_text(
        '{"role":"train_gate_only"}', encoding="utf-8"
    )
    (detector_dir / "complete.json").write_text(
        '{"checkpoint_set":"domain_adaptation","role":"train_gate_only"}',
        encoding="utf-8",
    )
    (detector_dir / "threshold.json").write_text(
        '{"selected_score_threshold":0.42}', encoding="utf-8"
    )
    for fold, epoch in enumerate((51, 57, 69)):
        fold_dir = detector_dir / "folds" / f"fold{fold}"
        fold_dir.mkdir(parents=True)
        (fold_dir / "history.json").write_text(
            json.dumps(
                [
                    {
                        "epoch": epoch,
                        "validation_loss": 1.0,
                        "detector_quality_key": [1.0, float(epoch)],
                    }
                ]
            ),
            encoding="utf-8",
        )
        adapted_dir = detector_dir / "domain-adaptation" / "folds" / f"fold{fold}"
        adapted_dir.mkdir(parents=True)
        (adapted_dir / "history.json").write_text(
            json.dumps(
                [
                    {
                        "epoch": 3,
                        "validation_loss": 0.1,
                        "detector_quality_key": [9.0],
                    }
                ]
            ),
            encoding="utf-8",
        )

    calls = []

    def fake_train(namespace):
        calls.append(namespace)
        best = namespace.output_dir / "best"
        best.mkdir(parents=True, exist_ok=True)
        (best / "config.json").write_text("{}", encoding="utf-8")
        (best / "model.safetensors").write_bytes(f"stage-{len(calls)}".encode())
        (namespace.output_dir / "run.json").write_text(
            json.dumps(
                {
                    "arguments": {"seed": namespace.seed},
                    "optimizer_recipe": detector_optimizer_recipe(namespace),
                }
            ),
            encoding="utf-8",
        )
        (namespace.output_dir / "history.json").write_text(
            json.dumps(
                [{"epoch": epoch, "train_loss": 1.0} for epoch in range(1, namespace.epochs + 1)]
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_worker_gate._detector_phase_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_worker_gate._baseline_detector_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_worker_gate._train_gate_complete",
        lambda *_args, **_kwargs: (True, {"role": "train_gate_only"}),
    )
    monkeypatch.setattr("bixolon_scanner.training.rpc_worker_gate.train_detector", fake_train)
    args = Namespace(
        output_dir=tmp_path / "output",
        dataset_root=dataset_root,
        config=config_path,
    )
    checkpoint = train_final_detector(args, config, resume=False)

    assert checkpoint == detector_dir / "final" / "stage-a-base" / "best"
    assert [call.epochs for call in calls] == [57]
    complete = json.loads((detector_dir / "final" / "complete.json").read_text(encoding="utf-8"))
    assert complete["contract"] == "rpc-final-detector-baseline-val-all-v1"
    assert complete["target_adaptation_stage"] == "disabled_train_gate_only"
    assert complete["stage_a_checkpoint_sha256"] == sha256_file(checkpoint / "model.safetensors")
    assert complete["active_detector_complete_sha256"] == sha256_file(
        detector_dir / "baseline" / "complete.json"
    )


def test_training_order_is_nested_balanced_and_reproducible():
    records = []
    for category in (1, 2):
        for barcode in ("a", "b"):
            for camera in range(4):
                for view in range(3):
                    records.append(
                        {
                            "sample_id": f"{category}:{barcode}:{camera}:{view}",
                            "category_id": category,
                            "barcode": barcode,
                            "camera": camera,
                        }
                    )
    first = _balanced_training_order(records, 10)
    again = _balanced_training_order(records, 10)
    other = _balanced_training_order(records, 11)
    assert first == again
    assert first != other
    for category in ("1", "2"):
        assert len(first[category]) == len(set(first[category])) == 24
        cameras = {int(value.split(":")[2]) for value in first[category][:4]}
        assert cameras == {0, 1, 2, 3}
        assert set(first[category][:5]) < set(first[category][:10])


def test_validation_partition_keeps_checkout_groups_together():
    records = []
    for group in range(10):
        for category in range(3):
            records.append(
                {
                    "group_id": str(group),
                    "target": category,
                    "image_id": group * 10 + category,
                    "level": ("easy", "medium", "hard")[category],
                }
            )
    partition = _validation_partition(records, 3, 123, 0.5)
    assert len(partition) == 10
    assert list(partition.values()).count("calibration") == 5
    assert list(partition.values()).count("selection") == 5
    assert partition == _validation_partition(records, 3, 123, 0.5)


def test_oof_fold_assignment_is_group_safe_balanced_and_reproducible():
    records = []
    for group in range(12):
        for category in (1, 2, 3):
            records.append(
                {
                    "capture_session_id": str(group),
                    "level": ("easy", "medium", "hard")[group % 3],
                    "annotations": [{"category_id": category}],
                }
            )
    first = assign_oof_folds(records, 3)
    assert first == assign_oof_folds(records, 3)
    assert set(first.values()) == {0, 1, 2}
    counts = np.bincount(list(first.values()), minlength=3)
    assert counts.max() - counts.min() <= 1


def test_visual_farthest_order_prefers_distinct_appearance_and_is_nested():
    records = [
        {
            "sample_id": f"s{index}",
            "category_id": 1,
            "camera": index % 4,
            "surface": "front" if index < 3 else "back",
            "barcode": "a" if index < 3 else "b",
            "view_id": index,
        }
        for index in range(6)
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.999, 0.001, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    order = _visual_farthest_order(
        records, embeddings, seed=7, anchor_pool_size=1, tie_tolerance=1e-9
    )
    assert order == _visual_farthest_order(
        records, embeddings, seed=7, anchor_pool_size=1, tie_tolerance=1e-9
    )
    assert not ({"s0", "s1"} <= set(order[:3]))
    assert set(order[:2]) < set(order[:5])


def test_worker_gate_separates_recapture_and_exact_alignment():
    record = {
        "width": 100,
        "height": 100,
        "annotations": [{"bbox_xywh": [20, 20, 40, 40], "category_id": 1, "annotation_id": 1}],
    }
    options = {
        "score_threshold": 0.5,
        "max_queries": 300,
        "nms_iou_threshold": 0.7,
        "match_iou_threshold": 0.5,
        "uncertainty_score_threshold": 0.2,
        "uncertainty_min_area_ratio": 0.01,
        "uncertainty_match_iou_threshold": 0.5,
        "min_object_area_ratio": 0.005,
    }
    normal = postprocess_worker_gate(
        record,
        {"boxes_xyxy": [[20, 20, 60, 60]], "scores": [0.9]},
        options,
    )
    assert not normal["recapture_reasons"]
    assert len(normal["matches"]) == 1
    recapture = postprocess_worker_gate(
        record,
        {"boxes_xyxy": [[20, 20, 60, 60], [70, 70, 95, 95]], "scores": [0.9, 0.3]},
        options,
    )
    assert recapture["recapture_reasons"] == ["DETECTOR_UNCERTAIN_OBJECT"]


def test_adapted_detector_selection_uses_calibration_threshold_without_tuning(
    monkeypatch,
):
    records = [
        {"source": "val", "image_id": 1, "role": "calibration"},
        {"source": "val", "image_id": 2, "role": "selection"},
    ]
    predictions = [
        {"sample_key": "val:1"},
        {"sample_key": "val:2"},
    ]
    calls = []

    def fake_metrics(selected_records, selected_predictions, **kwargs):
        calls.append((selected_records, selected_predictions, kwargs))
        return {"recall": 0.995, "precision": 0.9, "count_accuracy": 0.8}

    monkeypatch.setattr("bixolon_scanner.training.rpc_worker_gate._metrics", fake_metrics)
    report = evaluate_frozen_detector_threshold_selection(
        records,
        predictions,
        {
            "selected_score_threshold": 0.42,
            "target_recall": 0.99,
            "target_recall_satisfied": True,
            "calibration_metrics": {"recall": 0.999},
        },
        {
            "target_recall": 0.99,
            "nms_iou_threshold": 0.7,
            "match_iou_threshold": 0.5,
            "max_queries": 300,
        },
    )

    assert len(calls) == 1
    assert [row["image_id"] for row in calls[0][0]] == [2]
    assert calls[0][2]["score_threshold"] == 0.42
    assert report["selection_score_threshold"] == 0.42
    assert report["selection_metrics"]["recall"] == 0.995
    assert report["frozen_threshold_selection_gate"] is True

    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_worker_gate._metrics",
        lambda *_args, **_kwargs: {
            "recall": 0.98,
            "precision": 1.0,
            "count_accuracy": 1.0,
        },
    )
    failed = evaluate_frozen_detector_threshold_selection(
        records,
        predictions,
        {"selected_score_threshold": 0.42},
        {
            "target_recall": 0.99,
            "nms_iou_threshold": 0.7,
            "match_iou_threshold": 0.5,
            "max_queries": 300,
        },
    )
    assert failed["selection_score_threshold"] == 0.42
    assert failed["frozen_threshold_selection_gate"] is False


def test_train_rejection_reason_counter_counts_each_reason_once_per_image():
    counts: Counter[str] = Counter()

    _update_unique_reason_counts(counts, ["NO_DETECTION", "NO_DETECTION", "DATA_ALIGNMENT_REJECT"])
    _update_unique_reason_counts(counts, ["NO_DETECTION"])

    assert counts == {"NO_DETECTION": 2, "DATA_ALIGNMENT_REJECT": 1}


def test_domain_adaptation_subset_is_balanced_diverse_and_group_oof(tmp_path):
    records = []
    (tmp_path / "train2019").mkdir()
    for category_id in range(1, 7):
        for index in range(8):
            image_path = (
                f"train2019/code{category_id}{'-back' if index % 2 else ''}"
                f"_camera{index % 4}-{index + 1}.jpg"
            )
            Image.new(
                "RGB",
                (8, 8),
                (category_id * 30, index * 30, (category_id + index) * 15),
            ).save(tmp_path / image_path)
            records.append(
                {
                    "source": "rpc_train2019",
                    "image_id": category_id * 100 + index,
                    "image_path": image_path,
                    "physical_group": f"code{category_id}",
                    "prediction_fold": (category_id - 1) % 3,
                    "annotations": [{"category_id": category_id}],
                }
            )

    first = _domain_adaptation_train_subset(
        records,
        dataset_root=tmp_path,
        samples_per_surface_camera=1,
        seed=20260810,
        fold_count=3,
    )
    second = _domain_adaptation_train_subset(
        records,
        dataset_root=tmp_path,
        samples_per_surface_camera=1,
        seed=20260810,
        fold_count=3,
    )

    assert [row["image_id"] for row in first] == [row["image_id"] for row in second]
    assert Counter(row["annotations"][0]["category_id"] for row in first) == {
        category_id: 4 for category_id in range(1, 7)
    }
    for row in first:
        category_id = row["annotations"][0]["category_id"]
        assert row["fold"] == (category_id - 1) % 3
        assert row["split"] == "development"


def test_domain_adaptation_namespace_uses_its_local_optimizer_and_freeze_recipe(
    tmp_path,
):
    checkpoint = tmp_path / "baseline" / "best"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"baseline")
    options = {
        "pretrained_name": "pretrained",
        "image_size": 640,
        "batch_size": 2,
        "workers": 0,
        "epochs": 100,
        "patience": 20,
        "learning_rate": 1e-5,
        "head_lr_multiplier": 10.0,
        "class_head_prior_probability": 0.01,
        "warmup_epochs": 0,
        "weight_decay": 1e-4,
        "seed": 1,
        "min_score_threshold": 0.05,
        "max_score_threshold": 0.95,
        "threshold_steps": 91,
        "nms_iou_threshold": 0.7,
        "match_iou_threshold": 0.5,
        "target_recall": 0.99,
        "max_queries": 300,
        "domain_adaptation": {
            "epochs": 1,
            "patience": 1,
            "learning_rate": 2.5e-7,
            "head_lr_multiplier": 2.0,
            "weight_decay": 0.0,
            "freeze_mode": "classification_heads_only",
            "frozen_modules_eval": True,
            "skip_epoch_validation": True,
            "workers": 1,
            "seed": 7,
        },
    }

    namespace = _domain_adaptation_namespace(
        options,
        tmp_path / "manifest.jsonl",
        tmp_path,
        tmp_path / "adapted",
        0,
        checkpoint,
        {"identity": "fixed"},
        resume=True,
    )
    recipe = detector_optimizer_recipe(namespace)

    assert namespace.fixed_epoch_checkpoint is True
    assert namespace.initial_checkpoint_sha256 == sha256_file(checkpoint / "model.safetensors")
    assert recipe["total_epochs"] == 1
    assert recipe["base_learning_rate"] == 2.5e-7
    assert recipe["head_lr_multiplier"] == 2.0
    assert recipe["weight_decay"] == 0.0
    assert recipe["freeze_mode"] == "classification_heads_only"
    assert recipe["frozen_modules_eval"] is True
    assert recipe["skip_epoch_validation"] is True
    assert namespace.workers == 1
    assert recipe["workers"] == 1


def test_domain_adaptation_source_replay_has_unique_train_only_identities():
    originals = [
        {"source": "rpc_val2019", "image_id": 1, "fold": 0},
        {"source": "rpc_val2019", "image_id": 2, "fold": 1},
    ]

    replay = _domain_adaptation_source_replay(originals, 8)

    assert len(replay) == 14
    assert len({record["adaptation_replay_key"] for record in replay}) == 14
    assert all(record["adaptation_replay_only"] is True for record in replay)
    assert {record["fold"] for record in replay} == {0, 1}
    assert [record["image_id"] for record in originals] == [1, 2]


def test_progressive_fold_gate_reuses_exact_predictions_without_policy_rebinding(
    tmp_path, monkeypatch
):
    detector_dir = tmp_path / "detector"
    detector_dir.mkdir()
    (detector_dir / "threshold.json").write_text(
        json.dumps(
            {
                "threshold_policy": "calibration_oof_only",
                "selected_score_threshold": 0.42,
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "attempt" / "folds" / "fold0" / "best"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "instances_val2019.json").write_text("{}", encoding="utf-8")
    (tmp_path / "instances_train2019.json").write_text("{}", encoding="utf-8")
    source = [
        {
            "source": "rpc_val2019",
            "image_id": 1,
            "image_path": "val.jpg",
            "width": 100,
            "height": 100,
            "fold": 0,
            "annotations": [{"category_id": 1, "bbox_xywh": [10, 10, 20, 20]}],
        }
    ]
    target = [
        {
            **source[0],
            "source": "rpc_train2019",
            "image_id": 2,
            "image_path": "train.jpg",
            "prediction_fold": 0,
        }
    ]
    calls = []

    def fake_predict(_checkpoint, records, *_args, **_kwargs):
        calls.append([record["image_id"] for record in records])
        return [
            {
                "sample_key": f"{record['source']}:{record['image_id']}",
                "image_id": record["image_id"],
                "boxes_xyxy": [[10, 10, 30, 30]],
                "scores": [0.9],
            }
            for record in records
        ]

    monkeypatch.setattr("bixolon_scanner.training.rpc_worker_gate.predict_records", fake_predict)
    options = {
        "inference_batch_size": 2,
        "min_score_threshold": 0.05,
        "nms_iou_threshold": 0.7,
        "match_iou_threshold": 0.5,
        "max_queries": 300,
        "min_object_area_ratio": 0.001,
        "train_gate_policy": {
            "target_bbox_recall": 0.95,
            "target_exact_normal_rate": 0.20,
            "target_class_coverage": 1.0,
            "target_min_accepted_per_class": 1,
        },
    }
    adaptation = {"recipe": "fixed"}

    first = _adaptation_progressive_fold_gate(
        adaptation_dir=tmp_path / "attempt",
        detector_dir=detector_dir,
        dataset_root=tmp_path,
        checkpoint=checkpoint,
        fold=0,
        source_records=source,
        target_records=target,
        options=options,
        adaptation=adaptation,
        resume=False,
    )
    options["train_gate_policy"]["target_exact_normal_rate"] = 0.10
    second = _adaptation_progressive_fold_gate(
        adaptation_dir=tmp_path / "attempt",
        detector_dir=detector_dir,
        dataset_root=tmp_path,
        checkpoint=checkpoint,
        fold=0,
        source_records=source,
        target_records=target,
        options=options,
        adaptation=adaptation,
        resume=False,
    )

    assert first["passes"] is True
    assert first["score_threshold"] == 0.42
    assert first["policy"] == ("baseline_calibration_frozen_threshold_progressive_fold_gate")
    assert first["target_gate"]["class_coverage"] == 1.0
    assert second["passes"] is True
    assert second["train_gate_policy_sha256"] != first["train_gate_policy_sha256"]
    assert second["target_prediction_sha256"] == first["target_prediction_sha256"]
    assert calls == [[1], [2]]


def test_domain_adaptation_farthest_first_is_nested_and_hash_deduplicated(tmp_path):
    (tmp_path / "train2019").mkdir()
    records = []
    for view_id in (1, 6, 11, 21, 31, 40):
        image_path = f"train2019/code1_camera0-{view_id}.jpg"
        Image.new("RGB", (8, 8), (view_id, 0, 0)).save(tmp_path / image_path)
        records.append(
            {
                "source": "rpc_train2019",
                "image_id": view_id,
                "image_path": image_path,
                "physical_group": "code1",
                "prediction_fold": 0,
                "annotations": [{"category_id": 1}],
            }
        )
    duplicate_path = tmp_path / "train2019" / "code1_camera0-2.jpg"
    duplicate_path.write_bytes((tmp_path / records[0]["image_path"]).read_bytes())
    records.append(dict(records[0], image_id=2, image_path="train2019/code1_camera0-2.jpg"))

    prefixes = [
        _domain_adaptation_train_subset(
            records,
            dataset_root=tmp_path,
            samples_per_surface_camera=count,
            seed=7,
            fold_count=1,
        )
        for count in (1, 2, 3, 6)
    ]

    ids = [[row["image_id"] for row in selected] for selected in prefixes]
    assert ids[0] == ids[1][:1] == ids[2][:1] == ids[3][:1]
    assert ids[1] == ids[2][:2] == ids[3][:2]
    assert ids[2] == ids[3][:3]
    assert len(prefixes[-1]) == 6
    assert len({row["source_image_sha256"] for row in prefixes[-1]}) == 6


def test_baseline_marker_migration_never_infers_from_adapted_active_marker(tmp_path, monkeypatch):
    detector_dir = tmp_path / "detector"
    detector_dir.mkdir()
    active = detector_dir / "complete.json"
    active.write_text(json.dumps({"checkpoint_set": "domain_adaptation"}), encoding="utf-8")
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_worker_gate._detector_phase_complete",
        lambda *_args, **_kwargs: True,
    )
    assert not _baseline_detector_complete(detector_dir, tmp_path, {})
    assert not (detector_dir / "baseline" / "complete.json").exists()

    active.write_text(json.dumps({"checkpoint_set": "baseline"}), encoding="utf-8")
    assert _baseline_detector_complete(detector_dir, tmp_path, {})
    migrated = json.loads((detector_dir / "baseline" / "complete.json").read_text(encoding="utf-8"))
    assert migrated["contract"] == "rpc-detector-baseline-complete-v1"
    assert migrated["checkpoint_set"] == "baseline"


def test_existing_baseline_marker_is_read_only_and_tamper_fails_before_training(
    tmp_path, monkeypatch
):
    baseline = tmp_path / "detector" / "baseline" / "complete.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({"checkpoint_set": "baseline"}), encoding="utf-8")
    args = Namespace(output_dir=tmp_path, dataset_root=tmp_path, resume=False)
    config = {"detector": {}}
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_worker_gate.train_oof_detectors",
        lambda *_args, **_kwargs: pytest.fail("immutable baseline must not retrain"),
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_worker_gate._baseline_detector_complete",
        lambda *_args, **_kwargs: True,
    )
    assert prepare_detector_phase(args, config)["checkpoint_set"] == "baseline"

    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_worker_gate._baseline_detector_complete",
        lambda *_args, **_kwargs: False,
    )
    with pytest.raises(ValueError, match="immutable baseline"):
        prepare_detector_phase(args, config)


def test_train_gate_legacy_completion_migrates_only_after_exact_lineage_validation(
    tmp_path,
):
    detector_dir = tmp_path / "detector"
    attempt = detector_dir / "adaptation-attempts" / "recipe"
    adaptation = {"enabled": True, "epochs": 1}
    policy = {
        "target_bbox_recall": 0.95,
        "target_exact_normal_rate": 0.20,
        "target_class_coverage": 1.0,
        "target_min_accepted_per_class": 1,
    }
    config = {
        "detector": {
            "domain_adaptation": adaptation,
            "train_gate_policy": policy,
            "nms_iou_threshold": 0.7,
            "match_iou_threshold": 0.5,
            "max_queries": 300,
            "target_recall": 0.99,
        }
    }
    record = {
        "source": "rpc_val2019",
        "image_id": 1,
        "role": "selection",
        "annotations": [{"category_id": 1, "bbox_xywh": [10, 10, 20, 20]}],
    }
    prediction = {
        "sample_key": "rpc_val2019:1",
        "image_id": 1,
        "boxes_xyxy": [[10, 10, 30, 30]],
        "scores": [0.9],
    }
    manifest_path = detector_dir / "manifest" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    threshold_path = detector_dir / "threshold.json"
    threshold_path.write_text(
        json.dumps(
            {
                "threshold_policy": "calibration_oof_only",
                "selected_score_threshold": 0.42,
                "target_recall": 0.99,
                "target_recall_satisfied": True,
                "calibration_metrics": {"recall": 1.0},
            }
        ),
        encoding="utf-8",
    )
    val_predictions = detector_dir / "predictions" / "val_oof.jsonl"
    val_predictions.parent.mkdir(parents=True)
    val_predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    val_metadata = val_predictions.with_name("val_oof.jsonl.metadata.json")
    val_metadata.write_text("{}\n", encoding="utf-8")
    baseline_path = detector_dir / "baseline" / "complete.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(
        json.dumps(
            {
                "checkpoint_set": "baseline",
                "val_predictions_sha256": sha256_file(val_predictions),
                "val_predictions_metadata_sha256": sha256_file(val_metadata),
            }
        ),
        encoding="utf-8",
    )
    artifact_paths = {
        "adaptation_manifest_sha256": attempt / "manifest" / "manifest.jsonl",
        "adaptation_manifest_metadata_sha256": attempt / "manifest" / "metadata.json",
        "train_predictions_sha256": attempt / "predictions" / "train_assigned.jsonl",
        "train_predictions_metadata_sha256": attempt
        / "predictions"
        / "train_assigned.jsonl.metadata.json",
        "target_oof_gate_sha256": attempt / "target_oof_gate.json",
        "progressive_fold0_gate_sha256": attempt / "progressive-gate" / "fold0.json",
        "threshold_sha256": attempt / "threshold.json",
    }
    for path in artifact_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    marker = {
        "role": "train_gate_only",
        "checkpoint_set": "domain_adaptation",
        "artifact_root": "adaptation-attempts/recipe",
        "adaptation_config_sha256": hashlib.sha256(
            _canonical_json(adaptation).encode()
        ).hexdigest(),
        "train_gate_policy_sha256": hashlib.sha256(_canonical_json(policy).encode()).hexdigest(),
        "baseline_complete_sha256": sha256_file(baseline_path),
        "baseline_threshold_sha256": sha256_file(threshold_path),
        "target_oof_gate": {"passes": True},
        "progressive_fold0_gate": {"passes": True},
        **{key: sha256_file(path) for key, path in artifact_paths.items()},
    }
    marker["train_predictions_sha256"] = "0" * 64
    marker_path = detector_dir / "train-gate" / "complete.json"
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert _train_gate_complete(detector_dir, config) == (False, None)
    assert not (attempt / "baseline_frozen_selection_gate.json").exists()

    marker["train_predictions_sha256"] = sha256_file(artifact_paths["train_predictions_sha256"])
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    valid, migrated = _train_gate_complete(detector_dir, config)

    assert valid is True
    assert migrated is not None
    selection_path = attempt / "baseline_frozen_selection_gate.json"
    assert migrated["baseline_frozen_selection_gate_sha256"] == sha256_file(selection_path)
    assert migrated["baseline_frozen_selection_gate"]["frozen_threshold_selection_gate"] is True


def test_active_adaptation_same_recipe_is_immutable_on_retry(tmp_path, monkeypatch):
    detector_dir = tmp_path / "detector"
    detector_dir.mkdir()
    active = {"checkpoint_set": "domain_adaptation", "artifact_root": "attempt"}
    (detector_dir / "complete.json").write_text(json.dumps(active), encoding="utf-8")
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_worker_gate._baseline_detector_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_worker_gate._detector_phase_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_worker_gate._train_gate_complete",
        lambda *_args, **_kwargs: (True, active),
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_worker_gate.train_detector",
        lambda *_args, **_kwargs: pytest.fail("active adaptation must not retrain"),
    )

    result = prepare_detector_domain_adaptation(
        Namespace(output_dir=tmp_path, dataset_root=tmp_path, resume=False),
        {"detector": {"domain_adaptation": {"enabled": True}}},
    )

    assert result == active


def test_target_oof_gate_requires_bbox_exact_normal_and_full_class_coverage():
    records = [
        {"annotations": [{"category_id": 1}]},
        {"annotations": [{"category_id": 2}]},
    ]
    normal = {
        "recapture_reasons": [],
        "matches": [{}],
        "unmatched_detection_indices": [],
    }
    adaptation = {
        "target_bbox_recall": 0.99,
        "target_exact_normal_rate": 0.99,
        "target_class_coverage": 1.0,
    }

    passing = _target_oof_gate_report(
        records,
        [normal, normal],
        {"recall": 0.99},
        adaptation,
        expected_class_count=2,
        score_threshold=0.42,
    )
    failed = _target_oof_gate_report(
        records[:1],
        [dict(normal, recapture_reasons=["NO_DETECTION"], matches=[])],
        {"recall": 0.98},
        adaptation,
        expected_class_count=2,
        score_threshold=0.42,
    )
    missing_normal_class = _target_oof_gate_report(
        records,
        [normal, dict(normal, recapture_reasons=["NO_DETECTION"], matches=[])],
        {"recall": 0.99},
        dict(adaptation, target_exact_normal_rate=0.5),
        expected_class_count=2,
        score_threshold=0.42,
    )

    assert passing["passes"] is True
    assert passing["failure_reasons"] == []
    assert failed["passes"] is False
    assert set(failed["failure_reasons"]) == {
        "TARGET_BBOX_RECALL",
        "TARGET_EXACT_NORMAL_RATE",
        "TARGET_CLASS_COVERAGE",
        "TARGET_MIN_ACCEPTED_PER_CLASS",
    }
    assert missing_normal_class["class_count"] == 1
    assert missing_normal_class["class_coverage"] == 0.5
    assert missing_normal_class["bbox_recall"] == 0.99
    assert missing_normal_class["failure_reasons"] == ["TARGET_CLASS_COVERAGE"]


def test_train_detector_physical_group_fold_keeps_front_and_back_together(tmp_path):
    images = []
    annotations = []
    for index, filename in enumerate(
        [
            "111_camera0-1.jpg",
            "111-back_camera2-2.jpg",
            "222_camera1-3.jpg",
            "222-back_camera3-4.jpg",
        ],
        start=1,
    ):
        images.append({"id": index, "file_name": filename, "width": 100, "height": 100})
        annotations.append(
            {
                "id": index,
                "image_id": index,
                "category_id": 1 if index <= 2 else 2,
                "bbox": [20, 20, 40, 40],
                "area": 1600,
            }
        )
    _write_coco(tmp_path, "train", [{"id": 1}, {"id": 2}], images, annotations)

    records = _train_records(tmp_path, fold_count=3, fold_assignment="physical_group")
    by_group: dict[str, set[int]] = {}
    for record in records:
        by_group.setdefault(record["physical_group"], set()).add(record["prediction_fold"])

    assert set(by_group) == {"111", "222"}
    assert all(len(folds) == 1 for folds in by_group.values())


def test_evaluation_reports_class_difficulty_and_operational_metrics():
    logits = np.asarray([[8, 0], [0, 8], [7, 0], [0, 7], [6, 0], [0, 6]], dtype=float)
    targets = np.asarray([0, 1, 0, 1, 0, 1])
    predictions = {
        "logits": logits,
        "targets": targets,
        "levels": np.asarray(["easy", "easy", "medium", "medium", "hard", "hard"]),
        "groups": np.asarray(["a", "a", "b", "b", "c", "c"]),
        "sample_ids": np.asarray([str(index) for index in range(6)]),
    }
    calibration = {
        "temperature": 1.0,
        "approval_threshold": 0.9,
        "risk_control_satisfied": True,
    }
    report = evaluate_logits(
        predictions, calibration, category_count=2, bootstrap_repetitions=20, bootstrap_seed=1
    )
    assert report["overall_top1_accuracy"] == 1.0
    assert report["macro_top1_accuracy"] == 1.0
    assert report["approved_precision"] == 1.0
    assert report["difficulty"]["hard"]["sample_count"] == 2
    assert report["difficulty"]["hard"]["recognition_rate"] == 1.0
    assert report["difficulty"]["hard"]["misrecognition_rate"] == 0.0
    assert report["difficulty"]["hard"]["processing_speed_p95_ms"] is None


def test_evaluation_counts_unmatched_approval_and_excludes_border_recapture_image():
    predictions = {
        "logits": np.asarray([[8, 0], [8, 0], [0, 8], [0, 0]], dtype=float),
        "targets": np.asarray([0, -1, 1, 0]),
        "levels": np.asarray(["easy", "easy", "hard", "hard"]),
        "groups": np.asarray(["a", "a", "b", "c"]),
        "sample_ids": np.asarray(["1", "2", "3", "4"]),
        "image_ids": np.asarray([10, 10, 20, 30]),
        "touches_border": np.asarray([False, False, False, True]),
    }
    calibration = {
        "temperature": 1.0,
        "approval_threshold": 0.9,
        "risk_control_satisfied": True,
    }
    report = evaluate_logits(
        predictions, calibration, category_count=2, bootstrap_repetitions=10, bootstrap_seed=3
    )
    assert report["classifier_border_recapture_images"] == 1
    assert report["unmatched_detector_count"] == 1
    assert report["approved_precision"] == pytest.approx(2 / 3)


def test_ground_truth_worker_outcomes_partition_every_box():
    classifier_report = {
        "classifier_border_recapture_image_ids": [3],
        "approved_correct_count": 2,
        "approved_wrong_matched_count": 1,
        "approved_unmatched_count": 1,
        "unknown_top3_correct_count": 1,
        "unknown_top3_missing_count": 1,
        "unknown_unmatched_count": 0,
        "unmatched_detector_count": 1,
    }
    detector_report = {
        "validation_image_outcomes": [
            {
                "image_id": 1,
                "role": "selection",
                "ground_truth_count": 2,
                "missed_count": 0,
                "recapture_reasons": ["DETECTOR_UNCERTAIN_OBJECT"],
            },
            {
                "image_id": 2,
                "role": "selection",
                "ground_truth_count": 6,
                "missed_count": 1,
                "recapture_reasons": [],
            },
            {
                "image_id": 3,
                "role": "selection",
                "ground_truth_count": 3,
                "missed_count": 0,
                "recapture_reasons": [],
            },
        ]
    }
    outcomes = _ground_truth_worker_outcomes(classifier_report, detector_report, role="selection")
    assert outcomes["denominator"] == 11
    assert sum(outcomes["counts"].values()) == 11
    assert outcomes["counts"]["detector_recapture"] == 2
    assert outcomes["counts"]["classifier_border_recapture"] == 3


def test_difficulty_worker_metrics_adds_fixed_level_segmentation_denominators():
    classifier_report = {
        "difficulty": {
            level: {
                "recognition_rate": 1.0,
                "candidate_in_rate": 1.0,
                "candidate_out_rate": 0.0,
                "misrecognition_rate": 0.0,
                "processing_speed_p95_ms": None,
            }
            for level in ("easy", "medium", "hard")
        }
    }
    detector_report = {
        "validation_image_outcomes": [
            {
                "role": "selection",
                "level": "easy",
                "ground_truth_count": 100,
                "missed_count": 1,
            },
            {
                "role": "selection",
                "level": "medium",
                "ground_truth_count": 80,
                "missed_count": 2,
            },
            {
                "role": "selection",
                "level": "hard",
                "ground_truth_count": 50,
                "missed_count": 5,
            },
        ]
    }

    metrics = _difficulty_worker_metrics(classifier_report, detector_report, role="selection")

    assert metrics["easy"]["segmentation_failure_rate"] == 0.01
    assert metrics["medium"]["segmentation_failure_rate"] == 0.025
    assert metrics["hard"]["segmentation_failure_rate"] == 0.1


def test_worker_taxonomy_separates_image_and_segment_recapture_without_double_count():
    predictions = {
        "logits": np.asarray(
            [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 0.0]],
            dtype=float,
        ),
        "targets": np.asarray([0, -1, 1, 1]),
        "levels": np.asarray(["easy", "easy", "easy", "easy"]),
        "image_ids": np.asarray([1, 1, 2, 3]),
        "touches_border": np.asarray([False, False, False, True]),
    }
    calibration = {
        "temperature": 1.0,
        "approval_threshold": 0.9,
        "risk_control_satisfied": True,
    }
    detector_report = {
        "validation_image_outcomes": [
            {
                "image_id": 1,
                "role": "selection",
                "level": "easy",
                "ground_truth_count": 2,
                "missed_count": 1,
                "recapture_reasons": [],
            },
            {
                "image_id": 2,
                "role": "selection",
                "level": "easy",
                "ground_truth_count": 1,
                "missed_count": 0,
                "recapture_reasons": ["DETECTOR_OBJECT_TOO_SMALL"],
            },
            {
                "image_id": 3,
                "role": "selection",
                "level": "easy",
                "ground_truth_count": 1,
                "missed_count": 0,
                "recapture_reasons": [],
            },
        ]
    }

    report = evaluate_worker_taxonomy(predictions, calibration, detector_report, role="selection")[
        "easy"
    ]

    assert report["image_recapture_count"] == 1
    assert report["segment_recapture_count"] == 1
    assert report["segment_recapture_image_count"] == 1
    assert report["segmentation_missed_count"] == 1
    assert report["segmentation_missed_rate"] == pytest.approx(1 / 3)
    assert report["segmentation_false_positive_count"] == 1
    assert report["recognition_target_count"] == 1
    assert report["recognition_rate"] == 1.0
    assert report["misrecognition_rate"] == pytest.approx(1 / 2)
    assert report["end_to_end_success_rate"] == 0.25
    assert report["segmentation_failure_image_count"] == 3
    assert report["segmentation_failure_image_rate"] == 1.0


def test_worker_taxonomy_applies_segment_quality_without_hiding_it_as_unknown():
    predictions = {
        "logits": np.asarray([[8.0, 0.0], [8.0, 0.0]], dtype=float),
        "targets": np.asarray([0, -1]),
        "levels": np.asarray(["easy", "easy"]),
        "image_ids": np.asarray([1, 1]),
        "touches_border": np.asarray([False, False]),
    }
    calibration = {
        "temperature": 1.0,
        "approval_threshold": 0.9,
        "risk_control_satisfied": True,
    }
    detector_report = {
        "validation_image_outcomes": [
            {
                "image_id": 1,
                "role": "selection",
                "level": "easy",
                "ground_truth_count": 1,
                "missed_count": 0,
                "recapture_reasons": [],
            }
        ]
    }

    report = evaluate_worker_taxonomy(
        predictions,
        calibration,
        detector_report,
        role="selection",
        segment_quality_scores=np.asarray([0.99, 0.01]),
        segment_quality_threshold=0.5,
    )["easy"]

    assert report["segment_recapture_count"] == 1
    assert report["recognition_target_count"] == 1
    assert report["correct_approved_count"] == 1
    assert report["wrong_approved_count"] == 0
    assert report["unknown_count"] == 0


def test_worker_taxonomy_can_force_valid_ambiguous_roi_to_unknown():
    predictions = {
        "logits": np.asarray([[8.0, 0.0]], dtype=float),
        "targets": np.asarray([1]),
        "levels": np.asarray(["easy"]),
        "image_ids": np.asarray([1]),
        "touches_border": np.asarray([False]),
    }
    calibration = {
        "temperature": 1.0,
        "approval_threshold": 0.9,
        "risk_control_satisfied": True,
    }
    detector_report = {
        "validation_image_outcomes": [
            {
                "image_id": 1,
                "role": "selection",
                "level": "easy",
                "ground_truth_count": 1,
                "missed_count": 0,
                "recapture_reasons": [],
            }
        ]
    }

    report = evaluate_worker_taxonomy(
        predictions,
        calibration,
        detector_report,
        role="selection",
        force_unknown_mask=np.asarray([True]),
    )["easy"]

    assert report["segment_recapture_count"] == 0
    assert report["recognition_target_count"] == 1
    assert report["approved_count"] == 0
    assert report["unknown_count"] == 1
    assert report["unknown_top3_in_count"] == 1
    assert report["segmentation_failure_image_count"] == 0


def test_prepare_builds_cache_without_reading_test_and_resume_reuses_it(tmp_path, monkeypatch):
    root = tmp_path / "rpc"
    categories = [
        {"id": 1, "name": "one", "supercategory": "x"},
        {"id": 2, "name": "two", "supercategory": "x"},
    ]
    train_images = []
    train_annotations = []
    for category in (1, 2):
        for camera in range(4):
            image_id = category * 10 + camera
            train_images.append(
                {
                    "id": image_id,
                    "file_name": f"barcode{category}_camera{camera}-0.jpg",
                    "width": 16,
                    "height": 16,
                }
            )
            train_annotations.append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": [2, 2, 10, 10],
                }
            )
    val_images = []
    val_annotations = []
    annotation_id = 100
    for group in range(4):
        image_id = 100 + group
        val_images.append(
            {
                "id": image_id,
                "file_name": f"20180101-00-00-0{group}-{group}.jpg",
                "width": 16,
                "height": 16,
                "level": ("easy", "medium", "hard", "easy")[group],
            }
        )
        for category in (1, 2):
            val_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": [2, 2, 10, 10],
                }
            )
            annotation_id += 1
    _write_coco(root, "train", categories, train_images, train_annotations)
    _write_coco(root, "val", categories, val_images, val_annotations)
    config = {
        "experiment": {
            "expected_num_classes": 2,
            "sample_sizes": [2],
            "seeds": [7],
            "validation_split_seed": 7,
            "calibration_fraction": 0.5,
            "noninferiority_margin": 0.01,
            "bootstrap_repetitions": 10,
        },
        "detector": {},
        "sampling": {
            "anchor_pool_size": 2,
            "tie_tolerance": 1e-9,
            "contact_sheet_first_n": 2,
        },
        "training": {
            "cache_size": 8,
            "image_size": 8,
            "train_margin_ratio": 0.08,
            "eval_margin_ratio": 0.05,
        },
    }
    output = tmp_path / "output"
    _write_detector_phase_completion(output)
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._detector_phase_complete",
        lambda *_args, **_kwargs: True,
    )

    gated_train = []
    for annotation in train_annotations:
        image = next(row for row in train_images if row["id"] == annotation["image_id"])
        camera = int(image["file_name"].split("camera")[1].split("-")[0])
        gated_train.append(
            {
                "sample_id": f"train:{image['id']}:{annotation['id']}",
                "split": "train",
                "image_id": image["id"],
                "annotation_id": annotation["id"],
                "image_path": f"train2019/{image['file_name']}",
                "bbox_xywh": annotation["bbox"],
                "category_id": annotation["category_id"],
                "target": annotation["category_id"] - 1,
                "barcode": f"barcode{annotation['category_id']}",
                "surface": "front",
                "camera": camera,
                "view_id": 0,
                "detector_score": 0.99,
            }
        )
    gated_val = []
    for annotation in val_annotations:
        image = next(row for row in val_images if row["id"] == annotation["image_id"])
        gated_val.append(
            {
                "sample_id": f"val:{image['id']}:det{annotation['id']}",
                "split": "val",
                "image_id": image["id"],
                "annotation_id": annotation["id"],
                "image_path": f"val2019/{image['file_name']}",
                "bbox_xywh": annotation["bbox"],
                "category_id": annotation["category_id"],
                "target": annotation["category_id"] - 1,
                "level": image["level"],
                "group_id": str(image["id"] % 100),
                "role": "calibration" if image["id"] % 2 == 0 else "selection",
                "touches_border": False,
            }
        )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale.load_worker_gated_records",
        lambda *_: (gated_train, gated_val, {"normal": True}),
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._extract_visual_embeddings",
        lambda records, *_: (
            np.eye(len(records), dtype=np.float32),
            [str(index) for index in range(len(records))],
        ),
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._render_sampling_audit",
        lambda *_: None,
    )
    args = Namespace(dataset_root=root, output_dir=output, weights=weights, resume=False)
    metadata = prepare(args, config)
    assert metadata["category_count"] == 2
    assert metadata["test_accessed"] is False
    array_path = output / "prepared" / "cache" / "images.npy"
    first_mtime = array_path.stat().st_mtime_ns
    args.resume = True
    prepare(args, config)
    assert array_path.stat().st_mtime_ns == first_mtime

    first_fingerprint = json.loads(
        (output / "prepared" / "cache" / "metadata.json").read_text(encoding="utf-8")
    )["fingerprint"]
    config["training"]["train_margin_ratio"] = 0.12
    prepare(args, config)
    second_fingerprint = json.loads(
        (output / "prepared" / "cache" / "metadata.json").read_text(encoding="utf-8")
    )["fingerprint"]
    assert second_fingerprint != first_fingerprint

    metadata_path = output / "prepared" / "experiment.json"
    sealed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sealed_metadata.update(
        {
            "test_accessed": True,
            "test_access_started_at": "2026-08-12T00:00:00+00:00",
            "test_access_model_lock_sha256": "1" * 64,
            "test_access_final_detector_complete_sha256": "2" * 64,
            "test_access_final_detector_checkpoint_sha256": "3" * 64,
        }
    )
    metadata_path.write_text(json.dumps(sealed_metadata), encoding="utf-8")
    args.resume = False

    with pytest.raises(RuntimeError, match="fresh output directory"):
        prepare(args, config)

    prepared_again = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert prepared_again["test_accessed"] is True
    for field in (
        "test_access_started_at",
        "test_access_model_lock_sha256",
        "test_access_final_detector_complete_sha256",
        "test_access_final_detector_checkpoint_sha256",
    ):
        assert prepared_again[field] == sealed_metadata[field]

    with pytest.raises(ValueError, match="cannot be downgraded"):
        _write_experiment_metadata(metadata_path, {**prepared_again, "test_accessed": False})
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["test_accessed"] is True

    full_output = tmp_path / "full_output"
    _write_detector_phase_completion(full_output)
    config["experiment"] = {
        "mode": "full_dataset",
        "expected_num_classes": 2,
        "seeds": [20260810],
        "validation_split_seed": 7,
        "calibration_fraction": 0.5,
        "bootstrap_repetitions": 10,
    }
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._extract_exact_roi_hashes",
        lambda records, *_: ["a", "a", "b", "c", "d", "e", "f", "g"],
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._extract_visual_embeddings",
        lambda *_: pytest.fail("full_dataset prepare must not build DINO embeddings"),
    )
    full_metadata = prepare(
        Namespace(dataset_root=root, output_dir=full_output, weights=weights, resume=False),
        config,
    )
    assert full_metadata["mode"] == "full_dataset"
    assert full_metadata["sample_sizes"] == []
    assert full_metadata["train_union_count"] == 7
    assert full_metadata["train_counts"] == {"1": 3, "2": 4}
    assert full_metadata["train_class_imbalance"]["max_to_min_ratio"] == pytest.approx(4 / 3)
    assert not (full_output / "prepared" / "embeddings" / "train.npy").exists()


def test_crop_clamps_margin_and_preserves_the_annotation_region():
    image = Image.new("RGB", (20, 12), "white")
    cropped = _crop(image, [1, 2, 8, 6], 0.5)
    assert cropped.size == (13, 11)


def test_stage_progress_restores_epoch_optimizer_scheduler_and_rng(tmp_path):
    torch = pytest.importorskip("torch")
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    generator = torch.Generator().manual_seed(11)
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "frozen_progress.pt"
    history = [{"stage": "frozen", "epoch": 1, "training_loss": 1.0}]
    _save_stage_progress(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        generator=generator,
        stage="frozen",
        completed_epochs=1,
        total_epochs=3,
        history=history,
        sample_size=5,
        seed=11,
    )
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    completed, restored_history = _load_stage_progress(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        generator=generator,
        stage="frozen",
        total_epochs=3,
        sample_size=5,
        seed=11,
    )
    assert completed == 1
    assert restored_history == history
    assert scheduler.last_epoch == 1
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())
    with pytest.raises(ValueError, match="identity mismatch"):
        _load_stage_progress(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            generator=generator,
            stage="partial",
            total_epochs=3,
            sample_size=5,
            seed=11,
        )


def test_incomplete_or_corrupt_run_is_not_treated_as_resumable_completion(tmp_path):
    (tmp_path / "complete.json").write_text('{"complete": true}', encoding="utf-8")
    assert not _run_complete(tmp_path, 2)
    (tmp_path / "best.pt").write_bytes(b"checkpoint")
    (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
    (tmp_path / "selection_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "selection_predictions.npz").write_bytes(b"corrupt")
    assert not _run_complete(tmp_path, 2)


def test_full_dataset_train_dispatches_one_run_without_sample_size(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    cache = prepared / "cache"
    cache.mkdir(parents=True)
    (prepared / "experiment.json").write_text(
        json.dumps({"mode": "full_dataset"}), encoding="utf-8"
    )
    (cache / "records.jsonl").write_text(
        json.dumps({"sample_id": "train:1", "role": "train"}) + "\n", encoding="utf-8"
    )
    calls = []
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._train_one",
        lambda *values: calls.append(values),
    )
    config = {"experiment": {"mode": "full_dataset", "seeds": [20260810]}}
    args = Namespace(output_dir=tmp_path)
    train_all(args, config)
    assert len(calls) == 1
    assert calls[0][-2:] == (None, 20260810)


@pytest.mark.parametrize(
    "phase",
    ("detector", "adapt-detector", "prepare", "train", "select", "all"),
)
def test_main_blocks_every_mutation_phase_before_dispatch_on_sealed_output(
    tmp_path, monkeypatch, phase
):
    output = tmp_path / "output"
    metadata_path = output / "prepared" / "experiment.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text('{"test_accessed":true}', encoding="utf-8")
    args = Namespace(
        config=tmp_path / "missing-config.json",
        dataset_root=tmp_path / "missing-dataset",
        weights=tmp_path / "missing-weights.pt",
        output_dir=output,
        phase=phase,
        resume=False,
    )
    monkeypatch.setattr("bixolon_scanner.training.rpc_data_scale._parse_args", lambda: args)
    monkeypatch.setattr("bixolon_scanner.training.rpc_data_scale._load_config", lambda _path: {})
    for name in (
        "prepare_detector_phase",
        "prepare_detector_domain_adaptation",
        "prepare",
        "train_all",
        "summarize",
        "test_selected",
        "_environment_snapshot",
    ):
        monkeypatch.setattr(
            f"bixolon_scanner.training.rpc_data_scale.{name}",
            lambda *_args, _name=name, **_kwargs: pytest.fail(f"sealed main dispatched {_name}"),
        )

    with pytest.raises(RuntimeError, match="fresh output directory"):
        rpc_main()


def test_main_allows_only_read_only_test_dispatch_on_sealed_output(tmp_path, monkeypatch):
    output = tmp_path / "output"
    metadata_path = output / "prepared" / "experiment.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text('{"test_accessed":true}', encoding="utf-8")
    args = Namespace(
        config=tmp_path / "config.json",
        dataset_root=tmp_path / "missing-dataset",
        weights=tmp_path / "missing-weights.pt",
        output_dir=output,
        phase="test",
        resume=False,
    )
    calls = []
    monkeypatch.setattr("bixolon_scanner.training.rpc_data_scale._parse_args", lambda: args)
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._load_config",
        lambda _path: {"experiment": {"mode": "full_dataset"}},
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale.test_selected",
        lambda *_args: calls.append("test"),
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._environment_snapshot",
        lambda *_args: pytest.fail("sealed test must not rewrite environment"),
    )

    rpc_main()

    assert calls == ["test"]


@pytest.mark.parametrize(
    "entrypoint",
    (prepare, train_all, prepare_detector_phase, prepare_detector_domain_adaptation),
)
def test_direct_mutation_entrypoints_reject_sealed_output_before_work(tmp_path, entrypoint):
    output = tmp_path / "output"
    metadata_path = output / "prepared" / "experiment.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text('{"test_accessed":true}', encoding="utf-8")
    args = Namespace(output_dir=output)

    with pytest.raises(RuntimeError, match="fresh output directory"):
        entrypoint(args, {})


def test_direct_final_test_record_entrypoint_cannot_reaccess_sealed_test(tmp_path):
    output = tmp_path / "output"
    metadata_path = output / "prepared" / "experiment.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text('{"test_accessed":true}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="sealed final-test report"):
        prepare_final_test_records(
            Namespace(output_dir=output),
            {},
            resume=False,
            model_lock_path=output / "model_lock.json",
        )


def test_full_dataset_summary_locks_the_single_model_without_selected_n(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    worker_gate = {
        "score_threshold": 0.5,
        "train_candidates": 7,
        "train_rejected": {},
        "validation_images": 4,
        "validation_normal_images": 4,
        "validation_recapture_images": 0,
        "validation_recapture_reasons": {},
        "validation_missed_boxes": 0,
        "validation_unmatched_boxes": 0,
    }
    (prepared / "worker_gate_report.json").write_text(json.dumps(worker_gate), encoding="utf-8")
    (prepared / "experiment.json").write_text(
        json.dumps(
            {
                "mode": "full_dataset",
                "train_counts": {"1": 3, "2": 4},
                "train_class_imbalance": {
                    "minimum": 3,
                    "maximum": 4,
                    "mean": 3.5,
                    "median": 3.5,
                    "max_to_min_ratio": 4 / 3,
                    "missing_category_ids": [],
                },
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "full" / "seed20260810"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"checkpoint")
    (run_dir / "calibration.json").write_text(
        json.dumps(
            {
                "temperature": 1.0,
                "approval_threshold": 0.9,
                "approved_count": 10,
                "approved_precision": 1.0,
                "approval_coverage": 0.9,
                "approved_false_rate_upper_95": 0.004,
                "risk_control_satisfied": True,
                "accuracy": 0.99,
                "top3_accuracy": 1.0,
                "matched_count": 10,
                "unmatched_detector_count": 0,
            }
        ),
        encoding="utf-8",
    )
    report = {
        "overall_top1_accuracy": 0.99,
        "overall_top3_accuracy": 1.0,
        "macro_top1_accuracy": 0.98,
        "macro_top3_accuracy": 1.0,
        "class_top1_min": 0.9,
        "class_top1_p05": 0.95,
        "approved_precision": 1.0,
        "approval_coverage": 0.9,
        "unknown_top3_accuracy": 1.0,
        "top1_cluster_bootstrap_95ci": [0.98, 1.0],
        "difficulty": {
            "easy": {"top1_accuracy": 1.0, "top3_accuracy": 1.0},
            "medium": {"top1_accuracy": 0.99, "top3_accuracy": 1.0},
            "hard": {"top1_accuracy": 0.98, "top3_accuracy": 1.0},
        },
    }
    (run_dir / "selection_report.json").write_text(json.dumps(report), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({"train_sample_count": 7}), encoding="utf-8")
    np.savez(
        run_dir / "selection_predictions.npz",
        logits=np.asarray([[4.0, 0.0], [0.0, 4.0]]),
        targets=np.asarray([0, 1]),
    )
    (run_dir / "complete.json").write_text(json.dumps({"complete": True}), encoding="utf-8")
    final_complete_path = tmp_path / "detector" / "final" / "complete.json"
    final_complete = {
        "contract": "rpc-final-detector-baseline-val-all-v1",
        "base_epochs": 57,
        "target_adaptation_stage": "disabled_train_gate_only",
        "config_sha256": "1" * 64,
        "active_detector_complete_sha256": "2" * 64,
        "active_threshold_sha256": "3" * 64,
        "train_gate_complete_sha256": "4" * 64,
        "stage_a_checkpoint_sha256": "6" * 64,
    }

    def fake_train_final(*_args, **_kwargs):
        final_complete_path.parent.mkdir(parents=True, exist_ok=True)
        final_complete_path.write_text(json.dumps(final_complete), encoding="utf-8")
        return final_complete_path.parent

    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale.train_final_detector",
        fake_train_final,
    )
    config = {
        "experiment": {"mode": "full_dataset", "seeds": [20260810]},
        "detector": {
            "domain_adaptation": {
                "epochs": 1,
                "patience": 1,
                "learning_rate": 2.5e-7,
                "seed": 20260810,
                "samples_per_surface_camera": 2,
            }
        },
    }
    summary = summarize(Namespace(output_dir=tmp_path), config)
    assert summary["status"] == "validation_passed"
    assert summary["model_run"] == "runs/full/seed20260810"
    assert "selected_n" not in summary
    lock = json.loads((tmp_path / "model_lock.json").read_text(encoding="utf-8"))
    assert lock["model_run"] == "runs/full/seed20260810"
    assert lock["final_detector_checkpoint_sha256"] == "6" * 64
    assert lock["active_detector_threshold_sha256"] == "3" * 64
    assert lock["operational_detector_role"] == "checkout_baseline_val_all_operational"
    assert lock["train_gate_role"] == "offline_roi_train_gate_only"
    assert "selected_n" not in lock
    assert not (tmp_path / "selected_n.json").exists()


def test_final_test_gate_rejects_zero_approved_or_unknown_evaluable_samples():
    calibration = {"risk_control_satisfied": True}
    passing_metrics = {
        "approved_precision": 1.0,
        "approved_count": 1,
        "unknown_top3_accuracy": 1.0,
        "unknown_matched_count": 1,
    }
    assert _test_operational_gate(calibration, passing_metrics)
    assert not _test_operational_gate(calibration, {**passing_metrics, "approved_count": 0})
    assert not _test_operational_gate(
        calibration,
        {
            **passing_metrics,
            "unknown_top3_accuracy": None,
            "unknown_matched_count": 0,
        },
    )


def test_validation_gate_requires_evaluable_unknown_top3_samples():
    calibration = {"risk_control_satisfied": True}
    assert not _operational_gate(
        calibration,
        {"approved_precision": 1.0, "unknown_top3_accuracy": None},
    )


def test_evaluate_logits_never_approves_when_calibration_risk_control_failed():
    predictions = {
        "logits": np.asarray(
            [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]],
            dtype=np.float64,
        ),
        "targets": np.asarray([0, 1, 2], dtype=np.int64),
        "levels": np.asarray(["easy", "medium", "hard"]),
        "groups": np.asarray(["g0", "g1", "g2"]),
        "image_ids": np.asarray([0, 1, 2], dtype=np.int64),
        "touches_border": np.zeros(3, dtype=bool),
    }
    calibration = {
        "temperature": 1.0,
        "approval_threshold": 1.0,
        "risk_control_satisfied": False,
    }

    report = evaluate_logits(
        predictions,
        calibration,
        category_count=3,
        bootstrap_repetitions=10,
        bootstrap_seed=20260810,
    )

    assert report["approved_count"] == 0
    assert report["approval_coverage"] == 0.0
    assert report["unknown_matched_count"] == 3


def test_failed_full_dataset_lock_blocks_before_test_access(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    metadata_path = prepared / "experiment.json"
    metadata_path.write_text('{"test_accessed":false}', encoding="utf-8")
    (tmp_path / "model_lock.json").write_text(
        json.dumps(
            {
                "mode": "full_dataset",
                "status": "validation_gate_failed",
                "operational_gate": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale.prepare_final_test_records",
        lambda *_args, **_kwargs: pytest.fail("test2019 must not be accessed"),
    )
    config = {"experiment": {"mode": "full_dataset"}}
    with pytest.raises(RuntimeError, match="validation/model lock gate"):
        run_final_test(Namespace(dataset_root=tmp_path, output_dir=tmp_path, resume=False), config)
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["test_accessed"] is False


def test_test_access_seal_is_persisted_before_a_test_read_crash(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    metadata_path = prepared / "experiment.json"
    metadata_path.write_text('{"test_accessed":false}', encoding="utf-8")
    run_dir = tmp_path / "runs" / "full" / "seed20260810"
    run_dir.mkdir(parents=True)
    artifacts = {
        "best.pt": b"checkpoint",
        "calibration.json": b"{}",
        "selection_report.json": b"{}",
    }
    for filename, contents in artifacts.items():
        (run_dir / filename).write_bytes(contents)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    summary_path = reports_dir / "selection_summary.json"
    summary_path.write_text(
        '{"status":"validation_passed","operational_gate":true}', encoding="utf-8"
    )
    lock = {
        "mode": "full_dataset",
        "status": "validation_passed",
        "operational_gate": True,
        "model_run": "runs/full/seed20260810",
        "seed": 20260810,
        "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "calibration_sha256": sha256_file(run_dir / "calibration.json"),
        "selection_report_sha256": sha256_file(run_dir / "selection_report.json"),
        "selection_summary_sha256": sha256_file(summary_path),
        "final_detector_complete_sha256": "d" * 64,
        "final_detector_checkpoint_sha256": "e" * 64,
    }
    (tmp_path / "model_lock.json").write_text(json.dumps(lock), encoding="utf-8")

    def crash_on_test_access(*_args, **kwargs):
        kwargs["before_test_access"]()
        assert json.loads(metadata_path.read_text(encoding="utf-8"))["test_accessed"] is True
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale.prepare_final_test_records",
        crash_on_test_access,
    )
    config = {"experiment": {"mode": "full_dataset"}}
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_final_test(Namespace(dataset_root=tmp_path, output_dir=tmp_path, resume=False), config)
    sealed = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert sealed["test_accessed"] is True
    assert sealed["test_access_started_at"]


def test_full_dataset_final_test_uses_locked_model_without_n_logic(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "experiment.json").write_text(
        json.dumps({"category_count": 2, "test_accessed": False}), encoding="utf-8"
    )
    run_dir = tmp_path / "runs" / "full" / "seed20260810"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"checkpoint")
    (run_dir / "calibration.json").write_text(
        json.dumps({"risk_control_satisfied": True}), encoding="utf-8"
    )
    (run_dir / "selection_report.json").write_text("{}", encoding="utf-8")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    summary_path = reports_dir / "selection_summary.json"
    summary_path.write_text(
        '{"status":"validation_passed","operational_gate":true}', encoding="utf-8"
    )
    lock = {
        "mode": "full_dataset",
        "status": "validation_passed",
        "operational_gate": True,
        "model_run": "runs/full/seed20260810",
        "seed": 20260810,
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "calibration_sha256": hashlib.sha256(
            (run_dir / "calibration.json").read_bytes()
        ).hexdigest(),
        "selection_report_sha256": hashlib.sha256(b"{}").hexdigest(),
        "selection_summary_sha256": sha256_file(summary_path),
        "final_detector_complete_sha256": "d" * 64,
        "final_detector_checkpoint_sha256": "e" * 64,
    }
    (tmp_path / "model_lock.json").write_text(json.dumps(lock), encoding="utf-8")

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            return None

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def device(value):
            return value

    class DummyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return index

    detector_report = {
        "test_annotation_sha256": "annotation-sha",
        "detector_checkpoint_sha256": "detector-sha",
        "image_count": 1,
        "normal_image_count": 1,
        "recapture_image_count": 0,
        "recapture_reasons": {},
        "ground_truth_count": 1,
        "matched_count": 1,
        "missed_count": 0,
        "unmatched_count": 0,
    }
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale.require_torch", lambda: FakeTorch()
    )

    def fake_prepare_final_test_records(*_args, **_kwargs):
        _kwargs["before_test_access"]()
        detector_report_path = tmp_path / "test" / "detector_report.json"
        detector_report_path.parent.mkdir(parents=True, exist_ok=True)
        detector_report_path.write_text(json.dumps(detector_report), encoding="utf-8")
        return ([{"sample_id": "test:1"}], detector_report)

    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale.prepare_final_test_records",
        fake_prepare_final_test_records,
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._build_cache",
        lambda _root, _cache, records, *_args, **_kwargs: records,
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale.RpcCachedDataset",
        lambda *_args, **_kwargs: DummyDataset(),
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._load_checkpoint_model",
        lambda *_args: (object(), {}),
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._infer", lambda *_args: {"logits": []}
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._save_predictions", lambda *_args: None
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale.evaluate_logits",
        lambda *_args, **_kwargs: {
            "overall_top1_accuracy": 1.0,
            "overall_top3_accuracy": 1.0,
            "approved_precision": 1.0,
            "approved_count": 1,
            "unknown_matched_count": 1,
            "unknown_top3_accuracy": 1.0,
        },
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale._ground_truth_worker_outcomes",
        lambda *_args, **_kwargs: {},
    )
    config = {
        "experiment": {
            "mode": "full_dataset",
            "seeds": [20260810],
            "bootstrap_repetitions": 10,
        },
        "training": {"image_size": 8, "batch_size": 1, "workers": 0},
    }
    final = run_final_test(
        Namespace(dataset_root=tmp_path, output_dir=tmp_path, resume=False), config
    )
    assert final["model_run"] == "runs/full/seed20260810"
    assert final["seed"] == 20260810
    assert "selected_n" not in final


def test_summary_selects_smallest_operational_noninferior_condition(tmp_path):
    config = {
        "experiment": {
            "sample_sizes": [5, 20],
            "seeds": [1, 2, 3],
            "noninferiority_margin": 0.01,
            "validation_split_seed": 1,
            "bootstrap_repetitions": 20,
        }
    }
    (tmp_path / "prepared").mkdir()
    (tmp_path / "prepared" / "worker_gate_report.json").write_text(
        json.dumps(
            {
                "score_threshold": 0.5,
                "train_candidates": 400,
                "train_rejected": {},
                "validation_images": 20,
                "validation_normal_images": 19,
                "validation_recapture_images": 1,
                "validation_recapture_reasons": {"DETECTOR_NO_OBJECT": 1},
                "validation_missed_boxes": 1,
                "validation_unmatched_boxes": 0,
            }
        ),
        encoding="utf-8",
    )
    for sample_size, top1 in ((5, 0.985), (20, 0.99)):
        for seed in (1, 2, 3):
            run_dir = tmp_path / "runs" / f"n{sample_size}" / f"seed{seed}"
            run_dir.mkdir(parents=True)
            (run_dir / "best.pt").write_bytes(b"checkpoint")
            (run_dir / "complete.json").write_text('{"complete": true}', encoding="utf-8")
            (run_dir / "calibration.json").write_text(
                json.dumps({"risk_control_satisfied": True}), encoding="utf-8"
            )
            report = {
                "overall_top1_accuracy": top1,
                "overall_top3_accuracy": 1.0,
                "macro_top1_accuracy": top1,
                "macro_top3_accuracy": 1.0,
                "class_top1_min": top1 - 0.1,
                "class_top1_p05": top1 - 0.05,
                "per_class_top3": [1.0, 1.0],
                "difficulty": {
                    "easy": {"top1_accuracy": top1, "top3_accuracy": 1.0},
                    "medium": {"top1_accuracy": top1, "top3_accuracy": 1.0},
                    "hard": {"top1_accuracy": top1, "top3_accuracy": 1.0},
                },
                "top1_cluster_bootstrap_95ci": [top1 - 0.01, top1 + 0.01],
                "approved_precision": 1.0,
                "approval_coverage": 0.9,
                "unknown_top3_accuracy": 1.0,
            }
            (run_dir / "selection_report.json").write_text(json.dumps(report), encoding="utf-8")
            np.savez(
                run_dir / "selection_predictions.npz",
                logits=np.asarray([[4.0, 0.0], [0.0, 4.0]]),
                targets=np.asarray([0, 1]),
                groups=np.asarray(["a", "b"]),
            )
    summary = summarize(Namespace(output_dir=tmp_path), config)
    assert summary["selected_n"] == 5
    assert summary["conditions"][0]["macro_top3"]["mean"] == 1.0
    assert len(summary["conditions"][0]["top1_hierarchical_bootstrap_95ci"]) == 2
    assert json.loads((tmp_path / "selected_n.json").read_text(encoding="utf-8"))["selected_n"] == 5
    markdown = (tmp_path / "reports" / "selection_summary.md").read_text(encoding="utf-8")
    assert "데이터 규모 실험" in markdown
    assert "�" not in markdown
