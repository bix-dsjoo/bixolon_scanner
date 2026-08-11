from __future__ import annotations

import hashlib
import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bixolon_scanner.training.rpc_data_scale import (
    _balanced_training_order,
    _crop,
    _ground_truth_worker_outcomes,
    _visual_farthest_order,
    _load_stage_progress,
    _run_complete,
    _save_stage_progress,
    _validation_partition,
    evaluate_logits,
    prepare,
    summarize,
    test_selected as run_final_test,
    train_all,
)
from bixolon_scanner.training.rpc_worker_gate import (
    assign_oof_folds,
    postprocess_worker_gate,
)


def _write_coco(root: Path, split: str, categories: list[dict], images: list[dict], annotations: list[dict]):
    (root / f"{split}2019").mkdir(parents=True, exist_ok=True)
    (root / f"instances_{split}2019.json").write_text(
        json.dumps({"categories": categories, "images": images, "annotations": annotations}),
        encoding="utf-8",
    )
    for image in images:
        canvas = Image.new("RGB", (image["width"], image["height"]), "white")
        canvas.save(root / f"{split}2019" / image["file_name"])


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
        "annotations": [
            {"bbox_xywh": [20, 20, 40, 40], "category_id": 1, "annotation_id": 1}
        ],
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
    outcomes = _ground_truth_worker_outcomes(
        classifier_report, detector_report, role="selection"
    )
    assert outcomes["denominator"] == 11
    assert sum(outcomes["counts"].values()) == 11
    assert outcomes["counts"]["detector_recapture"] == 2
    assert outcomes["counts"]["classifier_border_recapture"] == 3


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
                {"id": image_id, "image_id": image_id, "category_id": category, "bbox": [2, 2, 10, 10]}
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
                {"id": annotation_id, "image_id": image_id, "category_id": category, "bbox": [2, 2, 10, 10]}
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
    (output / "detector").mkdir(parents=True)
    (output / "detector" / "complete.json").write_text('{"complete": true}', encoding="utf-8")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")

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
        lambda records, *_: (np.eye(len(records), dtype=np.float32), [str(index) for index in range(len(records))]),
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

    full_output = tmp_path / "full_output"
    (full_output / "detector").mkdir(parents=True)
    (full_output / "detector" / "complete.json").write_text(
        '{"complete": true}', encoding="utf-8"
    )
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


def test_full_dataset_summary_locks_the_single_model_without_selected_n(tmp_path):
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
    (prepared / "worker_gate_report.json").write_text(
        json.dumps(worker_gate), encoding="utf-8"
    )
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
        json.dumps({"risk_control_satisfied": True}), encoding="utf-8"
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
    (run_dir / "run.json").write_text(
        json.dumps({"train_sample_count": 7}), encoding="utf-8"
    )
    np.savez(
        run_dir / "selection_predictions.npz",
        logits=np.asarray([[4.0, 0.0], [0.0, 4.0]]),
        targets=np.asarray([0, 1]),
    )
    (run_dir / "complete.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    config = {"experiment": {"mode": "full_dataset", "seeds": [20260810]}}
    summary = summarize(Namespace(output_dir=tmp_path), config)
    assert summary["status"] == "validation_passed"
    assert summary["model_run"] == "runs/full/seed20260810"
    assert "selected_n" not in summary
    lock = json.loads((tmp_path / "model_lock.json").read_text(encoding="utf-8"))
    assert lock["model_run"] == "runs/full/seed20260810"
    assert "selected_n" not in lock
    assert not (tmp_path / "selected_n.json").exists()


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
    lock = {
        "model_run": "runs/full/seed20260810",
        "seed": 20260810,
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "calibration_sha256": hashlib.sha256(
            (run_dir / "calibration.json").read_bytes()
        ).hexdigest(),
        "selection_report_sha256": hashlib.sha256(b"{}").hexdigest(),
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
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_data_scale.prepare_final_test_records",
        lambda *_args, **_kwargs: ([{"sample_id": "test:1"}], detector_report),
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
