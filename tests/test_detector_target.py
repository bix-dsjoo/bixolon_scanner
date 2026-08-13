from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from bixolon_scanner.training import detector_target
from bixolon_scanner.training.rpc_worker_gate import _prediction_fold


def test_held_out_prediction_fold_normalizes_null_to_sentinel():
    assert _prediction_fold({"fold": None}) == -1
    assert _prediction_fold({"fold": 2}) == 2
    assert _prediction_fold({"prediction_fold": 1, "fold": None}) == 1


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _metric(*, detector_u95=0.004, e2e_u95=0.004, silent=0, approved_errors=0):
    return {
        "detector_pass_risk_upper_95": detector_u95,
        "e2e_approved_risk_upper_95": e2e_u95,
        "gate_table": {"silent_failure": silent},
        "approved_error_count": approved_errors,
        "unknown_top3_accuracy": 0.95,
        "error_catch_recall": 1.0,
    }


def test_checked_in_025_config_has_locked_grid_and_seeds():
    config = detector_target._load_config(Path("configs/detector_target_0.2.5.json"))
    assert config["experiment"]["seeds"] == [20260812, 20260813, 20260814]
    assert config["policy_grid"]["score_thresholds"][0] == 0.05
    assert config["policy_grid"]["score_thresholds"][-1] == 0.95
    assert config["training"]["nms_iou_threshold"] == 0.7
    assert config["promotion"]["manual_waiver_allowed"] is False


def test_prepare_records_missing_independence_as_non_promotion_evidence(tmp_path):
    manifests = {}
    for offset, name in enumerate(("natural", "hard", "shift"), start=1):
        directory = tmp_path / name
        manifest = directory / "manifest.jsonl"
        row = {
            "record_type": "detection",
            "split": "development",
            "image_id": offset,
            "image_path": f"{name}.jpg",
            "width": 100,
            "height": 100,
            "annotations": [
                {
                    "annotation_id": offset,
                    "category_id": 1,
                    "bbox_xywh": [10, 10, 30, 30],
                }
            ],
        }
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        _write(directory / "metadata.json", {"dataset_version": f"{name}-v1"})
        manifests[name] = manifest
    args = Namespace(
        training_manifest=manifests["natural"],
        natural_manifest=manifests["natural"],
        hard_manifest=manifests["hard"],
        shift_manifest=manifests["shift"],
        output_dir=tmp_path / "output",
    )

    detector_target.prepare(args, {})

    audit = json.loads((args.output_dir / "prepared" / "audit.json").read_text(encoding="utf-8"))
    assert audit["test_accessed"] is False
    assert audit["sets"]["natural"]["promotion_evidence_ready"] is False


def test_train_resume_does_not_skip_an_incomplete_fold(monkeypatch, tmp_path):
    output = tmp_path / "output"
    fold = output / "detector" / "seed-20260812" / "fold-0"
    _write(fold / "history.json", [{"epoch": 1}])
    (fold / "training_progress.pt").write_bytes(b"checkpoint")
    calls = []
    monkeypatch.setattr(detector_target, "train_detector", calls.append)
    args = Namespace(
        output_dir=output,
        resume=True,
        training_manifest=tmp_path / "manifest.jsonl",
        training_dataset_root=tmp_path,
        detector_image_cache=None,
        cpu=True,
    )
    config = {
        "experiment": {"seeds": [20260812], "fold_count": 1},
        "training": {
            "pretrained_name": "checkpoint",
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
        },
        "policy_grid": {"max_queries": 300, "min_object_area_ratio": 0.005},
        "selection": {"maximum_risk_upper_95": 0.005},
    }

    detector_target.train(args, config)

    assert len(calls) == 1


def test_legacy_recipe_compatibility_only_accepts_neutral_provenance_fields():
    current = {
        "optimizer": "AdamW",
        "base_learning_rate": 1e-5,
        "freeze_mode": "none",
        "frozen_modules_eval": False,
        "skip_epoch_validation": False,
        "workers": 4,
    }
    legacy = {
        "optimizer": "AdamW",
        "base_learning_rate": 1e-5,
    }

    assert detector_target._legacy_recipe_is_neutral_extension(legacy, current)
    assert not detector_target._legacy_recipe_is_neutral_extension(
        legacy | {"base_learning_rate": 2e-5}, current
    )
    assert not detector_target._legacy_recipe_is_neutral_extension(
        legacy, current | {"freeze_mode": "classification_heads_only"}
    )


def test_classifier_cache_keeps_union_of_every_locked_nms_policy():
    prediction = {
        "boxes_xyxy": [
            [0, 0, 10, 10],
            [3, 0, 13, 10],
            [3.5, 0, 13.5, 10],
        ],
        "scores": [0.9, 0.8, 0.7],
    }

    assert detector_target._classification_candidate_indices(prediction, [0.8]) == [
        0,
        1,
    ]
    assert detector_target._classification_candidate_indices(prediction, [0.5, 0.8]) == [0, 1, 2]


def test_parity_combines_strict_classifier_and_detector_provider_evidence(monkeypatch, tmp_path):
    output = tmp_path / "output"
    reports = output / "reports"
    package_hash = "a" * 64
    required = (
        "pytorch_cpu_tolerance",
        "pytorch_cuda_tolerance",
        "cpu_cuda_tolerance",
        "top1_equal",
        "top3_set_and_order_equal",
        "final_state_equal",
    )
    classifier = tmp_path / "classifier.json"
    cpu = tmp_path / "cpu.json"
    cuda = tmp_path / "cuda.json"
    _write(
        classifier,
        {
            "checks": {name: True for name in required},
            "package_artifact_sha256": {"metadata.json": package_hash},
        },
    )
    for path, provider in ((cpu, "cpu"), (cuda, "cuda")):
        _write(
            path,
            {
                "provider": provider,
                "passes": True,
                "detector": {"passes": True},
                "package_artifact_sha256": {"metadata.json": package_hash},
            },
        )
    monkeypatch.setattr(detector_target, "verify_lock", lambda args: {})

    detector_target.parity(
        Namespace(
            output_dir=output,
            parity_report=[classifier, cpu, cuda],
        ),
        {},
    )

    report = json.loads((reports / "parity-gate.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["checks"] == {name: True for name in required}


def test_finalize_never_uses_manual_waiver(monkeypatch, tmp_path):
    output = tmp_path / "output"
    reports = output / "reports"
    natural = _metric(detector_u95=0.006)
    _write(
        reports / "locked-test.json",
        {
            "data_evidence_ready": True,
            "sets": {
                "natural": {"metrics": natural},
                "hard": {"metrics": _metric()},
                "shift": {"metrics": _metric()},
            },
        },
    )
    _write(reports / "parity-gate.json", {"passed": True})
    _write(reports / "benchmark-gate.json", {"passed": True})
    monkeypatch.setattr(
        detector_target,
        "verify_lock",
        lambda args: {"lock_sha256": "a" * 64},
    )

    detector_target.finalize(
        Namespace(output_dir=output),
        {"promotion": {"maximum_full_path_p95_ms": 100.0}},
    )

    report = json.loads((reports / "final-promotion.json").read_text(encoding="utf-8"))
    assert report["promotion_status"] == "experiment_only"
    assert report["manual_waiver_allowed"] is False
    assert report["failures"] == ["detector_pass_risk_u95"]
