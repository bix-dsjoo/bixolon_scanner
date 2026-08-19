from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bixolon_scanner.cli import COMMANDS
from bixolon_scanner.training.pipeline_contract import (
    PIPELINE_STAGES,
    ArtifactLock,
    PipelineContractError,
    TrainingPipelineContract,
    canonical_contract_sha256,
    stage_ledger_sha256,
    verify_pipeline_contract,
)

REVISION = "a" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _onnx(
    path: Path, component: str
) -> tuple[dict[str, list[int | str]], dict[str, list[int | str]]]:
    import onnx
    from onnx import TensorProto, helper

    path.parent.mkdir(parents=True, exist_ok=True)
    if component == "detector":
        inputs = {"pixel_values": ["batch", 3, 640, 640]}
        outputs = {"logits": ["batch", 300, 20], "pred_boxes": ["batch", 300, 4]}
    else:
        inputs = {"pixel_values": ["batch", 3, 224, 224], "view_affine": ["batch", 2, 3]}
        outputs = {"logits": ["batch", 20]}
    nodes = []
    for name, shape in outputs.items():
        fixed_shape = [1 if isinstance(value, str) else value for value in shape]
        tensor = helper.make_tensor(
            name=f"{name}_value",
            data_type=TensorProto.FLOAT,
            dims=fixed_shape,
            vals=[0.0] * int(__import__("math").prod(fixed_shape)),
        )
        nodes.append(helper.make_node("Constant", inputs=[], outputs=[name], value=tensor))
    graph = helper.make_graph(
        nodes,
        component,
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in inputs.items()
        ],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in outputs.items()
        ],
    )
    onnx.save(helper.make_model(graph), path)
    return inputs, outputs


def _repository(tmp_path: Path, component: str) -> tuple[dict[str, object], dict[str, Path]]:
    dataset_root = tmp_path / "datasets" / "bread_dataset"
    manifest_dir = tmp_path / "manifests" / "selected"
    manifest_dir.mkdir(parents=True)
    training_rows = []
    for index in range(2):
        image = dataset_root / "single_objects_3" / "bread_01" / f"image-{index}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image-{index}".encode())
        training_rows.append(
            {
                "image_path": image.relative_to(dataset_root).as_posix(),
                "image_sha256": _sha(image),
            }
        )
    manifest = manifest_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in training_rows), encoding="utf-8")
    evaluation = manifest_dir / "evaluation_manifest.jsonl"
    evaluation.write_text(
        json.dumps({"image_path": "multi_object_scenes/x.jpg", "training_allowed": False}) + "\n",
        encoding="utf-8",
    )
    _json(
        manifest_dir / "metadata.json",
        {
            "dataset_version": "bread-test-1",
            "manifest_sha256": _sha(manifest),
            "training_contract": {
                "allowed_directory": "single_objects_3",
                "original_image_count": 2,
                "derived_evaluation_images_are_training_forbidden": True,
            },
            "evaluation_sets": {
                "multi_object_scenes": {"training_allowed": False},
            },
        },
    )

    artifact_dir = tmp_path / "artifacts"
    checkpoint = artifact_dir / f"{component}.pt"
    checkpoint.parent.mkdir(parents=True)
    if component == "classifier":
        import torch

        torch.save(
            {
                "dataset_version": "bread-test-1",
                "manifest_sha256": _sha(manifest),
                "source_revision": REVISION,
                "source_weight_sha256": "e" * 64,
                "image_size": 224,
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "contrastive_weight": 0.1,
                    "contrastive_temperature": 0.1,
                    "seed": 2,
                },
                "challenger": {
                    "scope": "backbone.stages[-1]",
                    "learning_rate": 0.00001,
                    "l2_sp_weight": 0.001,
                },
            },
            checkpoint,
        )
    else:
        checkpoint.write_bytes(b"checkpoint")
    package = artifact_dir / "package"
    detector_inputs, detector_outputs = _onnx(package / "detector.onnx", "detector")
    classifier_inputs, classifier_outputs = _onnx(package / "classifier.onnx", "classifier")
    current_onnx = package / f"{component}.onnx"
    parity = artifact_dir / f"{component}-parity.json"
    _json(parity, {"passes": True})
    evidence_source = artifact_dir / "evidence.json"
    _json(evidence_source, {"passes": True})
    run_evidence = artifact_dir / f"{component}-run.json"
    _json(
        run_evidence,
        {
            "component": component,
            "pipeline_version": "1.0.0",
            "provenance_mode": "recovered",
            "stages": [
                {
                    "stage": stage,
                    "passes": True,
                    "artifact": {
                        "path": "artifacts/evidence.json",
                        "sha256": _sha(evidence_source),
                    },
                }
                for stage in PIPELINE_STAGES
            ],
        },
    )
    selection = manifest_dir / "classifier_selection.jsonl"
    selection.write_text(
        "".join(
            json.dumps({"group_id": f"group-{fold}", "fold": fold, "target": 0, "image_id": fold})
            + "\n"
            for fold in range(3)
        ),
        encoding="utf-8",
    )
    source = {
        "architecture": "test",
        "revision": REVISION,
        "weight_filename": checkpoint.name,
        "weight_sha256": _sha(checkpoint),
    }
    _json(
        package / "metadata.json",
        {
            "schema_version": "1.1",
            "worker_version": "1.0.0",
            "promotion_status": "production",
            "dataset_version": "bread-test-1",
            "detector": {
                "filename": "detector.onnx",
                "version": "1.0.0",
                "score_threshold": 0.5,
                "nms_iou_threshold": 0.7,
                "max_queries": 300,
            },
            "classifier": {
                "filename": "classifier.onnx",
                "version": "1.0.0",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "approval_threshold": 0.9,
                "temperature": 1.0,
                "labels": [{"class_id": "bread_01", "class_name": "Bread"}],
            },
            "quality": {},
            "checksums": {
                "detector.onnx": _sha(package / "detector.onnx"),
                "classifier.onnx": _sha(package / "classifier.onnx"),
            },
            "licenses": {"detector": "test", "classifier": "test"},
            "sources": {component: source},
            "promotion": {
                "decision": "approved",
                "method": "all_gates",
                "decided_on": "2026-08-14",
            },
        },
    )
    common = {
        "schema_version": "1.1",
        "component": component,
        "pipeline_version": "1.0.0",
        "lifecycle": "promoted",
        "dataset": {
            "root_path": "datasets/bread_dataset",
            "metadata_path": "manifests/selected/metadata.json",
            "manifest_path": "manifests/selected/manifest.jsonl",
            "evaluation_manifest_path": "manifests/selected/evaluation_manifest.jsonl",
            "dataset_version": "bread-test-1",
            "training_directory": "single_objects_3",
            "original_image_count": 2,
            "forbidden_sources": ["multi_object_scenes"],
        },
        "pretrained": {
            "repository": "test/repository",
            "revision": REVISION,
            "weight_sha256": "e" * 64,
        },
        "stages": list(PIPELINE_STAGES),
        "test_access_policy": "forbidden_until_cpu_cuda_parity",
        "run_evidence": {
            "path": f"artifacts/{component}-run.json",
            "sha256": _sha(run_evidence),
            "provenance_mode": "recovered",
        },
        "checkpoint": {"path": f"artifacts/{component}.pt", "sha256": _sha(checkpoint)},
        "onnx": {
            "path": f"artifacts/package/{component}.onnx",
            "sha256": _sha(current_onnx),
            "inputs": detector_inputs if component == "detector" else classifier_inputs,
            "outputs": detector_outputs if component == "detector" else classifier_outputs,
            "dynamic_batch": True,
        },
        "parity": {
            "path": f"artifacts/{component}-parity.json",
            "sha256": _sha(parity),
            "passes_key": "passes",
        },
        "package": {
            "path": "artifacts/package",
            "model_filename": f"{component}.onnx",
            "model_version": "1.0.0",
            "schema_policy": "legacy_external_lock",
        },
        "output_schema": {"status": "locked"},
    }
    if component == "detector":
        synthetic = artifact_dir / "synthetic"
        synthetic.mkdir()
        rows = [
            {
                "image_id": fold + 1,
                "fold": fold,
                "source_dataset": "bread-test-1",
                "annotations": [] if fold == 0 else [{"category_id": 1}],
            }
            for fold in range(3)
        ]
        synthetic_manifest = synthetic / "manifest.jsonl"
        synthetic_manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        synthetic_metadata = synthetic / "metadata.json"
        _json(
            synthetic_metadata,
            {
                "dataset_version": "bread-test-1",
                "synthetic_image_count": 3,
                "synthetic_empty_image_count": 1,
                "synthetic_annotation_count": 2,
                "seed": 7,
                "manifest_sha256": _sha(synthetic_manifest),
            },
        )
        coco = synthetic / "coco.json"
        _json(
            coco,
            {
                "source_manifest": "artifacts/synthetic/manifest.jsonl",
                "training_image_count": 2,
                "validation_image_count": 1,
                "training_annotation_count": 1,
                "validation_annotation_count": 1,
                "evaluation_images_used": False,
            },
        )
        train_coco = synthetic / "instances_train.json"
        validation_coco = synthetic / "instances_validation.json"
        _json(
            train_coco,
            {
                "images": [{"id": 1}, {"id": 2}],
                "annotations": [{"image_id": 2}],
            },
        )
        _json(
            validation_coco,
            {"images": [{"id": 3}], "annotations": [{"image_id": 3}]},
        )
        training_config = synthetic / "train.yml"
        training_config.write_text("epochs: 1\n", encoding="utf-8")
        common["recipe"] = {
            "kind": "detector",
            "architecture": "D-FINE-N HGNetv2",
            "input_size": [640, 640],
            "synthetic_image_count": 3,
            "synthetic_empty_image_count": 1,
            "synthetic_annotation_count": 2,
            "synthetic_seed": 7,
            "synthetic_metadata": {
                "path": "artifacts/synthetic/metadata.json",
                "sha256": _sha(synthetic_metadata),
            },
            "synthetic_manifest": {
                "path": "artifacts/synthetic/manifest.jsonl",
                "sha256": _sha(synthetic_manifest),
            },
            "coco_provenance": {"path": "artifacts/synthetic/coco.json", "sha256": _sha(coco)},
            "coco_train_annotations": {
                "path": "artifacts/synthetic/instances_train.json",
                "sha256": _sha(train_coco),
            },
            "coco_validation_annotations": {
                "path": "artifacts/synthetic/instances_validation.json",
                "sha256": _sha(validation_coco),
            },
            "training_config": {
                "path": "artifacts/synthetic/train.yml",
                "sha256": _sha(training_config),
            },
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "head_learning_rate_multiplier": 2.0,
            "weight_decay": 0.0,
        }
        common["selection"] = {
            "kind": "detector",
            "split": "group_aware_synthetic_validation",
            "group_key": "capture_session_id",
            "folds": 3,
            "validation_fold": 2,
            "selection_source": "validation_only",
            "benchmark_source": "multi_object_scenes",
            "independent_test_status": "pending_user_images",
        }
    else:
        common["recipe"] = {
            "kind": "classifier",
            "architecture": "DINOv3 ConvNeXt-Tiny",
            "input_size": [224, 224],
            "class_count": 20,
            "original_image_count": 2,
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "contrastive_weight": 0.1,
            "contrastive_temperature": 0.1,
            "seeds": [1, 2, 3],
            "challenger": "last_stage_l2sp",
            "finetune_scope": "backbone.stages[-1]",
            "challenger_learning_rate": 0.00001,
            "l2_sp_weight": 0.001,
            "export_policy": "staged_affine_view_tta",
            "final_views": ["base"],
            "top3_views": ["base"],
            "top3_aggregation": "top3_vote",
        }
        common["selection"] = {
            "kind": "classifier",
            "assignment": {
                "path": "manifests/selected/classifier_selection.jsonl",
                "sha256": _sha(selection),
            },
            "group_key": "group_id",
            "fold_key": "fold",
            "target_key": "target",
            "image_id_key": "image_id",
            "folds": 3,
            "selection_source": "multi_object_scenes_development_roi",
            "selected_recipe": "last_stage_l2sp",
            "selected_seed": 2,
            "mean_validation_top1": 0.9,
            "benchmark_source": "multi_object_scenes",
            "independent_test_status": "pending_user_images",
        }
    return common, {"checkpoint": checkpoint, "selection": selection, "run_evidence": run_evidence}


@pytest.mark.parametrize("component", ["detector", "classifier"])
def test_typed_contract_verifies_locked_artifact_chain(tmp_path: Path, component: str) -> None:
    raw, _ = _repository(tmp_path, component)
    result = verify_pipeline_contract(
        TrainingPipelineContract.model_validate(raw), repository_root=tmp_path
    )
    assert result["passed"] is True
    assert result["onnx"]["dynamic_batch"] is True


def test_classifier_selection_rejects_group_leakage(tmp_path: Path) -> None:
    raw, paths = _repository(tmp_path, "classifier")
    paths["selection"].write_text(
        "\n".join(
            json.dumps({"group_id": "same", "fold": fold, "target": 0, "image_id": fold})
            for fold in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    raw["selection"]["assignment"]["sha256"] = _sha(paths["selection"])
    with pytest.raises(PipelineContractError, match="leakage"):
        verify_pipeline_contract(
            TrainingPipelineContract.model_validate(raw), repository_root=tmp_path
        )


def test_original_image_byte_drift_is_rejected(tmp_path: Path) -> None:
    raw, _ = _repository(tmp_path, "classifier")
    image = next((tmp_path / "datasets" / "bread_dataset").rglob("*.jpg"))
    image.write_bytes(b"changed")
    with pytest.raises(PipelineContractError, match="training image checksum mismatch"):
        verify_pipeline_contract(
            TrainingPipelineContract.model_validate(raw), repository_root=tmp_path
        )


def test_recovered_classifier_accepts_historical_recipe_field_names(tmp_path: Path) -> None:
    import torch

    raw, paths = _repository(tmp_path, "classifier")
    payload = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    payload["training_config"] = payload.pop("training")
    payload["challenger"]["trainable_scope"] = payload["challenger"].pop("scope")
    torch.save(payload, paths["checkpoint"])
    raw["checkpoint"]["sha256"] = _sha(paths["checkpoint"])
    metadata_path = tmp_path / "artifacts" / "package" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sources"]["classifier"]["weight_sha256"] = _sha(paths["checkpoint"])
    _json(metadata_path, metadata)

    result = verify_pipeline_contract(
        TrainingPipelineContract.model_validate(raw), repository_root=tmp_path
    )
    assert result["checkpoint"]["provenance"]["mode"] == "recovered"


def test_dry_run_is_not_a_passed_artifact_verification(tmp_path: Path) -> None:
    raw, paths = _repository(tmp_path, "classifier")
    paths["checkpoint"].unlink()
    result = verify_pipeline_contract(
        TrainingPipelineContract.model_validate(raw), repository_root=tmp_path, dry_run=True
    )
    assert result["verification_scope"] == "contract_and_data_only"
    assert result["passed"] is None


def test_native_evidence_rejects_noncanonical_stage_hash(tmp_path: Path) -> None:
    raw, paths = _repository(tmp_path, "detector")
    artifact = ArtifactLock(
        path="artifacts/evidence.json", sha256=_sha(tmp_path / "artifacts" / "evidence.json")
    )
    previous = None
    stages = []
    for stage in PIPELINE_STAGES:
        stage_hash = stage_ledger_sha256(
            stage=stage,
            passes=True,
            artifact=artifact,
            previous_stage_sha256=previous,
        )
        stages.append(
            {
                "stage": stage,
                "passes": True,
                "artifact": artifact.model_dump(mode="json"),
                "previous_stage_sha256": previous,
                "stage_sha256": stage_hash,
            }
        )
        previous = stage_hash
    stages[-1]["stage_sha256"] = "f" * 64
    _json(
        paths["run_evidence"],
        {
            "component": "detector",
            "pipeline_version": "1.0.0",
            "provenance_mode": "native",
            "stages": stages,
        },
    )
    raw["run_evidence"] = {
        "path": "artifacts/detector-run.json",
        "sha256": _sha(paths["run_evidence"]),
        "provenance_mode": "native",
    }
    with pytest.raises(PipelineContractError, match="not canonical"):
        verify_pipeline_contract(
            TrainingPipelineContract.model_validate(raw), repository_root=tmp_path
        )


def test_component_recipe_types_cannot_be_swapped(tmp_path: Path) -> None:
    detector, _ = _repository(tmp_path, "detector")
    detector["component"] = "classifier"
    with pytest.raises(ValidationError, match="must match"):
        TrainingPipelineContract.model_validate(detector)


def test_contract_hashes_are_component_independent(tmp_path: Path) -> None:
    detector, _ = _repository(tmp_path / "detector", "detector")
    classifier, _ = _repository(tmp_path / "classifier", "classifier")
    left = TrainingPipelineContract.model_validate(detector)
    right = TrainingPipelineContract.model_validate(classifier)
    assert canonical_contract_sha256(left) != canonical_contract_sha256(right)


def test_unified_cli_exposes_pipeline_and_release_verification() -> None:
    assert ("train", "verify-pipeline") in COMMANDS
    assert ("release", "verify") in COMMANDS
