from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..contracts.model_package import SEMVER, load_model_package, sha256_file

PIPELINE_STAGES = (
    "dataset_audit",
    "group_split",
    "train",
    "validation_selection",
    "checkpoint_lock",
    "onnx_export",
    "cpu_cuda_parity",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


class PipelineContractError(ValueError):
    """Raised when a versioned training pipeline contract cannot be verified."""


class ArtifactLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("artifact sha256 must be a lowercase SHA-256 digest")
        return value


class DatasetLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_path: str = "datasets/bread_dataset"
    metadata_path: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    evaluation_manifest_path: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    training_directory: str = Field(min_length=1)
    original_image_count: int = Field(gt=0)
    forbidden_sources: list[Literal["multi_object_scenes", "scan_log_samples"]]

    @field_validator("forbidden_sources")
    @classmethod
    def validate_forbidden_sources(cls, value: list[str]) -> list[str]:
        if set(value) != {"multi_object_scenes", "scan_log_samples"}:
            raise ValueError("both evaluation sources must be forbidden from training")
        return value


class PretrainedLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository: str = Field(min_length=1)
    revision: str
    weight_sha256: str
    weight: ArtifactLock | None = None

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if not REVISION_PATTERN.fullmatch(value):
            raise ValueError("pretrained revision must be a pinned 40-character commit SHA")
        return value

    @field_validator("weight_sha256")
    @classmethod
    def validate_weight_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("pretrained weight must have a lowercase SHA-256 digest")
        return value


class DetectorRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["detector"]
    architecture: Literal["D-FINE-N HGNetv2"]
    input_size: tuple[int, int]
    synthetic_image_count: int = Field(gt=0)
    synthetic_empty_image_count: int = Field(ge=0)
    synthetic_annotation_count: int = Field(gt=0)
    synthetic_seed: int
    synthetic_metadata: ArtifactLock
    synthetic_manifest: ArtifactLock
    coco_provenance: ArtifactLock
    coco_train_annotations: ArtifactLock
    coco_validation_annotations: ArtifactLock
    training_config: ArtifactLock
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    head_learning_rate_multiplier: float = Field(gt=0)
    weight_decay: float = Field(ge=0)


class ClassifierRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["classifier"]
    architecture: Literal["DINOv3 ConvNeXt-Tiny"]
    input_size: tuple[int, int]
    class_count: int = Field(gt=1)
    original_image_count: int = Field(gt=0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    contrastive_weight: float = Field(ge=0)
    contrastive_temperature: float = Field(gt=0)
    seeds: list[int] = Field(min_length=1)
    challenger: Literal["last_stage_l2sp"]
    finetune_scope: Literal["backbone.stages[-1]"]
    challenger_learning_rate: float = Field(gt=0)
    l2_sp_weight: float = Field(gt=0)
    export_policy: Literal["staged_affine_view_tta"]
    final_views: list[str] = Field(min_length=1)
    top3_views: list[str] = Field(min_length=1)
    top3_aggregation: Literal["top3_vote"]


class DetectorSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["detector"]
    split: Literal["group_aware_synthetic_validation"]
    group_key: str = Field(min_length=1)
    folds: int = Field(ge=2)
    validation_fold: int = Field(ge=0)
    selection_source: Literal["validation_only", "multi_object_scenes_development_threshold"]
    selection_benchmark_overlap: bool = False
    benchmark_source: Literal["multi_object_scenes"]
    independent_test_status: Literal["pending_user_images"]

    @model_validator(mode="after")
    def validate_fold(self) -> "DetectorSelection":
        if self.validation_fold >= self.folds:
            raise ValueError("validation_fold must be inside the declared fold range")
        if (
            self.selection_source == "multi_object_scenes_development_threshold"
        ) != self.selection_benchmark_overlap:
            raise ValueError(
                "development threshold recovery must explicitly record benchmark overlap"
            )
        return self


class ClassifierSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["classifier"]
    assignment: ArtifactLock
    group_key: Literal["group_id"]
    fold_key: Literal["fold"]
    target_key: Literal["target"]
    image_id_key: Literal["image_id"]
    folds: Literal[3]
    selection_source: Literal["multi_object_scenes_development_roi"]
    selected_recipe: Literal["last_stage_l2sp"]
    selected_seed: int
    mean_validation_top1: float = Field(ge=0, le=1)
    benchmark_source: Literal["multi_object_scenes"]
    independent_test_status: Literal["pending_user_images"]


class OnnxLock(ArtifactLock):
    model_config = ConfigDict(extra="forbid")

    inputs: dict[str, list[int | str]]
    outputs: dict[str, list[int | str]]
    dynamic_batch: bool


class ParityLock(ArtifactLock):
    model_config = ConfigDict(extra="forbid")

    passes_key: str = Field(min_length=1)


class RunEvidenceLock(ArtifactLock):
    model_config = ConfigDict(extra="forbid")

    provenance_mode: Literal["native", "recovered"]


class DetectorSmokeLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["detector"]
    repository_path: str
    config_path: str


class ClassifierSmokeLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["classifier"]
    parity_tensors: ArtifactLock
    crop_scale: float = Field(gt=0, le=1)


class PackageLock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    model_filename: str = Field(min_length=1)
    model_version: str
    schema_policy: Literal["legacy_external_lock", "embedded_pipeline_lock"]

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, value: str) -> str:
        if not SEMVER.fullmatch(value):
            raise ValueError("model_version must use semantic versioning")
        return value


class TrainingPipelineContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.1"]
    component: Literal["detector", "classifier"]
    pipeline_version: str
    lifecycle: Literal["proposal", "active", "locked", "promoted", "rejected", "archived"]
    dataset: DatasetLock
    pretrained: PretrainedLock
    stages: list[str]
    test_access_policy: Literal["forbidden_until_cpu_cuda_parity"]
    recipe: DetectorRecipe | ClassifierRecipe
    selection: DetectorSelection | ClassifierSelection
    run_evidence: RunEvidenceLock
    smoke: DetectorSmokeLock | ClassifierSmokeLock | None = None
    checkpoint: ArtifactLock
    onnx: OnnxLock
    parity: ParityLock
    package: PackageLock
    output_schema: dict[str, Any]

    @field_validator("pipeline_version")
    @classmethod
    def validate_pipeline_version(cls, value: str) -> str:
        if not SEMVER.fullmatch(value):
            raise ValueError("pipeline_version must use semantic versioning")
        return value

    @model_validator(mode="after")
    def validate_component_contract(self) -> "TrainingPipelineContract":
        if tuple(self.stages) != PIPELINE_STAGES:
            raise ValueError(f"stages must exactly match {PIPELINE_STAGES}")
        if self.recipe.kind != self.component or self.selection.kind != self.component:
            raise ValueError("component, recipe, and selection kinds must match")
        if self.smoke is not None and self.smoke.kind != self.component:
            raise ValueError("component and smoke kinds must match")
        if self.lifecycle == "promoted" and self.package.schema_policy not in {
            "legacy_external_lock",
            "embedded_pipeline_lock",
        }:
            raise ValueError("promoted contracts require a package provenance policy")
        if not self.output_schema:
            raise ValueError("output_schema cannot be empty")
        return self


def canonical_contract_sha256(contract: TrainingPipelineContract) -> str:
    encoded = json.dumps(
        contract.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_pipeline_contract(path: Path) -> TrainingPipelineContract:
    try:
        return TrainingPipelineContract.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise PipelineContractError(f"invalid pipeline contract: {path}") from exc


def _repository_path(repository_root: Path, value: str) -> Path:
    root = repository_root.resolve()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PipelineContractError(f"contract path escapes repository: {value}") from exc
    return candidate


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise PipelineContractError(f"missing {label}: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError
        return value
    except Exception as exc:
        raise PipelineContractError(f"invalid {label}: {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise TypeError
                    records.append(value)
    except Exception as exc:
        raise PipelineContractError(f"invalid JSONL at {path}") from exc
    return records


def _verify_lock(lock: ArtifactLock, repository_root: Path, label: str) -> Path:
    path = _repository_path(repository_root, lock.path)
    _require_file(path, label)
    if sha256_file(path) != lock.sha256:
        raise PipelineContractError(f"{label} checksum mismatch")
    return path


def _verify_dataset(contract: TrainingPipelineContract, repository_root: Path) -> dict[str, Any]:
    dataset = contract.dataset
    data_root = _repository_path(repository_root, dataset.root_path)
    metadata_path = _repository_path(repository_root, dataset.metadata_path)
    manifest_path = _repository_path(repository_root, dataset.manifest_path)
    evaluation_path = _repository_path(repository_root, dataset.evaluation_manifest_path)
    metadata = _read_json(metadata_path, "dataset metadata")
    _require_file(manifest_path, "training manifest")
    _require_file(evaluation_path, "evaluation manifest")
    if metadata.get("dataset_version") != dataset.dataset_version:
        raise PipelineContractError("dataset version does not match the pipeline lock")
    training_contract = metadata.get("training_contract", {})
    for key, expected in {
        "allowed_directory": dataset.training_directory,
        "original_image_count": dataset.original_image_count,
        "derived_evaluation_images_are_training_forbidden": True,
    }.items():
        if training_contract.get(key) != expected:
            raise PipelineContractError(f"dataset training contract mismatch: {key}")
    manifest_sha = sha256_file(manifest_path)
    if metadata.get("manifest_sha256") != manifest_sha:
        raise PipelineContractError("training manifest checksum does not match dataset metadata")
    records = _read_jsonl(manifest_path)
    if len(records) != dataset.original_image_count:
        raise PipelineContractError("training manifest image count does not match the lock")
    allowed_prefix = f"{dataset.training_directory}/"
    for record in records:
        image_path = str(record.get("image_path", "")).replace("\\", "/")
        if not image_path.startswith(allowed_prefix):
            raise PipelineContractError("training manifest contains a non-canonical source")
        if any(source in image_path.split("/") for source in dataset.forbidden_sources):
            raise PipelineContractError("evaluation data appears in the training manifest")
        source_path = (data_root / image_path).resolve()
        try:
            source_path.relative_to(data_root.resolve())
        except ValueError as exc:
            raise PipelineContractError("training image escapes the dataset root") from exc
        _require_file(source_path, "training image")
        if sha256_file(source_path) != record.get("image_sha256"):
            raise PipelineContractError(f"training image checksum mismatch: {image_path}")
    evaluation_sets = metadata.get("evaluation_sets", {})
    for source in dataset.forbidden_sources:
        source_metadata = evaluation_sets.get(source)
        if (
            not isinstance(source_metadata, dict)
            or source_metadata.get("training_allowed") is not False
        ):
            raise PipelineContractError(f"evaluation source is not locked as forbidden: {source}")
    evaluation_records = _read_jsonl(evaluation_path)
    if not evaluation_records or any(
        row.get("training_allowed") is not False for row in evaluation_records
    ):
        raise PipelineContractError("evaluation manifest contains a training-enabled record")
    return {
        "manifest_sha256": manifest_sha,
        "training_records": len(records),
        "image_bytes_verified": len(records),
        "evaluation_records": len(evaluation_records),
    }


def _verify_selection(contract: TrainingPipelineContract, repository_root: Path) -> dict[str, Any]:
    selection = contract.selection
    if isinstance(selection, DetectorSelection):
        return {
            "kind": "detector",
            "folds": selection.folds,
            "validation_fold": selection.validation_fold,
            "group_key": selection.group_key,
        }
    path = _verify_lock(selection.assignment, repository_root, "classifier selection assignment")
    records = _read_jsonl(path)
    if not records:
        raise PipelineContractError("classifier selection assignment cannot be empty")
    group_folds: dict[str, set[int]] = {}
    seen_folds: set[int] = set()
    for row in records:
        try:
            group = str(row[selection.group_key])
            fold = int(row[selection.fold_key])
            int(row[selection.target_key])
            int(row[selection.image_id_key])
        except Exception as exc:
            raise PipelineContractError("classifier selection assignment schema mismatch") from exc
        group_folds.setdefault(group, set()).add(fold)
        seen_folds.add(fold)
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise PipelineContractError("classifier selection group leakage detected")
    if seen_folds != set(range(selection.folds)):
        raise PipelineContractError("classifier selection fold coverage mismatch")
    return {
        "kind": "classifier",
        "records": len(records),
        "groups": len(group_folds),
        "folds": sorted(seen_folds),
        "assignment_sha256": selection.assignment.sha256,
    }


def _verify_detector_recipe(
    recipe: DetectorRecipe,
    repository_root: Path,
    *,
    dataset_version: str,
    validation_fold: int,
) -> dict[str, Any]:
    metadata_path = _verify_lock(recipe.synthetic_metadata, repository_root, "synthetic metadata")
    manifest_path = _verify_lock(recipe.synthetic_manifest, repository_root, "synthetic manifest")
    provenance_path = _verify_lock(recipe.coco_provenance, repository_root, "COCO provenance")
    train_annotations_path = _verify_lock(
        recipe.coco_train_annotations, repository_root, "COCO training annotations"
    )
    validation_annotations_path = _verify_lock(
        recipe.coco_validation_annotations,
        repository_root,
        "COCO validation annotations",
    )
    _verify_lock(recipe.training_config, repository_root, "Detector training config")
    metadata = _read_json(metadata_path, "synthetic metadata")
    if metadata.get("dataset_version") != dataset_version:
        raise PipelineContractError("synthetic detector source dataset mismatch")
    for key, expected in {
        "synthetic_image_count": recipe.synthetic_image_count,
        "synthetic_empty_image_count": recipe.synthetic_empty_image_count,
        "synthetic_annotation_count": recipe.synthetic_annotation_count,
        "seed": recipe.synthetic_seed,
        "manifest_sha256": recipe.synthetic_manifest.sha256,
    }.items():
        if metadata.get(key) != expected:
            raise PipelineContractError(f"synthetic detector evidence mismatch: {key}")
    rows = _read_jsonl(manifest_path)
    if len(rows) != recipe.synthetic_image_count:
        raise PipelineContractError("synthetic detector manifest count mismatch")
    if sum(not row.get("annotations") for row in rows) != recipe.synthetic_empty_image_count:
        raise PipelineContractError("synthetic detector empty count mismatch")
    if sum(len(row.get("annotations", [])) for row in rows) != recipe.synthetic_annotation_count:
        raise PipelineContractError("synthetic detector annotation count mismatch")
    if any(row.get("source_dataset") is None for row in rows):
        raise PipelineContractError("synthetic detector source provenance is missing")
    provenance = _read_json(provenance_path, "COCO provenance")
    if provenance.get("source_manifest", "").replace("\\", "/") != recipe.synthetic_manifest.path:
        raise PipelineContractError("COCO provenance source manifest mismatch")
    if provenance.get("evaluation_images_used") is not False:
        raise PipelineContractError("evaluation images entered Detector training")
    if (
        provenance.get("training_image_count", 0) + provenance.get("validation_image_count", 0)
        != recipe.synthetic_image_count
    ):
        raise PipelineContractError("COCO split image count mismatch")
    train_payload = _read_json(train_annotations_path, "COCO training annotations")
    validation_payload = _read_json(validation_annotations_path, "COCO validation annotations")
    train_ids = {int(image["id"]) for image in train_payload.get("images", [])}
    validation_ids = {int(image["id"]) for image in validation_payload.get("images", [])}
    expected_validation_ids = {
        int(row["image_id"]) for row in rows if int(row["fold"]) == validation_fold
    }
    expected_train_ids = {int(row["image_id"]) for row in rows} - expected_validation_ids
    if train_ids != expected_train_ids or validation_ids != expected_validation_ids:
        raise PipelineContractError("COCO annotations do not match the group-aware fold split")
    for payload, ids, count_key, annotation_count_key in (
        (train_payload, train_ids, "training_image_count", "training_annotation_count"),
        (
            validation_payload,
            validation_ids,
            "validation_image_count",
            "validation_annotation_count",
        ),
    ):
        annotations = payload.get("annotations", [])
        if len(ids) != provenance.get(count_key) or len(annotations) != provenance.get(
            annotation_count_key
        ):
            raise PipelineContractError("COCO annotation counts do not match provenance")
        if any(int(annotation["image_id"]) not in ids for annotation in annotations):
            raise PipelineContractError("COCO annotation references the wrong split")
    return {
        "synthetic_images": len(rows),
        "synthetic_empty_images": recipe.synthetic_empty_image_count,
        "synthetic_annotations": recipe.synthetic_annotation_count,
    }


def _nested_value(payload: dict[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise PipelineContractError(f"evidence key is missing: {dotted_key}")
        value = value[key]
    return value


def stage_ledger_sha256(
    *,
    stage: str,
    passes: bool,
    artifact: ArtifactLock,
    previous_stage_sha256: str | None,
) -> str:
    """Return the canonical hash for one immutable native-run ledger entry."""
    payload = {
        "artifact": artifact.model_dump(mode="json"),
        "passes": passes,
        "previous_stage_sha256": previous_stage_sha256,
        "stage": stage,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _verify_run_evidence(
    contract: TrainingPipelineContract, repository_root: Path
) -> dict[str, Any]:
    path = _verify_lock(contract.run_evidence, repository_root, "run evidence")
    payload = _read_json(path, "run evidence")
    if (
        payload.get("component") != contract.component
        or payload.get("pipeline_version") != contract.pipeline_version
    ):
        raise PipelineContractError("run evidence component or version mismatch")
    if payload.get("provenance_mode") != contract.run_evidence.provenance_mode:
        raise PipelineContractError("run evidence provenance mode mismatch")
    stages = payload.get("stages")
    if not isinstance(stages, list) or [row.get("stage") for row in stages] != list(
        PIPELINE_STAGES
    ):
        raise PipelineContractError("run evidence stage order mismatch")
    previous: str | None = None
    for row in stages:
        if row.get("passes") is not True:
            raise PipelineContractError(f"run evidence stage did not pass: {row.get('stage')}")
        artifact = ArtifactLock.model_validate(row.get("artifact"))
        _verify_lock(artifact, repository_root, f"{row['stage']} evidence")
        if contract.run_evidence.provenance_mode == "native":
            if row.get("previous_stage_sha256") != previous:
                raise PipelineContractError("native stage ledger hash-chain is broken")
            declared = row.get("stage_sha256")
            if not isinstance(declared, str) or not SHA256_PATTERN.fullmatch(declared):
                raise PipelineContractError("native stage ledger hash is invalid")
            expected = stage_ledger_sha256(
                stage=str(row["stage"]),
                passes=True,
                artifact=artifact,
                previous_stage_sha256=previous,
            )
            if declared != expected:
                raise PipelineContractError("native stage ledger hash is not canonical")
            previous = declared
    return {
        "path": contract.run_evidence.path,
        "provenance_mode": contract.run_evidence.provenance_mode,
    }


def _verify_onnx(lock: OnnxLock, repository_root: Path) -> dict[str, Any]:
    path = _verify_lock(lock, repository_root, "ONNX")
    try:
        import onnx

        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise PipelineContractError("ONNX model validation failed") from exc

    def shape(value: Any) -> list[int | str]:
        result: list[int | str] = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param:
                result.append(dim.dim_param)
            elif dim.HasField("dim_value"):
                result.append(int(dim.dim_value))
            else:
                result.append("?")
        return result

    inputs = {value.name: shape(value) for value in model.graph.input}
    outputs = {value.name: shape(value) for value in model.graph.output}
    if inputs != lock.inputs or outputs != lock.outputs:
        raise PipelineContractError(
            f"ONNX input/output schema mismatch: inputs={inputs}, outputs={outputs}"
        )
    dynamic = all(
        bool(dims) and isinstance(dims[0], str) for dims in [*inputs.values(), *outputs.values()]
    )
    if dynamic != lock.dynamic_batch:
        raise PipelineContractError("ONNX dynamic batch contract mismatch")
    return {"sha256": lock.sha256, "inputs": inputs, "outputs": outputs, "dynamic_batch": dynamic}


def _verify_parity(lock: ParityLock, repository_root: Path) -> dict[str, Any]:
    path = _verify_lock(lock, repository_root, "parity report")
    report = _read_json(path, "parity report")
    if _nested_value(report, lock.passes_key) is not True:
        raise PipelineContractError("parity report does not pass")
    return {"path": lock.path, "sha256": lock.sha256, "passes": True}


def _verify_checkpoint_provenance(
    contract: TrainingPipelineContract,
    checkpoint_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    if contract.pretrained.weight is not None:
        pretrained_path = _verify_lock(
            contract.pretrained.weight, repository_root, "pretrained weight"
        )
        if sha256_file(pretrained_path) != contract.pretrained.weight_sha256:
            raise PipelineContractError("pretrained weight lock mismatch")
    if contract.component == "detector" and contract.run_evidence.provenance_mode == "recovered":
        return {
            "mode": contract.run_evidence.provenance_mode,
            "pretrained_weight_sha256": contract.pretrained.weight_sha256,
        }
    try:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise PipelineContractError(f"{contract.component} checkpoint cannot be inspected") from exc
    if not isinstance(payload, dict):
        raise PipelineContractError(f"{contract.component} checkpoint schema mismatch")
    if contract.component == "detector":
        if not isinstance(contract.recipe, DetectorRecipe):
            raise PipelineContractError("Detector recipe schema mismatch")
        provenance = payload.get("bixolon_training_provenance")
        if not isinstance(provenance, dict):
            raise PipelineContractError("native Detector checkpoint provenance is missing")
        expected = {
            "dataset_version": contract.dataset.dataset_version,
            "manifest_sha256": sha256_file(
                _repository_path(repository_root, contract.dataset.manifest_path)
            ),
            "source_revision": contract.pretrained.revision,
            "source_weight_sha256": contract.pretrained.weight_sha256,
            "synthetic_manifest_sha256": contract.recipe.synthetic_manifest.sha256,
            "coco_provenance_sha256": contract.recipe.coco_provenance.sha256,
            "training_config_sha256": contract.recipe.training_config.sha256,
            "epochs": contract.recipe.epochs,
            "batch_size": contract.recipe.batch_size,
            "backbone_learning_rate": contract.recipe.learning_rate,
            "head_learning_rate": (
                contract.recipe.learning_rate * contract.recipe.head_learning_rate_multiplier
            ),
            "weight_decay": contract.recipe.weight_decay,
            "synthetic_seed": contract.recipe.synthetic_seed,
        }
        for key, value in expected.items():
            if provenance.get(key) != value:
                raise PipelineContractError(f"Detector checkpoint provenance mismatch: {key}")
        return {
            "mode": "native",
            "source_revision": provenance["source_revision"],
            "source_weight_sha256": provenance["source_weight_sha256"],
            "manifest_sha256": provenance["manifest_sha256"],
        }
    if not isinstance(contract.recipe, ClassifierRecipe):
        raise PipelineContractError("classifier checkpoint schema mismatch")
    expected = {
        "dataset_version": contract.dataset.dataset_version,
        "manifest_sha256": sha256_file(
            _repository_path(repository_root, contract.dataset.manifest_path)
        ),
        "source_revision": contract.pretrained.revision,
        "source_weight_sha256": contract.pretrained.weight_sha256,
        "image_size": contract.recipe.input_size[0],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PipelineContractError(f"classifier checkpoint provenance mismatch: {key}")
    training = payload.get("training")
    if training is None and contract.run_evidence.provenance_mode == "recovered":
        training = payload.get("training_config")
    if not isinstance(training, dict):
        raise PipelineContractError("classifier checkpoint training recipe is missing")
    for key, value in {
        "epochs": contract.recipe.epochs,
        "batch_size": contract.recipe.batch_size,
        "learning_rate": contract.recipe.learning_rate,
        "weight_decay": contract.recipe.weight_decay,
        "contrastive_weight": contract.recipe.contrastive_weight,
        "contrastive_temperature": contract.recipe.contrastive_temperature,
        "seed": contract.selection.selected_seed,
    }.items():
        if training.get(key) != value:
            raise PipelineContractError(f"classifier checkpoint recipe mismatch: {key}")
    challenger = payload.get("challenger")
    if not isinstance(challenger, dict):
        raise PipelineContractError("classifier checkpoint challenger provenance is missing")
    challenger_scope = challenger.get("scope")
    if challenger_scope is None and contract.run_evidence.provenance_mode == "recovered":
        challenger_scope = challenger.get("trainable_scope")
    if (
        challenger_scope != contract.recipe.finetune_scope
        or challenger.get("learning_rate") != contract.recipe.challenger_learning_rate
        or challenger.get("l2_sp_weight") != contract.recipe.l2_sp_weight
    ):
        raise PipelineContractError("classifier checkpoint challenger recipe mismatch")
    return {
        "mode": contract.run_evidence.provenance_mode,
        "source_revision": payload["source_revision"],
        "source_weight_sha256": payload["source_weight_sha256"],
        "manifest_sha256": payload["manifest_sha256"],
    }


def _verify_package(contract: TrainingPipelineContract, repository_root: Path) -> dict[str, Any]:
    package_path = _repository_path(repository_root, contract.package.path)
    try:
        package = load_model_package(package_path)
    except Exception as exc:
        raise PipelineContractError("production package failed validation") from exc
    metadata = package.metadata
    if metadata.promotion_status != "production":
        raise PipelineContractError("package is not promoted to production")
    model = getattr(metadata, contract.component)
    if (
        model.version != contract.package.model_version
        or model.filename != contract.package.model_filename
    ):
        raise PipelineContractError(
            "package component identity does not match the pipeline contract"
        )
    if metadata.checksums.get(model.filename) != contract.onnx.sha256:
        raise PipelineContractError("package ONNX checksum does not match pipeline lock")
    source = metadata.sources.get(contract.component)
    if (
        source is None
        or source.revision != contract.pretrained.revision
        or source.weight_sha256 != contract.checkpoint.sha256
    ):
        raise PipelineContractError("package source provenance does not match pipeline lock")
    if contract.package.schema_policy == "embedded_pipeline_lock":
        manifest_path = _repository_path(repository_root, contract.dataset.manifest_path)
        if (
            source.training_pipeline_version != contract.pipeline_version
            or source.training_contract_sha256 != canonical_contract_sha256(contract)
            or source.training_dataset_version != contract.dataset.dataset_version
            or source.training_manifest_sha256 != sha256_file(manifest_path)
        ):
            raise PipelineContractError("package embedded training provenance mismatch")
    return {
        "path": contract.package.path,
        "schema_version": metadata.schema_version,
        "worker_version": metadata.worker_version,
        "model_version": model.version,
        "external_training_lock": contract.package.schema_policy == "legacy_external_lock",
    }


def _run_framework_smoke(contract: TrainingPipelineContract) -> dict[str, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise PipelineContractError(
            "framework smoke requires torch, onnx, numpy, and onnxruntime"
        ) from exc

    model = (
        torch.nn.Conv2d(3, 4, 1)
        if contract.component == "detector"
        else torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 20)
        )
    )
    sample = torch.randn(1, 3, 32, 32)
    model(sample).mean().backward()
    with tempfile.TemporaryDirectory(prefix=f"bixolon-{contract.component}-framework-") as temp:
        output = Path(temp) / "smoke.onnx"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(
                model.eval(),
                sample,
                output,
                input_names=["pixel_values"],
                output_names=["logits"],
                dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
        onnx.checker.check_model(onnx.load(output))
        values = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
            None, {"pixel_values": sample.numpy().astype(np.float32)}
        )
    return {"component": contract.component, "scope": "framework_only", "outputs": len(values)}


def _run_smoke(contract: TrainingPipelineContract, repository_root: Path) -> dict[str, Any]:
    """Run the real pinned source-model load, forward, backward, and export path."""
    if contract.smoke is None:
        raise PipelineContractError("real smoke configuration is missing")
    checkpoint = _verify_lock(contract.checkpoint, repository_root, "checkpoint")
    onnx_path = _verify_lock(contract.onnx, repository_root, "ONNX")
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
    except Exception as exc:
        raise PipelineContractError("real smoke dependencies are unavailable") from exc
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise PipelineContractError("checkpoint is not a mapping")
    with tempfile.TemporaryDirectory(prefix=f"bixolon-{contract.component}-native-") as temp:
        exported = Path(temp) / f"{contract.component}.onnx"
        try:
            if isinstance(contract.smoke, DetectorSmokeLock):
                repository = _repository_path(repository_root, contract.smoke.repository_path)
                config = _repository_path(repository_root, contract.smoke.config_path)
                revision = subprocess.run(
                    [
                        "git",
                        "-c",
                        f"safe.directory={repository.as_posix()}",
                        "-C",
                        str(repository),
                        "rev-parse",
                        "HEAD",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if revision != contract.pretrained.revision:
                    raise PipelineContractError("D-FINE checkout revision mismatch")
                import sys

                sys.path.insert(0, str(repository))
                try:
                    from src.core import YAMLConfig

                    from .dfine_export import (
                        checkpoint_model_state,
                        compatible_checkpoint_state,
                        export_dfine_onnx,
                    )

                    cfg = YAMLConfig(str(config), resume=str(checkpoint))
                    cfg.yaml_cfg["eval_spatial_size"] = list(contract.recipe.input_size)
                    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
                    state = compatible_checkpoint_state(
                        cfg.model.state_dict(), checkpoint_model_state(payload)
                    )
                    cfg.model.load_state_dict(state, strict=False)
                    model = cfg.model.train()
                    sample = torch.zeros(1, 3, *contract.recipe.input_size)
                    targets = [
                        {
                            "labels": torch.tensor([0], dtype=torch.int64),
                            "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),
                        }
                    ]
                    result = model(sample, targets)
                    loss = sum(
                        value.float().mean()
                        for value in result.values()
                        if isinstance(value, torch.Tensor) and value.requires_grad
                    )
                    loss.backward()
                    deployed = cfg.model.eval().deploy()
                    with torch.inference_mode():
                        reference = deployed(sample)
                    pytorch_outputs = [
                        reference["pred_logits"].detach().cpu().numpy(),
                        reference["pred_boxes"].detach().cpu().numpy(),
                    ]
                    export_dfine_onnx(
                        repository=repository,
                        config=config,
                        checkpoint=checkpoint,
                        output=exported,
                        input_size=contract.recipe.input_size,
                    )
                finally:
                    if sys.path and sys.path[0] == str(repository):
                        sys.path.pop(0)
            else:
                _verify_lock(
                    contract.smoke.parity_tensors,
                    repository_root,
                    "classifier smoke tensors",
                )
                from .fewshot_adapter import (
                    adapter_spec_from_dict,
                    build_ten_shot_classifier,
                    compatible_proxy_state_dict,
                )
                from .staged_classifier_export import build_staged_view_model, view_affine

                classifier = build_ten_shot_classifier(
                    backbone_kind=str(payload["backbone_kind"]),
                    weights_path=None,
                    hub_repository=f"facebookresearch/dinov3:{payload['source_revision']}",
                    spec=adapter_spec_from_dict(payload["adapter_spec"]),
                )
                classifier.load_state_dict(compatible_proxy_state_dict(payload["model_state_dict"]))
                staged = build_staged_view_model(
                    torch,
                    classifier,
                    input_size=int(payload["image_size"]),
                    crop_scale=contract.smoke.crop_scale,
                )
                sample = torch.zeros(1, 3, *contract.recipe.input_size)
                affine = torch.from_numpy(view_affine("base"))[None]
                staged.train()
                staged(sample, affine).mean().backward()
                staged.eval()
                with torch.inference_mode():
                    pytorch_outputs = [staged(sample, affine).detach().cpu().numpy()]
                torch.onnx.export(
                    staged,
                    (sample, affine),
                    exported,
                    input_names=["pixel_values", "view_affine"],
                    output_names=["logits"],
                    dynamic_axes={
                        "pixel_values": {0: "batch"},
                        "view_affine": {0: "batch"},
                        "logits": {0: "batch"},
                    },
                    opset_version=20,
                    dynamo=False,
                )
            import onnx

            onnx.checker.check_model(onnx.load(exported))
            exported_sha256 = sha256_file(exported)
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            feeds: dict[str, np.ndarray] = {}
            for item in session.get_inputs():
                dimensions = [
                    1 if isinstance(value, str) or value is None else int(value)
                    for value in item.shape
                ]
                feeds[item.name] = np.zeros(dimensions, dtype=np.float32)
            outputs = session.run(None, feeds)
            if len(outputs) != len(pytorch_outputs) or any(
                left.shape != right.shape for left, right in zip(pytorch_outputs, outputs)
            ):
                raise PipelineContractError("PyTorch and production ONNX output shapes differ")
            if any(not np.isfinite(value).all() for value in [*pytorch_outputs, *outputs]):
                raise PipelineContractError("PyTorch or production ONNX output is non-finite")
            pytorch_cpu_max_abs_error = max(
                float(np.max(np.abs(left - right))) for left, right in zip(pytorch_outputs, outputs)
            )
            raw_query_parity_enforced = contract.component != "detector"
            if raw_query_parity_enforced and pytorch_cpu_max_abs_error > 0.01:
                raise PipelineContractError(
                    "PyTorch and production CPU ONNX parity failed: "
                    f"max_abs_error={pytorch_cpu_max_abs_error:.8f}"
                )
        except PipelineContractError:
            raise
        except Exception as exc:
            raise PipelineContractError("real artifact smoke failed") from exc
    return {
        "component": contract.component,
        "scope": "pinned_source_checkpoint_forward_backward_export_and_production_onnx",
        "checkpoint_keys": sorted(payload),
        "temporary_export_sha256": exported_sha256,
        "onnx_output_count": len(outputs),
        "pytorch_cpu_max_abs_error": pytorch_cpu_max_abs_error,
        "pytorch_cpu_tolerance": 0.01,
        "raw_query_parity_enforced": raw_query_parity_enforced,
        "detector_semantic_parity_report": (
            contract.parity.path if contract.component == "detector" else None
        ),
        "full_retraining": False,
    }


def verify_pipeline_contract(
    contract: TrainingPipelineContract,
    *,
    repository_root: Path,
    dry_run: bool = False,
    smoke: bool = False,
    framework_smoke: bool = False,
) -> dict[str, Any]:
    if sum((dry_run, smoke, framework_smoke)) > 1:
        raise PipelineContractError("verification modes are mutually exclusive")
    dataset = _verify_dataset(contract, repository_root)
    selection = _verify_selection(contract, repository_root)
    result: dict[str, Any] = {
        "schema_version": "1.1",
        "evaluation": "training_pipeline_contract_verification",
        "component": contract.component,
        "pipeline_version": contract.pipeline_version,
        "contract_sha256": canonical_contract_sha256(contract),
        "dataset_version": contract.dataset.dataset_version,
        "verification_scope": "contract_and_data_only" if dry_run else "full_locked_artifact_chain",
        "dataset": dataset,
        "selection": selection,
    }
    if isinstance(contract.recipe, DetectorRecipe):
        result["detector_recipe"] = _verify_detector_recipe(
            contract.recipe,
            repository_root,
            dataset_version=contract.dataset.dataset_version,
            validation_fold=contract.selection.validation_fold,
        )
    if dry_run:
        result.update({"artifacts": "not_checked", "package": "not_checked", "passed": None})
        return result
    result["run_evidence"] = _verify_run_evidence(contract, repository_root)
    checkpoint_path = _verify_lock(
        contract.checkpoint, repository_root, f"{contract.component} checkpoint"
    )
    result["checkpoint"] = {"path": contract.checkpoint.path, "sha256": contract.checkpoint.sha256}
    result["checkpoint"]["provenance"] = _verify_checkpoint_provenance(
        contract,
        checkpoint_path,
        repository_root,
    )
    result["onnx"] = _verify_onnx(contract.onnx, repository_root)
    result["parity"] = _verify_parity(contract.parity, repository_root)
    result["package"] = _verify_package(contract, repository_root)
    result["smoke"] = (
        _run_smoke(contract, repository_root)
        if smoke
        else _run_framework_smoke(contract)
        if framework_smoke
        else None
    )
    result["passed"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a Detector or Classifier training pipeline contract"
    )
    parser.add_argument("--component", choices=("detector", "classifier"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--framework-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = load_pipeline_contract(args.contract)
    if contract.component != args.component:
        raise PipelineContractError("--component does not match the contract")
    result = verify_pipeline_contract(
        contract,
        repository_root=args.repository_root,
        dry_run=args.dry_run,
        smoke=args.smoke,
        framework_smoke=args.framework_smoke,
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
