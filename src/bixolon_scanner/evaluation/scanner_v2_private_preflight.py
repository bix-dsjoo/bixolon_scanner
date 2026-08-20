from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..contracts.artifact import assert_release_not_revoked
from ..contracts.catalog import SHA256, sha256_file

Endpoint = Literal[
    "approval_safety",
    "detector_fn",
    "detector_fp",
    "top3_safety",
    "ood_false_approval",
    "image_recapture_recall",
    "unnecessary_image_recapture",
    "invalid_roi_action",
]

MINIMUM_ZERO_ERROR_TRIALS: dict[str, int] = {
    "approval_safety": 2_995,
    "detector_fn": 2_995,
    "detector_fp": 2_995,
    "top3_safety": 2_995,
    "ood_false_approval": 2_995,
    "image_recapture_recall": 299,
    "unnecessary_image_recapture": 299,
    "invalid_roi_action": 299,
}


@lru_cache(maxsize=None)
def difference_hash(path: Path, *, size: int = 8) -> int:
    if size < 2:
        raise ValueError("difference hash size must be at least two")
    with Image.open(path) as source:
        image = (
            ImageOps.exif_transpose(source)
            .convert("L")
            .resize((size + 1, size), Image.Resampling.LANCZOS)
        )
        values = np.asarray(image, dtype=np.int16)
    bits = values[:, 1:] >= values[:, :-1]
    result = 0
    for value in bits.flat:
        result = (result << 1) | int(value)
    return result


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


class PrivateTrial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: Endpoint
    group_id: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    annotation_id: str | None = None


class PrivateTestPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    dataset_id: str = Field(min_length=1)
    immutable_revision: str = Field(min_length=1)
    review_status: Literal["locked"]
    release_lock_sha256: str
    manifest_sha256: str
    image_count: int = Field(gt=0)
    store_count: int = Field(gt=0)
    trials: list[PrivateTrial] = Field(min_length=1)

    @field_validator("release_lock_sha256", "manifest_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("private test checksums must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_unique_trials(self) -> "PrivateTestPlan":
        keys = [(trial.endpoint, trial.group_id) for trial in self.trials]
        if len(keys) != len(set(keys)):
            raise ValueError("each endpoint can use a certification group only once")
        return self


class PrivateAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annotation_id: str = Field(min_length=1)
    bbox_xywh: tuple[float, float, float, float]
    target_class_id: str = Field(min_length=1)
    physical_object_id: str = Field(min_length=1)
    catalog_membership: Literal["in_catalog", "ood"]
    expected_item_status: Literal["APPROVED", "SEGMENT_RECAPTURE"]

    @field_validator("bbox_xywh")
    @classmethod
    def validate_box(cls, value: tuple[float, float, float, float]) -> tuple[float, ...]:
        if value[0] < 0 or value[1] < 0 or value[2] <= 0 or value[3] <= 0:
            raise ValueError("private GT boxes must have non-negative origin and positive size")
        return value

    @model_validator(mode="after")
    def validate_expected_action(self) -> "PrivateAnnotation":
        if self.catalog_membership == "ood" and self.expected_item_status != "SEGMENT_RECAPTURE":
            raise ValueError("OOD objects must require SEGMENT_RECAPTURE")
        return self


class PrivateImageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    image_sha256: str
    perceptual_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    store_id: str = Field(min_length=1)
    capture_session_id: str = Field(min_length=1)
    expected_image_status: Literal["SEGMENTATION", "IMAGE_RECAPTURE"]
    annotations: list[PrivateAnnotation]

    @field_validator("image_sha256")
    @classmethod
    def validate_image_sha256(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("private image checksum must be lowercase SHA-256")
        return value

    @field_validator("image_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("private image paths must stay inside the mounted dataset root")
        return value

    @model_validator(mode="after")
    def validate_image_target(self) -> "PrivateImageRecord":
        annotation_ids = [annotation.annotation_id for annotation in self.annotations]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("annotation IDs must be unique within each image")
        if self.expected_image_status == "SEGMENTATION" and not self.annotations:
            raise ValueError("SEGMENTATION targets require at least one GT object")
        if self.expected_image_status == "IMAGE_RECAPTURE" and self.annotations:
            raise ValueError("IMAGE_RECAPTURE targets cannot contain judgeable GT objects")
        return self


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def release_lock_self_sha256(payload: dict) -> str:
    expected = payload.get("lock_sha256")
    body = {key: value for key, value in payload.items() if key != "lock_sha256"}
    actual = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if expected != actual:
        raise ValueError("pre-private release lock self-checksum is invalid")
    return actual


def load_private_manifest(path: Path) -> list[PrivateImageRecord]:
    records = [
        PrivateImageRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    ids = [record.image_id for record in records]
    hashes = [record.image_sha256 for record in records]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise ValueError("private image IDs and hashes must be unique")
    return records


def _development_identities(paths: list[Path]) -> tuple[set[str], set[str]]:
    sha256s: set[str] = set()
    perceptual: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            value = row.get("image_sha256")
            if isinstance(value, str):
                sha256s.add(value)
            value = row.get("perceptual_hash")
            if isinstance(value, int):
                perceptual.add(f"{value:016x}")
            elif isinstance(value, str):
                perceptual.add(value.lower().removeprefix("0x").zfill(16))
    return sha256s, perceptual


def validate_development_identity_manifests(release_lock: dict, paths: list[Path]) -> None:
    required = release_lock.get("private_test_policy", {}).get(
        "required_development_identity_manifests", []
    )
    required_hashes = {
        row.get("sha256") for row in required if isinstance(row, dict) and row.get("sha256")
    }
    observed_hashes = {sha256_file(path) for path in paths}
    if (
        not required_hashes
        or len(observed_hashes) != len(paths)
        or observed_hashes != required_hashes
    ):
        raise ValueError(
            "private preflight must include the exact locked development identity lineage"
        )


def validate_locked_gate_tool(release_lock: dict, path: Path) -> str:
    matches = [
        row
        for row in release_lock.get("supply_chain", {}).get("private_gate_tools", [])
        if Path(str(row.get("path", ""))).name == path.name
    ]
    if len(matches) != 1:
        raise ValueError(f"release lock does not identify exactly one gate tool named {path.name}")
    observed = sha256_file(path)
    if matches[0].get("sha256") != observed:
        raise ValueError(f"private gate tool differs from the release lock: {path.name}")
    return observed


def validate_private_trials(
    plan: PrivateTestPlan,
    records: list[PrivateImageRecord],
) -> Counter[str]:
    by_image = {record.image_id: record for record in records}
    counts: Counter[str] = Counter()
    image_endpoints = {
        "detector_fn",
        "detector_fp",
        "image_recapture_recall",
        "unnecessary_image_recapture",
    }
    provenance_to_group: dict[tuple[str, str, str], str] = {}
    group_to_provenance: dict[tuple[str, str, str], str] = {}
    for trial in plan.trials:
        record = by_image.get(trial.image_id)
        if record is None:
            raise ValueError("private trial references an unknown image")
        if trial.endpoint in image_endpoints:
            if trial.annotation_id is not None:
                raise ValueError("image-level certification trials cannot reference an annotation")
            if (
                trial.endpoint == "image_recapture_recall"
                and record.expected_image_status != "IMAGE_RECAPTURE"
            ):
                raise ValueError("image recapture trials must reference recapture GT")
            if (
                trial.endpoint != "image_recapture_recall"
                and record.expected_image_status != "SEGMENTATION"
            ):
                raise ValueError("eligible-image trials must reference SEGMENTATION GT")
            provenance = record.capture_session_id
        else:
            annotation = next(
                (
                    value
                    for value in record.annotations
                    if value.annotation_id == trial.annotation_id
                ),
                None,
            )
            if annotation is None:
                raise ValueError("object-level certification trial requires a known annotation")
            if trial.endpoint == "ood_false_approval":
                if annotation.catalog_membership != "ood":
                    raise ValueError("OOD trials must reference OOD GT")
            elif trial.endpoint == "invalid_roi_action":
                if annotation.expected_item_status != "SEGMENT_RECAPTURE":
                    raise ValueError("invalid-ROI trials must require SEGMENT_RECAPTURE")
            elif (
                annotation.catalog_membership != "in_catalog"
                or annotation.expected_item_status != "APPROVED"
            ):
                raise ValueError("approval and Top-3 trials require valid registered GT")
            provenance = annotation.physical_object_id
        provenance_key = (trial.endpoint, record.store_id, provenance)
        group_key = (trial.endpoint, record.store_id, trial.group_id)
        previous_group = provenance_to_group.setdefault(provenance_key, trial.group_id)
        previous_provenance = group_to_provenance.setdefault(group_key, provenance)
        if previous_group != trial.group_id or previous_provenance != provenance:
            raise ValueError(
                "certification groups must map one-to-one to capture sessions or physical objects"
            )
        counts[trial.endpoint] += 1
    return counts


def preflight(args: argparse.Namespace) -> dict:
    release_lock = _read_json(args.release_lock)
    lock_sha256 = release_lock_self_sha256(release_lock)
    assert_release_not_revoked(args.release_lock, lock_sha256)
    gate_tool_sha256 = validate_locked_gate_tool(release_lock, Path(__file__))
    if (
        release_lock.get("status") != "owner_private_test_pending"
        or release_lock.get("production_eligible") is not False
        or release_lock.get("remaining_gates") != ["owner_private_locked_production_test"]
    ):
        raise ValueError("candidate is not in the pre-private locked state")
    plan = PrivateTestPlan.model_validate(_read_json(args.plan))
    if plan.release_lock_sha256 != lock_sha256:
        raise ValueError("private plan does not target the locked candidate")
    if plan.manifest_sha256 != sha256_file(args.manifest):
        raise ValueError("private manifest checksum does not match the locked plan")
    validate_development_identity_manifests(release_lock, args.development_identity_manifest)
    records = load_private_manifest(args.manifest)
    if len(records) != plan.image_count:
        raise ValueError("private plan image count does not match the manifest")
    stores = {record.store_id for record in records}
    if len(stores) != plan.store_count:
        raise ValueError("private plan store count does not match the manifest")

    dataset_root = args.dataset_root.resolve()
    for record in records:
        path = (dataset_root / record.image_path).resolve()
        try:
            path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError("private image path escaped the dataset root") from exc
        if not path.is_file() or sha256_file(path) != record.image_sha256:
            raise ValueError("private image is missing or does not match its checksum")

    development_sha256s, development_perceptual = _development_identities(
        args.development_identity_manifest
    )
    private_sha256s = {record.image_sha256 for record in records}
    private_perceptual: set[str] = set()
    for record in records:
        path = (dataset_root / record.image_path).resolve()
        computed = f"{difference_hash(path):016x}"
        if record.perceptual_hash != computed:
            raise ValueError("private manifest perceptual hash does not match the image")
        private_perceptual.add(computed)
    if private_sha256s & development_sha256s:
        raise ValueError("private test contains an exact development image identity")
    near_overlap_count = sum(
        any(
            hamming_distance(int(value, 16), int(known, 16)) <= 2
            for known in development_perceptual
        )
        for value in private_perceptual
    )
    if near_overlap_count:
        raise ValueError("private test contains a near-duplicate development image identity")

    counts = validate_private_trials(plan, records)
    insufficient = {
        endpoint: {"observed": counts[endpoint], "required": minimum}
        for endpoint, minimum in MINIMUM_ZERO_ERROR_TRIALS.items()
        if counts[endpoint] < minimum
    }
    if insufficient:
        raise ValueError(f"private test has insufficient certification trials: {insufficient}")

    report = {
        "schema_version": "2.0",
        "evaluation": "scanner_2_0_owner_private_preflight",
        "dataset_id": plan.dataset_id,
        "immutable_revision": plan.immutable_revision,
        "release_candidate": release_lock["release_candidate"],
        "release_lock_sha256": lock_sha256,
        "gate_tool_sha256": gate_tool_sha256,
        "plan_sha256": sha256_file(args.plan),
        "manifest_sha256": sha256_file(args.manifest),
        "image_count": len(records),
        "store_count": len(stores),
        "judgeable_gt_object_count": sum(len(record.annotations) for record in records),
        "certification_trial_counts": dict(sorted(counts.items())),
        "development_exact_identity_overlap_count": 0,
        "development_perceptual_identity_overlap_count": near_overlap_count,
        "image_checksums_verified": True,
        "production_inference_executed": False,
        "eligible_for_single_private_run": True,
        "privacy": {
            "image_paths_recorded": False,
            "image_bytes_recorded": False,
            "per_image_identifiers_recorded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preflight the owner-private Scanner 2.0 production test without inference"
    )
    parser.add_argument("--release-lock", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--development-identity-manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    preflight(parser.parse_args(argv))


if __name__ == "__main__":
    main()
