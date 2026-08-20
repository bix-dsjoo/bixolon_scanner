from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import PackageValidationError
from .package_files import resolve_package_file, validate_package_filename

SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CatalogState(StrEnum):
    COLLECTING = "collecting"
    VALIDATING = "validating"
    ACTIVE = "active"
    ACTIVE_RESTRICTED = "active_restricted"
    SUPERSEDED = "superseded"


class CatalogLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)
    support_offset: int = Field(ge=0)
    support_count: int = Field(ge=10, le=10)
    compactness: float = Field(ge=-1.0, le=1.0)
    nearest_class_id: str | None = None
    nearest_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    approval_restricted: bool = False


class CatalogRestrictedPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_ids: tuple[str, str]
    prototype_similarity: float = Field(ge=-1.0, le=1.0)
    reason: Literal["CATALOG_CONFUSABLE_PAIR"] = "CATALOG_CONFUSABLE_PAIR"

    @field_validator("class_ids")
    @classmethod
    def validate_pair(cls, value: tuple[str, str]) -> tuple[str, str]:
        if len(set(value)) != 2 or tuple(sorted(value)) != value:
            raise ValueError("restricted pair IDs must be two distinct sorted values")
        return value


class CatalogActivation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["active", "active_restricted"]
    restricted_class_ids: list[str] = Field(default_factory=list)
    restricted_pairs: list[CatalogRestrictedPair] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_state(self) -> "CatalogActivation":
        restricted = bool(self.restricted_class_ids or self.restricted_pairs)
        if restricted != (self.state == CatalogState.ACTIVE_RESTRICTED):
            raise ValueError("catalog activation state must match its approval restrictions")
        if self.restricted_class_ids != sorted(set(self.restricted_class_ids)):
            raise ValueError("restricted class IDs must be unique and sorted")
        pairs = [pair.class_ids for pair in self.restricted_pairs]
        if pairs != sorted(set(pairs)):
            raise ValueError("restricted pairs must be unique and sorted")
        return self


class CatalogMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    authentication: Literal["HMAC-SHA256", "CHECKSUM-SHA256"] = "HMAC-SHA256"
    catalog_version: str
    store_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    embedder_id: str = Field(min_length=1)
    embedder_version: str
    classifier_policy_version: str
    embedding_dimension: int = Field(gt=0)
    l2_normalized: Literal[True] = True
    support_count_per_class: int = Field(ge=10, le=10)
    support_count: int = Field(gt=0)
    labels: list[CatalogLabel] = Field(min_length=1)
    source_manifest_sha256: str
    decision_head: Literal["exact_retrieval", "ridge_adapter"] = "exact_retrieval"
    adapter_filename: str | None = None

    _validate_adapter_filename = field_validator("adapter_filename")(validate_package_filename)

    @field_validator("catalog_version", "embedder_version", "classifier_policy_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not SEMVER.fullmatch(value):
            raise ValueError("catalog and component versions must use semantic versioning")
        return value

    @field_validator("source_manifest_sha256")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("source manifest checksum must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_labels(self) -> "CatalogMetadata":
        ids = [label.class_id for label in self.labels]
        if ids != sorted(set(ids)):
            raise ValueError("catalog labels must have unique sorted class IDs")
        expected_offset = 0
        for label in self.labels:
            if label.support_offset != expected_offset:
                raise ValueError("catalog support offsets must be contiguous")
            expected_offset += label.support_count
        if expected_offset != self.support_count:
            raise ValueError("catalog support count does not match labels")
        if (self.decision_head == "ridge_adapter") != (self.adapter_filename is not None):
            raise ValueError("ridge Catalogs require exactly one adapter file")
        return self


class CatalogSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["HMAC-SHA256"] = "HMAC-SHA256"
    key_id: str = Field(min_length=1)
    signed_file: Literal["checksums.json"] = "checksums.json"
    digest: str

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("catalog signature must be lowercase SHA-256")
        return value


@dataclass(frozen=True)
class StoreCatalogPackage:
    root: Path
    metadata: CatalogMetadata
    activation: CatalogActivation
    supports_path: Path
    prototypes_path: Path
    statistics_path: Path
    source_manifest_path: Path
    adapter_path: Path | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageValidationError from exc


def load_store_catalog_package(
    root: Path,
    *,
    signing_key: bytes | None = None,
    expected_store_id: str | None = None,
    expected_key_id: str | None = None,
) -> StoreCatalogPackage:
    catalog_root = root.resolve()
    try:
        metadata = CatalogMetadata.model_validate(_load_json(catalog_root / "catalog.json"))
        activation = CatalogActivation.model_validate(_load_json(catalog_root / "activation.json"))
        checksums_path = catalog_root / "checksums.json"
        checksums_bytes = checksums_path.read_bytes()
        checksums = _load_json(checksums_path)
    except (OSError, ValueError) as exc:
        raise PackageValidationError from exc
    if expected_store_id is not None and metadata.store_id != expected_store_id:
        raise PackageValidationError
    if metadata.authentication == "HMAC-SHA256":
        try:
            signature = CatalogSignature.model_validate(_load_json(catalog_root / "signature.json"))
        except (OSError, ValueError) as exc:
            raise PackageValidationError from exc
        if not signing_key:
            raise PackageValidationError
        if expected_key_id is not None and signature.key_id != expected_key_id:
            raise PackageValidationError
        expected_signature = hmac.new(signing_key, checksums_bytes, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature.digest, expected_signature):
            raise PackageValidationError
    elif expected_key_id is not None or (catalog_root / "signature.json").exists():
        raise PackageValidationError
    if not isinstance(checksums, dict) or not checksums:
        raise PackageValidationError
    required = {
        "catalog.json",
        "activation.json",
        "supports.bin",
        "prototypes.bin",
        "statistics.json",
        "source-manifest.jsonl",
    }
    if metadata.adapter_filename is not None:
        required.add(metadata.adapter_filename)
    if set(checksums) != required:
        raise PackageValidationError
    resolved_files: dict[str, Path] = {}
    for filename, expected in checksums.items():
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise PackageValidationError
        path = resolve_package_file(catalog_root, filename)
        if sha256_file(path) != expected:
            raise PackageValidationError
        resolved_files[filename] = path
    source_manifest = resolved_files["source-manifest.jsonl"]
    if sha256_file(source_manifest) != metadata.source_manifest_sha256:
        raise PackageValidationError
    return StoreCatalogPackage(
        root=catalog_root,
        metadata=metadata,
        activation=activation,
        supports_path=resolved_files["supports.bin"],
        prototypes_path=resolved_files["prototypes.bin"],
        statistics_path=resolved_files["statistics.json"],
        source_manifest_path=source_manifest,
        adapter_path=(
            None if metadata.adapter_filename is None else resolved_files[metadata.adapter_filename]
        ),
    )
