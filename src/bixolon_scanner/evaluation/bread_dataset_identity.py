from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ..contracts.model_package import sha256_file
from ..training.bread_cv import difference_hash


def resolve_coco_image(dataset_root: Path, annotation_path: Path, file_name: str) -> Path:
    """Resolve a COCO image while keeping it inside the declared dataset root."""
    root = dataset_root.resolve()
    candidates = (
        root / file_name,
        root / "images" / file_name,
        annotation_path.resolve().parent / file_name,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(root) and resolved.is_file():
            return resolved
    raise FileNotFoundError(file_name)


def load_coco_image_identities(
    dataset_root: Path,
    annotation_path: Path,
) -> list[dict[str, Any]]:
    """Read deterministic cryptographic and perceptual identities from a COCO dataset."""
    root = dataset_root.resolve()
    annotation = annotation_path.resolve()
    payload = json.loads(annotation.read_text(encoding="utf-8-sig"))
    identities: list[dict[str, Any]] = []
    for image in sorted(payload["images"], key=lambda row: int(row["id"])):
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        path = resolve_coco_image(root, annotation, file_name)
        with Image.open(path) as source:
            actual_size = ImageOps.exif_transpose(source).size
        identities.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "path": path,
                "image_sha256": sha256_file(path).lower(),
                "perceptual_hash": difference_hash(path),
                "actual_width": actual_size[0],
                "actual_height": actual_size[1],
                "declared_width": int(image["width"]),
                "declared_height": int(image["height"]),
                "declared_sha256": image.get("exported_image_sha256") or image.get("image_sha256"),
            }
        )
    return identities


def identity_manifest_sha256(identities: list[dict[str, Any]]) -> str:
    """Hash the independent-preflight image identity contract."""
    lines = [
        f"{int(row['image_id']):06d} {row['image_sha256']} {row['file_name']}\n"
        for row in identities
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
