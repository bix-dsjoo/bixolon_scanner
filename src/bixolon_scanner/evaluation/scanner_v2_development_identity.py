from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..contracts.catalog import sha256_file
from .scanner_v2_private_preflight import difference_hash


def _normalize_perceptual_hash(value: object) -> str | None:
    if isinstance(value, int):
        return f"{value:016x}"
    if isinstance(value, str):
        normalized = value.lower().removeprefix("0x").zfill(16)
        if len(normalized) == 16 and all(
            character in "0123456789abcdef" for character in normalized
        ):
            return normalized
    return None


def build_identity_lineage(
    source_manifests: list[Path], dataset_root: Path, output: Path
) -> dict[str, Any]:
    if not source_manifests:
        raise ValueError("development identity lineage requires source manifests")
    root = dataset_root.resolve()
    identities: dict[str, str] = {}
    source_locks = []
    for manifest in source_manifests:
        source_locks.append({"sha256": sha256_file(manifest), "row_count": 0})
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            source_locks[-1]["row_count"] += 1
            row = json.loads(line)
            image_sha256 = str(row.get("image_sha256", "")).lower()
            if len(image_sha256) != 64:
                raise ValueError("development source row is missing image SHA-256")
            perceptual_hash = _normalize_perceptual_hash(row.get("perceptual_hash"))
            if perceptual_hash is None:
                relative = Path(str(row.get("image_path", "")))
                path = (root / relative).resolve()
                try:
                    path.relative_to(root)
                except ValueError as exc:
                    raise ValueError("development source image escaped the dataset root") from exc
                if not path.is_file() or sha256_file(path) != image_sha256:
                    raise ValueError("development source image is missing or has changed")
                perceptual_hash = f"{difference_hash(path):016x}"
            previous = identities.setdefault(image_sha256, perceptual_hash)
            if previous != perceptual_hash:
                raise ValueError("one development image SHA-256 has conflicting perceptual hashes")
    rows = [
        {"image_sha256": image_sha256, "perceptual_hash": perceptual_hash}
        for image_sha256, perceptual_hash in sorted(identities.items())
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": "2.0",
        "evaluation": "scanner_2_0_development_identity_lineage",
        "source_manifests": source_locks,
        "unique_image_count": len(rows),
        "output_sha256": sha256_file(output),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the complete Scanner 2.0 development image identity lineage"
    )
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build_identity_lineage(args.source_manifest, args.dataset_root, args.output)


if __name__ == "__main__":
    main()
