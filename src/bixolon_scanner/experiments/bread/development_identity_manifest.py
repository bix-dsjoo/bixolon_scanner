from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...contracts.model_package import sha256_file
from ...evaluation.bread_dataset_identity import load_coco_image_identities


def build_development_identity_manifest(
    *,
    dataset_root: Path,
    annotation_path: Path,
    dataset_version: str,
    evaluation_set: str,
) -> list[dict[str, object]]:
    identities = load_coco_image_identities(dataset_root, annotation_path)
    return [
        {
            "record_type": "detection_image_identity",
            "dataset_version": dataset_version,
            "evaluation_set": evaluation_set,
            "image_id": row["image_id"],
            "image_path": row["file_name"],
            "image_sha256": row["image_sha256"],
            "perceptual_hash": row["perceptual_hash"],
        }
        for row in identities
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a portable image-identity manifest for Bread development lineage"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--evaluation-set", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_development_identity_manifest(
        dataset_root=args.dataset_root,
        annotation_path=args.annotation,
        dataset_version=args.dataset_version,
        evaluation_set=args.evaluation_set,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "row_count": len(rows),
                "sha256": sha256_file(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
