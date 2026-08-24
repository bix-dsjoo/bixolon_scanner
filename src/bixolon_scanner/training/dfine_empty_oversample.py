from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def oversample_empty_coco(payload: dict[str, Any], repeats: int) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("empty-image repeats must be positive")
    images = list(payload.get("images", []))
    annotations = list(payload.get("annotations", []))
    annotated_ids = {int(row["image_id"]) for row in annotations}
    empty_images = [row for row in images if int(row["id"]) not in annotated_ids]
    if not empty_images:
        raise ValueError("COCO payload has no empty images to oversample")
    next_id = max((int(row["id"]) for row in images), default=0) + 1
    added = []
    for source in empty_images:
        for repeat_index in range(1, repeats):
            clone = deepcopy(source)
            clone["id"] = next_id
            clone["oversampled_from_image_id"] = int(source["id"])
            clone["oversample_repeat_index"] = repeat_index
            added.append(clone)
            next_id += 1
    result = deepcopy(payload)
    result["images"] = images + added
    result["empty_image_oversampling"] = {
        "unique_image_count": len(images),
        "unique_empty_image_count": len(empty_images),
        "empty_image_repeats": repeats,
        "added_training_record_count": len(added),
        "training_record_count": len(images) + len(added),
    }
    return result


def create_empty_oversampled_coco(
    source: Path, output: Path, provenance_output: Path, repeats: int
) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    result = oversample_empty_coco(payload, repeats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "1.0",
        "recipe": "empty_image_training_sampler_oversampling",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        **result["empty_image_oversampling"],
        "unique_dataset_changed": False,
        "selection_scope": "development_dataset_only",
        "independent_test_claimed": False,
    }
    provenance_output.parent.mkdir(parents=True, exist_ok=True)
    provenance_output.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a D-FINE training-only COCO view with repeated empty images"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            create_empty_oversampled_coco(
                args.source, args.output, args.provenance_output, args.repeats
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
