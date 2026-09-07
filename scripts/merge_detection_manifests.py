from __future__ import annotations

import argparse
import json
from pathlib import Path

from bixolon_scanner.contracts.catalog import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge checksummed detection manifests")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = []
    image_paths: set[str] = set()
    for source in args.input:
        for line in source.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            image_path = str(row["image_path"])
            if image_path in image_paths:
                raise ValueError(f"duplicate image path: {image_path}")
            image_paths.add(image_path)
            rows.append(row)
    rows.sort(key=lambda row: (str(row["image_path"]), int(row["image_id"])))
    args.output_dir.mkdir(parents=True)
    manifest = args.output_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "1.0",
        "dataset_version": args.dataset_version,
        "image_count": len(rows),
        "annotation_count": sum(len(row.get("annotations", [])) for row in rows),
        "manifest_sha256": sha256_file(manifest),
        "sources": [
            {"path": source.as_posix(), "sha256": sha256_file(source)} for source in args.input
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
