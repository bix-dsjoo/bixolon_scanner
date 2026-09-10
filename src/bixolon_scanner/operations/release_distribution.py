"""Collect verified release outputs in a versioned handoff directory."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from ..configuration import load_json_config
from ..contracts.artifact import directory_content_manifest
from ..contracts.catalog import sha256_file


def collect(root: Path, version: str, sdk_version: str) -> Path:
    config = load_json_config(root / f"configs/versions/{version}.json")
    if config["version"] != version:
        raise ValueError("distribution version identity mismatch")
    output = root / f"artifacts/distributions/{version}"
    store = config["catalog"]["store_id"]
    sources = {
        "installers": [
            root / f"artifacts/installers/{version}/BixolonBakeryAIScanner-{version}-Setup.exe",
            root / f"artifacts/lite/{version}/BixolonBakeryAIScannerLite-{version}-Setup.exe",
        ],
        "developer": [
            root / f"artifacts/installers/{version}/BixolonBakeryAIScanner-{version}-Worker.zip",
            root
            / f"artifacts/external-sdk/{sdk_version}/BIXOLON-Scanner-SDK-Windows-x64-{sdk_version}.zip",
        ],
        "models": [
            root
            / f"artifacts/store-models/{store}/{version}/BIXOLON-Store-Model-{store}-{version}.zip",
        ],
    }
    for folder, paths in sources.items():
        destination = output / folder
        destination.mkdir(parents=True, exist_ok=True)
        for source in paths:
            target = destination / source.name
            if not target.exists() or sha256_file(target) != sha256_file(source):
                shutil.copy2(source, target)
            if sha256_file(source) != sha256_file(target):
                raise ValueError("distribution copy changed payload")
    portable = output / "portable"
    portable.mkdir(exist_ok=True)
    archive_sources = {
        f"BixolonBakeryAIScanner-{version}-CPU-Portable": root
        / f"artifacts/installers/{version}/windows-payload",
        f"BixolonBakeryAIScanner-{version}-CUDA-Portable": root
        / f"artifacts/versions/{version}/bixolon-bakery-ai-scanner-{version}",
        f"BixolonBakeryAIScannerLite-{version}-CPU-Portable": root
        / f"artifacts/lite/{version}/payload",
    }
    for name, source in archive_sources.items():
        source_manifest = directory_content_manifest(source)
        archive = portable / f"{name}.zip"
        receipt = portable / f"{name}.source.json"
        if archive.exists():
            previous = load_json_config(receipt)
            if previous["source_manifest_sha256"] != source_manifest["manifest_sha256"] or previous[
                "zip_sha256"
            ] != sha256_file(archive):
                raise ValueError("existing portable archive differs from its source or checksum")
            continue
        print(f"Compressing {name}", flush=True)
        temporary = portable / f"{name}.partial"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
            for row in source_manifest["files"]:
                path = source / row["path"]
                if sha256_file(path) != row["sha256"]:
                    raise ValueError("portable source changed during compression")
                bundle.write(path, f"{name}/{row['path']}")
        temporary.replace(archive)
        receipt.write_text(
            json.dumps(
                {
                    "source_manifest_sha256": source_manifest["manifest_sha256"],
                    "zip_sha256": sha256_file(archive),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    for path in output.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".exe", ".zip"}:
            path.with_suffix(path.suffix + ".sha256").write_text(
                f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--sdk-version", required=True)
    args = parser.parse_args()
    print(collect(args.repository_root.resolve(), args.version, args.sdk_version))


if __name__ == "__main__":
    main()
