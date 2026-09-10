"""Verify all handoff archives and component manifests, then seal the file index."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.operations.lite_bundle import verify as verify_lite


def check_manifest(root: Path, name: str) -> int:
    report = load_json_config(root / name)
    for row in report["files"]:
        path = root / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Component manifest mismatch: {path}")
    return len(report["files"])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/distributions/0.1.18"
    version = root / "artifacts/versions/0.1.18"
    if (
        not load_json_config(output / "reports/final-evaluation.json")["regression_passed"]
        or not load_json_config(output / "reports/packaged-confirmation.json")["passed"]
        or not load_json_config(output / "reports/model-binary-parity.json")["passed"]
    ):
        raise ValueError("Release evidence does not support the selected CPU change")
    for row in load_json_config(output / "reports/evidence-sources.json")["files"]:
        if sha256_file(output / "reports" / row["path"]) != row["sha256"]:
            raise ValueError("Copied evidence changed after reporting")
    checked = {}
    deliverables = sorted(p for p in output.rglob("*") if p.suffix in {".zip", ".exe"})
    if len(deliverables) != 8:
        raise ValueError("Expected exactly eight release deliverables")
    for path in deliverables:
        if sha256_file(path) != path.with_suffix(path.suffix + ".sha256").read_text().split()[0]:
            raise ValueError("Release download checksum mismatch")
        if path.suffix == ".zip":
            print(f"Checking ZIP: {path.name}", flush=True)
            with zipfile.ZipFile(path) as bundle:
                if bundle.testzip() is not None:
                    raise ValueError("Release archive CRC mismatch")
                names = bundle.namelist()
                if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
                    raise ValueError("Unsafe ZIP member path")
                checked[path.name] = {"zip_crc_passed": True, "file_count": len(names)}
    components = [
        (root / "artifacts/installers/0.1.18/windows-payload", "installer-payload-manifest.json"),
        (
            root / "artifacts/installers/0.1.18/BixolonBakeryAIScanner-0.1.18-Worker",
            "worker-manifest.json",
        ),
        (
            root / "artifacts/external-sdk/1.1.2/BIXOLON-Scanner-SDK-Windows-x64-1.1.2",
            "integration-manifest.json",
        ),
        (
            root
            / "artifacts/store-models/three_bakery/0.1.18/BIXOLON-Store-Model-three_bakery-0.1.18",
            "bundle-manifest.json",
        ),
    ]
    for path, name in components:
        checked[path.name] = {"manifest_passed": True, "file_count": check_manifest(path, name)}
    verify_lite(root / "artifacts/lite/0.1.18/payload")
    for provider, worker in (
        ("cpu", root / "artifacts/installers/0.1.18/windows-payload/worker"),
        ("cuda", version / "bixolon-bakery-ai-scanner-0.1.18/worker"),
    ):
        smoke = load_json_config(version / f"packaged-{provider}-smoke.json")
        if (
            not smoke["passes"]
            or directory_content_manifest(worker)["manifest_sha256"]
            != smoke["worker_artifact_content_manifest_sha256"]
        ):
            raise ValueError("The packaged Worker changed since its successful smoke")
    report = {"passed": True, "deliverable_count": len(deliverables), "checks": checked}
    (output / "reports/distribution-verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    files = [
        {
            "path": p.relative_to(output).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name not in {"distribution-manifest.json", "SHA256SUMS.txt"}
    ]
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{r['sha256']}  {r['path']}\n" for r in files), encoding="utf-8"
    )
    (output / "distribution-manifest.json").write_text(
        json.dumps(
            {
                "product_version": "0.1.18",
                "sdk_version": "1.1.2",
                "source_candidate": "ssdlite-margin-dense-20260908-retained-cpu-sleep",
                "files": files,
                "checksum_index_sha256": sha256_file(output / "SHA256SUMS.txt"),
                "self_exclusion": "distribution-manifest.json",
                "final_benchmark_target_met": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
