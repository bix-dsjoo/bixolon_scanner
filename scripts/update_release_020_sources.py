"""Advance active release surfaces while preserving the 0.1.18 contracts verbatim."""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative, archived in (
        ("configs/versions/0.1.18.json", "configs/archive/versions/0.1.18.json"),
        (
            "docs/contracts/worker-integration-0.1.18.md",
            "docs/archive/contracts/worker-integration-0.1.18.md",
        ),
        ("docs/contracts/examples/0.1.18", "docs/archive/contracts/examples/0.1.18"),
    ):
        source, destination = root / relative, root / archived
        source.resolve().relative_to(root)
        destination.resolve().relative_to(root)
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError("archive already exists; inspect before replacing")
            source.rename(destination)
    archived_status = root / "docs/archive/status/0.1.18.md"
    if not archived_status.exists():
        shutil.copy2(root / "docs/status/current.md", archived_status)
    examples = root / "docs/contracts/examples/0.2.0"
    if not examples.exists():
        shutil.copytree(root / "docs/archive/contracts/examples/0.1.18", examples)
    integration = root / "docs/contracts/worker-integration-0.2.0.md"
    if not integration.exists():
        shutil.copy2(root / "docs/archive/contracts/worker-integration-0.1.18.md", integration)
    paths = [
        "pyproject.toml",
        "AGENTS.md",
        "README.md",
        "configs/README.md",
        "docs/README.md",
        "docs/contracts/api.md",
        "docs/contracts/flutter-worker-client-example.md",
        "docs/contracts/worker-integration-0.2.0.md",
        "src/bixolon_scanner/__init__.py",
        "src/bixolon_scanner/operations/lite_bundle.py",
        "scripts/build_app.ps1",
        "scripts/build_windows_installer.ps1",
        "scripts/build_lite.ps1",
        "scripts/build_external_sdk.ps1",
        "installer/windows/INSTALL-KO.txt",
        "installer/windows/WORKER-KO.txt",
        "apps/product_scanner/pubspec.yaml",
        "apps/product_scanner/README.md",
        "apps/product_scanner/windows/CMakeLists.txt",
        "apps/bakery_scanner_lite/pubspec.yaml",
        "apps/bakery_scanner_lite/README.md",
        "sdk/external/README-KO.md",
        "sdk/external/deployment/active-bundle.example.json",
        "sdk/flutter/bixolon_scanner_sdk/pubspec.yaml",
        "tests/test_documentation.py",
        "tests/test_windows_installer.py",
        "tests/test_lite_bundle.py",
        "tests/test_external_sdk_packaging.py",
        "tests/test_handoff_artifacts.py",
    ]
    selected = {root / p for p in paths}
    selected.update(examples.glob("*.json"))
    for app in ("product_scanner", "bakery_scanner_lite"):
        for directory in ("lib", "test", "integration_test"):
            selected.update((root / "apps" / app / directory).rglob("*.dart"))
    for path in sorted(selected):
        if not path.is_file():
            continue
        old = path.read_text(encoding="utf-8")
        new = old.replace("0.1.18+21", "0.2.0+23").replace("0.1.18", "0.2.0")
        if path.name in {
            "README.md",
            "README-KO.md",
            "pubspec.yaml",
            "build_external_sdk.ps1",
            "test_external_sdk_packaging.py",
        }:
            new = new.replace("1.1.2", "1.2.0")
        if path.name == "test_documentation.py":
            new = new.replace("test_only_0118_", "test_only_020_").replace(
                '"app_build": 21', '"app_build": 23'
            )
        if path.name == "test_lite_bundle.py":
            new = new.replace("app_build=21", "app_build=23")
        if old != new:
            path.write_text(new, encoding="utf-8")
    print("Active 0.2.0 surfaces updated; 0.1.18 contracts archived.")


if __name__ == "__main__":
    main()
