from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")


def _documentation_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "AGENTS.md"]
    files.extend((ROOT / "docs").rglob("*.md"))
    files.extend(
        [
            ROOT / "apps" / "product_scanner" / "README.md",
            ROOT / "apps" / "product_scanner" / "DESIGN_SYSTEM.md",
        ]
    )
    files.extend((ROOT / "apps" / "product_scanner" / "docs").rglob("*.md"))
    return sorted(set(files))


def test_documentation_internal_links_resolve() -> None:
    broken: list[str] = []

    for document in _documentation_files():
        for raw_target in LINK_PATTERN.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>").split("#", maxsplit=1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue

            resolved = (document.parent / unquote(target)).resolve()
            if not resolved.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")

    assert not broken, "깨진 문서 링크:\n" + "\n".join(broken)


def test_documented_versions_match_single_version_source() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    python_version = pyproject["project"]["version"]
    package_init = (ROOT / "src" / "bixolon_scanner" / "__init__.py").read_text(encoding="utf-8")
    flutter_pubspec = (ROOT / "apps" / "product_scanner" / "pubspec.yaml").read_text(
        encoding="utf-8"
    )
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    current_status = (ROOT / "docs" / "status" / "current.md").read_text(encoding="utf-8")
    version_config = (ROOT / "configs" / "versions" / "0.0.2.json").read_text(encoding="utf-8")

    assert f'__version__ = "{python_version}"' in package_init
    assert python_version == "0.0.2"
    assert '"version": "0.0.2"' in version_config
    assert "version: 0.0.2+2" in flutter_pubspec
    assert "`0.0.2+2`" in root_readme
    assert "`0.0.2`" in current_status
    assert "`2.0.1-rc.3`" in current_status


def test_windows_bundle_uses_single_version_root() -> None:
    cmake = (ROOT / "apps" / "product_scanner" / "windows" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    build_script = (ROOT / "scripts" / "build_app.ps1").read_text(encoding="utf-8")

    normalized_cmake = cmake.replace("\\", "/")

    assert "SCANNER_VERSION_ROOT" in cmake
    assert "artifacts/versions/0.0.2" in normalized_cmake
    assert "staging/runtime" in normalized_cmake
    assert "staging/catalog" in normalized_cmake
    assert "staging/cuda-runtime" in normalized_cmake
    assert "SCANNER_RELEASE_COMPOSITION" not in cmake
    assert "production_release" not in cmake
    assert 'EXISTS "${SCANNER_WORKER_RUNTIME_DIR}/bixolon-worker.exe"' in cmake
    assert 'DESTINATION "${CMAKE_INSTALL_PREFIX}/worker"' in cmake
    assert 'DESTINATION "${CMAKE_INSTALL_PREFIX}/worker/store-catalog"' in cmake
    assert "configs/versions/$Version.json" in build_script.replace("\\", "/")
    assert "bixolon-scanner-$Version" in build_script
