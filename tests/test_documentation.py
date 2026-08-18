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


def test_documented_versions_match_release_sources() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    python_version = pyproject["project"]["version"]
    package_init = (ROOT / "src" / "bixolon_scanner" / "__init__.py").read_text(encoding="utf-8")
    flutter_pubspec = (ROOT / "apps" / "product_scanner" / "pubspec.yaml").read_text(
        encoding="utf-8"
    )
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    current_status = (ROOT / "docs" / "status" / "current.md").read_text(encoding="utf-8")

    assert f'__version__ = "{python_version}"' in package_init
    assert f"`bixolon-scanner {python_version}`" in root_readme
    assert "version: 1.0.0+3" in flutter_pubspec
    assert "`1.0.0+3`" in root_readme
    assert "`bread-worker-0.1.1`" in root_readme
    assert "`bread-worker-0.1.1`" in current_status
    assert "`0.2.5` | `experiment_only`" in current_status


def test_windows_bundle_keeps_promoted_model_package() -> None:
    cmake = (ROOT / "apps" / "product_scanner" / "windows" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert "artifacts/packages/bread-worker-1.1.0" in cmake.replace("\\", "/")
    assert "SCANNER_PREVIOUS_MODEL_PACKAGE_DIR" in cmake
    assert "SCANNER_LEGACY_MODEL_PACKAGE_DIR" in cmake
    assert 'CACHE PATH "Promoted BIXOLON Worker model package" FORCE' in cmake
    assert "artifacts/worker/bixolon-worker" in cmake.replace("\\", "/")
    assert 'EXISTS "${SCANNER_WORKER_RUNTIME_DIR}/bixolon-worker.exe"' in cmake
    assert 'DESTINATION "${CMAKE_INSTALL_PREFIX}/worker"' in cmake
