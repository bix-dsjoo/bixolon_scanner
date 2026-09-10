from __future__ import annotations

import json
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
    version_config = (ROOT / "configs" / "versions" / "0.1.18.json").read_text(encoding="utf-8")

    assert f'__version__ = "{python_version}"' in package_init
    assert python_version == "0.1.18"
    assert '"version": "0.1.18"' in version_config
    assert '"app_build": 21' in version_config
    assert "version: 0.1.18+21" in flutter_pubspec
    assert "`0.1.18+21`" in root_readme
    assert "`0.1.18`" in current_status
    assert "BIXOLON Bakery AI Scanner" in root_readme
    assert "BIXOLON Bakery AI Scanner" in current_status
    assert "`ssdlite-margin-dense-20260908`" in current_status


def test_only_0118_is_exposed_as_an_active_product_contract() -> None:
    version_files = {path.name for path in (ROOT / "configs" / "versions").glob("*.json")}
    example_versions = {
        path.name
        for path in (ROOT / "docs" / "contracts" / "examples").iterdir()
        if path.is_dir() and any(path.glob("*.json"))
    }
    integration_specs = {
        path.name for path in (ROOT / "docs" / "contracts").glob("worker-integration-*.md")
    }

    assert version_files == {"0.1.18.json"}
    assert example_versions == {"0.1.18"}
    assert integration_specs == {"worker-integration-0.1.18.md"}

    active_surfaces = [
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "status" / "current.md",
        ROOT / "docs" / "contracts" / "api.md",
        ROOT / "apps" / "product_scanner" / "README.md",
    ]
    for path in active_surfaces:
        assert "0.1.18" in path.read_text(encoding="utf-8"), path


def test_013_build6_packaged_worker_smoke_is_preserved() -> None:
    evidence = json.loads(
        (ROOT / "docs" / "diagnostics" / "packaged-worker-0.1.3-build6-smoke.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["product_version"] == "0.1.3"
    assert evidence["app_build"] == 6
    assert evidence["passes"] is True
    assert evidence["cases"]["ready"]["all_non_null_versions"] == "0.1.3"
    assert evidence["cases"]["corrupt_image"]["status"] == "ERROR"
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in evidence["artifacts"].values())


def test_014_build7_packaged_worker_smoke_passes() -> None:
    evidence = json.loads(
        (ROOT / "docs" / "diagnostics" / "packaged-worker-0.1.4-build7-smoke.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["product_version"] == "0.1.4"
    assert evidence["app_build"] == 7
    assert evidence["passes"] is True
    assert evidence["cases"]["ready"]["all_non_null_versions"] == "0.1.4"
    assert evidence["cases"]["provided_overlap_image"]["status"] == "IMAGE_RECAPTURE"
    assert evidence["cases"]["corrupt_image"]["status"] == "ERROR"
    assert re.fullmatch(r"[0-9a-f]{64}", evidence["bundle_manifest_sha256"])


def test_017_build10_packaged_worker_smoke_passes() -> None:
    evidence = json.loads(
        (
            ROOT / "docs" / "archive" / "diagnostics" / "packaged-worker-0.1.7-build10-smoke.json"
        ).read_text(encoding="utf-8")
    )

    assert evidence["product_version"] == "0.1.7"
    assert evidence["app_build"] == 10
    assert evidence["passes"] is True
    assert evidence["cases"]["ready"]["all_non_null_versions"] == "0.1.7"
    assert evidence["cases"]["operational_20260827"]["expected_status_mismatch_count"] == 0
    assert evidence["cases"]["operational_20260827"]["false_positive_count"] == 0
    assert evidence["cases"]["operational_20260827"]["false_negative_count"] == 0
    assert evidence["cases"]["existing_415_regression"]["semantic_diff_count"] == 0
    assert evidence["cases"]["corrupt_image"]["status"] == "ERROR"
    assert evidence["cases"]["missing_image"]["status"] == "ERROR"
    assert evidence["cases"]["unsupported_image"]["status"] == "ERROR"
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in evidence["artifacts"].values())


def test_0111_build14_packaged_worker_smoke_passes() -> None:
    evidence = json.loads(
        (ROOT / "docs" / "diagnostics" / "packaged-worker-0.1.11-build14-smoke.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["product_version"] == "0.1.11"
    assert evidence["app_build"] == 14
    assert evidence["passes"] is True
    assert evidence["cases"]["ready"]["all_non_null_versions"] == "0.1.11"
    assert evidence["cases"]["valid_scan"]["status"] == "SEGMENTATION"
    assert evidence["cases"]["corrupt_image"]["status"] == "ERROR"
    assert evidence["cases"]["missing_image"]["status"] == "ERROR"
    assert evidence["cases"]["unsupported_image"]["status"] == "ERROR"
    assert evidence["cases"]["manifests"]["setup_sha256_file_matches"] is True
    assert evidence["cases"]["manifests"]["worker_zip_sha256_file_matches"] is True
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in evidence["artifacts"].values())


def test_0112_build15_packaged_worker_smoke_passes() -> None:
    evidence = json.loads(
        (ROOT / "docs" / "diagnostics" / "packaged-worker-0.1.12-build15-smoke.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["product_version"] == "0.1.12"
    assert evidence["app_build"] == 15
    assert evidence["passes"] is True
    assert evidence["cases"]["ready"]["all_non_null_versions"] == "0.1.12"
    assert evidence["cases"]["valid_scan"]["status"] == "SEGMENTATION"
    assert evidence["cases"]["corrupt_image"]["status"] == "ERROR"
    assert evidence["cases"]["missing_image"]["status"] == "ERROR"
    assert evidence["cases"]["unsupported_image"]["status"] == "ERROR"
    assert evidence["cases"]["manifests"]["setup_sha256_file_matches"] is True
    assert evidence["cases"]["manifests"]["worker_zip_sha256_file_matches"] is True
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in evidence["artifacts"].values())


def test_0114_build17_packaged_worker_smoke_passes() -> None:
    evidence = json.loads(
        (ROOT / "docs" / "diagnostics" / "packaged-worker-0.1.14-build17-smoke.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["product_version"] == "0.1.14"
    assert evidence["app_build"] == 17
    assert evidence["passes"] is True
    assert evidence["cases"]["ready"]["all_non_null_versions"] == "0.1.14"
    assert evidence["cases"]["valid_scan"]["status"] == "SEGMENTATION"
    assert evidence["cases"]["corrupt_image"]["status"] == "ERROR"
    assert evidence["cases"]["missing_image"]["status"] == "ERROR"
    assert evidence["cases"]["unsupported_image"]["status"] == "ERROR"
    assert evidence["cases"]["manifests"]["bundle_verify"] is True
    assert evidence["cases"]["manifests"]["setup_sha256_file_matches"] is True
    assert evidence["cases"]["manifests"]["worker_zip_sha256_file_matches"] is True
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in evidence["artifacts"].values())


def test_017_source_candidate_n100_measurement_is_pinned_with_limits() -> None:
    evidence = json.loads(
        (ROOT / "docs" / "diagnostics" / "n100-0.1.5-measurement-package.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["product_version"] == "0.1.5"
    received = evidence["received_n100_measurement"]
    assert re.fullmatch(r"[0-9a-f]{64}", received["sha256"])
    assert received["sample_count"] == 100
    assert received["semantic_mismatch_count"] == 0
    assert received["hybrid_full_path_ms"]["mean"] < received["cpu_only_full_path_ms"]["mean"]
    candidate = evidence["optimization_candidate"]
    assert candidate["zip"]["size_bytes"] > 0
    assert re.fullmatch(r"[0-9a-f]{64}", candidate["zip"]["sha256"])
    assert candidate["package_manifest"]["checksum_mismatch_count"] == 0
    assert candidate["execution_contract"]["object_presence_verifier"].endswith(":GPU")
    assert candidate["execution_contract"]["object_presence_execution"] == (
        "parallel_with_detector"
    )
    assert candidate["n100_measurement_received"] is True
    measurement = candidate["n100_measurement"]
    assert measurement["sha256"] == (
        "eb421557aa794e56726decd9eb53f9117860a91612e31309f3b617782de7e6aa"
    )
    assert measurement["semantic_mismatch_count"] == 0
    assert measurement["passes"] is False
    assert measurement["client_full_path_ms"]["hybrid_p95"] > 500
    assert (
        evidence["local_validation"]["packaged_cpu_fallback_regression"]["maximum_confidence_delta"]
        == 0.0
    )


def test_windows_bundle_uses_single_version_root() -> None:
    cmake = (ROOT / "apps" / "product_scanner" / "windows" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    build_script = (ROOT / "scripts" / "build_app.ps1").read_text(encoding="utf-8")

    normalized_cmake = cmake.replace("\\", "/")

    assert "SCANNER_VERSION_ROOT" in cmake
    assert "artifacts/versions/0.1.18" in normalized_cmake
    assert "staging/runtime" in normalized_cmake
    assert "staging/catalog" in normalized_cmake
    assert "staging/cuda-runtime" in normalized_cmake
    assert "SCANNER_RELEASE_COMPOSITION" not in cmake
    assert "production_release" not in cmake
    assert 'EXISTS "${SCANNER_WORKER_RUNTIME_DIR}/bixolon-worker.exe"' in cmake
    assert 'DESTINATION "${CMAKE_INSTALL_PREFIX}/worker"' in cmake
    assert 'DESTINATION "${CMAKE_INSTALL_PREFIX}/worker/store-catalog"' in cmake
    assert "configs/versions/$Version.json" in build_script.replace("\\", "/")
    assert "bixolon-bakery-ai-scanner-$Version" in build_script
    assert "[switch]$Force" in build_script
    assert "Use -Force to replace it safely" in build_script
    assert "[System.IO.Directory]::Move($targetBundle, $previousBundle)" in build_script
    assert "[System.IO.Directory]::Move($previousBundle, $targetBundle)" in build_script
