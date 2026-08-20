from __future__ import annotations

from pathlib import Path

import pytest

from bixolon_scanner.operations.scanner_v2_release_lock import (
    _validate_evidence,
    parse_requirement_pins,
    validate_isolated_build_environment,
)


def _evidence() -> dict:
    version = "2.0.1-rc.1"
    runtime_sha = "a" * 64
    catalog_sha = "b" * 64
    return {
        "version": version,
        "runtime_metadata_sha256": runtime_sha,
        "catalog_metadata_sha256": catalog_sha,
        "development": {
            "evaluation": "scanner_2_0_development_300",
            "promotion_evidence": False,
            "dataset": {"image_count": 300},
            "development_gates": {"all_met": True},
            "versions": {
                "worker": version,
                "detector": version,
                "embedder": version,
                "classifier_policy": version,
                "catalog": version,
            },
        },
        "parity": {
            "evaluation": "scanner_2_0_runtime_cpu_cuda_parity",
            "image_count": 300,
            "final_status_class_rank_parity_exact": True,
            "passes": True,
        },
        "embedder_parity": {
            "evaluation": "scanner_2_0_embedder_parity",
            "passes": True,
        },
        "worker_smoke": {
            "evaluation": "scanner_2_0_real_worker_smoke",
            "runtime_metadata_sha256": runtime_sha,
            "catalog_metadata_sha256": catalog_sha,
            "passes": True,
        },
        "packaged_worker_smoke": {
            "evaluation": "scanner_2_0_packaged_worker_smoke",
            "worker_artifact_content_manifest_sha256": "c" * 64,
            "prohibited_runtime_module_path_count": 0,
            "unlocked_bundled_distribution_count": 0,
            "passes": True,
        },
        "worker_artifact_content_manifest_sha256": "c" * 64,
        "reliability": {
            "evaluation": "scanner_2_0_accelerated_reliability_gate",
            "runtime_metadata_sha256": runtime_sha,
            "catalog_metadata_sha256": catalog_sha,
            "request_count": 10_000,
            "decision_mismatch_image_count": 0,
            "non_200_count": 0,
            "passes": True,
        },
        "vulnerability_scan": {
            "dependencies": [{"name": "pillow", "version": "12.3.0", "vulns": []}]
        },
        "requirement_pins": {"pillow": "12.3.0"},
        "sbom": {"components": [{"name": "pillow", "version": "12.3.0"}]},
    }


def test_parse_requirement_pins_reads_hashed_continuations(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "pillow==12.3.0 \\\n+    --hash=sha256:"
        + "a" * 64
        + "\nonnxruntime-gpu==1.28.0 \\\n+    --hash=sha256:"
        + "b" * 64
        + "\n",
        encoding="utf-8",
    )

    assert parse_requirement_pins(lock) == {
        "pillow": "12.3.0",
        "onnxruntime-gpu": "1.28.0",
    }


def test_pre_private_release_evidence_accepts_only_locked_pending_state() -> None:
    checks = _validate_evidence(**_evidence())

    assert all(checks.values())


def test_pre_private_release_evidence_rejects_short_reliability_run() -> None:
    evidence = _evidence()
    evidence["reliability"]["request_count"] = 9_999

    with pytest.raises(ValueError, match="accelerated_reliability"):
        _validate_evidence(**evidence)


def test_pre_private_release_evidence_rejects_known_vulnerability() -> None:
    evidence = _evidence()
    evidence["vulnerability_scan"]["dependencies"][0]["vulns"] = [{"id": "CVE-test"}]

    with pytest.raises(ValueError, match="dependency_vulnerability_scan"):
        _validate_evidence(**evidence)


def test_isolated_worker_build_environment_allows_only_locked_runtime_and_build_tools() -> None:
    observed = {
        "Pillow": "12.3.0",
        "PyInstaller": "6.11.1",
        "pyinstaller-hooks-contrib": "2026.6",
        "pip": "24.0",
        "setuptools": "65.5.0",
    }

    tools = validate_isolated_build_environment(observed, {"pillow": "12.3.0"})

    assert tools["pyinstaller"] == "6.11.1"
    with pytest.raises(ValueError, match="unexpected"):
        validate_isolated_build_environment({**observed, "torch": "2.8.0"}, {"pillow": "12.3.0"})
