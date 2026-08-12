import json

import pytest

from bixolon_scanner.benchmark import (
    _latency_summary,
    _manifest_evidence,
    _resolve_gpu_selection,
)
from bixolon_scanner.package import sha256_file


def test_latency_summary_reports_tail_percentiles():
    summary = _latency_summary([10.0, 20.0, 30.0, 40.0])
    assert summary["sample_count"] == 4
    assert summary["p50_ms"] == 25.0
    assert summary["p95_ms"] > summary["p50_ms"]
    assert summary["p99_ms"] >= summary["p95_ms"]


def test_manifest_evidence_binds_the_exact_benchmark_images(tmp_path):
    root = tmp_path / "benchmark"
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    image = image_dir / "1.jpg"
    image.write_bytes(b"rpc-image")
    image_sha = sha256_file(image)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "image_path": "images/1.jpg",
                        "source_image_sha256": image_sha,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    checksums = root / "checksums.json"
    checksums.write_text(
        json.dumps(
            {
                "phase": "benchmark-manifest",
                "outputs": {
                    "manifest.json": sha256_file(manifest),
                    "images/1.jpg": image_sha,
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = _manifest_evidence(image_dir, [image], manifest, checksums)
    assert evidence["benchmark_manifest_sha256"] == sha256_file(manifest)
    assert evidence["image_artifact_sha256"] == {"images/1.jpg": image_sha}

    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image checksum"):
        _manifest_evidence(image_dir, [image], manifest, checksums)


def test_multi_gpu_numeric_visible_device_is_not_mapped_to_physical_index():
    rows = [
        {
            "physical_index": index,
            "uuid": f"GPU-{index}",
            "name": f"GPU {index}",
            "driver_version": "1",
            "memory_total_mib": 16_000.0,
        }
        for index in range(2)
    ]

    selected, source = _resolve_gpu_selection(rows, "0")

    assert selected["physical_index"] == -1
    assert selected["uuid"] == ""
    assert source == "ambiguous_CUDA_VISIBLE_DEVICES"


def test_multi_gpu_uuid_visible_device_binds_physical_gpu():
    rows = [
        {
            "physical_index": index,
            "uuid": f"GPU-uuid-{index}",
            "name": f"GPU {index}",
            "driver_version": "1",
            "memory_total_mib": 16_000.0,
        }
        for index in range(2)
    ]

    selected, source = _resolve_gpu_selection(rows, "GPU-uuid-1")

    assert selected["physical_index"] == 1
    assert selected["uuid"] == "GPU-uuid-1"
    assert source == "CUDA_VISIBLE_DEVICES_UUID"
