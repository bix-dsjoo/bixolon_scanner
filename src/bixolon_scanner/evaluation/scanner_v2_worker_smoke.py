from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from ..contracts.catalog import sha256_file
from ..worker.api import create_app
from ..worker.settings import WorkerSettings


def evaluate(args: argparse.Namespace) -> dict:
    signing_key = os.environ.get(args.signing_key_env, "") or None
    settings = WorkerSettings(
        package_dir=args.runtime,
        catalog_dir=args.catalog,
        catalog_store_id=args.store_id,
        catalog_key_id=args.key_id,
        catalog_signing_key=signing_key,
        provider=args.provider,
        cuda_dll_dir=args.cuda_dll_dir,
    )
    app = create_app(settings=settings)
    image_bytes = args.image.read_bytes()
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        scan = client.post(
            "/v1/scan",
            files={"image": ("smoke.jpg", image_bytes, "image/jpeg")},
        )
        missing = client.post("/v1/scan")
        corrupt = client.post(
            "/v1/scan",
            files={"image": ("corrupt.jpg", b"not-an-image", "image/jpeg")},
        )
        unsupported = client.post(
            "/v1/scan",
            files={
                "image": (
                    "unsupported.gif",
                    bytes.fromhex(
                        "47494638396101000100800000000000ffffff21f90401000000002c00000000"
                        "010001000002024401003b"
                    ),
                    "image/gif",
                )
            },
        )
    ready_body = ready.json()
    scan_body = scan.json()
    cases = {
        "ready": {
            "http_status": ready.status_code,
            "body": ready_body,
        },
        "valid_scan": {
            "http_status": scan.status_code,
            "status": scan_body.get("status"),
            "segmentation_count": len(scan_body.get("segmentations", [])),
            "versions": {
                key: scan_body.get(key)
                for key in (
                    "worker_version",
                    "detector_version",
                    "classifier_version",
                    "embedder_version",
                    "detector_policy_version",
                    "classifier_policy_version",
                    "catalog_version",
                )
            },
        },
        "missing_multipart": {
            "http_status": missing.status_code,
            "status": missing.json().get("status"),
            "reason_codes": missing.json().get("reason_codes"),
        },
        "corrupt_image": {
            "http_status": corrupt.status_code,
            "status": corrupt.json().get("status"),
            "reason_codes": corrupt.json().get("reason_codes"),
        },
        "unsupported_image": {
            "http_status": unsupported.status_code,
            "status": unsupported.json().get("status"),
            "reason_codes": unsupported.json().get("reason_codes"),
        },
    }
    passes = (
        ready.status_code == 200
        and ready_body.get("status") == "ready"
        and scan.status_code == 200
        and scan_body.get("status") in {"SEGMENTATION", "IMAGE_RECAPTURE"}
        and missing.status_code == 422
        and missing.json().get("status") == "ERROR"
        and 400 <= corrupt.status_code < 500
        and corrupt.json().get("status") == "ERROR"
        and 400 <= unsupported.status_code < 500
        and unsupported.json().get("status") == "ERROR"
        and unsupported.json().get("reason_codes") == ["UNSUPPORTED_IMAGE_FORMAT"]
    )
    report = {
        "schema_version": "2.0",
        "evaluation": "scanner_2_0_real_worker_smoke",
        "runtime_metadata_sha256": sha256_file(args.runtime / "metadata.json"),
        "catalog_metadata_sha256": sha256_file(args.catalog / "catalog.json"),
        "provider": args.provider,
        "cases": cases,
        "passes": passes,
        "privacy": {
            "image_path_recorded": False,
            "image_bytes_recorded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passes:
        raise RuntimeError("real Worker smoke gate failed")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the real Scanner 2.0 Worker smoke gate")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store-id")
    parser.add_argument("--key-id")
    parser.add_argument("--signing-key-env", default="BIXOLON_CATALOG_SIGNING_KEY")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
