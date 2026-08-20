from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from ..contracts.artifact import directory_content_manifest
from ..contracts.catalog import sha256_file


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _request(
    url: str, *, body: bytes | None = None, content_type: str | None = None
) -> tuple[int, dict]:
    headers = {} if content_type is None else {"Content-Type": content_type}
    request = urllib.request.Request(
        url, data=body, headers=headers, method="POST" if body else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _multipart_image(content: bytes, filename: str) -> tuple[bytes, str]:
    boundary = f"bixolon-{uuid.uuid4().hex}"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + content
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "==" not in line or line.lstrip().startswith("#") or line[0].isspace():
            continue
        names.add(line.split("==", 1)[0].lower().replace("_", "-"))
    if not names:
        raise ValueError("packaged Worker smoke requires an exact dependency lock")
    return names


def _bundled_distribution_names(root: Path) -> set[str]:
    names: set[str] = set()
    for metadata in root.rglob("*.dist-info/METADATA"):
        for line in metadata.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Name: "):
                names.add(line.removeprefix("Name: ").strip().lower().replace("_", "-"))
                break
    return names


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    signing_key = os.environ.get(args.signing_key_env, "")
    executable = args.executable.resolve()
    worker_artifact = args.worker_artifact.resolve()
    try:
        executable.relative_to(worker_artifact)
    except ValueError as exc:
        raise ValueError(
            "packaged Worker executable must be inside its artifact directory"
        ) from exc
    manifest = directory_content_manifest(worker_artifact)
    locked_distributions = _requirement_names(args.requirements_lock)
    bundled_distributions = _bundled_distribution_names(worker_artifact)
    unlocked_distributions = sorted(bundled_distributions - locked_distributions)
    prohibited_names = {"torch", "torchvision", "scipy", "pytest"}
    prohibited = sorted(
        row["path"]
        for row in manifest["files"]
        if any(part.lower() in prohibited_names for part in Path(row["path"]).parts)
    )
    port = _free_local_port()
    environment = os.environ.copy()
    environment.update(
        {
            "BIXOLON_PACKAGE_DIR": str(args.runtime.resolve()),
            "BIXOLON_CATALOG_DIR": str(args.catalog.resolve()),
            "BIXOLON_PROVIDER": args.provider,
            "BIXOLON_HOST": "127.0.0.1",
            "BIXOLON_PORT": str(port),
        }
    )
    if args.key_id is not None:
        environment["BIXOLON_CATALOG_KEY_ID"] = args.key_id
    if args.store_id is not None:
        environment["BIXOLON_CATALOG_STORE_ID"] = args.store_id
    if signing_key:
        environment["BIXOLON_CATALOG_SIGNING_KEY"] = signing_key
    if args.cuda_dll_dir is not None:
        environment["BIXOLON_CUDA_DLL_DIR"] = str(args.cuda_dll_dir.resolve())
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [str(executable)],
        cwd=executable.parent,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    base_url = f"http://127.0.0.1:{port}"
    ready_status = 0
    ready_body: dict[str, Any] = {}
    try:
        deadline = time.monotonic() + args.startup_timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("packaged Worker exited before readiness")
            try:
                ready_status, ready_body = _request(f"{base_url}/health/ready")
                if ready_status == 200 and ready_body.get("status") == "ready":
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        else:
            raise TimeoutError("packaged Worker readiness timed out")
        multipart, content_type = _multipart_image(args.image.read_bytes(), "smoke.jpg")
        scan_status, scan_body = _request(
            f"{base_url}/v1/scan", body=multipart, content_type=content_type
        )
        corrupt_body, corrupt_type = _multipart_image(b"not-an-image", "corrupt.jpg")
        corrupt_status, corrupt_response = _request(
            f"{base_url}/v1/scan", body=corrupt_body, content_type=corrupt_type
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    passes = (
        ready_status == 200
        and ready_body.get("status") == "ready"
        and scan_status == 200
        and scan_body.get("status") in {"SEGMENTATION", "IMAGE_RECAPTURE"}
        and 400 <= corrupt_status < 500
        and corrupt_response.get("status") == "ERROR"
        and not prohibited
        and not unlocked_distributions
    )
    report = {
        "schema_version": "2.0",
        "evaluation": "scanner_2_0_packaged_worker_smoke",
        "worker_artifact_content_manifest_sha256": manifest["manifest_sha256"],
        "worker_artifact_file_count": manifest["file_count"],
        "executable_sha256": sha256_file(executable),
        "provider": args.provider,
        "cases": {
            "ready": {"http_status": ready_status, "status": ready_body.get("status")},
            "valid_scan": {
                "http_status": scan_status,
                "status": scan_body.get("status"),
                "worker_version": scan_body.get("worker_version"),
                "catalog_version": scan_body.get("catalog_version"),
            },
            "corrupt_image": {
                "http_status": corrupt_status,
                "status": corrupt_response.get("status"),
                "reason_codes": corrupt_response.get("reason_codes"),
            },
        },
        "prohibited_runtime_module_path_count": len(prohibited),
        "bundled_distribution_names": sorted(bundled_distributions),
        "unlocked_bundled_distribution_names": unlocked_distributions,
        "unlocked_bundled_distribution_count": len(unlocked_distributions),
        "passes": passes,
        "privacy": {"image_paths_recorded": False, "image_bytes_recorded": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passes:
        raise RuntimeError("packaged Worker smoke gate failed")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Smoke the frozen standalone Scanner 2.0 Worker")
    parser.add_argument("--worker-artifact", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requirements-lock", type=Path, required=True)
    parser.add_argument("--store-id")
    parser.add_argument("--key-id")
    parser.add_argument("--signing-key-env", default="BIXOLON_CATALOG_SIGNING_KEY")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--startup-timeout-seconds", type=float, default=120.0)
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
