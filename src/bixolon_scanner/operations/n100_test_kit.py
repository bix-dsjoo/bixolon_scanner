"""Standalone Windows N100 HTTP measurement runner; no PowerShell or installed Python."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from collections import Counter
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.operations.process_memory import process_memory


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Invalid kit manifest path")
    return path


def verify_kit(root: Path) -> dict:
    manifest = load_json_config(root / "KIT-MANIFEST.json")
    for row in manifest["files"]:
        path = child(root, row["path"])
        if not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError(f"Missing or changed file: {row['path']}")
    if len(manifest["inputs"]) != 132:
        raise ValueError("Exactly 132 frozen inputs are required")
    for row in manifest["inputs"]:
        if digest(child(root, row["path"])) != row["image_sha256"]:
            raise ValueError("Frozen input hash mismatch")
    return manifest


def stage_benchmark(root: Path, manifest: dict) -> Path:
    """Run the measured models from local storage, even when launched from USB."""
    if os.name != "nt" or not os.environ.get("LOCALAPPDATA"):
        return root
    identity = digest(root / "KIT-MANIFEST.json")[:24]
    cache = Path(os.environ["LOCALAPPDATA"]) / "BixolonN100Benchmark" / identity
    directories = [manifest["benchmark"], *manifest.get("benchmark_dependencies", [])]
    prefixes = []
    for directory in directories:
        relative = child(root, directory).relative_to(root.resolve())
        if relative == Path("."):
            raise ValueError("A benchmark dependency must name a subdirectory")
        prefixes.append(relative.as_posix().rstrip("/") + "/")
    entries = [row for row in manifest["files"] if row["path"].startswith(tuple(prefixes))]
    if not entries:
        raise ValueError("No benchmark files in manifest")
    print(
        "Preparing verified local benchmark cache; first copy may take a few minutes...", flush=True
    )
    for row in entries:
        source, target = child(root, row["path"]), child(cache, row["path"])
        if target.is_file() and digest(target) == row["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != row["sha256"]:
            raise ValueError("Local benchmark copy checksum mismatch")
    return cache


def overlay_runtime(base: Path, overlay: Path, destination: Path) -> Path:
    """Materialize a complete package using verified local files and small overlays."""
    sources = {
        path.relative_to(folder).as_posix(): path
        for folder in (base, overlay)
        for path in folder.rglob("*")
        if path.is_file()
    }
    for relative, source in sources.items():
        target = child(destination, relative)
        if target.is_file() and digest(target) == digest(source):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Cached packages are read-only inputs; replace links rather than writing through.
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise ValueError("Experiment runtime materialization failed")
    return destination


def timing(values: list[float]) -> dict:
    ordered = sorted(values)

    def percentile(q):
        if not ordered:
            return None
        position = (len(ordered) - 1) * q
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (position - low) * (ordered[high] - ordered[low])

    return {
        "count": len(values),
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": max(values) if values else None,
        "within_1000ms_count": sum(value <= 1000 for value in values),
        "over_1000ms_count": sum(value > 1000 for value in values),
    }


def summarize(rows):
    return {
        "latency": timing([r["elapsed_ms"] for r in rows]),
        "full_path": timing(
            [r["elapsed_ms"] for r in rows if r["response"]["status"] == "SEGMENTATION"]
        ),
        "image_recapture": timing(
            [r["elapsed_ms"] for r in rows if r["response"]["status"] == "IMAGE_RECAPTURE"]
        ),
        "error": timing([r["elapsed_ms"] for r in rows if r["response"]["status"] == "ERROR"]),
        "image_status_counts": dict(Counter(r["response"]["status"] for r in rows)),
        "item_status_counts": dict(
            Counter(s["status"] for r in rows for s in r["response"]["segmentations"])
        ),
    }


def request(port: int, data: bytes | None = None) -> tuple[int, dict, float]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=130 if data else 3)
    boundary = "bixolon-n100-test-boundary"
    body = (
        None
        if data is None
        else (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="input.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n".encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
    )
    try:
        started = time.perf_counter()
        connection.request(
            "GET" if data is None else "POST",
            "/health/ready" if data is None else "/v1/scan",
            body=body,
            headers={}
            if body is None
            else {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        return response.status, payload, (time.perf_counter() - started) * 1000
    finally:
        connection.close()


def worker_environment(
    worker: Path,
    profile: dict,
    port: int,
    *,
    runtime: Path | None = None,
    catalog: Path | None = None,
) -> dict:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("BIXOLON_")}
    environment.update(
        {
            "BIXOLON_PACKAGE_DIR": str(runtime or worker / "model-package"),
            "BIXOLON_CATALOG_DIR": str(catalog or worker / "store-catalog"),
            "BIXOLON_CATALOG_STORE_ID": "three_bakery",
            "BIXOLON_PROVIDER": profile["provider"],
            "BIXOLON_EMBEDDER_PROVIDER": profile["embedder_provider"],
            "BIXOLON_EMBEDDER_FALLBACK_PROVIDER": profile["embedder_fallback_provider"],
            "BIXOLON_PROVIDER_EXECUTION_CPU_FALLBACK": str(
                profile["provider_execution_cpu_fallback"]
            ).lower(),
            "BIXOLON_VERIFIER_PROVIDER": profile["verifier_provider"],
            "BIXOLON_CPU_DETECTOR_WORKERS": "1",
            "BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS": str(profile["cpu_detector_intra_op_threads"]),
            "BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS": str(profile["cpu_embedder_intra_op_threads"]),
            "BIXOLON_OPENVINO_GPU_PRECISION": profile.get("openvino_gpu_precision", "f32"),
            "BIXOLON_LOG_MODEL_TIMINGS": str(profile.get("log_model_timings", False)).lower(),
            "BIXOLON_REUSE_VERIFIER_EMBEDDINGS": str(
                profile.get("reuse_verifier_embeddings", True)
            ).lower(),
            "BIXOLON_PARALLEL_VERIFICATION": str(
                profile.get("parallel_verification", False)
            ).lower(),
            "BIXOLON_HOST": "127.0.0.1",
            "BIXOLON_PORT": str(port),
            "BIXOLON_REQUEST_TIMEOUT_SECONDS": "120",
            "BIXOLON_LOG_TO_STDERR": "1",
        }
    )
    # PyInstaller changes DLL lookup for itself. Workers must find their own DLLs.
    return environment


def windows_inventory() -> dict:
    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if os.name != "nt":
        return result
    import winreg

    for label, key_path in {
        "cpu": r"HARDWARE\DESCRIPTION\System\CentralProcessor",
        "display": r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}",
    }.items():
        rows = []
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as parent:
                for index in range(winreg.QueryInfoKey(parent)[0]):
                    name = winreg.EnumKey(parent, index)
                    with winreg.OpenKey(parent, name) as key:
                        row = {}
                        for field in (
                            "ProcessorNameString",
                            "DriverDesc",
                            "DriverVersion",
                            "DriverDate",
                            "ProviderName",
                        ):
                            try:
                                value = winreg.QueryValueEx(key, field)[0]
                                row[field] = value if isinstance(value, (str, int)) else str(value)
                            except OSError:
                                pass
                        if row:
                            rows.append(row)
        except OSError:
            pass
        result[label] = rows
    return result


def ensure_idle() -> None:
    if os.name != "nt":
        return
    tool = Path(os.environ["SystemRoot"]) / "System32/tasklist.exe"
    output = subprocess.check_output(
        [str(tool), "/FO", "CSV", "/NH"], creationflags=subprocess.CREATE_NO_WINDOW
    )
    names = {
        row[0].lower() for row in csv.reader(io.StringIO(output.decode(errors="replace"))) if row
    }
    if names & {"bixolon-worker.exe", "bakery_scanner_lite.exe", "product_scanner.exe"}:
        raise RuntimeError("Close Scanner Lite and all other Scanner Workers, then run again.")


def acquire_measurement_lock():
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateMutexW(None, True, "Local\\BixolonN100Measurement")
    error = ctypes.get_last_error()
    if not handle:
        raise OSError(error, "Cannot create measurement lock")
    if error == 183:
        kernel.CloseHandle(handle)
        raise RuntimeError("Another N100 measurement is already running. Wait for it to finish.")
    return handle


def release_measurement_lock(handle):
    if handle is None:
        return
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.ReleaseMutex(handle)
    kernel.CloseHandle(handle)


def run_worker(
    root: Path,
    version: str,
    relative: str,
    inputs: list[dict],
    output: Path,
    *,
    smoke: bool,
    experiment: dict | None = None,
    repetitions: int = 3,
) -> dict:
    output.mkdir(parents=True)
    package = child(root, relative)
    base_worker = package / "worker"
    worker = (
        child(root, experiment["worker"])
        if experiment and experiment.get("worker")
        else base_worker
    )
    profile = load_json_config(package / "provenance.json")["execution_profile"]
    profile.update((experiment or {}).get("profile", {}))
    runtime = (
        base_worker / "model-package"
        if not experiment or not experiment.get("runtime")
        else child(root, experiment["runtime"])
    )
    if experiment and experiment.get("runtime_overlay"):
        runtime = overlay_runtime(
            runtime,
            child(root, experiment["runtime_overlay"]),
            child(root, "prepared/" + experiment["id"]),
        )
    catalog = (
        child(root, experiment["catalog"])
        if experiment and experiment.get("catalog")
        else base_worker / "store-catalog"
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with (output / "worker.log").open("wb") as log:
        process = subprocess.Popen(
            [str(worker / "bixolon-worker.exe")],
            cwd=worker,
            env=worker_environment(worker, profile, port, runtime=runtime, catalog=catalog),
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            startup_started = time.perf_counter()
            print(
                f"Starting {version}; first GPU compilation may take several minutes...", flush=True
            )
            deadline = time.monotonic() + 600
            ready = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Worker exited before readiness; see worker.log")
                try:
                    status, payload, _ = request(port)
                    if status == 200 and payload.get("status") == "ready":
                        ready = payload
                        break
                except (OSError, ValueError, http.client.HTTPException):
                    pass
                time.sleep(1)
            if ready is None or ready.get("worker_version") != version:
                raise RuntimeError("Worker readiness/version check failed")
            startup_ms = (time.perf_counter() - startup_started) * 1000
            write_json(
                output / "environment.json",
                {
                    "hardware": windows_inventory(),
                    "ready": ready,
                    "profile": profile,
                    "experiment": experiment,
                    "startup_ms": startup_ms,
                },
            )
            print(f"{version}: actual provider = {ready['provider']}", flush=True)
            for index in range(1 if smoke else 10):
                code, warm, _ = request(port, inputs[index % len(inputs)]["bytes"])
                if code != 200 or warm.get("status") == "ERROR":
                    raise RuntimeError("Warmup failed; see worker.log")
            rows = []
            with (output / "responses.jsonl").open("w", encoding="utf-8") as stream:
                for repeat in range(1 if smoke else repetitions):
                    for index, entry in enumerate(inputs):
                        code, response, elapsed = request(port, entry["bytes"])
                        row = {
                            "repetition": repeat,
                            "image_id": entry["image_id"],
                            "image_sha256": entry["image_sha256"],
                            "elapsed_ms": elapsed,
                            "http_status": code,
                            "response": response,
                            "memory_after_request": process_memory(process.pid),
                        }
                        rows.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        if (index + 1) % 10 == 0 or index == len(inputs) - 1:
                            print(
                                f"{version} repeat {repeat + 1}: {index + 1}/{len(inputs)}",
                                flush=True,
                            )
            _, ready_end, _ = request(port)
            summary = {
                "version": version,
                "repetitions": [
                    summarize([r for r in rows if r["repetition"] == i])
                    for i in range(1 if smoke else repetitions)
                ],
                "latency": timing([r["elapsed_ms"] for r in rows]),
                "image_status_counts": dict(Counter(r["response"]["status"] for r in rows)),
                "item_status_counts": dict(
                    Counter(s["status"] for r in rows for s in r["response"]["segmentations"])
                ),
                "ready": ready,
                "ready_end": ready_end,
                "accuracy": "Requires frozen GT matching; APPROVED counts alone are not accuracy.",
            }
            write_json(output / "summary.json", summary)
            for i, result in enumerate(summary["repetitions"], 1):
                print(
                    f"Repeat {i}: p50/p95/p99 = {result['latency']['p50_ms']:.2f}/{result['latency']['p95_ms']:.2f}/{result['latency']['p99_ms']:.2f} ms",
                    flush=True,
                )
            return summary
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(15)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit-root", type=Path)
    parser.add_argument("--verify-only", "-VerifyOnly", action="store_true")
    parser.add_argument("--matrix", action="store_true", help="Run the frozen experiment matrix")
    parser.add_argument("--profile", help="Run one matrix profile by its ID")
    parser.add_argument("--repetitions", type=int, choices=[1, 3], default=None)
    parser.add_argument(
        "--smoke", action="store_true", help="Three images/version; not a performance result"
    )
    args = parser.parse_args(argv)
    root = (
        args.kit_root
        or (Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd())
    ).resolve()
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetDllDirectoryW(None)
    print("N100 measurement kit: no PowerShell or Python installation required.", flush=True)
    output = root / "results" / (time.strftime("N100-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    success = False
    measurement_lock = None
    try:
        print("Verifying kit checksums...", flush=True)
        manifest = verify_kit(root)
        print("All checksums verified.", flush=True)
        if args.verify_only:
            write_json(output / "verification.json", {"passed": True})
            success = True
            return 0
        measurement_lock = acquire_measurement_lock()
        ensure_idle()
        execution_root = stage_benchmark(root, manifest)
        inputs = [dict(r, bytes=child(root, r["path"]).read_bytes()) for r in manifest["inputs"]]
        if args.smoke:
            inputs = inputs[:3]
        if args.matrix or args.profile:
            experiments = manifest.get("experiments", [])
            if args.profile:
                experiments = [item for item in experiments if item["id"] == args.profile]
            if not experiments:
                raise ValueError("No matching matrix experiments")
            summary = []
            for experiment in experiments:
                identifier = experiment["id"]
                destination = child(output, identifier)
                print(f"EXPERIMENT {identifier}: {experiment['description']}", flush=True)
                try:
                    result = run_worker(
                        execution_root,
                        manifest["version"],
                        manifest["benchmark"],
                        inputs,
                        destination,
                        smoke=args.smoke,
                        experiment=experiment,
                        repetitions=args.repetitions or 1,
                    )
                    summary.append({"id": identifier, "summary": result})
                except Exception as exc:
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / "FAILED.txt").write_text(str(exc), encoding="utf-8")
                    summary.append({"id": identifier, "failed": str(exc)})
                    print(f"Experiment failed; logs kept: {identifier}: {exc}", flush=True)
                write_json(output / "matrix-progress.json", summary)
        else:
            summary = run_worker(
                execution_root,
                manifest["version"],
                manifest["benchmark"],
                inputs,
                output / "measurement",
                smoke=args.smoke,
                repetitions=args.repetitions or 3,
            )
        write_json(
            output / "measurement-summary.json",
            {
                "smoke_only": args.smoke,
                "benchmark_staged_to_local_cache": execution_root != root,
                "experiment": manifest.get("experiment"),
                "summary": summary,
                "note": "Inspect Worker logs for provider transitions. Accuracy requires GT matching.",
            },
        )
        success = not isinstance(summary, list) or not any("failed" in item for item in summary)
    except Exception as exc:
        (output / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(f"ERROR: {exc}", flush=True)
    finally:
        release_measurement_lock(measurement_lock)
        archive = output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in output.rglob("*"):
                if path.is_file():
                    bundle.write(path, path.relative_to(output.parent))
        print(
            f"Result ZIP: {archive}\n{'DONE' if success else 'FAILED - logs preserved'}", flush=True
        )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
