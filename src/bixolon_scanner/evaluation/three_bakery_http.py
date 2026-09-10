"""Real HTTP Worker measurements with fixed image denominators and repeat parity."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

import httpx

from ..configuration import load_json_config
from ..contracts import ScanResponse
from ..contracts.artifact import directory_content_manifest
from ..contracts.catalog import sha256_file
from ..training.three_bakery_data import read_jsonl, source_path, write_json, write_jsonl
from .three_bakery import score_response, summarize


def interpreter_identity(python: Path) -> dict:
    script = (
        "import json,sys,onnxruntime; from importlib.metadata import version; "
        "print(json.dumps({'python':sys.version,'onnxruntime':onnxruntime.__version__,"
        "'packages':{name:version(name) for name in ['numpy','pillow','fastapi','pydantic','uvicorn']}}))"
    )
    result = subprocess.run(
        [str(python), "-c", script],
        capture_output=True,
        text=True,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return json.loads(result.stdout)


def measurement_environment() -> dict:
    """Refuse overlapping experiment work before starting the HTTP measurement."""
    import onnxruntime
    import psutil

    own = psutil.Process()
    excluded = {own.pid, *(p.pid for p in own.parents())}
    conflicting = []
    suspended = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid in excluded or "python" not in (process.info["name"] or "").lower():
            continue
        arguments = process.info["cmdline"] or []
        text = " ".join(arguments)
        training = "bixolon_scanner.training." in text
        benchmark = "bixolon_scanner.evaluation.three_bakery_parity" in text or (
            "bixolon_scanner.experiments.bread.three_bakery" in text
            and any(stage in arguments for stage in ("train", "compare", "evaluate", "export"))
        )
        if training or benchmark:
            if process.status() == psutil.STATUS_STOPPED:
                suspended.append(process.pid)
            else:
                conflicting.append(process.pid)
    if conflicting:
        raise RuntimeError(
            f"CPU measurement cannot overlap other training/benchmarks: {conflicting}"
        )
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "onnxruntime": onnxruntime.__version__,
        "overlapping_experiment_processes": conflicting,
        "suspended_experiment_processes": suspended,
        "overlap_check_scope": "visible Python experiment command lines plus serial orchestration",
        "recorded_at_unix": time.time(),
    }


@contextmanager
def worker_server(
    candidate: Path,
    python: Path,
    provider: str,
    output: Path,
    cuda_dll_dir: Path | None = None,
    cpu_profile: tuple[int, int] = (4, 4),
    worker_executable: Path | None = None,
):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = {k: v for k, v in os.environ.items() if not k.startswith("BIXOLON_")}
    environment.update(
        {
            "BIXOLON_PACKAGE_DIR": str((candidate / "runtime").resolve()),
            "BIXOLON_CATALOG_DIR": str((candidate / "catalog").resolve()),
            "BIXOLON_CATALOG_STORE_ID": "three_bakery",
            "BIXOLON_PROVIDER": provider,
            "BIXOLON_HOST": "127.0.0.1",
            "BIXOLON_PORT": str(port),
            "BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS": str(cpu_profile[0]),
            "BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS": str(cpu_profile[1]),
            "BIXOLON_REQUEST_TIMEOUT_SECONDS": "120",
            "BIXOLON_LOG_LEVEL": "INFO",
        }
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [environment.get("PYTHONPATH"), str(Path(__file__).resolve().parents[2])])
    )
    if cuda_dll_dir is not None:
        environment["BIXOLON_CUDA_DLL_DIR"] = str(cuda_dll_dir.resolve())
    output.mkdir(parents=True, exist_ok=True)
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(worker_executable.resolve())]
            if worker_executable is not None
            else [str(python), "-c", "from bixolon_scanner.worker.cli import serve; serve()"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=130, trust_env=False
            ) as client:
                deadline = time.monotonic() + 180
                ready = None
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            "Worker exited during initialization; inspect evaluation worker.log"
                        )
                    try:
                        response = client.get("/health/ready")
                        if response.status_code == 200:
                            ready = response.json()
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.25)
                if ready is None or ready["provider"] != provider:
                    raise RuntimeError("Worker did not become ready with the requested provider")
                yield client, ready
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)


def response_signature(response: dict) -> dict:
    return {
        "status": response["status"],
        "reason_codes": response["reason_codes"],
        "segmentations": [
            {
                "status": value["status"],
                "reason_codes": value["reason_codes"],
                "class_id": (value.get("prediction") or {}).get("class_id"),
                "top3": [candidate["class_id"] for candidate in value["top3"]],
            }
            for value in response["segmentations"]
        ],
    }


def measure(
    candidate: Path,
    records: list[dict],
    output: Path,
    *,
    python: Path,
    provider: str,
    cuda_dll_dir: Path | None = None,
    warmup: int = 10,
    repetitions: int = 1,
    minimum_complete_images: int | None = None,
    match_iou: float = 0.5,
    input_identity: dict | None = None,
    cpu_profile: tuple[int, int] = (4, 4),
    maximum_p95_ms: float | None = None,
    worker_executable: Path | None = None,
) -> dict:
    if not records or repetitions < 1 or warmup < 0:
        raise ValueError("invalid measurement budget")
    contract = {
        "runtime_sha256": sha256_file(candidate / "runtime/metadata.json"),
        "catalog_checksums_sha256": sha256_file(candidate / "catalog/checksums.json"),
        "input_identity": input_identity,
        "provider": provider,
        "cpu_profile": list(cpu_profile),
        "maximum_p95_ms": maximum_p95_ms,
        "concurrent_requests": 1,
        "python_executable": str(python.resolve()),
        "interpreter_identity": interpreter_identity(python),
        "warmup": warmup,
        "repetitions": repetitions,
        "image_count": len(records),
        "records_sha256": sha256(
            json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "match_iou": match_iou,
        "minimum_complete_images": minimum_complete_images,
        "code_sha256": sha256_file(Path(__file__)),
        "scoring_code_sha256": sha256_file(Path(__file__).with_name("three_bakery.py")),
        "inference_code_sha256": {
            path.relative_to(Path(__file__).resolve().parents[1]).as_posix(): sha256_file(path)
            for directory in ("contracts", "pipeline", "runtime", "worker")
            for path in sorted((Path(__file__).resolve().parents[1] / directory).glob("*.py"))
        },
    }
    if worker_executable is not None:
        contract["packaged_worker"] = {
            "executable": str(worker_executable.resolve()),
            "executable_sha256": sha256_file(worker_executable),
            "directory_manifest_sha256": directory_content_manifest(worker_executable.parent)[
                "manifest_sha256"
            ],
        }
    if (output / "report.json").exists():
        existing = load_json_config(output / "report.json")
        if (
            existing["contract"] != contract
            or sha256_file(output / "responses.jsonl") != existing["responses_sha256"]
        ):
            raise ValueError("measurement resume input/output mismatch")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "contract.json", contract)
    environment = measurement_environment() if provider == "cpu" else None
    rows = []
    with worker_server(
        candidate, python, provider, output, cuda_dll_dir, cpu_profile, worker_executable
    ) as (
        client,
        ready,
    ):
        for index in range(warmup):
            record = records[index % len(records)]
            path = Path(record["image_path"])
            data = path.read_bytes()
            if sha256(data).hexdigest() != record["image_sha256"]:
                raise ValueError("warmup image differs from the frozen input manifest")
            warm = client.post("/v1/scan", files={"image": (path.name, data, "image/jpeg")})
            ScanResponse.model_validate(warm.json())
        with (output / "responses.jsonl").open("w", encoding="utf-8") as stream:
            for repetition in range(repetitions):
                for index, record in enumerate(records):
                    path = Path(record["image_path"])
                    data = path.read_bytes()
                    if sha256(data).hexdigest() != record["image_sha256"]:
                        raise ValueError("evaluation image differs from the frozen input manifest")
                    started = time.perf_counter()
                    transport_error = None
                    try:
                        raw = client.post(
                            "/v1/scan", files={"image": (path.name, data, "image/jpeg")}
                        )
                        response = ScanResponse.model_validate(raw.json())
                        http_status = raw.status_code
                    except (httpx.TransportError, ValueError) as exc:
                        transport_error = type(exc).__name__
                        http_status = None
                        response = ScanResponse(
                            request_id=f"measurement-error-{repetition}-{index}",
                            status="ERROR",
                            reason_codes=["MODEL_EXECUTION_FAILED"],
                            segmentations=[],
                            processing_time_ms=0,
                            worker_version=ready["worker_version"],
                            detector_version=None,
                            classifier_version=None,
                        )
                    elapsed = (time.perf_counter() - started) * 1000
                    row = {
                        "image_id": record["image_id"],
                        "image_sha256": record["image_sha256"],
                        "repetition": repetition,
                        "difficulty": record.get("difficulty", "unspecified"),
                        "elapsed_ms": elapsed,
                        "http_status": http_status,
                        "transport_error": transport_error,
                        "response": response.model_dump(mode="json"),
                        "metrics": score_response(
                            response, record["annotations"], threshold=match_iou
                        ),
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    if (index + 1) % 50 == 0:
                        print(
                            f"{provider} {output.name} repeat {repetition + 1}: {index + 1}/{len(records)}",
                            flush=True,
                        )
    per_repeat = [
        summarize(
            [r for r in rows if r["repetition"] == i],
            minimum_complete_images=minimum_complete_images,
            maximum_p95_ms=maximum_p95_ms,
        )
        for i in range(repetitions)
    ]
    signatures = defaultdict(list)
    for row in rows:
        signatures[row["image_id"]].append(response_signature(row["response"]))
    unstable = [
        image_id
        for image_id, values in signatures.items()
        if any(value != values[0] for value in values[1:])
    ]
    result = {
        "contract": contract,
        "ready": ready,
        "environment": environment,
        "summary": per_repeat[0],
        "repetitions": per_repeat,
        "latency": summarize(rows)["latency"],
        "repeat_status_rank_mismatch_image_ids": unstable,
        "target_met_all_repetitions": all(v["target_met"] for v in per_repeat)
        if minimum_complete_images is not None
        else None,
        "responses_sha256": sha256_file(output / "responses.jsonl"),
    }
    write_json(output / "report.json", result)
    return result


def final_records(config: dict, freeze: Path, output: Path) -> list[dict]:
    if not freeze.is_file():
        raise ValueError("final benchmark cannot be read before candidate/policy freeze")
    settings = config["evaluation"]
    root = Path(settings["dataset_root"]).resolve()
    annotations_path = source_path(root, settings["annotations"])
    access_path = output / "benchmark-access.json"
    access = {
        "freeze_sha256": sha256_file(freeze),
        "annotation_sha256": sha256_file(annotations_path),
        "accessed_at_unix": time.time(),
        "evaluation_role": "historically_used_fixed_benchmark",
    }
    if access_path.exists():
        previous = load_json_config(access_path)
        if any(previous[key] != access[key] for key in ("freeze_sha256", "annotation_sha256")):
            raise ValueError("previously accessed benchmark or frozen candidate changed")
        access = previous
    # Even a failed GT parse/count check is benchmark access and must prohibit training.
    write_json(access_path, access)
    coco = load_json_config(annotations_path)
    grouped = defaultdict(list)
    for annotation in coco["annotations"]:
        grouped[annotation["image_id"]].append(annotation)
    rows = []
    for image in sorted(coco["images"], key=lambda value: value["id"]):
        relative = Path(image["file_name"])
        if relative.parts[0] != settings["image_directory"]:
            relative = Path(settings["image_directory"]) / relative
        path = source_path(root, str(relative))
        path.relative_to(root / settings["image_directory"])
        rows.append(
            {
                "image_id": image["id"],
                "image_path": str(path),
                "image_sha256": sha256_file(path),
                "annotations": grouped[image["id"]],
                "difficulty": image.get("difficulty")
                or f"object_count_{len(grouped[image['id']])}",
                "difficulty_source": "annotation"
                if image.get("difficulty")
                else "ground_truth_object_count",
                "width": image["width"],
                "height": image["height"],
            }
        )
    if (
        len(rows) != settings["expected_images"]
        or sum(len(r["annotations"]) for r in rows) != settings["expected_objects"]
    ):
        raise ValueError("fixed benchmark image/object count mismatch")
    manifest = output / "final-inputs.jsonl"
    if manifest.exists() and (
        sha256_file(manifest) != access.get("input_manifest_sha256") or read_jsonl(manifest) != rows
    ):
        raise ValueError("previously frozen benchmark image manifest changed")
    write_jsonl(manifest, rows)
    write_json(access_path, {**access, "input_manifest_sha256": sha256_file(manifest)})
    return rows


def provider_parity(cpu_path: Path, cuda_path: Path) -> dict:
    cpu = {(r["repetition"], r["image_id"]): r for r in read_jsonl(cpu_path)}
    cuda = {(r["repetition"], r["image_id"]): r for r in read_jsonl(cuda_path)}
    if cpu.keys() != cuda.keys():
        raise ValueError("provider parity requires the same image IDs and repetitions")
    mismatches = []
    for key, row in cpu.items():
        other = cuda[key]
        if row["image_sha256"] != other["image_sha256"]:
            raise ValueError("provider parity image checksum mismatch")
        if response_signature(row["response"]) != response_signature(other["response"]):
            mismatches.append({"repetition": key[0], "image_id": key[1]})
    return {
        "image_count": len({key[1] for key in cpu}),
        "compared_request_count": len(cpu),
        "status_rank_mismatch_image_ids": sorted({row["image_id"] for row in mismatches}),
        "mismatches": mismatches,
        "status_rank_parity": not mismatches,
    }
