"""Profile a frozen candidate in process; these are not HTTP or N100 target timings."""

import argparse
import json
import time
from pathlib import Path

from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.evaluation.three_bakery import score_response
from bixolon_scanner.evaluation.three_bakery_http import summarize
from bixolon_scanner.runtime.imaging import decode_image
from bixolon_scanner.runtime.inference_timing import collect_inference_timings
from bixolon_scanner.worker.runtime_factory import build_worker_runtime
from bixolon_scanner.worker.settings import WorkerSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--embedder-provider", choices=["same", "openvino_gpu"], default="openvino_gpu"
    )
    parser.add_argument("--precision", choices=["f32", "f16"], default="f16")
    parser.add_argument("--detector-threads", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    runtime = build_worker_runtime(
        WorkerSettings(
            package_dir=args.candidate / "runtime",
            catalog_dir=args.candidate / "catalog",
            provider="cpu",
            embedder_provider=args.embedder_provider,
            verifier_provider="cpu",
            openvino_gpu_precision=args.precision,
            cpu_detector_intra_op_threads=args.detector_threads,
            cpu_embedder_intra_op_threads=4,
        )
    )
    rows = []
    try:
        with (args.output / "responses.jsonl").open("w", encoding="utf-8") as stream:
            for index, record in enumerate(records):
                path = Path(record["image_path"])
                if sha256_file(path) != record["image_sha256"]:
                    raise ValueError("Frozen input hash mismatch")
                with decode_image(
                    path.read_bytes(),
                    max_bytes=30_000_000,
                    max_pixels=50_000_000,
                    jpeg_draft_size=runtime.jpeg_draft_size,
                ) as image:
                    with collect_inference_timings() as calls:
                        started = time.perf_counter()
                        response = runtime.scan(image, f"profile-static-{index}")
                        elapsed = (time.perf_counter() - started) * 1000
                row = {
                    "image_id": record["image_id"],
                    "image_sha256": record["image_sha256"],
                    "elapsed_ms": elapsed,
                    "response": response.model_dump(mode="json"),
                    "model_calls": calls,
                    "metrics": score_response(response, record["annotations"], threshold=0.5),
                }
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
        report = {
            "timing_scope": "Build-host in-process pipeline, excludes HTTP and decode",
            "actual_provider": runtime.provider,
            "gpu_precision_hint": args.precision,
            "input_manifest_sha256": sha256_file(args.manifest),
            "summary": summarize(rows),
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
