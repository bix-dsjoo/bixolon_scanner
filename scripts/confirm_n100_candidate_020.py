"""Serial build-host CPU/CUDA confirmation of the zero-error local proposal policy."""

from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.evaluation.three_bakery_http import measure
from bixolon_scanner.training.three_bakery_data import read_jsonl


def main():
    config = load_json_config(Path("configs/experiments/bread/n100_020.json"))
    work = Path(config["work"])
    candidate = work / "candidates/dfine-dense-local-recapture"
    rows = read_jsonl(Path(config["log_manifest"]))
    for provider, threads in [("cpu", (12, 8)), ("cpu", (8, 8)), ("cuda", (8, 8))]:
        python = (
            Path("artifacts/build-envs")
            / ("worker-cuda-py311" if provider == "cuda" else "worker-cpu-0.1.17-py311")
            / "Scripts/python.exe"
        )
        result = measure(
            candidate,
            rows,
            work / f"confirmation/{provider}-{threads[0]}-{threads[1]}",
            python=python,
            provider=provider,
            cpu_profile=threads,
            warmup=10,
            repetitions=3 if provider == "cuda" else 1,
            maximum_p95_ms=200,
            cuda_dll_dir=Path("artifacts/runtime/cuda-13.0-cudnn-9.13.1-win-x64")
            if provider == "cuda"
            else None,
        )
        print(provider, threads, result["summary"], flush=True)


if __name__ == "__main__":
    main()
