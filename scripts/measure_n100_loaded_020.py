"""Measure the Intel UHD execution profile on the build host, without claiming N100 timing."""

from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.evaluation.three_bakery_http import measure
from bixolon_scanner.training.three_bakery_data import read_jsonl


def main():
    config = load_json_config(Path("configs/experiments/bread/n100_020.json"))
    work = Path(config["work"])
    report = measure(
        work / "candidates/dfine-dense-selective",
        read_jsonl(Path(config["log_manifest"])),
        work / "measurements/build-host-openvino-loaded",
        python=Path("artifacts/build-envs/worker-openvino-gpu-py311/Scripts/python.exe"),
        provider="cpu",
        embedder_provider="openvino_gpu",
        embedder_fallback_provider="same",
        execution_cpu_fallback=True,
        cpu_profile=(8, 4),
        warmup=10,
        repetitions=1,
        maximum_p95_ms=200,
        input_identity={
            "hardware": "Core Ultra 9 285K and RTX 5080; not N100; effective provider recorded in ready/ready_end"
        },
    )
    print(report["ready"], report["ready_end"], report["summary"], flush=True)


if __name__ == "__main__":
    main()
