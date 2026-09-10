"""Post-development build-host compatibility check; never label as N100 performance."""

from pathlib import Path

from bixolon_scanner.evaluation.three_bakery_http import measure, provider_parity
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main():
    output = Path("artifacts/retraining/recapture-0.2.1/n100-build-host")
    records = read_jsonl(
        Path("artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl")
    )
    report = measure(
        Path("artifacts/versions/0.2.1/staging"),
        records,
        output,
        python=Path("artifacts/build-envs/worker-openvino-gpu-py311/Scripts/python.exe"),
        worker_executable=Path("artifacts/n100/0.2.1/worker-payload/worker/bixolon-worker.exe"),
        provider="cpu",
        embedder_provider="openvino_gpu",
        embedder_fallback_provider="same",
        execution_cpu_fallback=True,
        verifier_provider="cpu",
        cpu_profile=(2, 4),
        warmup=10,
        repetitions=1,
        input_identity={
            "scope": "Post-development compatibility/fallback smoke on Core Ultra 9 285K; NOT N100 hardware measurement"
        },
    )
    summary = report["summary"]
    parity = provider_parity(
        output.parent / "packaged/cpu/0.2.1/responses.jsonl", output / "responses.jsonl"
    )
    write_json(
        output / "compatibility.json",
        {
            "n100_hardware_measured": False,
            "summary": summary,
            "cpu_parity": parity,
            "accuracy_passed": summary["correct_approved_count"] >= 1078
            and summary["wrong_approved_count"] == 0
            and summary["missed_count"] == 0,
            "scope": "Actual N100-targeted EXE on build host; inspect readiness for actual provider and fallback",
        },
    )
    print(summary, parity, flush=True)


if __name__ == "__main__":
    main()
