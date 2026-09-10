"""Assemble a separate, reproducible N100 experiment matrix without changing a release."""

import argparse
import json
import shutil
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.operations.n100_test_kit import child, digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-kit", type=Path, default=Path("D:/N100"))
    args = parser.parse_args()
    work = Path("artifacts/n100/optimization-r2")
    previous = Path("artifacts/n100/optimization/static1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = output / "Experiment"
    worker = package / "worker"
    shutil.copytree(work / "worker-build/bixolon-worker", worker)
    shutil.copytree(work / "cpu-worker-build/bixolon-worker", package / "cpu-worker")
    shutil.copytree(previous / "runtime", worker / "model-package")
    shutil.copytree(previous / "catalog", worker / "store-catalog")
    shutil.copytree("artifacts/n100/0.2.1/worker-payload/licenses", package / "licenses")
    variants = {}
    for name in ("folded", "int8", "static2", "dynamic"):
        source = work / name / "runtime"
        overlay = package / "overlays" / name
        overlay.mkdir(parents=True)
        changes = []
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            baseline = previous / "runtime" / relative
            if baseline.is_file() and digest(path) == digest(baseline):
                continue
            target = overlay / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            changes.append({"path": relative.as_posix(), "sha256": digest(path)})
        variants[name] = changes
    base = {
        "provider": "cpu",
        "embedder_provider": "openvino_gpu",
        "embedder_fallback_provider": "same",
        "provider_execution_cpu_fallback": True,
        "verifier_provider": "cpu",
        "cpu_detector_intra_op_threads": 4,
        "cpu_embedder_intra_op_threads": 4,
        "openvino_gpu_precision": "f16",
        "log_model_timings": True,
        "reuse_verifier_embeddings": False,
        "parallel_verification": False,
    }
    experiments = []

    def trial(identifier, description, changes=None, overlay=None):
        row = {"id": identifier, "description": description, "profile": base | (changes or {})}
        if overlay:
            row["runtime_overlay"] = f"Experiment/overlays/{overlay}"
        experiments.append(row)

    reuse = {"reuse_verifier_embeddings": True}
    parallel = reuse | {"parallel_verification": True}
    trial("A0_gpu_reference", "Current static1 GPU FP16 execution, without reuse")
    trial("A1_reuse", "Exact verifier input reuse within each request", reuse)
    trial("A2_parallel", "Reuse plus overlapping GPU rotation and CPU verification", parallel)
    trial(
        "B_cpu_threads2",
        "CPU detector and verifier use two threads, GPU classifier unchanged",
        parallel | {"cpu_detector_intra_op_threads": 2, "cpu_embedder_intra_op_threads": 2},
    )
    trial(
        "C_cpu_verifier",
        "Native CPU detector, GPU classifier, folded OpenVINO CPU FP32 verifier",
        parallel | {"verifier_provider": "openvino"},
        "folded",
    )
    trial(
        "D_int8_verifier",
        "CPU detector, GPU classifier, INT8 MatMul CPU verifier",
        parallel,
        "int8",
    )
    trial(
        "E_gpu_batch2",
        "GPU primary batch2 with reuse and parallel verification",
        parallel,
        "static2",
    )
    trial(
        "F_cpu_reference",
        "Entire inference on native CPU, without reuse",
        {"embedder_provider": "same"},
    )
    trial(
        "G_cpu_dynamic",
        "ONNX Runtime 1.24 CPU with full ROI batches and verifier reuse",
        reuse | {"embedder_provider": "same"},
        "dynamic",
    )
    trial(
        "H_cpu_ort129_dynamic",
        "ONNX Runtime 1.29 CPU with full ROI batches and verifier reuse",
        reuse | {"embedder_provider": "same"},
        "dynamic",
    )
    experiments[-1]["worker"] = "Experiment/cpu-worker"
    trial("A9_gpu_reference_repeat", "Repeat initial reference to expose time/thermal drift")
    write_json(
        package / "provenance.json",
        {
            "version": "0.2.1",
            "experiment": "n100-matrix-r2-20260909",
            "purpose": "Experimental Worker, not an installed release or N100 performance claim",
            "execution_profile": base,
            "variant_files": variants,
            "note": "INT8 changes verifier weights; other policies, classes and objects are unchanged.",
        },
    )
    old = load_json_config(args.existing_kit / "KIT-MANIFEST.json")
    frozen = [
        json.loads(line)
        for line in Path("artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if {(r["image_id"], r["image_sha256"]) for r in old["inputs"]} != {
        (r["image_id"], r["image_sha256"]) for r in frozen
    } or len(old["inputs"]) != 132:
        raise ValueError("Frozen input set mismatch")
    for row in old["inputs"]:
        target = child(output, row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child(args.existing_kit, row["path"]), target)
    shutil.copy2(work / "runner-build/N100-EXPERIMENTS.exe", output / "N100-EXPERIMENTS.exe")
    for name, arguments in (
        ("1_RUN_ALL.cmd", "--matrix %*"),
        ("2_VERIFY_FILES.cmd", "--verify-only %*"),
    ):
        (output / name).write_bytes(
            (
                f'@echo off\r\nsetlocal\r\n"%~dp0N100-EXPERIMENTS.exe" {arguments}\r\nset "RESULT=%ERRORLEVEL%"\r\necho.\r\npause\r\nexit /b %RESULT%\r\n'
            ).encode("ascii")
        )
    (output / "README_KO.txt").write_text(
        "N100 전체 파이프라인 실험 R2 — 0.2.1 기반, 정식 설치본과 별도\n\n"
        "Scanner Lite와 다른 Worker를 닫고 1_RUN_ALL.cmd를 한 번 실행하세요.\n"
        "PowerShell/Python 설치나 앱 재설치가 필요 없습니다. 이 폴더의 새 Worker를 실행합니다.\n"
        "11개 설정을 순서대로 각각 warmup10회 + 132장 1회 측정합니다(총 1,452개).\n"
        "약 30~60분을 예상하지만 CPU 속도와 최초 컴파일에 따라 더 걸릴 수 있습니다.\n"
        "화면 절전/잠자기를 피하고 다른 작업을 멈춘 상태로 전원에 연결해 주세요.\n"
        "중간 실패도 기록하고 다음 후보를 실행합니다. 다른 측정과 중복 실행하지 마세요.\n"
        "마지막 Result ZIP 경로에 표시된 ZIP 한 개를 가져오세요.\n\n"
        "A0/A9는 시작·종료 기준, A1은 중복 재사용, A2는 CPU/GPU 검증 동시 실행입니다.\n"
        "B는 CPU 2스레드, C는 CPU 검증 엔진, D는 INT8 검증 모델, E는 GPU batch2입니다.\n"
        "F/G/H는 GPU 없이 전체 CPU 경로에서 ROI 전체 batch 및 ORT 버전을 비교합니다.\n"
        "모든 후보에 같은 객체·판정 정책을 적용하며 정확도는 반환 결과를 GT와 대조합니다.\n"
        "실행 시간이 빨라도 오류·오승인·미검출이 생긴 후보는 채택하지 않습니다.\n"
        "GPU 후보의 CPU fallback은 유지하며 실제 사용 provider와 전환 로그를 보존합니다.\n"
        "p50/p95/p99/max, 1초 초과 수, 상태별 수, 메모리, 모델별 시간과 응답을 기록합니다.\n"
        "시작/복사/컴파일 시간은 HTTP 시간과 별도로 기록합니다.\n"
        "전체 매트릭스는 후보 탐색입니다. 선정 후보는 3회 이상 다시 측정합니다.\n"
        "원래 0.2.1 설치 파일과 기존 측정 결과는 변경하지 않습니다.\n"
        "%LOCALAPPDATA%\\BixolonN100Benchmark의 캐시는 실험 종료 후 삭제할 수 있습니다.\n"
        "checksum은 파일 손상 검증이며 발행자 인증은 아닙니다.\n",
        encoding="utf-8-sig",
    )
    write_json(
        output / "KIT-MANIFEST.json",
        {
            "version": "0.2.1",
            "experiment": "n100-matrix-r2-20260909",
            "benchmark": "Experiment",
            "powershell_required": False,
            "inputs": old["inputs"],
            "experiments": experiments,
            "files": [
                {
                    "path": p.relative_to(output).as_posix(),
                    "sha256": digest(p),
                    "size_bytes": p.stat().st_size,
                }
                for p in sorted(output.rglob("*"))
                if p.is_file()
            ],
        },
    )
    print(f"Prepared {output}: {len(experiments)} profiles / 132 images")


if __name__ == "__main__":
    main()
