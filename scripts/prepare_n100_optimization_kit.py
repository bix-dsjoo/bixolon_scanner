"""Assemble a new experimental kit; never overwrite the released 0.2.1 payload."""

import argparse
import json
import shutil
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.operations.static_batch_candidate import specialize_runtime


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-kit", type=Path, default=Path("D:/N100"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = Path("artifacts/n100/optimization")
    source = Path("artifacts/versions/0.2.1/staging")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = output / "Optimization"
    worker = package / "worker"
    shutil.copytree(work / "worker-build/bixolon-worker", worker)
    specialization = specialize_runtime(source / "runtime", worker / "model-package")
    # The selected candidate's graphs are the exact ones used by the regression runs.
    for item in specialization["models"]:
        if item["candidate_sha256"] != sha256_file(work / "static1/runtime" / item["model"]):
            raise ValueError("Reproduced model differs from selected candidate")
    shutil.copytree(source / "catalog", worker / "store-catalog")
    shutil.copytree(Path("artifacts/n100/0.2.1/worker-payload/licenses"), package / "licenses")
    profile = {
        "provider": "cpu",
        "embedder_provider": "openvino_gpu",
        "embedder_fallback_provider": "same",
        "provider_execution_cpu_fallback": True,
        "verifier_provider": "cpu",
        "cpu_detector_intra_op_threads": 4,
        "cpu_embedder_intra_op_threads": 4,
        "openvino_gpu_precision": "f16",
        "log_model_timings": True,
    }
    write_json(
        package / "provenance.json",
        {
            "version": "0.2.1",
            "experiment": "n100-static1-f16-20260909",
            "purpose": "Experimental benchmark, not a new product release or proof of N100 target attainment",
            "execution_profile": profile,
            "specialization": specialization,
            "original_runtime": directory_content_manifest(source / "runtime"),
            "selection_sha256": sha256_file(work / "selection.json"),
        },
    )
    shutil.copy2(work / "runner-build-final/N100-OPTIMIZE.exe", output / "N100-OPTIMIZE.exe")
    previous = load_json_config(args.existing_kit / "KIT-MANIFEST.json")
    frozen_inputs = [
        json.loads(line)
        for line in Path("artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected = {(row["image_id"], row["image_sha256"]) for row in frozen_inputs}
    actual = {(row["image_id"], row["image_sha256"]) for row in previous["inputs"]}
    if len(previous["inputs"]) != 132 or len(expected) != 132 or actual != expected:
        raise ValueError("Kit input identities differ from frozen log132")
    for row in previous["inputs"]:
        src, dst = args.existing_kit / row["path"], output / row["path"]
        if not dst.resolve().is_relative_to(output):
            raise ValueError("Input path escapes kit")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256_file(dst) != row["image_sha256"]:
            raise ValueError("Copied test image hash mismatch")
    common = '@echo off\nsetlocal\n"%~dp0N100-OPTIMIZE.exe" {args}\nset "RESULT=%ERRORLEVEL%"\necho.\npause\nexit /b %RESULT%\n'
    for name, argument in [("2_MEASURE.cmd", "%*"), ("3_VERIFY_FILES.cmd", "--verify-only %*")]:
        (output / name).write_bytes(
            common.replace("{args}", argument).replace("\n", "\r\n").encode("ascii")
        )
    (output / "README_KO.txt").write_text(
        "N100 1초 목표 최적화 시험 키트 — 제품 0.2.1 기반 실험\n\n"
        "1. Scanner Lite와 다른 Scanner Worker를 닫으세요.\n"
        "2. 2_MEASURE.cmd를 실행하세요. PowerShell·Python 설치는 필요 없습니다.\n"
        "3. 첫 실행에는 로컬 모델 복사·검증과 GPU 준비에 몇 분이 걸릴 수 있습니다.\n"
        "4. warmup10회 뒤 132장×3회를 측정합니다. results의 새 ZIP을 가져오세요.\n\n"
        "CPU 검출 → GPU 기본/상세 분류 → CPU verifier와 CPU fallback을 유지합니다.\n"
        "고정 batch1, GPU FP16 힌트, CPU 검출4스레드 후보입니다. 판정 임계값은 그대로입니다.\n"
        "모델별 시간, 메모리, 전체 응답, p50/p95/p99/max 및 1초 초과 개수를 보존합니다.\n"
        "실제 N100에서 1초 달성 여부는 아직 미확인입니다. 개발 PC 결과를 N100 실측으로 표시하지 않습니다.\n"
        "1_INSTALL.cmd가 있다면 기존 정식 0.2.1 설치본입니다. 이번 최적화 시험에 재설치는 필요 없습니다.\n"
        "Optimization 폴더는 검증 중인 실험용 Worker이며 정식 새 제품 배포물이 아닙니다.\n"
        "원래 Benchmark와 이전 결과는 보존합니다. 0.2.0 비교는 하지 않습니다.\n"
        "로컬 캐시는 %LOCALAPPDATA%\\BixolonN100Benchmark에 있으며 시험 후 삭제할 수 있습니다.\n"
        "3_VERIFY_FILES.cmd는 현재 시험 파일의 checksum을 확인합니다. 발행자 인증은 아닙니다.\n",
        encoding="utf-8-sig",
    )
    files = [
        {
            "path": p.relative_to(output).as_posix(),
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
        }
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    write_json(
        output / "KIT-MANIFEST.json",
        {
            "version": "0.2.1",
            "experiment": "n100-static1-f16-20260909",
            "benchmark": "Optimization",
            "powershell_required": False,
            "image_count": 132,
            "warmup": 10,
            "repetitions": 3,
            "files": files,
            "inputs": previous["inputs"],
            "n100_target_measured": False,
        },
    )
    print(f"Prepared {output} with {len(files)} verified files", flush=True)


if __name__ == "__main__":
    main()
