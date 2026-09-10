"""Package the N100-selected combinations and exact static-batch routing experiment."""

import argparse
import shutil
from pathlib import Path

from bixolon_scanner.operations.n100_test_kit import child, digest, verify_kit, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = Path("artifacts/n100/optimization-r3")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = output / "Experiment"
    worker = package / "worker"
    shutil.copytree(work / "worker-build/bixolon-worker", worker)
    baseline = work / "combined2/runtime"
    shutil.copytree(baseline, worker / "model-package")
    shutil.copytree(work / "combined2/catalog", worker / "store-catalog")
    shutil.copytree("artifacts/n100/0.2.1/worker-payload/licenses", package / "licenses")
    variants = {}
    for name, source in {
        "reference": Path("artifacts/n100/optimization-r2/static2/runtime"),
        "pool2": work / "pool2/runtime",
        "pool4": work / "pool4/runtime",
    }.items():
        overlay = package / "overlays" / name
        overlay.mkdir(parents=True)
        variants[name] = []
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            original = baseline / relative
            if original.is_file() and digest(path) == digest(original):
                continue
            target = overlay / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            variants[name].append({"path": relative.as_posix(), "sha256": digest(path)})
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
        "reuse_verifier_embeddings": True,
        "parallel_verification": True,
    }
    experiments = []

    def trial(identifier, description, overlay=None, **changes):
        entry = {"id": identifier, "description": description, "profile": profile | changes}
        if overlay:
            entry["runtime_overlay"] = f"Experiment/overlays/{overlay}"
        experiments.append(entry)

    trial("A0_gpu_reference", "R2 best p95: fixed batch2 and FP32 CPU verifier", "reference")
    trial("B_combined_int8", "Fixed batch2 plus INT8 CPU verifier")
    trial("C_exact_batch2", "Batch2/1 without padding plus INT8 verifier", "pool2")
    trial("D_exact_batch4", "Batch4/2/1 without padding plus INT8 verifier", "pool4")
    trial(
        "E_batch4_serial",
        "Batch4/2/1 and INT8 with serial verification",
        "pool4",
        parallel_verification=False,
    )
    trial("A9_gpu_reference_repeat", "Repeat R2 best p95 to measure time drift", "reference")
    write_json(
        package / "provenance.json",
        {
            "version": "0.2.1",
            "experiment": "n100-r3-exact-batch-20260909",
            "execution_profile": profile,
            "variant_files": variants,
            "purpose": "Separate experimental Worker, not a replacement installed release",
            "note": "INT8 verifier weights are experimental; all states and ROI policies are preserved.",
        },
    )
    old_root = Path("artifacts/n100/optimization-r2/kit")
    old = verify_kit(old_root)
    for row in old["inputs"]:
        target = child(output, row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child(old_root, row["path"]), target)
    shutil.copy2(old_root / "N100-EXPERIMENTS.exe", output / "N100-EXPERIMENTS.exe")
    for name, arguments in (
        ("1_RUN_ALL.cmd", "--matrix --repetitions 3 %*"),
        ("2_VERIFY_FILES.cmd", "--verify-only %*"),
    ):
        (output / name).write_bytes(
            (
                "@echo off\r\nsetlocal\r\n"
                f'"%~dp0N100-EXPERIMENTS.exe" {arguments}\r\n'
                'set "RESULT=%ERRORLEVEL%"\r\necho.\r\npause\r\nexit /b %RESULT%\r\n'
            ).encode("ascii")
        )
    (output / "README_KO.txt").write_text(
        "N100 실험 R3: 0.2.1 기반 새 Worker / INT8 검증 / batch 1·2·4 선택\n\n"
        "Scanner Lite와 기존 Worker를 종료하고 1_RUN_ALL.cmd를 한 번 실행하세요.\n"
        "앱 재설치가 아니라 이 폴더의 새 Worker EXE와 새 모델을 직접 실행합니다.\n"
        "PowerShell과 Python 설치는 필요하지 않습니다. 중복 실행은 차단됩니다.\n"
        "전원에 연결하고 다른 작업과 절전 없이 실행하세요.\n"
        "6개 설정 × 132장 × 3회 = 2,376회 HTTP 측정, 설정별 warmup 10회입니다.\n"
        "약 45~75분 예상이며 최초 GPU 컴파일에 따라 더 걸릴 수 있습니다.\n"
        "끝나면 Result ZIP 경로에 표시된 ZIP 한 개를 전달하세요.\n\n"
        "A0/A9: 이전 N100 p95 최선 batch2+FP32 검증, 시작·종료 반복\n"
        "B: batch2+INT8 검증 결합\n"
        "C: batch2/1 선택으로 빈 칸 추론 제거+INT8\n"
        "D: batch4/2/1 선택+INT8\n"
        "E: D와 같고 GPU 회전 검증/CPU 검증을 순차 실행\n"
        "모든 실제 객체와 판정 정책을 보존하고 CPU fallback도 유지합니다.\n"
        "HTTP p50/p95/p99/max, 1초 초과 요청, 상태, 모델별 시간, 메모리를 수집합니다.\n"
        "파일 복사와 최초 컴파일/시작 시간은 HTTP 시간과 별도 기록합니다.\n"
        "원본 0.2.1 설치 파일과 R2 결과를 유지합니다. 아직 전 요청 1초 달성은 아닙니다.\n"
        "정확도는 결과 반환 후 GT와 대조합니다. checksum은 발행자 인증이 아닙니다.\n"
        "%LOCALAPPDATA%\\BixolonN100Benchmark 캐시는 실험 종료 후 삭제할 수 있습니다.\n",
        encoding="utf-8-sig",
    )
    write_json(
        output / "KIT-MANIFEST.json",
        {
            "version": "0.2.1",
            "experiment": "n100-r3-exact-batch-20260909",
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
    manifest = verify_kit(output)
    print(
        f"Prepared {len(experiments)} profiles, {sum(r['size_bytes'] for r in manifest['files']):,} bytes"
    )


if __name__ == "__main__":
    main()
