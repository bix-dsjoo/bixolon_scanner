"""Build a standalone N100 structural-model experiment with per-model Catalogs."""

import argparse
import shutil
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.catalog import load_store_catalog_package
from bixolon_scanner.contracts.runtime_package_v2 import load_runtime_package_v2
from bixolon_scanner.experiments.bread.structural_regression import assemble_full_student
from bixolon_scanner.operations.n100_test_kit import child, digest, verify_kit, write_json
from bixolon_scanner.operations.static_batch_candidate import specialize_runtime
from bixolon_scanner.runtime.catalog import load_resolution_fallback_catalog


def prepare_models(config):
    root = Path(config["output"])
    models = {}
    for name, identifier in [("repvit", "repvit_m0_9-supervised")]:
        source = root / "students" / identifier
        package = root / "deploy-candidates-final" / name
        if not (package / "runtime/metadata.json").exists():
            dynamic = root / "deploy-candidates-final" / (name + "-dynamic")
            assemble_full_student(config, source / "regression/assembly", dynamic)
            package.mkdir(parents=True, exist_ok=True)
            provenance = specialize_runtime(
                dynamic / "runtime",
                package / "runtime",
                primary_batch_size=2,
                primary_batch_variants=(1,),
            )
            shutil.copytree(dynamic / "catalog", package / "catalog")
            write_json(package / "specialization.json", provenance)
        runtime = load_runtime_package_v2(package / "runtime")
        catalog = load_store_catalog_package(package / "catalog")
        load_resolution_fallback_catalog(runtime, catalog)
        models[name] = package
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/bread/n100_structural.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models-only", action="store_true")
    args = parser.parse_args()
    config = load_json_config(args.config)
    models = prepare_models(config)
    if args.models_only:
        return
    if args.output is None:
        parser.error("--output is required for a kit")
    root = Path(config["output"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    experiment = output / "Experiment"
    worker = experiment / "worker"
    shutil.copytree(root / "worker-build-final/bixolon-worker", worker)
    shutil.copytree(root / "licenses", experiment / "licenses")
    baseline = Path("artifacts/n100/optimization-r2/static2")
    shutil.copytree(baseline / "runtime", worker / "model-package")
    shutil.copytree(baseline / "catalog", worker / "store-catalog")
    overlays = {}
    for name, package in models.items():
        overlay = experiment / "overlays" / name
        overlay.mkdir(parents=True)
        overlays[name] = []
        for path in sorted((package / "runtime").rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(package / "runtime")
            original = baseline / "runtime" / relative
            if original.exists() and digest(original) == digest(path):
                continue
            target = overlay / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            overlays[name].append({"path": relative.as_posix(), "sha256": digest(path)})
        shutil.copytree(package / "catalog", experiment / "catalogs" / name)
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
    trials = []

    def trial(identifier, description, model=None, **changes):
        row = {"id": identifier, "description": description, "profile": profile | changes}
        if model is not None:
            row.update(
                runtime_overlay=f"Experiment/overlays/{model}",
                catalog=f"Experiment/catalogs/{model}",
            )
        trials.append(row)

    trial("A0_reference", "Current DINO primary, static batch2, FP32 independent verifier")
    trial(
        "B_repvit", "RepViT supervised primary; original detail and independent verifier", "repvit"
    )
    trial(
        "D_repvit_cpu",
        "RepViT same ONNX and policy on CPU",
        "repvit",
        embedder_provider="same",
        parallel_verification=False,
    )
    trial("A9_reference_repeat", "Reference repeat to quantify time drift")
    write_json(
        experiment / "provenance.json",
        {
            "version": "0.2.1",
            "experiment": "n100-structural-r4",
            "execution_profile": profile,
            "variant_files": overlays,
            "purpose": "Experimental new Worker and retrained ONNX models; does not replace the installed release",
        },
    )
    old_root = Path("artifacts/n100/optimization-r2/kit")
    old = verify_kit(old_root)
    for row in old["inputs"]:
        target = child(output, row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child(old_root, row["path"]), target)
    shutil.copy2(root / "runner-build/N100-EXPERIMENTS.exe", output / "N100-EXPERIMENTS.exe")
    for name, arguments in [
        ("1_RUN_ALL.cmd", "--matrix --repetitions 3 %*"),
        ("2_VERIFY_FILES.cmd", "--verify-only %*"),
    ]:
        (output / name).write_bytes(
            (
                f'@echo off\r\nsetlocal\r\n"%~dp0N100-EXPERIMENTS.exe" {arguments}\r\n'
                'set "RESULT=%ERRORLEVEL%"\r\necho.\r\npause\r\nexit /b %RESULT%\r\n'
            ).encode("ascii")
        )
    (output / "README_KO.txt").write_text(
        "N100 구조 실험 R4 — 새 Worker EXE와 새 학습 모델\n\n"
        "Scanner 앱과 기존 Worker를 종료하고 1_RUN_ALL.cmd를 한 번 실행하세요.\n"
        "PowerShell/Python 설치는 필요 없습니다. 중복 실행은 자동 차단됩니다.\n"
        "앱 재설치 대신 이 폴더의 새 EXE와 모델을 직접 실행하는 실험입니다.\n"
        "A0/A9 기존 DINO, B RepViT, D 동일 RepViT의 CPU 실행입니다.\n"
        "검출 CPU → 분류 Intel GPU → 독립 검증 CPU 및 명시적 CPU fallback을 유지합니다.\n"
        "4개 설정 × 132장 × 3회 = 1,584회 HTTP 요청입니다. 설정별 warmup 10회가 별도 실행됩니다.\n"
        "전원 연결, 절전 해제 상태에서 다른 무거운 작업 없이 실행하세요.\n"
        "완료 후 Result ZIP 경로에 표시된 ZIP 한 개를 전달하세요.\n"
        "N100의 모든 요청이 1초 이내인지 아직 확인되지 않았습니다. p95와 최대 시간을 구분합니다.\n"
        "파일 검증만 하려면 2_VERIFY_FILES.cmd를 실행하세요. 기존 설치본과 결과는 유지합니다.\n"
        "파일 checksum은 변조·손상 탐지이며 발행자 진위 인증이 아닙니다.\n"
        "%LOCALAPPDATA%\\BixolonN100Benchmark의 해당 실험 캐시는 종료 후 정리할 수 있습니다.\n",
        encoding="utf-8-sig",
    )
    write_json(
        output / "KIT-MANIFEST.json",
        {
            "version": "0.2.1",
            "experiment": "n100-structural-r4",
            "benchmark": "Experiment",
            "powershell_required": False,
            "inputs": old["inputs"],
            "experiments": trials,
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
        f"Prepared {len(trials)} profiles, {sum(r['size_bytes'] for r in manifest['files']):,} bytes",
        flush=True,
    )


if __name__ == "__main__":
    main()
