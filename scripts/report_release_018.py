"""Write the Korean 0.1.18 handoff from measured evidence, without changing selection."""

from __future__ import annotations

import shutil
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def table_row(label: str, value: dict) -> str:
    latency = value["latency"]["all"]
    return (
        f"| {label} | {value['correct_approved_count']}/{value['ground_truth_count']} "
        f"({value['correct_approved_rate']:.2%}) | {value['wrong_approved_count']} | "
        f"{value['complete_images']}/{value['image_count']} | "
        f"{latency['p50_ms']:.2f} | {latency['p95_ms']:.2f} | "
        f"{latency['p99_ms']:.2f} | {latency['max_ms']:.2f} | {latency['count']} |"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    version = root / "artifacts/versions/0.1.18"
    study = root / "artifacts/retraining/three-bakery-improvement-0.1.18"
    output = root / "artifacts/distributions/0.1.18"
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    final = load_json_config(version / "final-evaluation/report.json")
    confirmation = load_json_config(version / "packaged-confirmation.json")
    cuda_confirmation = load_json_config(version / "cuda-interleaved/report.json")
    if not final["regression_passed"] or not confirmation["passed"]:
        raise ValueError("Measured release regression or packaged confirmation failed")
    log = load_json_config(version / "packaged-log-cpu/report.json")
    for report in final["reports"].values():
        for value in report["repetitions"]:
            if (
                value["image_count"],
                value["ground_truth_count"],
                value["correct_approved_count"],
                value["wrong_approved_count"],
            ) != (300, 1410, 1352, 4):
                raise ValueError("The retained-model narrative differs from measured final results")
    for value in log["repetitions"]:
        if (
            value["image_count"],
            value["ground_truth_count"],
            value["correct_approved_count"],
            value["wrong_approved_count"],
        ) != (132, 1096, 1082, 4):
            raise ValueError("The retained-model narrative differs from measured log results")
    selection = load_json_config(study / "model-selection.json")
    if selection["selected_model"] != "0.1.17_baseline":
        raise ValueError("This report describes the retained model only")
    interleaved = load_json_config(study / "cpu-interleaved/report.json")
    source = load_json_config(study / "cpu-sleep-comparison.json")["source"]
    rows = [
        "# 0.1.18 검증 결과",
        "",
        "**정확도 개선 후보는 기각하고 기존 모델을 유지했다. 검증된 CPU 실행 변경만 배포한다.**",
        "CPU ONNX session의 idle spinning을 비활성화했다. 모델·Catalog payload, 승인 임계값,",
        "ROI 재촬영 head, 선택 detail/Frozen ViT 및 회전 합의는 0.1.17과 같다.",
        "",
        "## 학습 미사용 로그와 후보 비교",
        "",
        "확정 로그132장·1,096개 객체 전체를 학습에서 제외했다. 원본302장과 SHA-256 중복은0개지만",
        "같은 실물 컬렉션이므로 독립 일반화 검증이 아니다. 임의 사진 분할도 하지 않았다.",
        "로그 정답은 category와 visible bbox 확정이며 SAM mask 확정은 아니다. GT를 수정하지 않았다.",
        "",
        "| 후보 | 로그 정답 승인 | 로그 오승인 | 원본 정답 승인 | 원본 오승인 | 선택 |",
        "|---|---|---|---|---|---|",
        "| 0.1.17 유지 | 1082/1096 | 4 | 645/649 | 0 | 채택 |",
    ]
    for candidate in selection["assessments"]:
        measured = study / "measurements" / candidate
        a = load_json_config(measured / "log-cpu/report.json")["repetitions"][0]
        b = load_json_config(measured / "source-cpu/report.json")["repetitions"][0]
        rows.append(
            f"| {candidate} | {a['correct_approved_count']}/1096 | {a['wrong_approved_count']} | "
            f"{b['correct_approved_count']}/649 | {b['wrong_approved_count']} | 기각 |"
        )
    rows += [
        "",
        "D-FINE 두 후보는 올바른 승인이 감소했다. SSDLite6epoch 재학습은 로그 정답2개 증가·",
        "오승인1개 감소였지만 원본에서 새로운 오승인1개가 생겼다. 측정 전에 정한 회귀 기준에 따라",
        "기각했다. 새 학습은 기존 원본302장과 해당 원본 파생 합성1,200장만 사용했고 원본별96회",
        "소비 이력을 남겼다. 로그와 최종300장은 새 학습에 사용하지 않았다.",
        "",
        "오승인4건은 로그83,123(2건),128의 부분·중복·추가 박스다. 품목 오분류 승인은0건이다.",
        "GT 미대응이라는 집계가 사진에 실제 빵이 없다는 뜻은 아니다.",
        "",
        "- [83번 GT/예측](log-083.jpg)",
        "- [123번 GT/예측](log-123.jpg)",
        "- [128번 GT/예측](log-128.jpg)",
        "",
        "## CPU 교차 측정과 배포 EXE 확인",
        "",
        "AB,BA,AB 순서로 같은132장을 비교했다. 외부 Lite Worker가 시작·종료한 두 측정은",
        "간섭 가능성으로 제외하고 원본 기록을 보존했다. 사용한3회는 외부 Worker CPU 소비≤0.5초,",
        "프로세스 변화 없음 조건을 만족했다. p50은 일부 반복에서 느려졌으므로 모든 요청이",
        "빨라졌다고 표현하지 않는다.",
        "",
        "| 반복 | 0.1.17 p95(ms) | CPU 변경 p95(ms) | 감소 | 상태·품목 순위 |",
        "|---|---|---|---|---|",
    ]
    for item in interleaved["repetitions"]:
        directory = root / item["measurement_directory"]
        old = load_json_config(directory / "baseline/report.json")["summary"]
        new = load_json_config(directory / "sleep/report.json")["summary"]
        rows.append(
            f"| {item['repetition']} | {old['latency']['all']['p95_ms']:.2f} | "
            f"{new['latency']['all']['p95_ms']:.2f} | "
            f"{1 - item['p95_ratios']['all']:.2%} | 132/132 일치 |"
        )
    rows += [
        "",
        f"기존 원본302장 비교는 {source['parity']['compared_request_count']}회 상태·품목 순위가 같고 "
        "645/649 정답 승인·오승인0을 유지했다. 이 수치는 학습 원본 진단이다.",
        "아래는 최종0.1.18 **배포 CPU Worker EXE**로 로그를 다시 실행한 결과다.",
        "",
        "| 반복 | 정답 승인/GT | 오승인 | 완전 성공 이미지 | p50(ms) | p95(ms) | p99(ms) | 최대(ms) | n |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rows += [table_row(str(i), value) for i, value in enumerate(log["repetitions"], 1)]
    rows += [
        "",
        f"기존 로그 판정과 {confirmation['parity']['compared_request_count']}회 상태·품목 순위 일치. "
        "최종 EXE 확인의 외부 Worker 시작/종료 snapshot은 JSON에 보존했다. 연속 시스템 감시는 아니다.",
        "",
        "## 고정300장 실제 EXE 회귀",
        "",
        "모델·Catalog·정책·Worker·CPU 설정·평가 코드 해시를 고정한 뒤 접근했다. 이전 평가 이력이",
        "있는 고정 benchmark이며 독립 validation 성적이 아니다. 이후 재학습·임계값 변경은 없다.",
        "정답 대응은 클래스 무관 bbox IoU≥0.5의 최대 대응 수 우선·총 IoU 차선 일대일 대응이다.",
        "누락·UNKNOWN·재촬영·ERROR를 정답 승인에서 제외한다. 반복900회를 독립900장으로 세지 않는다.",
        "",
        "| 버전/provider/반복 | 정답 승인/GT | 오승인 | 완전 성공 이미지 | p50(ms) | p95(ms) | p99(ms) | 최대(ms) | n |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, report in final["reports"].items():
        rows += [
            table_row(f"{name}/{i}", value) for i, value in enumerate(report["repetitions"], 1)
        ]
    rows += [
        "",
        "CUDA block 측정은 한 반복의 p95가10.6% 높았다. CUDA 경로와 모델은 그대로지만",
        "이를 숨기지 않고 추가 AB,BA,AB 회귀 진단을 고정하여 모두 측정했다. 추가 결과는 최초",
        "측정을 대체하거나 빠른 반복만 고른 값이 아니다. 최종 GT로 모델·정책을 조정하지 않았다.",
        "",
        "| 추가 CUDA 반복 | 0.1.17 p95(ms) | 0.1.18 p95(ms) | 신/구 비율 | 상태·품목 순위 |",
        "|---|---|---|---|---|",
    ]
    for value in cuda_confirmation["repetitions"]:
        rows.append(
            f"| {value['repeat']} | {value['old']['latency']['all']['p95_ms']:.2f} | "
            f"{value['new']['latency']['all']['p95_ms']:.2f} | {value['p95_ratio']:.4f} | "
            f"{value['parity']['status_rank_parity']} |"
        )
    rows += [
        "",
        "추가3회 모두 CUDA p95가 구버전의1.05배 이내인지: "
        f"**{cuda_confirmation['all_p95_ratios_at_most_1_05']}**. "
        "이 단일 장비의 반복 측정으로 다른 GPU의 성능이나 SLA를 보장하지 않는다.",
        "",
        "**정답 승인99%·오승인0의 동시 목표는 미달이다.**",
        "각 provider의 구버전/신버전 상태·품목 순위900회 일치, 신버전 CPU/CUDA900회 일치다.",
        "전체 요청과 full-path 통계는 JSON에 따로 기록했다. 최종300장에 IMAGE_RECAPTURE/ERROR가",
        "없어 두 집단의 p95는 null이고 전체 요청과 full-path는 같다. 원본 배경2장은 조기 종료로",
        "따로 집계했다. 모델 시작 로딩과 사전 파일 읽기를 제외한 HTTP 업로드·디코드·추론·응답",
        "시간이다. Intel Core Ultra9 285K, CPU ORT1.29.0, detector8/embedder12 threads, 동시요청1,",
        "warmup10, 3회 반복. CUDA는 RTX5080·ORT1.28.0이다. 빌드·학습·다른 벤치마크와 겹치지 않았다.",
        "",
        "오승인 상세는 [이번 EXE 집계](final-false-approvals.json)에 기록했다. 기존 실패 유형을",
        "해결했다고 주장하지 않는다. PyTorch/ORT는 동일 모델 binary의0.1.17 증거를 재사용한다.",
        "원시 detector query 순서 차이 때문에 raw box tensor 전체 일치를 주장하지 않으며,",
        "bbox 대응 후 수치 비교와 최종 상태·순위 parity 증거를 구분한다.",
        "",
        "## 빌드와 필수 검사",
        "",
        "전체 Python1159개, 일반 Scanner Flutter193개, Lite24개, SDK6개 테스트 통과.",
        "세 Flutter analyze, Ruff check/format, git diff --check 통과. 실제 CPU/CUDA EXE의",
        "readiness·정상 스캔·손상422 ERROR·누락422 ERROR·미지원415 ERROR smoke 통과.",
        "앱·Worker·Runtime·Catalog는0.1.18, 앱 build21, SDK Core1.1.2다. Worker/runtime에",
        "PyTorch를 넣지 않았다. source 및 bundle checksum과 동일 모델 payload hash를 검증했다.",
        "CPU+GPU 혼합 및 초기화 CPU fallback 코드는 유지하되 기본값은 비활성이다. CPU 설치본과",
        "CUDA portable을 별도로 제공한다. 요청 실행 장애는 ERROR로 유지한다.",
        "",
        "설치 파일 생성·버전·checksum·packaged Worker 실행을 검증했다. 시스템에 실제 설치/제거하거나",
        "실제 카메라 하드웨어를 시험하지는 않았다. EXE는 Authenticode 미서명이며 checksum은",
        "손상·변경 탐지 기능으로 발행자 인증을 제공하지 않는다. 이전 버전 산출물은 보존했다.",
        "",
    ]
    (reports / "verification.md").write_text("\n".join(rows), encoding="utf-8")
    for image_id in (83, 123, 128):
        shutil.copy2(study / f"diagnosis/baseline/log-{image_id:03d}.jpg", reports)
    failures = [
        {"image_id": row["image_id"], **item}
        for row in read_jsonl(version / "final-evaluation/0.1.18/cpu/responses.jsonl")
        if row["repetition"] == 0
        for item in row["metrics"]["items"]
        if item["status"] == "APPROVED" and item["predicted_class_id"] != item["target_class_id"]
    ]
    write_json(reports / "final-false-approvals.json", failures)
    copies = {
        "version-config.json": root / "configs/versions/0.1.18.json",
        "final-evaluation.json": version / "final-evaluation/report.json",
        "final-candidate.json": version / "final-evaluation/final-candidate.json",
        "packaged-log-cpu.json": version / "packaged-log-cpu/report.json",
        "packaged-confirmation.json": version / "packaged-confirmation.json",
        "model-binary-parity.json": version / "model-binary-parity.json",
        "cuda-interleaved.json": version / "cuda-interleaved/report.json",
        "cuda-interleaved-config.json": version / "cuda-interleaved/config.json",
        "runtime-source-diff.txt": version / "runtime-source-diff.txt",
        "model-selection.json": study / "model-selection.json",
        "release-selection.json": study / "release-selection.json",
        "data-audit.json": study / "data-audit.json",
        "cpu-interleaved.json": study / "cpu-interleaved/report.json",
        "cpu-sleep-comparison.json": study / "cpu-sleep-comparison.json",
        "study-method.md": root / "docs/experiments/log-improvement-0.1.18.md",
        "0.1.17-vs-0.1.18.md": root / "docs/architecture/scanner-0.1.18.md",
    }
    for path in version.glob("*.log"):
        if path.name != "distribution-verification.log":
            copies[f"checks/{path.name}"] = path
    for name in (
        "packaged-cpu-smoke.json",
        "packaged-cuda-smoke.json",
        "flutter-checks.json",
        "pe-versions.json",
    ):
        copies[name] = version / name
    for name in (
        "pytorch-ort.json",
        "ort-cpu-cuda.json",
        "ssdlite-active-cuda.json",
        "ssdlite-active-torch.json",
    ):
        copies[f"inherited-0.1.17/{name}"] = root / "artifacts/distributions/0.1.17/reports" / name
    receipts = []
    for target, original in copies.items():
        path = reports / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, path)
        if target == "0.1.17-vs-0.1.18.md":
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "../experiments/log-improvement-0.1.18.md", "study-method.md"
                ),
                encoding="utf-8",
            )
        receipts.append({"path": target, "source": str(original), "sha256": sha256_file(path)})
    write_json(reports / "evidence-sources.json", {"files": receipts})
    final_values = final["reports"]["0.1.18/cpu"]["repetitions"]
    p95 = " / ".join(f"{v['latency']['all']['p95_ms']:.2f}" for v in final_values)
    log_p95 = " / ".join(f"{v['latency']['all']['p95_ms']:.2f}" for v in log["repetitions"])
    (output / "README.md").write_text(
        f"""# BIXOLON Bakery AI Scanner 0.1.18

0.1.17 모델을 유지하고 **CPU 대기 spinning을 비활성화**한 Windows x64 배포물입니다.
앱·Worker·Runtime·Catalog 0.1.18, Flutter build21, SDK Core1.1.2입니다.

학습 미사용 로그132장은 정답1082/1096(98.72%)·오승인4건으로 기존과 같습니다.
교차 측정 CPU p95는 약59~64% 감소했고, 최종 EXE 로그 p95는 **{log_p95}ms**입니다.
고정300장 최종 CPU p95는 **{p95}ms**입니다. 정확도는1352/1410(95.89%)·오승인4건으로
기존과 같으며 **정답 승인99%·오승인0의 동시 목표는 미달**입니다.

## 설치해서 사용

- [일반 Scanner CPU 설치](installers/BixolonBakeryAIScanner-0.1.18-Setup.exe)
- [Scanner Lite CPU 설치](installers/BixolonBakeryAIScannerLite-0.1.18-Setup.exe)

Windows10 1809 이상/Windows11 x64용입니다. CPU 설치본은 Python·Flutter·CUDA 별도 설치 없이 사용합니다.

## 압축을 풀어서 사용

- [일반 Scanner CPU](portable/BixolonBakeryAIScanner-0.1.18-CPU-Portable.zip): `start-bixolon-scanner.ps1` 실행.
- [일반 Scanner CUDA](portable/BixolonBakeryAIScanner-0.1.18-CUDA-Portable.zip): `product_scanner.exe` 실행.
- [Scanner Lite CPU](portable/BixolonBakeryAIScannerLite-0.1.18-CPU-Portable.zip): `bakery_scanner_lite.exe` 실행.

전체 압축을 해제해야 합니다. 휴대형은 Microsoft Visual C++ x64 Runtime이 필요합니다.
CPU detector8/embedder12 threads, 동시 요청1입니다. CPU+GPU 혼합 및 CPU fallback은 기본 비활성입니다.

## 외부 프로그램 연동

- [CPU Worker 전체](developer/BixolonBakeryAIScanner-0.1.18-Worker.zip): `RUN-BIXOLON-WORKER.cmd`, `POST /v1/scan`.
- [SDK Core1.1.2](developer/BIXOLON-Scanner-SDK-Windows-x64-1.1.2.zip)
- [Store Model0.1.18](models/BIXOLON-Store-Model-three_bakery-0.1.18.zip)

SDK Core와 Store Model은 함께 사용합니다. 공개 API·Frozen ViT 선택 검증·회전 합의·재촬영 계약은 유지했습니다.

## 변경 내용과 검증

- [0.1.17 대비 차이](reports/0.1.17-vs-0.1.18.md)
- [정확도·성능·후보 기각·한계 보고서](reports/verification.md)
- [학습 미사용 로그 오승인83번](reports/log-083.jpg), [123번](reports/log-123.jpg), [128번](reports/log-128.jpg)
- [전체 SHA-256](SHA256SUMS.txt), [배포 manifest](distribution-manifest.json)

전체 Python1159개, Flutter Scanner193/Lite24/SDK6개 및 analyze·Ruff·diff 검사 통과.
실제 CPU/CUDA EXE와 버전·checksum을 검증했습니다. 시스템 설치/제거 및 카메라 하드웨어는 미시험입니다.
로그는 같은 실물 컬렉션이고 최종300장은 과거 평가 이력이 있어 독립 일반화 검증이 아닙니다.
모델 개선 후보3개는 회귀 때문에 기각했습니다. 임계값이나 GT를 결과에 맞춰 바꾸지 않았습니다.

이전0.1.17 배포물은 보존했습니다. 소스는 `C:/workspace/bixolon_scanner_release_0_1_18`입니다.
EXE는 Authenticode 미서명이며 SHA-256은 손상·변경 탐지 기능으로 발행자 인증을 제공하지 않습니다.
""",
        encoding="utf-8",
    )
    print(reports / "verification.md")


if __name__ == "__main__":
    main()
