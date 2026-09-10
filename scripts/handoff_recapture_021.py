"""Collect 0.2.1 measurement evidence and verify the finished distribution."""

import argparse
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.operations.lite_bundle import verify as verify_bundle
from bixolon_scanner.operations.version_bundle import load_version_config, verify_prepared_version
from bixolon_scanner.training.three_bakery_data import write_json

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/retraining/recapture-0.2.1"
VERSION = ROOT / "artifacts/versions/0.2.1"
OUTPUT = ROOT / "artifacts/distributions/0.2.1"


def latency_row(label, summary):
    t = summary["latency"]["all"]
    return f"| {label} | {t['count']} | {t['p50_ms']:.2f} | {t['p95_ms']:.2f} | {t['p99_ms']:.2f} |"


def report():
    reports = OUTPUT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    suites = (
        ET.parse(VERSION / "verification/python-tests-final.xml").getroot().findall(".//testsuite")
    )
    assert suites and not any(
        int(s.get(k, "0")) for s in suites for k in ("failures", "errors", "skipped")
    )
    python_count = sum(int(s.get("tests", "0")) for s in suites)
    compatibility = load_json_config(STUDY / "n100-build-host/compatibility.json")
    assert compatibility["accuracy_passed"] and compatibility["cpu_parity"]["status_rank_parity"]
    for name in (
        "confirmation-context",
        "final300-context",
        "packaged",
        "visual-context-report",
        "n100-build-host",
    ):
        shutil.copytree(STUDY / name, reports / name, dirs_exist_ok=True)
    for path in STUDY.glob("case-*.jpg"):
        shutil.copy2(path, reports / path.name)
    for name in ("recapture-causes.json", "source-baseline.json", "selection-context.json"):
        shutil.copy2(STUDY / name, reports / name)
    shutil.copytree(VERSION / "verification", reports / "verification", dirs_exist_ok=True)
    shutil.copytree(VERSION / "source/evidence", reports / "frozen", dirs_exist_ok=True)
    for p in VERSION.glob("packaged-*-smoke.json"):
        shutil.copy2(p, reports / p.name)
    shutil.copy2(ROOT / "docs/experiments/recapture-0.2.1.md", OUTPUT / "CHANGES-KO.md")
    shutil.copy2(ROOT / "docs/experiments/n100-0.2.1.md", OUTPUT / "N100-KO.md")
    shutil.copy2(ROOT / "artifacts/third_party/D-FINE/LICENSE", OUTPUT / "D-FINE-LICENSE.txt")
    (OUTPUT / "MODELS-KO.md").write_text(
        "# 모델 출처\n\nD-FINE HGNetV2-S 640: Peterande/D-FINE, Apache 2.0. "
        "D-FINE-LICENSE.txt 원문을 포함한다. DINOv3 ConvNeXt-Tiny / ViT-B/16: "
        "배포 Runtime의 DINOV3-LICENSE.md를 따른다. OpenVINO: N100 번들의 OPENVINO-LICENSE.txt.\n\n"
        "0.2.0 대비 모델 graph·weight·adapter·support·prototype bytes는 동일하다. "
        "Runtime에 보존된 과거 THIRD_PARTY_MODELS.md의 SSDLite 설명은 현재 D-FINE 검출기가 아닌 역사적 기록이다.\n",
        encoding="utf-8",
    )
    lines = [
        "# 0.2.1 검증 보고서",
        "",
        "제품 0.2.1 · Flutter build 24 · 외부 SDK 1.2.1. 0.2.0 대비 재촬영과 추가 검출을 줄였다.",
        "로그132장·GT 1,096개는 정책 선택에 사용한 노출된 개발 세트이며 독립 일반화 검증이 아니다.",
        "GT·원본 SHA-256·IoU ≥0.5 일대일 대응·분모를 유지했다. 오류와 재촬영도 제외하지 않았다.",
        "",
        "## 로그 132장 정확도",
        "",
        "| 지표 | 0.2.0 | 0.2.1 |",
        "|---|---:|---:|",
        "| APPROVED / 정답 승인 | 1,078 | 1,078 |",
        "| UNKNOWN | 17 | 15 |",
        "| SEGMENT_RECAPTURE | 38 | 19 |",
        "| 오승인 | 0 | 0 |",
        "| 미검출 | 0 | 0 |",
        "| 추가 검출 | 37 | 16 |",
        "| 전체 출력 박스 | 1,133 | 1,112 |",
        "| SEGMENTATION 이미지 | 132 | 132 |",
        "| IMAGE_RECAPTURE / ERROR 이미지 | 0 / 0 | 0 / 0 |",
        "| 이미지 완전 성공 | 99 | 108 |",
        "",
        "정답 승인 1,078/1,096(98.36%)을 유지했다. GT 객체별 상태도 그대로다.",
        "제거한 21개는 모두 GT 미대응 추가 박스이며 기존 재촬영 19개와 UNKNOWN 2개다.",
        "재촬영은 50%, 추가 검출은 56.76% 줄었다. 실제 객체 삭제나 상태 재명명은 없다.",
        "",
        "## 동일 장비·provider HTTP 시간",
        "",
        "개발 장비 Core Ultra 9 285K / RTX 5080. CPU detector/embedder 8/12 threads, CUDA는 같은 ORT/CUDA 환경.",
        "warmup 10, 동시 요청 1, 입력 bytes 준비·해시 검증은 시간 밖이다. 단위 ms.",
        "소스 확인은 AB/BA/AB 순서 3쌍이며 모든 쌍에서 p95가 증가하지 않았다. 각 실행 표본은 132개다.",
        "",
        "| provider / 쌍 / 버전 | n | p50 | p95 | p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for provider in ("cpu", "cuda"):
        comparison = load_json_config(
            STUDY / f"confirmation-context/log/{provider}/comparison.json"
        )
        assert comparison["accepted"]
        for pair in range(1, 4):
            for role in ("baseline", "candidate"):
                summary = load_json_config(
                    STUDY / f"confirmation-context/log/{provider}/pair-{pair}/{role}/report.json"
                )["summary"]
                lines.append(
                    latency_row(
                        f"{provider.upper()} / {pair} / {'0.2.0' if role == 'baseline' else '0.2.1'}",
                        summary,
                    )
                )
    lines += [
        "",
        "실제 배포 EXE로 별도 132장 1쌍을 확인했다. 포함된 Runtime/Catalog와 측정용 staging의 전체 파일 hash가 같음을 먼저 검증했다.",
        "",
        "| packaged provider / 버전 | n | p50 | p95 | p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for provider in ("cpu", "cuda"):
        assert load_json_config(STUDY / f"packaged/{provider}/comparison.json")["accepted"]
        for version in ("0.2.0", "0.2.1"):
            lines.append(
                latency_row(
                    f"{provider.upper()} / {version}",
                    load_json_config(STUDY / f"packaged/{provider}/{version}/report.json")[
                        "summary"
                    ],
                )
            )
    lines += [
        "",
        "## 원본·고정300장 회귀",
        "",
        "원본302장 / GT649개: 정답 승인646·오승인0·미검출0 유지, UNKNOWN4 유지, 재촬영15→4, 추가 검출16→5.",
        "후보 고정 후 고정300장 / GT1,410개: 정답 승인1,356·기존 오승인4·미검출0 유지, UNKNOWN46→43, 재촬영67→39, 추가 검출63→32.",
        "고정300장의 기존 오승인4건을 해결했다고 주장하지 않는다. 해당 세트로 가중치나 임계값을 조정하지 않았다.",
        "",
        "| 고정300 provider / 버전 | n | p50 | p95 | p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    regression = load_json_config(STUDY / "final300-context/comparison.json")
    assert regression["no_regression"]
    for provider, versions in regression["reports"].items():
        for version, summary in versions.items():
            lines.append(latency_row(f"{provider.upper()} / {version}", summary))
    lines += [
        "",
        "로그132장은 CPU/CUDA 상태·class rank가 일치한다. 고정300장은 이미지134의 UNKNOWN Top-3 세 번째가",
        "CPU bread_07 / CUDA bread_10으로 다르며, 같은 차이가 0.2.0에도 있었다. 엄격 parity는 실패로 보존했다.",
        "각 provider 안에서 남겨진 박스의 상태·reason·품목 순위는 0.2.0과 같으며 신규 provider 불일치는 없다.",
        "",
        "## 분석·변경·기각",
        "",
        "[38개 영역 원본·GT·예측 비교](reports/visual-context-report/comparison.html), [전체 변경 기록](CHANGES-KO.md).",
        "검출 context 0.025를 유지하고 출력 후보 점수0.04를 별도 metadata로 둔다. 기존 classify_selected가 전체 context를 보면서 선택 ROI를 분류한다.",
        "가중치·Catalog·승인·품질 임계값과 NMS는 유지한다. 기존 metadata에 새 필드가 없으면 기존 동작이다.",
        "검출 자체를0.04로 올린 첫 후보는 고정300 오승인4→5로 기각했다. 이웃 마스크 해제, bias0, containment NMS도 악화해 기각했다.",
        "로그의 남은 재촬영19개는 실제 GT9개(낮은 검출점수4·다중 객체 거부4·경계1)와 추가 박스10개다.",
        "",
        "## 배포·검증·한계",
        "",
        "[전체 검증 로그](reports/verification/)와 [배포 파일 검증](reports/distribution-verification.json)에 결과를 보존한다.",
        f"Python 전체 {python_count}개 통과. ruff check·ruff format --check·git diff --check 통과.",
        "Flutter 6개 프로젝트 analyze 통과; 전체 테스트는 Product193·Lite28·SDK6·Camera Lab4개 통과(총231개).",
        "첫 Python 전체 실행에서 시간 제한 관련 테스트1개가 실패했다. 실패 로그를 유지했으며 해당 모듈18개 재실행과 최종 전체 재실행을 확인했다.",
        "CPU·CUDA EXE는 정상 이미지와 손상/누락/미지원 입력, readiness, 단일0.2.1 버전, 불필요한 학습 dependency 제외를 확인했다.",
        "N100용 CPU detector2 → Intel UHD GPU primary/detail → CPU verifier4와 명시적 CPU fallback을 유지했다.",
        "실제 N100이 연결되지 않아 N100 성능은 미측정이다. 개발 완료 후 [측정 절차](N100-KO.md)대로0.2.0과 동일 장비/provider에서 측정해야 한다.",
        "빌드 PC에서 N100 설정을 실행한 결과는 호환성·fallback 확인이며 N100 GPU 성능이 아니다.",
        "요청 provider는 OpenVINO GPU였지만 초기화 ProviderExecutionError로 CPU fallback이 발생했다. 시작·종료 readiness는 cpu다.",
        "이 CPU fallback의132장 p50/p95/p99는332.05/413.26/471.59ms이며 정답 승인1,078·오승인0·미검출0·재촬영19개다. CPU 배포물과 상태·품목 순위가 일치했다.",
        "이 수치는 ORT OpenVINO 패키지의 빌드 PC CPU2/4 threads 결과이며, N100 성능이나 GPU 가속 성능으로 비교하지 않는다.",
        "Checksum은 손상·파일 변경 탐지이며 발행자 진위 인증은 제공하지 않는다. 독립 일반화 성능·SLA·인증을 주장하지 않는다.",
        "진단 이미지와 trace는 명시적 개발 산출물이며 Worker 기본 로그에 포함하지 않는다. 검토 종료 후 reports의 진단 이미지와 로컬 study를 삭제할 수 있다.",
        "",
    ]
    (OUTPUT / "RESULTS-KO.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "README-KO.md").write_text(
        "# BIXOLON Scanner 0.2.1\n\n"
        "재촬영38→19, 추가 검출37→16. 로그132장 정답 승인1,078/1,096·오승인0·미검출0.\n\n"
        "- 일반 CPU 앱: [Setup](installers/BixolonBakeryAIScanner-0.2.1-Setup.exe)\n"
        "- Lite CPU: [Setup](installers/BixolonBakeryAIScannerLite-0.2.1-Setup.exe)\n"
        "- N100 Intel UHD Lite: [Setup](installers/BixolonBakeryAIScannerLite-0.2.1-N100-Setup.exe)\n"
        "- 설치 없는 CPU/CUDA/N100 앱과 Worker: portable 폴더\n"
        "- 외부 연동 Worker·SDK1.2.1: developer 폴더 / 모델0.2.1: models 폴더\n\n"
        "[검증 보고서](RESULTS-KO.md) · [변경과 남은 한계](CHANGES-KO.md) · [실패 영역 비교](reports/visual-context-report/comparison.html)\n\n"
        "N100 실행 구성을 유지했으며 실제 N100 성능은 개발 완료 후 대상 장비에서 측정한다. 현재 미측정이다.\n"
        "SHA256SUMS.txt와 distribution-manifest.json으로 파일 변경을 확인한다. Checksum은 발행자 인증이 아니다.\n",
        encoding="utf-8",
    )


def verify():
    config = load_version_config(ROOT / "configs/versions/0.2.1.json")
    verify_prepared_version(config, repository_root=ROOT)
    raw_config = load_json_config(ROOT / "configs/versions/0.2.1.json")
    for evidence in raw_config["evaluation_evidence"]:
        assert sha256_file(ROOT / evidence["path"]) == evidence["sha256"]
    for component in ("runtime", "catalog"):
        lock = raw_config[component]
        assert (
            directory_content_manifest(ROOT / lock["path"])["manifest_sha256"]
            == lock["manifest_sha256"]
        )
    checked = {}
    for component in ("runtime", "catalog"):
        ignored = (
            {"metadata.json"} if component == "runtime" else {"catalog.json", "checksums.json"}
        )

        def payload(path):
            return {
                r["path"]: r["sha256"]
                for r in directory_content_manifest(path)["files"]
                if Path(r["path"]).name not in ignored
            }

        assert payload(VERSION / f"source/{component}") == payload(VERSION / f"staging/{component}")
        assert payload(ROOT / f"artifacts/versions/0.2.0/staging/{component}") == payload(
            VERSION / f"staging/{component}"
        )
        checked[component] = {"payload_unchanged_from_020": True}
    for profile, worker in {
        "cpu": ROOT / "artifacts/installers/0.2.1/windows-payload/worker",
        "cuda": VERSION / "bixolon-bakery-ai-scanner-0.2.1/worker",
        "n100": ROOT / "artifacts/n100/0.2.1/worker-payload/worker",
    }.items():
        for component, target in (("runtime", "model-package"), ("catalog", "store-catalog")):
            assert directory_content_manifest(worker / target) == directory_content_manifest(
                VERSION / f"staging/{component}"
            )
        if profile != "n100":
            smoke = load_json_config(VERSION / f"packaged-{profile}-smoke.json")
            assert smoke["passes"]
            assert (
                directory_content_manifest(worker)["manifest_sha256"]
                == smoke["worker_artifact_content_manifest_sha256"]
            )
    for path in (
        ROOT / "artifacts/lite/0.2.1/payload",
        ROOT / "artifacts/n100/0.2.1/worker-payload",
        ROOT / "artifacts/n100/0.2.1/lite-payload",
    ):
        verify_bundle(path)
    deliverables = sorted(p for p in OUTPUT.rglob("*") if p.suffix in {".zip", ".exe"})
    assert len(deliverables) == 11
    for path in deliverables:
        assert (
            sha256_file(path)
            == path.with_suffix(path.suffix + ".sha256").read_text(encoding="utf-8").split()[0]
        )
        if path.suffix == ".zip":
            print("ZIP CRC:", path.name, flush=True)
            with zipfile.ZipFile(path) as bundle:
                assert bundle.testzip() is None
                assert all(
                    not Path(n).is_absolute() and ".." not in Path(n).parts
                    for n in bundle.namelist()
                )
                checked[path.name] = {"zip_crc_passed": True, "file_count": len(bundle.namelist())}
    write_json(
        OUTPUT / "reports/distribution-verification.json",
        {
            "passed": True,
            "deliverable_count": len(deliverables),
            "checks": checked,
            "n100_hardware_measured": False,
        },
    )
    files = [
        r
        for r in directory_content_manifest(OUTPUT)["files"]
        if r["path"] not in {"distribution-manifest.json", "SHA256SUMS.txt"}
    ]
    (OUTPUT / "SHA256SUMS.txt").write_text(
        "".join(f"{r['sha256']}  {r['path']}\n" for r in files), encoding="utf-8"
    )
    write_json(
        OUTPUT / "distribution-manifest.json",
        {
            "product_version": "0.2.1",
            "sdk_version": "1.2.1",
            "files": files,
            "checksum_index_sha256": sha256_file(OUTPUT / "SHA256SUMS.txt"),
            "n100_hardware_measured": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "verify"))
    args = parser.parse_args()
    report() if args.command == "report" else verify()
