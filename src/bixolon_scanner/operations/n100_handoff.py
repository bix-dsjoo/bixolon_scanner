"""Report measured 0.2.0 results and verify its complete local handoff."""

from __future__ import annotations

import argparse
import html
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from ..configuration import load_json_config
from ..contracts.artifact import canonical_sha256, directory_content_manifest
from ..contracts.catalog import load_store_catalog_package, sha256_file
from ..contracts.runtime_package_v2 import load_runtime_package_v2
from ..evaluation.log_diagnosis import diagnose
from ..evaluation.three_bakery_http import provider_parity
from ..training.three_bakery_data import write_json
from .lite_bundle import verify as verify_bundle

VERSION = "0.2.0"


def _row(label: str, value: dict) -> str:
    timing = value["latency"]["all"]
    return (
        f"| {label} | {value['correct_approved_count']}/{value['ground_truth_count']} "
        f"({value['correct_approved_rate']:.2%}) | {value['wrong_approved_count']} | "
        f"{value['missed_count']} | {value['extra_count']} | "
        f"{timing['p50_ms']:.2f} | {timing['p95_ms']:.2f} | "
        f"{timing['p99_ms']:.2f} | {timing['max_ms']:.2f} | {timing['count']} |"
    )


def report(root: Path) -> None:
    version = root / f"artifacts/versions/{VERSION}"
    study = root / f"artifacts/retraining/n100-{VERSION}"
    output = root / f"artifacts/distributions/{VERSION}"
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    suites = ET.parse(version / "python-tests.xml").getroot().findall(".//testsuite")
    if not suites or any(
        int(s.get(k, "0")) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise ValueError("Full Python test evidence is missing or unsuccessful")
    python_count = sum(int(s.get("tests", "0")) for s in suites)
    measured = {
        p: load_json_config(version / f"log132/{p}-packaged/report.json")
        for p in ("cpu", "cuda", "n100")
    }
    objectives = {
        p: load_json_config(version / f"log132/{p}-packaged/objective.json") for p in measured
    }
    parity = {
        p: provider_parity(
            version / "log132/cpu-packaged/responses.jsonl",
            version / f"log132/{p}-packaged/responses.jsonl",
        )
        for p in ("cuda", "n100")
    }
    write_json(reports / "provider-parity.json", parity)
    write_json(reports / "objectives.json", objectives)
    names = {
        "cpu": "285K CPU 8/12 threads",
        "cuda": "RTX 5080 CUDA",
        "n100": "빌드 PC의 N100 설정 / 실제 provider 별도 확인",
    }
    lines = [
        "# 0.2.0 배포 및 측정 결과",
        "",
        "**N100에서 200ms를 달성한 제품으로 판정하지 않는다. N100 실측 장비가 연결되지 않았다.**",
        "현재 PC에서 실행한 배포 Worker HTTP 측정값을 아래에 기록한다. CPU 결과와 GPU 결과를",
        "분리하며 OpenVINO 초기화 실패 뒤의 CPU fallback 시간을 GPU 성능으로 표현하지 않는다.",
        "",
        "## 로그 132장 / 객체 1,096개",
        "",
        "로그 이미지는 모델 가중치 학습에 사용하지 않았다. 전역 검출 임계값과 지역 재촬영 정책",
        "선택에는 이 로그의 결과를 사용했다. 같은 실물 컬렉션의 노출된 개발 진단이며 독립 검증이",
        "아니다. 과거 최종 300장은 이번 후보 선택에서 제외했고, 모델 확정 후 전체 테스트를 실행했다.",
        "",
        "목표는 정답 APPROVED ≥1,042/1,096(95%), 오승인 0건, 미검출 0건, 각 반복 HTTP",
        "전체 요청과 full-path p95 ≤200ms다. 분모에서 UNKNOWN·재촬영·ERROR를 제외하지 않는다.",
        "매칭은 클래스와 무관한 bbox IoU ≥0.5이며 최대 대응 수를 먼저, 총 IoU를 다음으로 적용한다.",
        "오승인은 오분류·배경·중복 승인을 포함한다. 추가 segmentation도 별도 집계한다.",
        "",
        "warmup 10회, 동시 요청 1개, 132장씩 3회다. 모델 로딩·사전 파일 읽기는 제외하고 업로드·",
        "디코딩·추론·응답 처리를 포함한다. 학습·빌드·다른 벤치마크와 겹치지 않게 측정했다.",
        "시간 단위는 ms이며 full-path/IMAGE_RECAPTURE/ERROR의 분리 통계는 각 JSON에 있다.",
        "",
        "| 실행 / 반복 | 정답 승인 | 오승인 | 미검출 | 추가 박스 | p50 | p95 | p99 | 최대 | N |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for profile, data in measured.items():
        lines.extend(
            _row(f"{names[profile]} / {i + 1}", s) for i, s in enumerate(data["repetitions"])
        )
    lines += ["", "## 판정과 남은 한계", ""]
    for profile, data in measured.items():
        verdicts = objectives[profile]["repetitions"]
        lines.append(
            f"- {names[profile]}: 인식률 조건 {all(v['recognition_met'] for v in verdicts)}, "
            f"오승인 0건 {all(v['zero_wrong_approval_met'] for v in verdicts)}, "
            f"미검출 0건 {all(v['zero_missed_met'] for v in verdicts)}, "
            f"200ms 조건 {all(v['http_p95_met'] for v in verdicts)}. "
            f"실제 readiness provider={data['ready'].get('provider')}, "
            f"종료 provider={data['ready_end'].get('provider')}."
        )
    first = measured["cpu"]["summary"]
    lines += [
        "",
        f"CPU 첫 반복의 완전 성공 이미지는 {first['complete_images']}/132장이다. "
        f"객체별 상태는 {first['item_status_counts']}이며 추가 박스는 {first['extra_count']}개다.",
        "미검출 0건은 모든 GT에 공개 박스가 하나씩 대응했다는 뜻이다. 추가 박스가 없거나",
        "모든 이미지가 재촬영 없이 성공했다는 뜻은 아니다. 해당 로그와 측정 환경에서 관측된 결과다.",
        "",
        f"CPU/CUDA 상태·품목 순위 parity: {parity['cuda']['status_rank_parity']}. "
        f"CPU/빌드 PC N100 설정 parity: {parity['n100']['status_rank_parity']}.",
        "구체적 불일치 image ID와 반복은 provider-parity.json에 기록한다. 원시 detector query 순서와",
        "공개 상태·순위 parity는 별도 개념이다. 이번 패키징에서 ONNX를 재학습·재수출하지 않았다.",
        "원본 302장 PyTorch/ORT 비교에서는 241번의 추가 ROI가 ORT UNKNOWN(포함 중복),",
        "PyTorch SEGMENT_RECAPTURE로 달랐다. 해당 ROI 높이는 819/818px였으며 정답 승인 품목들은",
        "같았다. 따라서 전체 PyTorch/ORT 상태 parity 통과로 표시하지 않는다.",
        "모든 분류 tensor의 품목 순위는 일치했으나 detector 원시 tensor와 bbox 대응 후 수치 비교는",
        "atol 1e-4 / rtol 1e-3에서 일부 미달했다. query 순서 차이만으로 전부 설명하지 않는다.",
        "전체 수치와 241번 응답은 tensor-parity 증빙에 보존했다. 이 진단 뒤 정책을 변경하지 않았다.",
        "",
        "## 0.1.18에서 변경한 내용",
        "",
        "- SSDLite320에서 기존에 학습한 D-FINE HGNetV2-S 640 후보로 변경했다.",
        "- 검출 score 0.025, NMS IoU 0.5로 후보를 포함하고 detector score 0.25 미만인 ROI는",
        "  높은 분류 확신에도 SEGMENT_RECAPTURE로 반환한다. 이 전역 정책은 로그 개발 진단에서 선택했다.",
        "- ConvNeXt-Tiny 192 primary, 선택적 224 detail, Frozen ViT-B/16 160 검증과 회전 합의를 유지한다.",
        "  품목별 예외는 없으며 승인 0.8 / detail·verifier 경계 상한 0.85를 유지한다.",
        "- N100 전용 CPU detector → OpenVINO GPU primary/detail → CPU Frozen ViT 구성을 추가했다.",
        "  GPU 초기화·warmup 실패 시 CPU로 시작한다. 실행 장애 요청은 ERROR이며 CPU 복구 후 다음",
        "  요청부터 CPU를 사용한다. 복구 중·실패 후 readiness는 503이고 전환 사실을 로그에 기록한다.",
        "- Lite의 APPROVED 품목 ID·이름 표시와 로그·내보내기를 포함한다. 제품 0.2.0+23, SDK 1.2.0이다.",
        "",
        "## 모델별 학습 입력",
        "",
        "| 구성 | 실제 입력 / 이번 사용 |",
        "|---|---|",
        "| D-FINE dense detector | 단일 200 + 다중 100 + 배경 2 전량, 원본별 48회 소비 / 합성 6,000장, 12 epoch 기존 후보 |",
        "| Tiny margin dense 192·224 | 정정본 649개 객체 crop 전량 + 원본 파생 증강, 8 epoch 기존 후보 |",
        "| Catalog·support·prototype | 정정본 649개 crop에서 중복 제거·품목별 10개 support 선택, 원본 해시 추적 |",
        "| Frozen ViT verifier | 일반 사전학습 backbone 고정, 이번 원본에서 만든 support·Catalog 사용 |",
        "| ROI 무결성 head | 기존 원본에서 정상·겹침 ROI를 만든 학습 결과 유지, multi-object 거부 임계값 0.8 |",
        "| 가림 SSDLite 실험(기각) | 원본 302장 각각 64회 + 신규 합성 1,200장, 8 epoch |",
        "",
        "단일 cutout 176개와 원본 배경 2장·절차 배경으로 합성했다. 기존 dense 합성은 6,000장,",
        "20% 빈 장면, 양성당 1~8개다. 새 가림 실험은 1,200장, 빈 장면 240장, 양성 960장,",
        "양성당 4~12개, 최소 가시 면적 20%, 크기 16~40%였으나 최종 채택하지 않았다.",
        "원본·crop·합성의 부모 SHA-256 및 실제 입력 소비 이력은 frozen training evidence에 보존했다.",
        "223 객체4 꽃빵(C07), 284 객체3 와플(C03), 271 누락 객체 추가 정정본과 ID 1–302를 유지했다.",
        "",
        "## 후보 비교 기록",
        "",
        "아래는 개발 도중 서로 다른 설정·반복 수로 측정한 진단이다. 속도의 엄밀한 동일 조건 비교는",
        "위 최종 배포 Worker 표를 사용한다. 저점수 후보를 이미지 전체 재촬영으로 막아 얻은 짧은",
        "지연을 목표 달성으로 세지 않았다. 선택 뒤 모델·정책을 다시 바꾸지 않았다.",
        "",
        "| 후보 | 정답 승인 | 오승인 | 미검출 | 추가 박스 | p50 | p95 | p99 | 최대 | N |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in (
        "ssdlite-occlusion",
        "dfine-dense-sensitive",
        "dfine-dense-recall",
        "dfine-dense-local-recapture",
        "dfine-dense-selective",
    ):
        lines.append(
            _row(name, load_json_config(study / f"measurements/{name}/report.json")["summary"])
        )
    lines += [
        "",
        "## 검증 / 배포 선택",
        "",
        f"Python 전체 {python_count:,}개, Product Flutter 193개, Lite 28개, SDK 6개 테스트 및 analyze를 통과했다.",
        "CPU/CUDA packaged Worker의 정상 입력·손상·누락·미지원 형식 smoke도 통과했다.",
        "최종 Ruff·diff·ZIP CRC·checksum·버전·모델 payload 동일성 결과는 reports의 검증 기록에 있다.",
        "N100 Intel UHD 실제 GPU 실행·속도, 설치 마법사 전 과정의 육안 검수는 수행하지 않았다.",
        "",
        "N100에는 N100-Setup 또는 N100-Portable을, 일반 PC CPU에는 CPU 설치본을 사용한다.",
        "CUDA Portable은 NVIDIA CUDA 장비용이다. N100에는 적용하지 않는다. 기존 버전 배포물은 보존했다.",
        "N100 실측 명령과 fallback 확인 방법은 N100-KO.md에 있다. 모델·정책을 임의로 변경하지 말고",
        "실측 기록을 이 고정 GT에 대응하여 판정한다. Checksum은 손상 탐지이며 발행자 인증은 아니다.",
        "",
        "추가 박스·UNKNOWN·재촬영이 있는 이미지의 GT/예측 겹침 표시는 [검수 이미지](reports/diagnosis/index.html)에 있다.",
    ]
    (output / "RESULTS-KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(root / f"docs/experiments/n100-{VERSION}.md", output / "N100-KO.md")
    (output / "README-KO.md").write_text(
        "# BIXOLON Bakery AI Scanner 0.2.0\n\n"
        "N100용 설치본: `installers/BixolonBakeryAIScannerLite-0.2.0-N100-Setup.exe`.\n"
        "일반 CPU 앱·Lite 설치본은 같은 installers 폴더에 있다. 장비에 맞는 하나를 선택한다.\n\n"
        "portable에는 CPU/CUDA/N100 무설치 앱과 N100 Worker, developer에는 일반 CPU Worker와 SDK,\n"
        "models에는 three_bakery Store Model이 있다. SDK의 N100 실행 설정은 N100 Worker payload를 지정한다.\n\n"
        "**N100 실측은 미수행이며, 200ms 목표 달성으로 판정하지 않는다.**\n"
        "실제 정확도·오승인·미검출·속도는 [RESULTS-KO.md](RESULTS-KO.md), 실행·실측 절차는 "
        "[N100-KO.md](N100-KO.md)를 확인한다.\n\n"
        "SHA256SUMS.txt와 distribution-manifest.json은 전달 파일의 손상·변경 검사용이다.\n",
        encoding="utf-8",
    )
    evidence = {}
    evidence["roi-integrity-training.json"] = (
        root
        / "artifacts/retraining/three-bakery-revised300-roi-integrity"
        / "roi-integrity/margin-dense-20260908/report.json"
    )
    for name in ("pytorch-ort.json", "ort-cpu-cuda.json"):
        evidence[f"tensor-parity/{name}"] = version / f"parity-check/parity/{name}"
    evidence["tensor-parity/image-241.json"] = version / "parity-241/parity/pytorch-ort.json"
    for profile in measured:
        for name in ("report.json", "responses.jsonl", "objective.json", "worker.log"):
            evidence[f"{profile}/{name}"] = version / f"log132/{profile}-packaged/{name}"
    for name in (
        "packaged-cpu-smoke.json",
        "packaged-cuda-smoke.json",
        "python-tests.xml",
        "python-tests-final.log",
        "static-checks.log",
        "executable-versions.json",
        "flutter-product-analyze.log",
        "flutter-product-tests.log",
        "flutter-lite-analyze.log",
        "flutter-lite-tests.log",
        "flutter-sdk-analyze.log",
        "flutter-sdk-tests.log",
    ):
        evidence[name] = version / name
    for source in (version / "source/evidence").iterdir():
        if source.is_file():
            evidence[f"frozen/{source.name}"] = source
    copied = []
    for relative, source in evidence.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = reports / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append({"path": relative, "sha256": sha256_file(target), "source": str(source)})
    write_json(reports / "evidence-sources.json", {"files": copied})
    (output / "MODELS-0.2.0.md").write_text(
        "# 0.2.0 모델 출처\n\n"
        "Detector: D-FINE HGNetV2-S 640, Peterande/D-FINE 공식 구현과 COCO 사전학습에서 "
        "three_bakery 원본·합성으로 미세조정했다. 소스: https://github.com/Peterande/D-FINE . "
        "D-FINE 소스의 Apache 2.0 LICENSE 원문을 이 폴더에 포함한다.\n\n"
        "Classifier/verifier: DINOv3 ConvNeXt-Tiny 및 ViT-B/16. "
        "Runtime의 DINOV3-LICENSE.md를 유지한다.\n\n"
        "OpenVINO: N100 번들 licenses/OPENVINO-LICENSE.txt. "
        "과거 Runtime에서 유지한 THIRD_PARTY_MODELS.md의 0.1.14 SSDLite 설명은 역사적 기록이며 "
        "0.2.0 선택 detector는 위 D-FINE이다. 모델 graph·weight·support payload는 배포 중 변경하지 않았다.\n",
        encoding="utf-8",
    )
    shutil.copy2(root / "artifacts/third_party/D-FINE/LICENSE", output / "D-FINE-LICENSE.txt")
    diagnosis = diagnose(
        version / "source/evidence/log-inputs.jsonl",
        version / "log132/cpu-packaged",
        reports / "diagnosis",
    )
    cards = "\n".join(
        f"<figure><figcaption>로그 {r['image_id']} — "
        f"{html.escape(str({k: r['metrics'][k] for k in ('correct_approved_count', 'wrong_approved_count', 'missed_count', 'extra_count')}))}"
        f'</figcaption><img loading="lazy" src="log-{r["image_id"]:03d}.jpg"></figure>'
        for r in diagnosis["failure_images"]
    )
    (reports / "diagnosis/index.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>0.2.0 로그 검수</title>'
        "<style>body{font:16px sans-serif;margin:2rem;background:#f4f5f6}img{width:100%;max-width:1400px}"
        "figure{margin:2rem 0}figcaption{padding:1rem;background:white}</style>"
        "<h1>0.2.0 로그 검수</h1><p>초록 GT / 파랑 승인 / 주황 UNKNOWN·재촬영 / 빨강 오승인. "
        "완전 성공이 아닌 이미지 전체를 표시합니다. 오승인 0건과 추가 박스 0개는 다른 조건입니다.</p>"
        + cards
        + "</html>",
        encoding="utf-8",
    )


def verify(root: Path) -> dict:
    output = root / f"artifacts/distributions/{VERSION}"
    version = root / f"artifacts/versions/{VERSION}"
    config = load_json_config(root / f"configs/versions/{VERSION}.json")
    for lock in config["evaluation_evidence"]:
        if sha256_file(root / lock["path"]) != lock["sha256"]:
            raise ValueError("Frozen source evidence changed")
    checked = {}
    for component in ("runtime", "catalog"):
        source = root / config[component]["path"]
        if (
            directory_content_manifest(source)["manifest_sha256"]
            != config[component]["manifest_sha256"]
        ):
            raise ValueError("Frozen source model changed")
        excluded = (
            {"metadata.json"}
            if component == "runtime"
            else {"catalog.json", "checksums.json", "signature.json"}
        )
        before = {
            p.relative_to(source).as_posix(): sha256_file(p)
            for p in source.rglob("*")
            if p.is_file() and p.name not in excluded
        }
        stage = version / f"staging/{component}"
        after = {
            p.relative_to(stage).as_posix(): sha256_file(p)
            for p in stage.rglob("*")
            if p.is_file() and p.name not in excluded
        }
        if before != after:
            raise ValueError("Product version assignment changed model payload")
        checked[component] = {
            "immutable_file_count": len(before),
            "payload_sha256": canonical_sha256(before),
        }
    for profile, worker in {
        "cpu": root / f"artifacts/installers/{VERSION}/windows-payload/worker",
        "cuda": version / f"bixolon-bakery-ai-scanner-{VERSION}/worker",
        "n100": root / f"artifacts/n100/{VERSION}/worker-payload/worker",
    }.items():
        runtime = load_runtime_package_v2(worker / "model-package")
        catalog = load_store_catalog_package(
            worker / "store-catalog", expected_store_id="three_bakery"
        )
        if (
            runtime.metadata.worker_version != VERSION
            or catalog.metadata.catalog_version != VERSION
        ):
            raise ValueError("Packaged model version mismatch")
        for component, folder in (("runtime", "model-package"), ("catalog", "store-catalog")):
            if directory_content_manifest(worker / folder) != directory_content_manifest(
                version / f"staging/{component}"
            ):
                raise ValueError("Packaged model differs from measured staging payload")
        if profile != "n100":
            smoke = load_json_config(version / f"packaged-{profile}-smoke.json")
            if (
                not smoke["passes"]
                or directory_content_manifest(worker)["manifest_sha256"]
                != smoke["worker_artifact_content_manifest_sha256"]
            ):
                raise ValueError("Worker differs from its successful packaged smoke")
    for payload in (
        root / f"artifacts/lite/{VERSION}/payload",
        root / f"artifacts/n100/{VERSION}/worker-payload",
        root / f"artifacts/n100/{VERSION}/lite-payload",
    ):
        verify_bundle(payload)
    for row in load_json_config(output / "reports/evidence-sources.json")["files"]:
        if sha256_file(output / "reports" / row["path"]) != row["sha256"]:
            raise ValueError("Copied evidence changed")
    deliverables = sorted(p for p in output.rglob("*") if p.suffix in {".zip", ".exe"})
    if len(deliverables) != 11:
        raise ValueError("Expected eleven 0.2.0 deliverables")
    for path in deliverables:
        if sha256_file(path) != path.with_suffix(path.suffix + ".sha256").read_text().split()[0]:
            raise ValueError("Deliverable checksum mismatch")
        if path.suffix == ".zip":
            print(f"ZIP CRC: {path.name}", flush=True)
            with zipfile.ZipFile(path) as bundle:
                if bundle.testzip() is not None:
                    raise ValueError("ZIP CRC mismatch")
                if any(Path(n).is_absolute() or ".." in Path(n).parts for n in bundle.namelist()):
                    raise ValueError("Unsafe ZIP path")
                checked[path.name] = {"zip_crc_passed": True, "file_count": len(bundle.namelist())}
    for old, expected in {
        "0.1.17": "af21c1bcffaaef78fb479101c8412de6db0021331063d20fe82407a9b3b8a15a",
        "0.1.18": "389953874af2ab45063d8562aa7b7ceff5f9b5357b37dacecf5e11557aec491f",
    }.items():
        if (
            sha256_file(root / f"artifacts/distributions/{old}/distribution-manifest.json")
            != expected
        ):
            raise ValueError("Previous release manifest changed")
    result = {
        "passed": True,
        "deliverable_count": len(deliverables),
        "checks": checked,
        "n100_measured": False,
    }
    write_json(output / "reports/distribution-verification.json", result)
    files = [
        r
        for r in directory_content_manifest(output)["files"]
        if r["path"] not in {"distribution-manifest.json", "SHA256SUMS.txt"}
    ]
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{r['sha256']}  {r['path']}\n" for r in files), encoding="utf-8"
    )
    write_json(
        output / "distribution-manifest.json",
        {
            "product_version": VERSION,
            "sdk_version": "1.2.0",
            "files": files,
            "source_candidate": config["source_candidate"],
            "checksum_index_sha256": sha256_file(output / "SHA256SUMS.txt"),
            "n100_hardware_measured": False,
            "n100_target_met": None,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "verify"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    print(report(root) if args.command == "report" else verify(root))


if __name__ == "__main__":
    main()
