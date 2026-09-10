"""Collect actual R4 evidence without substituting probe accuracy for final scan results."""

import argparse
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.evaluation.structural_diagnostics import BASELINES, compare
from bixolon_scanner.operations.n100_test_kit import digest, write_json
from bixolon_scanner.training.three_bakery_data import read_jsonl


def counts(row):
    states = row["item_status_counts"]
    return (
        f"{row['correct_approved_count']} / {row['ground_truth_count']} | "
        f"{row['wrong_approved_count']} | {row['missed_count']} | "
        f"{states.get('APPROVED', 0)} / {states.get('UNKNOWN', 0)} / "
        f"{states.get('SEGMENT_RECAPTURE', 0)} | {row['extra_count']}"
    )


def run(root):
    evidence = {}
    baselines = {name: read_jsonl(Path(path)) for name, path in BASELINES.items()}
    for family in ["students", "detectors", "quantization"]:
        for path in sorted((root / family).glob("*/regression/*.json")):
            dataset = path.stem
            if dataset not in baselines:
                continue
            data = load_json_config(path)
            evidence[path.relative_to(root).as_posix()] = {
                "sha256": digest(path),
                "summary": data["summary"],
                "comparison": compare(baselines[dataset], data["rows"]),
                "timing_scope": data["scope"],
            }
    write_json(root / "all-regression-evidence.json", evidence)
    selected = {
        dataset: evidence[f"students/repvit_m0_9-supervised/regression/{dataset}.json"]
        for dataset in baselines
    }
    cpu = load_json_config(root / "paired-cpu-repvit.json")
    parity = load_json_config(root / "repvit-cpu-gpu-parity.json")
    lines = [
        "# N100 구조 최적화 R4 결과",
        "",
        "2026-09-10. 0.2.1 기반의 새 Worker·새 모델 실험 배포물이다. "
        "**N100 전 요청 HTTP 1초 이내 달성은 아직 확인되지 않았다.** "
        "현재 N100 측정 대상으로 남긴 분류기는 RepViT-M0.9 GT 일반 학습 모델이다. "
        "공식 설치본을 자동 교체하지 않는다.",
        "",
        "## 핵심 결과",
        "",
        "로그132에서 정답 승인이 1078→1085/1096, 오승인·미검출 0, "
        "SEGMENT_RECAPTURE 19→18이다. APPROVED/UNKNOWN/SEGMENT_RECAPTURE는 "
        "1085/9/18이며 추가 검출은 16개로 같다. detector·GT 객체를 삭제하거나 "
        "상태만 바꿔 만든 개선이 아니다.",
        "",
        "전체 734장에서 CPU와 실제 Intel GPU 실행의 박스·상태·reason·상품 ID·이름·Top-3 "
        "순위 차이는 0건이었다. GPU 측정에서 classifier fallback을 끈 상태로 확인했다. "
        "float confidence 자체의 비트 동일성을 의미하지 않는다. "
        "선정 후보의 734장 최상위 상태는 모두 SEGMENTATION이며 IMAGE_RECAPTURE·ERROR는 0개다.",
        "",
        "| 데이터 | 정답 승인 / GT | 오승인 | 미검출 | APPROVED / UNKNOWN / 재촬영 | 추가 검출 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, entry in selected.items():
        lines.append(f"| {dataset} | {counts(entry['summary'])} |")
    lines += ["", "개별 회귀도 함께 보존한다:", ""]
    for dataset, entry in selected.items():
        change = entry["comparison"]
        losses = ", ".join(str(row["image_id"]) for row in change["lost_correct"]) or "없음"
        lines.append(
            f"- {dataset}: 정답 승인 개선 {len(change['gained_correct'])}개 / "
            f"손실 {len(change['lost_correct'])}개(이미지 {losses}), "
            f"새 오승인 {len(change['new_wrong_approvals'])}개."
        )
    lines += [
        "",
        "final300의 오승인 4개는 기존과 같은 대상·상품이다. "
        "전체 데이터 오승인 0으로 표현하지 않는다. "
        "로그10·122의 정답 승인 2개는 재촬영으로 바뀌었으며, "
        "순증 +7이 이 손실을 없애는 것은 아니다. "
        "final222·250·290·299의 정답 승인 손실도 남겨 두었다.",
        "",
        "[원본·GT·기준·후보 실패 영역 비교](failure-images/index.html), "
        "[후보별 개별 변화 JSON](all-regression-evidence.json), "
        "[CPU/GPU parity](repvit-cpu-gpu-parity.json).",
        "",
        "## 1. 분류기 구조 교체",
        "",
        "RepViT-M0.9 / FastViT-T8에 일반 학습·teacher 증류·관계 증류를 각각 적용했다. "
        "동일 seed, source 7947 crop, 20 epoch 마지막 checkpoint를 사용했다. "
        "6개 모델 × 로그132+원본302+final300 = 4404회 최종 scan 회귀다.",
        "",
        "| 로그132 후보 | 정답 승인 / GT | 오승인 | 미검출 | 승인 / UNKNOWN / 재촬영 | 추가 검출 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in evidence.items():
        if key.startswith("students/") and key.endswith("/log.json"):
            lines.append(f"| {key.split('/')[1]} | {counts(value['summary'])} |")
    lines += [
        "",
        "RepViT 일반 학습만 N100 실험에 유지했다. RepViT 관계 증류는 로그 오승인, "
        "일반 FastViT는 최소 정답 수 미달, 나머지는 재촬영 악화나 더 넓은 데이터의 새 오승인으로 "
        "제외했다. FastViT 관계 증류는 로그만 보면 통과하지만 원본에서 새 오승인 1개, "
        "final300에서 기존 대비 새 오승인 4개가 확인됐다.",
        "",
        "## 2. 작은 detector",
        "",
        "D-FINE-N을 동일 source 2408 training row로 GT 일반 학습·GT 대응 teacher 증류 "
        "각 12 epoch 학습했다. teacher의 GT 미대응 검출은 학습 정답으로 추가하지 않았다. "
        "두 모델 모두 누락·오승인이 발생해 제외했다. 분류기는 이 실험에서 기존 DINO를 사용했다.",
        "",
        "| 로그132 후보 | 정답 승인 / GT | 오승인 | 미검출 | 승인 / UNKNOWN / 재촬영 | 추가 검출 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in evidence.items():
        if key.startswith("detectors/") and key.endswith("/log.json"):
            lines.append(f"| {key.split('/')[1]} | {counts(value['summary'])} |")
    lines += [
        "",
        "작은 graph 자체가 빨라도 추가 ROI가 늘면 후속 분류·검증 호출이 증가한다. "
        "현재 N 모델의 검출 품질은 S를 대체하기 부족하다. 모델 크기만으로 전체 속도를 판단하지 않았다.",
        "",
        "## 3. detector 특징 재사용",
        "",
        "기존 D-FINE-S의 stride 8 FPN 출력을 노출하고 ROIAlign 3×3 + 작은 identity/integrity "
        "head를 source 6000 ROI에 100 epoch 학습했다. detector 연산과 weight는 바꾸지 않았다.",
        "",
        "| 방식 | GT 수 | Top-1 정답 | Top-3 포함 |",
        "|---|---:|---:|---:|",
    ]
    for objective in ["supervised", "relational"]:
        probe = load_json_config(root / "shared-roi" / objective / "gt-log-probe.json")
        lines.append(
            f"| {objective} | {probe['gt_count']} | {probe['correct_top1']} | {probe['correct_top3']} |"
        )
    lines += [
        "",
        "이 수치는 **정답 위치를 미리 준 분류 probe**다. Catalog·최종 상태·검출 오류가 "
        "포함되지 않아 승인 수나 미검출 0으로 해석할 수 없다. 유리한 위치 조건에서도 identity "
        "구분이 부족해 현재 head는 제외했다. 향후에는 SKU 구분을 함께 학습한 backbone과 "
        "새 촬영 세션 검증이 필요하다.",
        "",
        "## 4. 정확도 보존형 INT8",
        "",
        "NNCF 3.3 ONNX accuracy restoration을 실행했다. RepViT는 고정 source 256 ROI의 "
        "Top-1·두 margin 경계·품질 경계가 각 입력에서 모두 유지되어야 한다. "
        "detector는 source 64장에서 기존 GT 대응을 잃지 않고 추가 검출을 늘리지 않아야 한다. "
        "이 callback은 독립 validation이 아니며, 통과해도 전체 scan 회귀가 별도로 필요하다.",
        "",
    ]
    for path in sorted((root / "quantization").glob("*/report.json")):
        report = load_json_config(path)
        elapsed = (
            f"{report['elapsed_seconds']:.1f}초 ({report.get('elapsed_scope', '복원·검증')})"
            if "elapsed_seconds" in report
            else "초기 PTQ 출력 진단"
        )
        lines.append(
            f"- {path.parent.name}: quantize node {report['quantize_nodes']}개, "
            f"{elapsed}. "
            f"[실행 증빙]({path.relative_to(root).as_posix()})"
        )
    for key, value in evidence.items():
        if key.startswith("quantization/") and key.endswith("/log.json"):
            row = value["summary"]
            lines.append(
                f"- {key.split('/')[1]} 로그132 최종 scan: 정답 {row['correct_approved_count']}, "
                f"오승인 {row['wrong_approved_count']}, 미검출 {row['missed_count']}, "
                f"재촬영 {row['item_status_counts'].get('SEGMENT_RECAPTURE', 0)}."
            )
    lines += [
        "",
        "RepViT PTQ는 약24분 자동 복원 후 quantize/dequantize node가 모두 0개가 되어 "
        "FP32로 돌아왔다. 정확도를 지켰지만 INT8 가속 성공이 아니다. "
        "detector 복원은 source64 중 최대62를 유지한 뒤 여러 전체 source 평가에서도 "
        "개선되지 않아 탐색을 중단했다. 200회 상한까지 전수 탐색한 결과가 아니며 "
        "INT8 detector가 원천적으로 불가능하다는 뜻도 아니다. "
        "detector-ptq는 복원 전 초기 PTQ 모델이며 복원 중 최고 모델과 구분한다.",
        "",
        "조건부 대조군으로 RepViT QAT+현재 FP32 student 증류를 source7947 ROI에 "
        "5 epoch, LR1e-5로 학습했다. observer는 첫 epoch 이후 고정했다. "
        "ONNX per-tensor scale/zero-point의 scalar 표현을 정리해 ORT에서 실행했지만 "
        "PyTorch 대비 embedding 최대 차이0.191(16 ROI), 원래 QDQ graph도0.206으로 "
        "parity를 통과하지 못했다. 로그 정답1070·재촬영30으로 회귀해 제외했다. "
        "threshold나 허용 오차를 낮춰 통과시키지 않았다.",
        "",
        "INT8 detector의 integral projection은 ORT CPU QLinearMatMul 커널의 "
        "zero-point shape 제약 때문에 FP32로 보존했다. NNCF의 weight 조회를 위해 상수 "
        "Identity alias를 동일 tensor로 표현했고, source 64장의 원본 출력 parity를 확인했다. "
        "실패 시도를 포함한 원문 로그도 보존했다.",
        "",
        "## 제어된 CPU 속도",
        "",
        "개발 PC Core Ultra 9 285K의 같은 논리 코어 4개, AB/BA 교대, 설정별 132장×2회. "
        "**HTTP·디코딩 제외 in-process scan이며 N100 수치가 아니다.** "
        "다른 학습·검증 작업이 없는 측정만 사용했다.",
        "",
        "| 설정 | p50 / p95 / p99 / 최대 ms | 표본 수 |",
        "|---|---:|---:|",
    ]
    for name, latency in cpu["latency"].items():
        lines.append(
            f"| {name} | "
            + " / ".join(f"{latency[k]:.2f}" for k in ["p50_ms", "p95_ms", "p99_ms", "maximum_ms"])
            + f" | {latency['count']} |"
        )
    tensor_path = root / "cpu-model-benchmarks.json"
    if tensor_path.exists():
        tensor = load_json_config(tensor_path)
        lines += [
            "",
            "모델 단독 CPU도 같은 논리 코어4개, warmup5회 뒤30회 측정했다. "
            "아래는 batch1 합성 tensor이며 decode·ROI 생성·최종 정책·HTTP를 제외한다. "
            "head 시간에는 detector와 ROIAlign 비용도 포함되지 않는다.",
            "",
            "| 모델 | p50 / p95 / p99 / 최대 ms | 표본 수 |",
            "|---|---:|---:|",
        ]
        for name, model in tensor["models"].items():
            latency = model["rows"][0]["latency"]
            lines.append(
                f"| {name} | "
                + " / ".join(
                    f"{latency[key]:.3f}" for key in ["p50_ms", "p95_ms", "p99_ms", "maximum_ms"]
                )
                + f" | {latency['count']} |"
            )
        lines += [
            "",
            "소형 detector는 연산만 보면 빨랐지만 검출 회귀로 제외했다. "
            "초기 detector INT8는 FP32보다 느렸고, QAT 분류기도 FP32 RepViT보다 빨라지지 않았다. "
            "[전체 batch1/4/8/12 tensor 결과](cpu-model-benchmarks.json).",
        ]
    lines += [
        "",
        "두 반복 모두 기준 정답1078·오승인0·미검출0, RepViT 정답1085·오승인0·미검출0이다. "
        "검사 작업과 겹친 이전 측정은 별도 파일에 보존하고 공식 비교에서 제외했다.",
        "",
        "## 새 Worker와 N100 실행",
        "",
        "새 primary feature 공간과 기존 detail feature 공간은 별도 Catalog를 사용한다. "
        "하위 Catalog 경로·checksum·버전·상품 순서를 검증하고 독립 verifier payload의 "
        "동일성도 검사한다. 잘못된 Catalog를 조용히 재사용하지 않는다. "
        "threshold·NMS·상품별 routing·상태 우선순위는 바꾸지 않았다.",
        "",
        "패키지의 `1_RUN_ALL.cmd`를 한 번 실행한다. A0/A9 기준 반복, B RepViT Intel GPU, "
        "D 동일 RepViT CPU의 4개 설정 ×132장×3회 =1584 HTTP 요청이며 설정별 warmup10회는 별도다. "
        "새 EXE·모델을 직접 실행하므로 앱 재설치는 필요 없다. 코드·모델이 바뀌지 않았다는 뜻이 아니다. "
        "PowerShell/Python 설치 없이 동작하고 중복 측정을 차단한다. "
        "CPU detector → Intel GPU classifier → CPU 독립 verifier 및 명시적 CPU fallback을 유지한다.",
        "",
        "**D:\\N100\\2_MEASURE.cmd**가 R4를 실행하도록 갱신됐다. 공통 파일165개를 "
        "검증해 재사용하고 약370MB만 추가했다. 공통 모델을 참조하므로 Experiments-R2/R3 폴더는 "
        "유지한다. 기존 설치본·실험·결과를 지우지 않았으며 이전 실행문·manifest는 "
        "previous-measurement-before-r4에 보관했다. 반환할 파일은 종료 화면의 Result ZIP 한 개다. "
        "p50/p95/p99/최대와 1초 초과 요청을 모두 분석하며 느린 요청을 제외하지 않는다.",
        "",
        "## 남은 한계",
        "",
        "source 원본302장/649객체와 합성 파생물은 같은 물리 상품·촬영 세션 그룹이다. "
        "이를 임의 train/validation으로 나누지 않았고 고정 마지막 epoch로 실험했다. "
        "로그132와 final300은 반복 사용한 회귀 진단이며 독립 일반화 성능이나 SLA 증거가 아니다. "
        "새 모델 margin은 독립 데이터로 재보정되지 않았다. 승인 손실·기존 final300 오승인4개, "
        "추가 검출16개, N100 실제 메모리·전력·최악 지연은 남아 있다. "
        "checksum은 파일 손상·변경 탐지이며 발행자 진위 인증이 아니다.",
        "",
        "## 참고한 원문",
        "",
        "- [RepViT 논문](https://arxiv.org/abs/2307.09283), "
        "[실제 pretrained model card](https://huggingface.co/timm/repvit_m0_9.dist_450e_in1k)",
        "- [FastViT 원저자 구현](https://github.com/apple/ml-fastvit)",
        "- [D-FINE 원저자 구현](https://github.com/Peterande/D-FINE)",
        "- [OpenVINO 2026 accuracy control](https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/quantizing-models-post-training/quantizing-with-accuracy-control.html)",
    ]
    assert sum(len(value["status_class_rank_differences"]) for value in parity.values()) == 0
    (root / "REPORT-KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/n100/structural-r4"))
    run(parser.parse_args().root)
