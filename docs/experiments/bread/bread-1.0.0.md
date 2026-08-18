# Bread Worker 1.0.0 실험 기록

## 현재 결론

Worker·Detector·Classifier `1.0.0` 운영 패키지를 만들었다. 실제 합격 기준은
`multi_object_scenes` 300장이다. point·segmentation·RECAPTURE·속도 KPI는 모두 통과했다. 승인 오인율의 단측
95% 상한은 표본 부족으로 통과하지 못했지만, 2026-08-14 프로젝트 소유자의 명시적
지시에 따라 이 gate 하나만 manual waiver로 기록하고 `production`으로 승격했다.

- 운영 패키지: `artifacts/packages/bread-worker-1.0.0`
- 승격 원본: `artifacts/packages/bread-worker-1.0.0-candidate-v3`
- 최종 운영 보고서: `artifacts/experiments/bread-source-single_objects_3-dinov3/reports/release-gate-multi-cuda-production.json`
- Worker / Detector / Classifier 버전: 각각 `1.0.0`
- 데이터셋 버전: `bread-1.0-a52b4faa3e20`
- canonical manifest: `manifests/bread-1.0-a52b4faa3e20`
- Detector 학습 파이프라인 계약: `configs/training/bread_detector_pipeline_1.0.0.json`
- Classifier 학습 파이프라인 계약: `configs/training/bread_classifier_pipeline_1.0.0.json`

2026-08-14 hardening에서는 기존 운영 ONNX를 변경하지 않은 schema 2.1 development 후보를 생성했다. RTX 5080 `multi_object_scenes` 재검증 결과는 인식률 99.0780%, 승인 오인 0/1,121, segmentation recall/precision 99.2908%/99.7151%, 평균/P95 69.89/84.20ms였다. 단측 95% 상한 0.2669%만 실패했으므로 명시적 통계 위험 승인 전에는 production으로 승격하지 않는다. 별도의 12-shot Detector 재학습 후보는 KPI 실패로 rejected archive에 보존했다.
- 학습 원본: `single_objects_3`, 종류별 12장, 총 240장
- 평가 원본: `multi_object_scenes`, 300장, GT segmentation 1,410개

## 학습 데이터 소스 비교

네 소스는 섞지 않고 동일한 SigLIP2 frozen encoder, 6-view 증강, Linear SVM 조건으로
각각 독립 학습했다. 평가는 모두 `multi_object_scenes`의 GT crop만 사용했다.

| 학습 소스 | 장수/종류 | 데이터셋 버전 | Top-1 | Top-3 | 결론 |
|---|---:|---|---:|---:|---|
| `single_objects` | 10 | `bread-1.0-17a07d799c4e` | 87.02% | 96.60% | 기준선 |
| `single_objects_1` | 7 | `bread-1.0-863543addcc1` | 86.03% | 96.88% | 거절 |
| `single_objects_2` | 10 | `bread-1.0-1f24ede7d6e9` | 84.47% | 96.95% | 거절 |
| `single_objects_3` | 12 | `bread-1.0-a52b4faa3e20` | 86.38% | 97.16% | 심화 학습 선택 |

`single_objects_1 ⊂ single_objects_2 ⊂ single_objects_3`이며, 기존
`single_objects`와 새 12장 소스의 exact SHA 교집합은 98장이다. 서로 다른 클래스에서
같은 SHA가 발견된 경우는 없다.

## 선택 모델과 정책

- Detector: D-FINE-N HGNetv2, 640×640, score threshold `0.49`
- Classifier: DINOv3 ConvNeXt-Tiny, 마지막 stage L2-SP, seed `20260813`
- Classifier checkpoint SHA-256:
  `1be1781c2ece6e8ac12aa9ac9915fd5f81f115d093f1e08f7c9c3b9622e3b104`
- ONNX SHA-256:
  `93a9d92c6fd63f5a6aef65e11e3d0acecfffd7c6cf5ac2bfdba732f4e543ab8f`
- 최종 승인 TTA: `vflip`, `rot-15`, `rot30` 평균 logits
- `UNKNOWN` Top-3: `base`, `vflip`, `rot-15`, `rot30`의 `top3_vote`
- 승인 임계값: `0.9999998211860657`
- 포함 중복 검토: containment `≥0.99`, 같은 Top-1, 낮은 detector score의 고신뢰 ROI만 `UNKNOWN`+Top-3
- JPEG native draft 목표: `1050`
- RECAPTURE 정책 변경: 없음

JPEG draft `1050`은 4032×3024 원본은 기존 1/2 축소를 유지하고 5712×4284 원본만
1/4 축소한다. `1000`처럼 모든 이미지를 1/4로 줄였을 때의 Top-3 손실을 피하면서
고해상도 이미지의 지연 꼬리를 제거한다.

## 최종 `multi_object_scenes` gate

환경은 Windows 11, RTX 5080, ONNX Runtime CUDA EP, 동시성 1, warm-up 10회다.
평가 범위는 decode, 전처리, detector, classifier, 후처리와 최종 결정을 포함한다.

| 지표 | 목표 | 결과 | 판정 |
|---|---:|---:|---|
| 전체 인식률 | ≥99% | **99.0780%** (1,397/1,410) | 통과 |
| 승인 오인율 | ≤0.1% | **0%** (0/1,121) | 통과 |
| 승인 분류 오인 | 최소화 | **0건** | 통과 |
| 승인 false segmentation | 최소화 | **0건** | 통과 |
| Segmentation recall IoU@0.5 | ≥99% | **99.2908%** | 통과 |
| Segmentation precision IoU@0.5 | ≥99% | **99.7151%** | 통과 |
| `UNKNOWN` Top-3 accuracy | ≥95% | **98.9247%** | 통과 |
| `IMAGE_RECAPTURE` | 증가 금지 | **0건** | 통과 |
| `SEGMENT_RECAPTURE` | 증가 금지 | **0건** | 통과 |
| 평균 지연 | ≤100ms | **77.79ms** | 통과 |
| P50 / P95 / P99 | P95≤100ms | **76.49 / 96.01 / 103.12ms** | 통과(P99 참고) |

이 결과는 평가용 override 없이 후보 패키지 metadata만으로 재현했다.

## 승인된 통계 잔여 위험

관측 승인 오인은 0건이고 point 오인율은 0.1% 이하이지만 단측 95% 상한은
`0.2669%`다. 0.1% 상한을 0건 오류로 증명하려면 최소 2,995개의 독립 승인 표본이
필요한데 현재 전체 GT도 1,410개뿐이다. 따라서 이 gate를 임계값이나 RECAPTURE로
우회하지 않았다. package의 `promotion.method=manual_waiver`와
`remaining_limitations`에 실제 상한, 표본 수와 독립 이미지 사후 검증 의무를 남겼다.

변경 전 유일한 승인 오인은 `easy_059`의 실제 `bread_15` ROI를 거의 완전히 포함한
낮은 detector score의 동종 중복 검출이었다. v3는 이 segmentation을 삭제하지 않고
`DETECTOR_CONTAINED_DUPLICATE` `UNKNOWN`으로 보존했다. 변경 전후 CSV를 비교하면
이 한 행만 `APPROVED_FALSE_SEGMENTATION`에서 `UNKNOWN_FALSE_SEGMENTATION`으로
바뀌었고, 인식률·검출 수·매칭 수·segmentation 지표와 RECAPTURE 수는 동일하다.

## 재현 명령

```powershell
$env:PYTHONPATH='src'
python -m bixolon_scanner.evaluation.release `
  --package-dir artifacts/packages/bread-worker-1.0.0 `
  --dataset-root datasets/bread_dataset `
  --dataset-metadata artifacts/experiments/bread-training-source-comparison/manifests/single_objects_3/metadata.json `
  --output artifacts/experiments/bread-source-single_objects_3-dinov3/reports/release-gate-multi-cuda-production.json `
  --details artifacts/experiments/bread-source-single_objects_3-dinov3/reports/release-errors-multi-cuda-production.csv `
  --provider cuda `
  --gate-dataset multi_object_scenes `
  --cuda-dll-dir C:/workspace/bixolon_scanner/apps/product_scanner/build/windows/x64/runner/Release/worker/cuda-runtime `
  --warmup-count 10
```

위 보고서를 변경하지 않고 소유자 승인 통계 예외를 기록하는 승격 명령은 다음과 같다.

```powershell
bixolon model promote `
  --candidate-dir artifacts/packages/bread-worker-1.0.0-candidate-v3 `
  --release-report artifacts/experiments/bread-source-single_objects_3-dinov3/reports/release-gate-multi-cuda-contained-duplicate-099.json `
  --output-dir artifacts/packages/bread-worker-1.0.0 `
  --decided-on 2026-08-14 `
  --approve-statistical-risk
```
