# RPC200 ambiguity gate v10 검증 결과

## 결론

`val2019` selection 전체에서 실제 classifier 오분류 승인을 Easy, Medium, Hard
모두 0건으로 줄였다. detector가 만든 unmatched 추가 박스 승인은 classifier 오류와
분리하여 `Segmentation 실패`로 집계했다. 이 기준의 실제 classifier 오인율은
`0 / 35,445 = 0.000%`이다.

이 결과는 validation 실패 사례를 분석해 반복 개선한 `validation_candidate`이며,
봉인된 `test2019` 결과나 운영 승격 결과가 아니다. `test2019`는 접근하지 않았다.

## 요청 형식 결과표

`Segmentation 실패 이미지`, `Top-3 Candidate`, `Candidate Out`, `오인율*` 네 구성비는
난이도별로 100%가 된다. 여기서 `Segmentation 실패 이미지`는 `IMAGE_RECAPTURE`,
`SEGMENT_RECAPTURE`, 승인된 detector unmatched 추가 박스를 합친 실패 outcome이다.
`오인율*`은 matched ROI의 실제 classifier 오분류만 센다.

| 난이도 | 인식률 | Segmentation 실패 이미지 | Top-3 Candidate | Candidate Out | 오인율* | 속도 평균 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 99.237% | 68.072% (113건) | 30.723% (51건) | 1.205% (2건) | 0.000% (0건) | 50.85 / 58.29ms |
| Medium | 99.603% | 78.475% (175건) | 21.525% (48건) | 0.000% (0건) | 0.000% (0건) | 60.38 / 68.63ms |
| Hard | 99.368% | 74.880% (313건) | 25.120% (105건) | 0.000% (0건) | 0.000% (0건) | 69.62 / 77.58ms |

독립 KPI는 다음과 같다.

| 난이도 | 실제 classifier 오분류 | 승인된 detector unmatched | APPROVED precision | E2E | SEGMENT_RECAPTURE |
|---|---:|---:|---:|---:|---:|
| Easy | 0건 | 7건 | 99.899% | 96.404% | 92건 |
| Medium | 0건 | 14건 | 99.884% | 97.512% | 153건 |
| Hard | 0건 | 47건 | 99.716% | 96.332% | 256건 |

모든 난이도에서 인식률 99% 이상, 실제 classifier 오인율 0%, RTX 5080 평균/P95
100ms 이하를 확인했다. unmatched를 포함한 전체 `APPROVED` precision도 모두
99.5% 이상이다.

## 실패 원인

v8의 실제 classifier 오분류 4건을 원본·detector crop·GT crop으로 확인했다.
네 ROI 모두 하나의 bbox에 둘 이상의 상품이 크게 포함되었고 classifier는 bbox 안의
다른 실제 상품을 높은 신뢰도로 선택했다.

- Easy 1건: detector ROI의 약 97.5%를 다른 GT 상품도 차지
- Medium 1건: 다른 GT 상품이 ROI의 약 60.2%를 차지
- Hard 2건: 다른 GT 상품이 각각 약 43.4%, 50.5%를 차지
- 네 건 모두 같은 이미지에 더 높은 detector score의 동일 예측 클래스 ROI가 존재

따라서 confidence threshold를 전역 상향하지 않고, 겹친 저순위 중복 ROI만
`SEGMENT_RECAPTURE`로 분리했다.

## v10 정책

버전 관리 설정은 `configs/rpc_zero_misapproval_v10.json`이다.

1. v8 class-aware NMS IoU `0.55`를 먼저 적용한다.
2. classifier가 계산된 뒤 같은 예측 클래스의 더 높은 detector score ROI가 있는지
   확인한다.
3. detector score 순위가 하위 15%이고 score가 `0.87` 이하이며, 같은 클래스 ROI와
   containment가 `0.45` 이상이면 `SEGMENT_RECAPTURE` 처리한다.
4. 같은 클래스가 반복되고 context quality가 기존 경계의 `1.118배` 미만인 하위 15%
   ROI도 `SEGMENT_RECAPTURE` 처리한다.

새 모델 추론은 추가하지 않는다. 이미 계산한 bbox, detector score, classifier Top-1,
context quality만 사용한다.

## 비열화 관리

초기 넓은 gate는 selection `SEGMENT_RECAPTURE`를 1,566건 늘려 E2E를 약
93~94%로 떨어뜨렸으므로 폐기했다. 최종 gate는 추가 recapture를 378건으로 줄였다.

v8 대비 E2E 변화:

- Easy: 96.642% → 96.404%, `-0.238pp`
- Medium: 97.812% → 97.512%, `-0.300pp`
- Hard: 96.746% → 96.332%, `-0.414pp`

오분류 4건을 제거하는 대신 전체 정답 영역 성공률이 최대 0.414pp 감소했다. 이
trade-off를 숨기지 않고 운영 승격 전 test gate에서 다시 확인해야 한다.

## RTX 5080 속도

난이도별 고정 SHA 순서 200장, warm-up 30장, ONNX Runtime CUDA EP에서 측정했다.
JPEG decode, detector, classifier, context ONNX, class-aware NMS와 v10 ambiguity
계산을 포함한다.

| 난이도 | 표본 | 평균 | P50 | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| Easy | 200 | 50.850ms | 50.323ms | 58.289ms | 60.636ms |
| Medium | 200 | 60.380ms | 60.432ms | 68.627ms | 71.919ms |
| Hard | 200 | 69.624ms | 69.178ms | 77.580ms | 81.461ms |

## 재현

기준 artifact 경로:
`artifacts/experiments/rpc-data-scale-diverse-worker-gated`

정확도 재평가:

```powershell
$env:PYTHONPATH = "src"
python -m bixolon_scanner.training.rpc_class_aware_nms `
  --config configs\rpc_data_scale.json `
  --output-dir artifacts\experiments\rpc-data-scale-diverse-worker-gated `
  --class-aware-nms-threshold 0.55 `
  --duplicate-overlap-threshold 0.45 `
  --duplicate-overlap-max-score 0.87 `
  --duplicate-low-quality-multiplier 1.118 `
  --duplicate-min-rank 0.85 `
  --report-version v10-zero-misapproval-final
```

주요 artifact:

- 전체 정확도: `runs/full/seed20260810/class-aware-nms-v10-zero-misapproval-final/report.json`
- 오류 contact sheet: `reports/classifier-errors-v10.jpg`
- calibration 오류 contact sheet: `reports/classifier-errors-calibration-v10.jpg`
- RTX 속도: `reports/validation-v10-zero-misapproval-*-benchmark.json`

## 승격 상태

현재 구현은 RPC200 validation 평가·benchmark 후보이다. 기존 bread Worker/API,
20종 label map과 운영 모델 패키지는 변경하지 않았다. 실제 운영 승격은 고정 정책을
모델 패키지 metadata에 봉인하고 Worker 상태 parity를 검증한 뒤, 봉인된 test를 한
번 평가하는 별도 단계로 남겨 둔다.
