# RPC200 Segmentation 실패 이미지 개선 v18

## 결론

`val2019` selection에서 v8을 기준으로 오류 원인을 분리하고 정책을 반복 검증했다.
v18은 실제 classifier 오분류 승인을 Easy, Medium, Hard 모두 0건으로 유지하면서,
중복 없는 `Segmentation 실패 이미지`를 v10 및 v8보다 줄였다. `test2019`는 접근하지
않았으며 이 결과는 운영 승격이 아닌 `validation_candidate`이다.

| 난이도 | 인식률 | Segmentation 실패 이미지 | Top-3 Candidate | Candidate Out | 오인율* | 속도 평균 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 99.010% | 49.64% (68건) | 48.17% (66건) | 2.19% (3건) | 0.00% (0건) | 55.63 / 70.66ms |
| Medium | 99.316% | 50.00% (83건) | 48.80% (81건) | 1.20% (2건) | 0.00% (0건) | 65.37 / 80.58ms |
| Hard | 99.103% | 49.32% (146건) | 50.68% (150건) | 0.00% (0건) | 0.00% (0건) | 77.95 / 96.80ms |

`오인율*`은 사용자가 지정한 원인 분리 규칙에 따라 matched ROI의 실제 classifier
오분류 승인만 계산한다. detector가 만든 unmatched 승인 박스는 classifier 오분류에
섞지 않고 `Segmentation 실패 이미지`에 포함한다.

네 결과 열은 난이도별 예외 outcome 구성비이며
`Segmentation 실패 이미지 + Top-3 Candidate + Candidate Out + 오인율 = 100%`다.
Easy는 `68+66+3+0=137`, Medium은 `83+81+2+0=166`, Hard는
`146+150+0+0=296`을 구성비 분모로 사용한다. 따라서 이 구성비를 전체 validation
이미지 대비 실패율로 해석하면 안 된다.

전체 validation 이미지 대비 중복 없는 Segmentation 실패율은 별도로 Easy
`68/1,006=6.759%`, Medium `83/1,000=8.300%`, Hard
`146/991=14.733%`다.

## 중복 없는 이미지 집계 정의

`Segmentation 실패 이미지`는 다음 image ID의 합집합을 전체 난이도 이미지 수로
나눈 값이다. 동일 이미지에 여러 실패 segment가 있어도 한 번만 센다.

1. `IMAGE_RECAPTURE` 이미지
2. 하나 이상의 `SEGMENT_RECAPTURE`가 있는 이미지
3. detector unmatched 박스가 `APPROVED`된 이미지

놓친 정답 영역과 전체 detector 오검출률은 별도 segment 지표로 유지하며 위 이미지
지표에 중복 합산하지 않는다.

## 동일 기준 버전 비교

| 버전 | 난이도 | 인식률 | 실패 이미지 | 실제 classifier 오인 | E2E |
|---|---|---:|---:|---:|---:|
| v8 | Easy | 99.224% | 8.350% (84) | 1건 | 96.642% |
| v8 | Medium | 99.587% | 10.700% (107) | 1건 | 97.812% |
| v8 | Hard | 99.359% | 17.861% (177) | 2건 | 96.746% |
| v10 | Easy | 99.237% | 9.642% (97) | 0건 | 96.404% |
| v10 | Medium | 99.603% | 13.100% (131) | 0건 | 97.512% |
| v10 | Hard | 99.368% | 21.998% (218) | 0건 | 96.332% |
| **v18** | **Easy** | **99.010%** | **6.759% (68)** | **0건** | **96.586%** |
| **v18** | **Medium** | **99.316%** | **8.300% (83)** | **0건** | **97.666%** |
| **v18** | **Hard** | **99.103%** | **14.733% (146)** | **0건** | **96.682%** |

v18은 v8 대비 실패 이미지를 Easy 16장, Medium 24장, Hard 31장 줄였고, v10
대비로는 각각 29장, 48장, 72장 줄였다. 세 난이도 모두 인식률 99% 이상을 유지한다.

## 실패 원인과 최종 정책

오분류 6건을 원본 이미지와 detector/GT crop으로 재검토했다. 모두 ROI 안에 이웃
상품이 크게 포함된 다중 상품 ambiguity였다.

- calibration 2건: class 160과 161이 한 이미지에서 서로 바뀐 상호 Top-2 교환
- selection 4건: 낮은 detector 순위 ROI가 이미 존재하는 다른 상품 class로 중복 예측

v10은 이를 전부 `SEGMENT_RECAPTURE`로 보내 정상 ROI도 많이 차단했다. v18은
segmentation 자체가 유효한 ambiguity를 `UNKNOWN+Top3`로 반환한다.

- class-aware NMS IoU: `0.55`
- context quality threshold: `0.00225`
- overlap: containment `>=0.452`, detector score `<=0.864`, quality `<=0.194`
- low quality duplicate: detector rank `>=0.857`, detector score `>=0.93`, quality `<0.0058`
- calibration confusion pair: Top-2 mutual pair `160↔161`
- ambiguity outcome: `UNKNOWN`

정책은 `configs/rpc_zero_misapproval_v18.json`에 고정했다.

## 폐기한 실험

- v11 Top-2 assignment conflict: selection 704 ROI를 차단하고 오류 3건이 남아 폐기
- v12 일반 mutual conflict: 405 ROI를 차단해 과도한 재촬영으로 폐기
- v13 confusion pair + overlap: 오인 0건이나 추가 차단 271 ROI로 개선 폭 부족
- v14 quality bounded overlap: 오인 0건, v10보다 개선됐으나 ambiguity를 재촬영 처리
- v17 context quality 0: 실패 이미지는 크게 감소했지만 Easy 98.798%, Hard 98.867%
- threshold `0.0020`: Easy 98.996%로 목표 미달

성능이 악화되거나 인식률 99%를 통과하지 못한 변경은 최고 결과로 채택하지 않았다.

## RTX 5080 벤치마크

Windows 11, RTX 5080, ONNX Runtime CUDA EP에서 난이도별 SHA 고정 200장,
warm-up 30장으로 디코딩부터 detector, classifier, context와 v18 후처리를 측정했다.

| 난이도 | 표본 | 평균 | P50 | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| Easy | 200 | 55.632ms | 53.665ms | 70.661ms | 76.867ms |
| Medium | 200 | 65.373ms | 64.040ms | 80.578ms | 88.493ms |
| Hard | 200 | 77.951ms | 76.393ms | 96.798ms | 104.537ms |

목표인 평균과 P95 100ms 이하는 모두 통과한다. Hard P99는 100ms를 초과하므로
참고값으로 그대로 기록한다.

## 재현

기준 artifact 경로는
`artifacts/experiments/rpc-data-scale-diverse-worker-gated`이다.

```powershell
$env:PYTHONPATH = "src"
python -m bixolon_scanner.training.rpc_class_aware_nms `
  --config configs\rpc_data_scale.json `
  --output-dir artifacts\experiments\rpc-data-scale-diverse-worker-gated `
  --class-aware-nms-threshold 0.55 `
  --context-threshold-override 0.00225 `
  --duplicate-overlap-threshold 0.452 `
  --duplicate-overlap-max-score 0.864 `
  --duplicate-overlap-max-quality 0.194 `
  --duplicate-low-quality-max-quality 0.0058 `
  --duplicate-low-quality-min-score 0.93 `
  --duplicate-min-rank 0.857 `
  --assignment-conflict-top-k 2 `
  --assignment-mutual-only `
  --assignment-mutual-pair 160:161 `
  --ambiguity-outcome unknown `
  --report-version v18-context-q00225-fixed
```

주요 Git 제외 artifact:

- 정확도: `runs/full/seed20260810/class-aware-nms-v18-context-q00225-fixed/report.json`
- v8 재집계: `runs/full/seed20260810/class-aware-nms-v8-image-metrics/report.json`
- v10 재집계: `runs/full/seed20260810/class-aware-nms-v10-image-metrics/report.json`
- 속도: `reports/validation-v18-context-q00225-*-benchmark.json`
- 오류 시각 감사: `reports/classifier-errors-v8-reaudit.jpg`
- calibration 감사: `reports/classifier-errors-calibration-v8-reaudit.jpg`

기존 bread Worker/API/20종 label map 및 운영 모델 패키지는 변경하지 않았다.
