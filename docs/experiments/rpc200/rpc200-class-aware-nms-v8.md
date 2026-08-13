# RPC200 class-aware NMS v8 검증 결과

## 원인 분리

기존 `context-logistic-v4`의 잘못된 `APPROVED` 117건은 다음 두 종류다.

- detector unmatched 오검출 승인: 113건
- 실제 classifier 오분류 승인: 4건

따라서 기존 표에서 117건 전체를 오인율로 표현한 것은 원인 분류상 부정확했다.
113건은 `Segmentation 실패`로, 실제 오인율 구성은 4건으로 분리한다.

| 난이도 | 인식률 | Segmentation 실패 이미지 | Top-3 Candidate | Candidate Out | 오인율* | 속도 평균 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 99.224% | 34.146% (28) | 62.195% (51) | 2.439% (2) | 1.220% (1) | 53.00 / 62.06ms |
| Medium | 99.587% | 43.820% (39) | 55.056% (49) | 0.000% (0) | 1.124% (1) | 63.32 / 73.97ms |
| Hard | 99.365% | 42.391% (78) | 56.522% (104) | 0.000% (0) | 1.087% (2) | 68.91 / 76.69ms |

`Segmentation 실패 이미지`는 기존 `IMAGE_RECAPTURE`와 승인된 detector unmatched
박스를 합산한 실패 원인 구성이다. 각 행의 네 실패 구성비 합은 100%다.

## v8 개선

시각 감사 결과 승인된 unmatched 박스 대부분은 배경 객체가 아니라 동일 상품을
중복 또는 부분 검출한 박스였다. 고신뢰 113건은 모두 어떤 GT와 IoU 0.1 이상이었고,
79건은 GT 최대 IoU가 0.45 이상이었다.

classifier가 이미 계산한 Top-1 class가 같고 bbox IoU가 0.55보다 큰 박스만
score 순으로 억제하는 `class-aware NMS`를 추가했다. 임계값 0.55는 calibration에서
선택했으며 selection은 정책 선택에 사용하지 않았다.

### 전체 selection 결과

| 난이도 | 인식률 | Segmentation 실패 이미지 | Top-3 Candidate | Candidate Out | 오인율* | 속도 평균 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 99.224% | 32.500% (26) | 63.750% (51) | 2.500% (2) | 1.250% (1) | 55.33 / 65.21ms |
| Medium | 99.587% | 37.500% (30) | 61.250% (49) | 0.000% (0) | 1.250% (1) | 64.08 / 73.74ms |
| Hard | 99.359% | 39.548% (70) | 59.322% (105) | 0.000% (0) | 1.130% (2) | 75.97 / 86.42ms |

- 승인된 detector unmatched: 113 → 94건, 19건 감소(16.8%)
- 실제 classifier 오분류: 4 → 4건, 변화 없음
- 인식률: Easy/Medium 동일, Hard -0.006 percentage point
- P95: Easy 65.21ms, Medium 73.74ms, Hard 86.42ms로 모두 100ms 이내
- detector 전체 false-positive count: Easy 70→63, Medium 119→95,
  Hard 201→164

## 폐기한 후보

- 일반 NMS/containment suppression: 겹쳐 놓인 정상 상품 recall을 크게 손상
- full-logit logistic objectness: 오승인 117→111건으로 효과 부족
- DINO embedding objectness: 오승인 117→111건으로 효과 부족

성능이 악화되거나 개선 폭이 작은 후보는 채택하지 않았고, v8 결과만 보존한다.

## Artifact

기준 경로는 `artifacts/experiments/rpc-data-scale-diverse-worker-gated`다.

- 평가: `runs/full/seed20260810/class-aware-nms-v8/report.json`
- 오검출 시각 감사: `reports/approved-unmatched-contact-sheet-v8.jpg`
- RTX benchmark: `reports/validation-class-aware-nms-v8-benchmark.json`

이 결과는 validation candidate이며 production 승격이나 봉인된 `test2019` 평가를
의미하지 않는다.
