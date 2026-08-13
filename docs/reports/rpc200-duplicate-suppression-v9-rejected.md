# RPC200 duplicate suppression v9 검토 결과

## 결론

v8 이후 남은 승인된 detector 추가 박스 94건을 줄이기 위해 bbox 포함관계와
detector/classifier 점수 차를 이용한 후처리를 실험했다. 강한 정책은 승인된 추가
박스를 94건에서 81건으로 줄였지만 정상 정답 박스도 제거하여 Easy, Medium, Hard의
End-to-End 성공률이 모두 하락했다. 따라서 이 변경은 폐기하고
`class-aware NMS v8`을 최고 결과로 계속 보존한다.

`test2019`는 접근하지 않았으며 모든 정책 선택과 비교는 calibration/selection에서만
수행했다.

## 기준 결과: 보존된 v8

아래 네 결과 구성비는 난이도별 합계가 100%이다. `오인율*`은 detector의 추가 박스가
아니라 실제 classifier 오분류만 센 값이다.

| 난이도 | 인식률 | Segmentation 실패 이미지 | Top-3 Candidate | Candidate Out | 오인율* | 속도 평균 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 99.224% | 32.500% (26건) | 63.750% (51건) | 2.500% (2건) | 1.250% (1건) | 55.33 / 65.21ms |
| Medium | 99.587% | 37.500% (30건) | 61.250% (49건) | 0.000% (0건) | 1.250% (1건) | 64.08 / 73.74ms |
| Hard | 99.359% | 39.548% (70건) | 59.322% (105건) | 0.000% (0건) | 1.130% (2건) | 75.97 / 86.42ms |

- 실제 classifier 오분류: 4 / 35,574 = 0.0112%
- 승인된 detector 추가 박스: 94건
- P95: 모든 난이도에서 100ms 이하

## 시도 1: 포함관계 + detector 점수 차

calibration에서 class-aware IoU 0.55를 유지하면서 다음 조합을 비교했다.

- bbox 포함률: 0.60, 0.70, 0.80, 0.90, 0.95
- detector 점수 차: 0.05, 0.10, 0.20, 0.30, 0.40

calibration의 `놓침 + 오검출` 합계는 기준 235건에서 219건으로 줄었다. 동률 중 더
보수적인 `포함률 > 0.90`, `점수 차 >= 0.05`를 고정하여 selection에 적용했다.

### 강한 정책 결과

| 난이도 | 인식률 | Segmentation 실패 이미지 | Top-3 Candidate | Candidate Out | 오인율* | 속도 평균 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 99.224% | 29.870% (23건) | 67.532% (52건) | 2.597% (2건) | 0.000% (0건) | 56.18 / 69.37ms |
| Medium | 99.579% | 36.250% (29건) | 62.500% (50건) | 0.000% (0건) | 1.250% (1건) | 65.82 / 76.97ms |
| Hard | 99.359% | 36.310% (61건) | 62.500% (105건) | 0.000% (0건) | 1.190% (2건) | 76.61 / 89.52ms |

표만 보면 승인된 추가 박스가 94건에서 81건으로 감소하고 classifier 오분류도
4건에서 3건으로 감소한다. 그러나 제거된 Easy 오분류 1건은 올바르게 고친 것이
아니라 해당 정답 박스 자체를 억제한 결과였다. 전체 정답 영역 기준 E2E는 v8 대비
Easy -0.042 percentage point, Medium -0.122 pp, Hard -0.052 pp 하락했다. 따라서
성능 개선으로 인정하지 않았다.

## 시도 2: classifier confidence gap 추가

정상 박스 제거를 막기 위해 낮은 박스의 classifier confidence가 높은 박스보다
0.02 이상 낮은 경우만 포함 박스를 억제했다.

- selection raw segmentation 오검출: 322 → 293건
- selection 놓침: 344 → 345건
- 승인된 detector 추가 박스: 94 → 94건
- 실제 classifier 오분류: 4 → 4건

raw 오검출은 줄었지만 사용자가 보는 승인 오검출과 classifier 오분류가 개선되지
않았다. 이 후보 역시 승격하지 않았다.

## 실제 classifier 오분류 4건 분석

selection의 실제 오분류는 Easy 1건, Medium 1건, Hard 2건이다. confidence는
0.9913~0.9992로 높아 단순 승인 threshold 상향만으로 안전하게 제거하기 어렵다.
특히 Medium 오류의 context quality score는 0.00573으로 현재 경계 0.00513에 매우
가깝지만, 나머지 세 오류는 0.0172~0.1936이어서 단일 quality threshold 상향은
정상 승인 coverage와 E2E를 더 크게 손상시킨다.

다음 개선은 validation 후처리 임계값을 더 탐색하는 대신, 이 네 오류와 남은
동일상품 중복 박스를 포함하는 group-safe hard-example 학습 또는 instance-level
중복 판별 모델로 진행해야 한다.

## 재현 artifact

Git 제외 기준 경로:
`artifacts/experiments/rpc-data-scale-diverse-worker-gated`

- v8 기준: `runs/full/seed20260810/class-aware-nms-v8/report.json`
- 포함관계 정책 후보: `runs/full/seed20260810/class-aware-nms-v9*/report.json`
- 오분류 audit: `runs/full/seed20260810/class-aware-nms-v9-error-audit/report.json`
- 정책 sweep: `reports/validation-duplicate-policy-sweep-v9.json`
- v9 RTX 측정: `reports/validation-class-aware-nms-v9-*-benchmark.json`

실패한 v9 코드는 작업 트리에서 제거했으며 운영 Worker/API/모델 패키지는 변경하지
않았다.
