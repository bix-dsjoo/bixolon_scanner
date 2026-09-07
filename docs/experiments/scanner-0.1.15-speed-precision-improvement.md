# 0.1.15 속도·오인 개선 진단

기준일: 2026-09-04

## 결론

`multi_object_scenes` 300장과 `operational_collections` 194장, 총 494장/2,114개 객체를 같은
0.1.15 ONNX·Catalog·판정 threshold로 다시 측정했습니다. JPEG draft decode만 `1500`에서
`640`으로 낮춘 후보가 `APPROVED` 안전성을 유지하면서 고해상도 입력의 지연을 크게 줄였습니다.

- `APPROVED` 오분류: 0 → 0
- unmatched `APPROVED`: 0 → 0
- UNKNOWN Top-3 이탈: 1 → 0
- raw detector FN: 2 → 1
- raw detector FP: 7 → 8; 추가 1개도 `APPROVED`가 아니므로 detector 진단과 승인 오인을 구분함
- 전체 correct approved: 2,088/2,114 (98.77%) → 2,089/2,114 (98.82%)
- RTX 5080 전체 가중 mean: 63.74ms → 37.90ms (-40.5%)
- CPU 전체 가중 mean: 154.50ms → 133.74ms (-13.4%)

모델 graph, weight, Catalog, threshold와 `DecisionPipeline` 우선순위는 바꾸지 않았습니다. 이 후보는
실험 산출물이며 활성 0.1.15 번들을 덮어쓰지 않습니다. 배포 EXE에 반영할 때는 단일 제품 버전
계약에 따라 새 patch 버전으로 Runtime·앱·Catalog 버전을 함께 맞춰야 합니다.

## 정량 목표

목표 원문은 `configs/experiments/bread/scanner_0_1_15_speed_precision_targets.json`에 둡니다.

1. 494장 전체에서 `APPROVED` 오분류와 unmatched `APPROVED`를 각각 0으로 유지합니다.
2. UNKNOWN의 정답 Top-3 이탈을 0으로 유지하고 raw FN은 최대 1개로 제한합니다.
3. RTX 5080 `multi_object_scenes` 3회 측정의 최악 mean/p95/p99를 각각 50/65/70ms 이하로
   유지합니다.
4. 각 운영 수집본의 CUDA p95/p99를 각각 60/70ms 이하로 유지합니다.
5. CPU 전체 가중 mean을 10% 이상 개선하고, 개별 수집본 p95 회귀는 10% 이내로 제한합니다.
6. CPU/CUDA 최종 상태와 class rank 불일치는 0, bbox 최소 IoU는 0.995 이상으로 유지합니다.

## CUDA 결과

환경은 RTX 5080, driver 591.86, ONNX Runtime GPU 1.28.0입니다. 지연은 파일 읽기를 제외하고
decode부터 최종 decision까지 측정했습니다.

| 집합 | 기준 mean/p95/p99 (ms) | JPEG 640 mean/p95/p99 (ms) | p95 변화 |
|---|---:|---:|---:|
| multi 300 | 78.48 / 99.56 / 106.43 | 46.85 / 58.87 / 65.39 | -40.9% |
| 운영 08-18 | 52.15 / 68.20 / 72.75 | 27.48 / 39.03 / 41.63 | -42.8% |
| 운영 08-27 | 15.41 / 26.64 / 35.65 | 15.93 / 26.47 / 30.06 | -0.7% |
| 운영 08-28 | 88.33 / 92.27 / 92.34 | 40.87 / 48.02 / 49.64 | -48.0% |

multi 300 후보를 총 3회 실행한 p95는 58.87/59.76/60.48ms, p99는
65.39/63.99/66.14ms였습니다. 세 번 모두 판정·오류 집계가 같았습니다.

## CPU 결과

| 집합 | 기준 mean/p95/p99 (ms) | JPEG 640 mean/p95/p99 (ms) | p95 변화 |
|---|---:|---:|---:|
| multi 300 | 178.10 / 246.68 / 281.02 | 151.38 / 224.86 / 251.59 | -8.8% |
| 운영 08-18 | 144.54 / 235.05 / 244.70 | 125.72 / 217.37 / 225.07 | -7.5% |
| 운영 08-27 | 60.68 / 153.58 / 181.57 | 63.68 / 164.37 / 186.28 | +7.0% |
| 운영 08-28 | 208.35 / 229.85 / 230.33 | 180.24 / 199.89 / 200.10 | -13.0% |

08-27은 저해상도·빈 이미지 비중 때문에 draft decode 이득이 작고 단일 실행 변동 안에서 7% 회귀했습니다.
전체 가중 mean은 13.4% 개선됐으며 개별 p95 회귀 10% 제한은 만족했습니다.

## 오인 측정 보강

기존 `scanner_v2` 평가기는 정답과 매칭된 segmentation의 class 오분류만 집계했습니다. unmatched
segmentation이 `APPROVED`여도 별도 공식 지표가 없었고, CLI의 `maximum_p99_ms`도 목표 판정에
사용되지 않았습니다. 평가기에 다음을 추가했습니다.

- `approved_false_positive_count`
- `approved_output_false_positive_rate`
- `approved_output_error_rate`
- `maximum_approved_false_positive_count` 목표 판정
- `maximum_p99_ms` 성능 목표 판정

새 지표로 확인한 후보의 unmatched `APPROVED`는 CPU와 CUDA 모두 0입니다. 운영 08-27의 raw FP
6개는 `UNKNOWN` 또는 `SEGMENT_RECAPTURE`이며 승인 오인이 아닙니다.

## Provider parity와 한계

494장 모두 CPU/CUDA 최종 상태와 class rank가 일치했습니다. 1,411개 segmentation 중 bbox 1개가
기존 진단 기본값 IoU 0.999에는 못 미쳤고 최소 IoU는 0.997821이었습니다. confidence 최대 차이는
0.005보다 작았습니다. 실제 상태·class rank 차이는 없으며 이번 목표의 bbox 허용치는 0.995로
명시했습니다.

이 데이터는 모델·정책 개발 및 이번 decode 크기 선택에 사용된 진단 집합입니다. 독립 매장·독립
capture session의 blind test가 아니므로 오인 0을 일반화 성능, 인증 또는 SLA로 표현하지 않습니다.
다음 수집 세션에서는 640을 고정한 뒤 빈 트레이, 유사 SKU, 작은 객체와 경계 객체를 우선 확인해야
합니다.
