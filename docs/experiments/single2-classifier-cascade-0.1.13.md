# single_objects_2 기반 classifier cascade 진단

## 결론

`single_objects_2`의 원본 200장만 출처로 사용해 5,000개 positive ROI를 만들고,
`ConvNeXt-Tiny 192 전체 → ConvNeXt-Tiny 224 선택 → frozen ViT-B/16 160 선택 검증`
구조를 다시 학습했다. source-view OOF로 선택된 후보는 prototype 초기화, backbone learning rate
`1e-5`, 3 epoch이다. 이 결과는 실험용 PyTorch 패키지이며 활성 `0.1.13` Runtime이나 ONNX로
승격하지 않았다.

고정된 detector trace의 실제 detector ROI 1,410개 진단에서는 detector Recall 100%, 잘못된
`APPROVED` 0건을 유지했지만 올바른 `APPROVED`는 95.74%, Top-3 누락은 1건이었다. 따라서 목표
전체를 달성한 운영 확정본이 아니다.

## 데이터와 분리 규칙

- 유일한 학습 원본: `datasets/bread_dataset/single_objects_2`, 20 class × 10 view = 200장
- positive 학습 ROI: clean 200 + appearance 1,600 + geometry 1,600 + context 1,600 = 5,000장
- 모델·정책 선택 probe: 600장, synthetic reject: 800장
- 동일 source view 파생물은 같은 fold에 두고, held-out source view 파생물만 OOF probe에 사용
- `multi_object_scenes`는 모델·epoch·threshold 선택에 사용하지 않고 마지막 진단에만 사용
- 20개 class가 각각 하나의 물리 상품 lineage만 가지므로 source-view OOF와 다중 장면 모두 독립
  상품·매장 일반화 평가가 아님

원본 manifest SHA-256은
`e25b4f5ac44de98e60f93b172e639d28bf7ed0443d4d8978926405dea8c45c8f`이다.

## 후보 선택 결과

| 후보 | epoch | OOF 192 Top-1 | OOF 192 Top-3 누락 | OOF 224 Top-1 | OOF 224 Top-3 누락 |
|---|---:|---:|---:|---:|---:|
| random FT, lr 1e-5 | 1 | 89.17% | 15 | 90.67% | 17 |
| prototype LP-FT, lr 1e-6 | 1 | 86.33% | 24 | 86.83% | 29 |
| prototype LP-FT, lr 1e-5 | 1 | 91.00% | 11 | 92.17% | 11 |
| prototype LP-FT, lr 3e-6 | 3 | 92.50% | 12 | 93.00% | 10 |
| **prototype LP-FT, lr 1e-5** | **3** | **93.00%** | **8** | **94.17%** | **9** |
| prototype LP-FT, lr 1e-5 | 5 | 93.17% | 10 | 93.50% | 10 |

선택 우선순위는 224 Top-3 누락, 224 Top-1 오류, 후보 선언 순서이며 다중 장면 결과를 보지
않았다. 5 epoch는 3 epoch보다 나빠져 선택되지 않았다.

선택 후보의 전체 selective OOF 600건은 올바른 `APPROVED` 489건(81.50%), 잘못된
`APPROVED` 1건, `UNKNOWN` 110건, Top-1 94.17%, Top-3 누락 9건이다. frozen ViT160 자체 OOF는
Top-1 77.50%, Top-3 88.83%여서 독립 분류기로 전수 사용하지 않고 ambiguity 후보에만 사용했다.

## 다중 객체 장면 진단

`multi_object_scenes` 300장, GT 객체 1,410개와 기존 고정 detector trace를 사용했다. detector를
다시 실행하지 않았고, classifier는 PyTorch로 실행했으므로 ONNX Runtime parity 결과가 아니다.

| 평가 범위 | 올바른 APPROVED | 잘못된 APPROVED | UNKNOWN | 최종 Top-1 | Top-3 누락 |
|---|---:|---:|---:|---:|---:|
| GT box, runtime preprocess | 1,354 / 1,410 (96.03%) | 0 | 56 | 99.22% | 1 |
| detector box, runtime preprocess | **1,350 / 1,410 (95.74%)** | **0** | **60** | **98.94%** | **1** |

detector box 기준 난이도별 올바른 승인율은 easy 99.51%, medium 97.40%, hard 91.00%다. 60개
`UNKNOWN`은 모두 224 detail 경로를 실행했고, 그중 45개는 최종 Top-1이 맞지만 안전 margin 또는
합의를 통과하지 못했다. verifier까지 실행된 `UNKNOWN`은 9개이므로 대다수 비승인은 verifier
자체가 아니라 primary/detail 저신뢰에서 발생했다.

가장 큰 class 병목은 `bread_02 croffle`로 `UNKNOWN` 18건 중 7건이 `bread_03 waffle`을
Top-1으로 선택했다. 유일한 Top-3 누락은 `hard_067`의 `bread_01 walnut_donut`이다. 이 ROI는
다른 빵에 크게 가려지고 detector/GT box 안에 전경의 긴 pastry가 함께 들어가므로 단일 객체 원본의
appearance 증강만으로 복원하기 어려운 가시성·ROI 구성 문제다.

## 목표 판정

| 목표 | 이번 진단 | 판정 |
|---|---:|---|
| 객체 검출 Recall 100% | 고정 trace 1,410 / 1,410, FP/FN 0 | 해당 trace에서만 달성 |
| 정상 ROI APPROVED 99% 이상 | detector ROI 전체 95.74%; easy 99.51% | 전체 미달 |
| 최종 FP/FN 0건 | detector FP/FN 0, classifier Top-1 오류 15건 | 미달 |
| 잘못된 APPROVED 0건 | 다중 장면 0건, source-view OOF 1건 | 일반화 기준 미달 |
| 정답 Top-3 누락 0건 | 다중 장면 1건, source-view OOF 9건 | 미달 |

현재 detector ROI의 최종 Top-1 정답은 1,395건이다. 99% 올바른 승인은 최소 1,396건이므로
threshold 완화만으로는 목표 달성이 불가능하고, 현재 `UNKNOWN`의 오분류 15건 중 적어도 1건을
먼저 교정해야 한다. 또한 threshold를 다중 장면에 맞추면 test leakage가 되므로 금지한다.

## 다음 데이터 수집 우선순위

1. 물리 상품을 새로 준비한 독립 session과 다른 조명·카메라·매장에서 class당 최소 30개 실제
   ROI를 수집한다.
2. `bread_02/bread_03`, `bread_04/bread_17`, `bread_19/bread_20` hard-negative pair를 실제
   인접 배치와 가림 단계별로 집중 수집한다.
3. 객체 가시율을 annotation에 추가하고, 심한 가림은 분류 증강이 아니라 detector quality 또는
   `SEGMENT_RECAPTURE` 정책의 별도 validation 대상으로 둔다.
4. 새 validation으로 모델과 전역 margin을 선택한 뒤, 손대지 않은 매장·session test에서 목표를
   다시 판정한다.

## 산출물

- 학습 보고서: `artifacts/experiments/single2-classifier-cascade-0.1.13/training-report.json`
- 실험 패키지: `artifacts/experiments/single2-classifier-cascade-0.1.13/classifier-package`
- 다중 장면 보고서: `artifacts/experiments/single2-classifier-multi-object-e3-0.1.13/report.json`
- 객체별 예측: `artifacts/experiments/single2-classifier-multi-object-e3-0.1.13/predictions.jsonl`
- 전체 비승인 오버레이: `artifacts/experiments/single2-classifier-multi-object-e3-0.1.13/non-approved-overlay-all.jpg`
- Top-3 누락 오버레이: `artifacts/experiments/single2-classifier-multi-object-e3-0.1.13/non-approved-overlay.jpg`
