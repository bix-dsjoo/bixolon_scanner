# BIX `single_objects` classifier cascade 실험

## 결론

`datasets/bix_bakery_dataset/single_objects`의 이미지 200장만 사용해 다음 classifier 구조를 실제로
학습하고 source-view grouped OOF로 평가했다.

`ConvNeXt-Tiny 192 전체 ROI → 같은 ConvNeXt-Tiny 224 선택 상세 → Frozen ViT-B/16 160 선택 검증`

ConvNeXt 마지막 stage와 하나의 cosine head를 192/224 paired loss로 1 epoch fine-tune했다. 안전
우선 nested calibration 결과는 오승인 0건과 Top-3 누락 0건을 기록했지만 정답 승인율은 91.17%로
99% 목표에 미달했다. 따라서 실험 checkpoint와 head는 생성했지만 0.1.13 Runtime에는 활성화하지
않는다.

## 입력과 누수 방지

- 원본: 20 class × 10 view = 200장
- 손상 이미지: 0장
- 원본 내부 SHA-256 중복: 0건
- 외부 이미지 접근: 없음
- 원본에서 생성한 모든 파생 이미지는 원본 fold를 그대로 상속
- 서로 다른 fold에 나타난 원본 SHA-256: 0건
- 파생 이미지 SHA-256: 6,200/6,200 unique

파일명에서 구분 가능한 normal/flipped, vertical, 30도와 60도 시점을 4/3/3 source-view fold로
분리했다. 물리 개체 ID와 capture session ID가 없으므로 미관측 물리 상품이나 매장 일반화는 평가할
수 없다.

## 데이터 구성

| 역할 | 수량 | 사용처 |
|---|---:|---|
| clean 원본 | 200 | ConvNeXt 학습, ViT160 support |
| appearance | 1,600 | ConvNeXt 학습 |
| geometry | 1,600 | ConvNeXt 학습 |
| context | 1,600 | ConvNeXt 학습, neighbor ownership mask 적용 |
| 공통 probe | 600 | 학습 금지, OOF 성능 평가 |
| synthetic reject | 800 | SKU positive 학습 금지, 품질 진단만 수행 |

저장된 positive ROI는 5,000장이다. 각 ROI를 192와 224로 모두 학습하므로 전체 데이터를 사용하는
최종 1 epoch의 입력 presentation은 10,000개다.

## 모델 학습

- ConvNeXt: DINOv3 ConvNeXt-Tiny, 마지막 stage 1개와 cosine head만 학습
- 입력: 같은 ROI의 192/224 paired view
- loss: 두 해상도의 classification loss 평균 + `0.2 × resolution consistency loss`
- 최종 전체 학습 정확도: 192 90.58%, 224 90.80%
- ViT: DINOv3 ViT-B/16 backbone은 frozen, clean 원본 support로 별도 linear/prototype head 구성
- routing threshold: outer test fold와 분리된 inner train/calibration fold에서만 선택

## OOF 평가 결과

600개 probe는 각 held-out 원본에서 appearance, geometry, context를 하나씩 생성했다.

| 경로 | Top-1 | Top-3 | 비고 |
|---|---:|---:|---|
| ConvNeXt 192 | 98.17% | 99.83% | 전체 ROI 경로 |
| ConvNeXt 224 | 98.50% | 100.00% | 상세 경로를 전수 계산한 진단 |
| ConvNeXt 192/224 equal fusion | 98.50% | 99.83% | 전수 계산 진단 |
| Frozen ViT160 | 92.83% | 98.33% | 독립 architecture 진단 |
| **선택 cascade** | **98.50%** | **100.00%** | 전역 nested threshold |

선택 cascade 상태 결과:

- 정답 `APPROVED`: 547/600, 91.17%
- 잘못된 `APPROVED`: 0/600
- `UNKNOWN`: 53/600
- 224 실행: 87/600, 14.50%
- ViT160 실행: 55/600, 9.17%
- Top-3 누락: 0/600

Probe 종류별 정답 승인율은 appearance 91.0%, geometry 91.5%, context 91.0%였다. 특정 변형군만
무너진 것이 아니라, 안전 threshold가 세 분포를 비슷하게 보수적으로 처리했다.

## 목표 판정

| 목표 | 결과 | 판정 |
|---|---:|---|
| 객체 검출 Recall 100% | 측정 불가 | 단일 ROI만 있어 detector 미평가 |
| 정상 ROI `APPROVED` ≥99% | 91.17% | 미달 |
| 최종 FP/FN 0건 | 측정 불가 | full scene과 detector annotation 없음 |
| 잘못된 `APPROVED` 0건 | 0/600 | source-derived OOF에서 충족 |
| 정답 Top-3 누락 0건 | 0/600 | source-derived OOF에서 충족 |

느슨한 threshold는 정답 승인 586/600, 97.67%까지 올라갔지만 잘못된 `APPROVED` 5건이 발생했다.
따라서 현재 데이터에서 99% 승인과 오승인 0건을 동시에 만족하는 근거는 없다.

## `RECAPTURE` 한계

synthetic blur, 과다/과소 노출, 큰 가림과 edge content loss 800개 중 안전 cascade가 승인하지 않은
표본은 159개, 19.88%뿐이었다. 이 변형들은 실제 품질 불량 ground truth가 아니며 SKU classifier의
confidence만으로 품질을 안정적으로 판별할 수 없음을 보여준다.

따라서 이 결과로 `SEGMENT_RECAPTURE` 정책을 활성화하지 않는다. 실제 초점 실패, 잘림, 반사,
역광, 손/포장 가림을 별도 capture session에서 수집해 detector quality 또는 별도 quality head를
평가해야 한다.

## 생성된 실험 패키지

- shared ConvNeXt 최종 checkpoint SHA-256:
  `b23e7b1099a2a0096b2b7c60191b247701f7d3b021b7b2766febb14b02a46535`
- ViT160 head SHA-256:
  `f50b2bcfb165b6cf5abbaf94256bcb73eb367c21f82718ce367929d135386c0b`
- calibrated 전역 margin: primary `0.32`, fallback `0.16`, verifier `0.04`
- 상태: `EXPERIMENTAL_NOT_RUNTIME_ACTIVATED`

ONNX export, Runtime/Catalog 조립과 PyTorch/ONNX parity는 실행하지 않았다. 승인율 99% 미달과 독립
데이터 부재가 해소되기 전에 배포 번들로 전환하지 않는다.

## `multi_object_scenes` 추가 진단

학습과 threshold 선택이 끝난 뒤 `datasets/bread_dataset/multi_object_scenes` 300장, 1,410개 객체에
고정 checkpoint와 고정 margin을 적용했다. 기존 0.1.13 detector trace는 이 범위에서 1,410개를 모두
검출했고 FP/FN과 `IMAGE_RECAPTURE`가 각각 0건이었다. 따라서 아래 실제 detector ROI 결과에는
detector box 위치 오차가 포함되지만 객체 누락 영향은 없다.

| 평가 조건 | 정답 APPROVED | UNKNOWN | 오승인 | Top-3 누락 |
|---|---:|---:|---:|---:|
| GT box, 학습 전처리(여백 5%, bias 0.0) | 1,348/1,410 (95.603%) | 62 | 0 | 6 |
| GT box, 현 Runtime 전처리(여백 0%, bias -0.1) | 1,348/1,410 (95.603%) | 59 | 3 | 7 |
| detector box, 현 Runtime 전처리 | 1,350/1,410 (95.745%) | 58 | 2 | 6 |
| detector box, 학습 전처리 추가 진단 | 1,353/1,410 (95.957%) | 55 | 2 | 8 |

실제 detector ROI의 난이도별 정답 승인율은 easy 99.512%, medium 98.600%, hard 89.800%였다.
`ConvNeXt192`, `ConvNeXt224`, `ViT160`의 전수 Top-1은 각각 97.872%, 97.943%, 97.801%였다.
224 상세 경로는 일부 오류를 수정하지만 hard의 극단적인 측면·얇은 투영·부분 가림을 해결하지
못했다.

현 Runtime 전처리는 GT box에서 학습 전처리 대비 최종 Top-1 18건과 승인 상태 25건을 바꾸고
오승인을 0건에서 3건으로 늘렸다. 그러나 detector box에 학습 전처리를 적용해도 오승인 2건과
Top-3 누락 8건이 남으므로 전처리 불일치가 유일한 원인은 아니다. 두 오승인은 croffle→waffle과
측면 croissant→almond campagne였다. 둘 다 primary 192가 높은 margin으로 바로 승인해 ViT 검증이
호출되지 않았고, 현 Runtime 전처리에서 ViT의 Top-1은 두 건 모두 정답이었다.

이 데이터의 annotation은 물리 대상을 `bread_xx:original_item`으로 기록한다. BIX 200장과 다른
촬영·상품 표현에 대한 유용한 분포 이동 진단이지만 새 매장·새 물리 상품의 독립 test는 아니다.
이 결과를 threshold 재선택에 사용하지 않았으며, 현재 활성 classifier 대비 정답 승인율과 안전성이
낮으므로 새 classifier package를 Runtime에 활성화하지 않는다.

추가 산출물:

- 보고서: `artifacts/experiments/bix-single-objects-classifier-multi-object-0.1.13/report.json`
- 객체별 판정: `artifacts/experiments/bix-single-objects-classifier-multi-object-0.1.13/predictions.jsonl`
- 오승인·Top-3 누락 오버레이:
  `artifacts/experiments/bix-single-objects-classifier-multi-object-0.1.13/classifier-issues-contact-sheet.jpg`

## 재현 경로

- 설정: `configs/experiments/bread/bix_single_objects_classifier_cascade_0.1.13.json`
- 데이터 생성·frozen 절제: `src/bixolon_scanner/experiments/bread/bix_classifier_cascade.py`
- paired fine-tune: `src/bixolon_scanner/experiments/bread/bix_classifier_cascade_finetune.py`
- nested calibration: `src/bixolon_scanner/experiments/bread/bix_classifier_cascade_calibration.py`
- 생성 데이터와 feature: `artifacts/experiments/bix-single-objects-classifier-cascade-0.1.13`
- fine-tuned checkpoint: `artifacts/experiments/bix-single-objects-classifier-cascade-finetune-0.1.13`
- 최종 평가와 안전 policy:
  `artifacts/experiments/bix-single-objects-classifier-calibrated-0.1.13`
