# BIX Bakery 원본 기반 재학습 실험 종료 보고서

## 종료 판정

이 실험은 **운영 미반영 상태로 종료**한다.

- 제품·추론 파이프라인 `0.1.14`는 변경하지 않았다.
- detector는 고정 300장 진단에서 FP 0, FN 0을 달성했지만, D-FINE 4회전 추론에 의존하고
  운영 지연시간을 측정하지 않아 배포 후보로 채택하지 않는다.
- classifier는 전체 정답 객체 1,410개 기준 정답 `APPROVED` 1,396개로 99%를 넘었지만
  오인 승인 12개가 남아 목표의 `오인 승인 0`을 충족하지 못했다.
- 고정 300장은 반복 개발 회귀에 사용했으므로 결과를 독립 일반화 성능, 인증 또는 SLA로
  해석하지 않는다.
- 실험 결과물은 `artifacts/retraining/fresh`에 보존하되 활성 Runtime, Catalog 또는 버전 번들에
  연결하지 않는다.

## 범위와 데이터 계약

빵 학습 이미지 원천은 `datasets/bix_bakery_dataset`만 사용했다. 과거 빵 detector/classifier
checkpoint, ONNX, Catalog support·prototype·adapter와 과거 threshold는 초기값으로 사용하지
않았다. 다만 일반 사물로 사전학습된 foundation/model weight는 아래처럼 별도 provenance로
취급했다.

- DINOv3 foundation backbone
- 공식 D-FINE HGNetV2-S 구성과 backbone
- SSDLite320 MobileNetV3-Large COCO weight 중 shape-compatible parameter

고정 회귀 `datasets/bread_dataset/multi_object_scenes`의 이미지나 label은 학습 manifest에 넣지
않았다. 이 데이터는 반복 진단에만 사용했다.

| 데이터 | 이미지 | 객체 | 역할 |
|---|---:|---:|---|
| BIX single bread | 200 | 200 | 실제 box·mask·SKU support |
| BIX empty/background | 10 | 0 | detector hard negative |
| BIX source multi | 20 | 81 | source-domain validation과 multi-object 보강 |
| dense 합성 | 12,000 | 71,854 | detector 밀집·가림·중복·경계 학습 |
| 고정 회귀 | 300 | 1,410 | 반복 진단 전용 |

원본 이미지 집합 SHA-256은
`9149ff156aa58b03a7d2d28c9059124f62633b03577d9c69673b572cffecba2f`이다.

## annotation과 mask QA

20개 class QA sheet와 200개 bbox/mask overlay를 전수 육안 확인했다. 확인 기준은 상품 전체
silhouette 포함, tray·table·가구·분리된 그림자 제외, tight bbox이다.

- 자동 paper-edge 오류 16건은 원본 SHA에 연결된 point/box override로 다시 분할했다.
- 수동 보정 mask 16개는 경계 불확실성을 피하려고 copy-paste 합성에서 제외했다.
- 프레임 경계에 닿은 cutout 56개도 합성에서 제외했다.
- 합성에 사용한 자동 interior cutout은 128개다.
- 6번 크루아상은 완전한 source view 6개만 합성에 사용했고, 경계에 걸린 source view 4개는
  제외했다.

근거는 `artifacts/retraining/fresh/source/annotation-review.json`과
`artifacts/retraining/fresh/synthetic-dense-v2/metadata.json`이다.

## Detector 실험

### 구성

Primary는 1-class D-FINE HGNetV2-S 640이고, 0°·90°·180°·270° 네 view에서 검출한 뒤 최소
3개 view가 지지하는 box를 mean fusion했다. Backup은 1-class SSDLite320
MobileNetV3-Large이며 primary 결과가 구조적으로 부족할 때만 후보를 추가하는 selective recovery로
평가했다. Detector class는 SKU 판정에 사용하지 않았다.

D-FINE 학습 입력은 실제 210장과 dense 합성 12,000장, 총 annotation 72,054개다. validation은
BIX source multi 20장/81개였으며 고정 300장 이미지는 사용하지 않았다.

SSDLite는 `pretrained_detector_transfer=true`로 학습했다. 이전 생성 report 안의
`pretrained_weights_used=false`라는 legacy 설명은 실제 setting 및 구현과 모순되는 오래된 문자열로,
이 종료 보고서에서는 source of truth로 사용하지 않는다.

### 결과

| 구성·평가 | 정답 객체 | matched | FP | FN | exact image |
|---|---:|---:|---:|---:|---:|
| D-FINE rot4, BIX source multi | 81 | 81 | 0 | 0 | 20/20 |
| D-FINE rot4, 고정 300장 | 1,410 | 1,409 | 0 | 1 | 299/300 |
| D-FINE rot4 + selective SSDLite, 고정 300장 | 1,410 | 1,410 | 0 | 0 | 300/300 |

SSDLite가 실제로 보완한 것은 고정 회귀 이미지 263의 ROI 1개다. 이 수치는 정확도 상한을 확인한
진단 결과다. D-FINE을 이미지당 네 번 실행하고 새 전체 경로의 p50/p95/p99를 측정하지 않았으므로
운영 detector로 승인하지 않는다. 또한 별도 합성 stress validation에서 D-FINE과 SSDLite 단독
recall이 각각 약 82.9%, 89.1%여서 미관측 난이도에 대한 안정성도 입증되지 않았다.

핵심 근거:

- `artifacts/retraining/fresh/evaluations/dfine-v1-epoch3-rot4-source-multi-v1.json`
- `artifacts/retraining/fresh/evaluations/dfine-v1-epoch3-rot4-support3-mean-fixed300.json`
- `artifacts/retraining/fresh/evaluations/dfine-v1-ssdlite-v7-standard-overlap-target-diagnostic-v6.json`
- `artifacts/retraining/fresh/evaluations/dfine-v1-epoch3-rot4-stress-validation-v1.json`
- `artifacts/retraining/fresh/evaluations/ssdlite-v7-stress-validation-v1.json`

## Classifier 실험

ConvNeXt-Tiny DINOv3 192 primary, 224 detail 및 Frozen DINOv3 ViT-B/16 verifier 조합을 비교했다.
고정 회귀에서 공통 비교가 가능한 ROI는 1,408개였고 최종 진단 기준은 다음과 같다.

| 항목 | 결과 |
|---|---:|
| 전체 정답 객체 기준 정답 `APPROVED` | 1,396 / 1,410 = 99.007% |
| 공통 비교 ROI 기준 정답 | 1,396 / 1,408 = 99.148% |
| 오인 승인 | 12 |
| quality rejector가 잡은 오분류 | 0 / 12 |
| 정상 객체 오재촬영 | 0 |

잔여 오분류는 다음과 같다.

| 이미지/annotation | 정답 → 예측 |
|---|---|
| 123/500 | 18 → 20 |
| 149/626 | 18 → 20 |
| 226/1022 | 2 → 17 |
| 232/1044 | 6 → 4 |
| 242/1086 | 19 → 20 |
| 245/1100 | 8 → 20 |
| 248/1116 | 2 → 17 |
| 260/1184 | 12 → 19 |
| 267/1232 | 1 → 4 |
| 267/1233 | 2 → 17 |
| 283/1299 | 6 → 4 |
| 288/1329 | 18 → 20 |

오분류는 confidence나 frozen-feature completeness risk만으로 안전하게 분리되지 않았다. 이 12개에
맞춰 threshold나 SKU별 예외를 추가하면 반복 회귀에 대한 과적합이 되므로 적용하지 않았다.
오분류 montage는
`artifacts/retraining/fresh/diagnostics/global-classifier-ensemble-current-12-errors.jpg`, quality
rejector 근거는 `artifacts/retraining/fresh/evaluations/feature-quality-rejector-v1.json`이다.

## 시도했지만 채택하지 않은 개선

- detector: raw D-FINE 단일 view, 여러 epoch, rot4 support 조건, D-FINE union, SSDLite 단독 및
  selective recovery
- classifier: 192/224 fine-tune, prototype·kNN·Ridge·diagonal LDA·trained head 조합,
  회전·margin·neighbor mask·부분 가림 합성
- verifier/rejector: ViT-B/16 회전 합의, completeness, class-conditional quality, frozen-feature
  nonlinear rejector, shape·scene geometry·source risk

일부 후보는 고정 300장의 label을 사용해 head나 threshold를 비교했으므로 진단 외 용도로 사용할 수
없다. source-only로 선택한 rejector는 12개 오분류를 하나도 잡지 못했다. 따라서 `RECAPTURE`로
오인만 숨기는 안전한 전역 규칙도 이번 데이터만으로는 만들지 못했다.

## 연구 근거

- [DINOv3](https://arxiv.org/abs/2508.10104): frozen dense feature와 소량 라벨 adapter
- [Synthetic Object Compositions (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_Synthetic_Object_Compositions_for_Scalable_and_Accurate_Learning_in_Detection_CVPR_2026_paper.html): 검수 object segment 기반 multi-object 합성
- [Prototype-based Contrastive Learning with Stage-wise Progressive Augmentation (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_Prototype-based_Contrastive_Learning_with_Stage-wise_Progressive_Augmentation_for_Self-Supervised_Fine-Grained_ICCV_2025_paper.html): fine-grained prototype와 단계적 증강
- [Selective Classification Under Distribution Shifts](https://arxiv.org/abs/2405.05160): 분포 이동에서 reject option
- [torchvision SSDLite 구현](https://pytorch.org/blog/torchvision-ssdlite-implementation/): SSDLite 기준 구조

## 보존과 비반영 범위

재현에 필요한 source annotation, 합성 recipe/config, 평가 script, report 및 QA overlay는 보존한다.
원본 이미지, 합성 이미지, checkpoint와 ONNX는 Git에 추가하지 않는다. 이번 실험용 Python 모듈은
`training`, `evaluation`, `experiments` 범위에만 두며 Worker/Runtime import 방향을 바꾸지 않는다.

다음 항목은 하지 않는다.

- `configs/versions/0.1.14.json`의 Runtime/Catalog/source candidate 변경
- `0.1.14` Runtime 또는 Catalog metadata 변경
- 새 EXE 또는 버전 번들 생성
- 고정 300장에 맞춘 threshold, SKU별 routing 예외 또는 hard-coded 승인 규칙 추가

## 종료 검증

- 이번 실험 관련 Python 테스트: 27개 통과
- 전체 `ruff check`: 통과
- 전체 `ruff format --check`: 통과
- `flutter analyze`: 통과
- Flutter 전체 테스트: 188개 통과
- `git diff --check`: 통과
- 전체 Python 테스트: 수집된 1,012개 중 1건 실패

전체 Python 테스트의 단일 실패는 이번 실험 코드가 아니라
`datasets/bread_dataset/single_objects_4/bread_19_pastry_bread`에 JPEG 10장과 ZIP 1개가 함께 있어
기존 데이터 감사 계약의 파일 수 10개를 위반한 것이다. 사용자 원본 데이터인 ZIP을 임의 삭제하지
않고 그대로 보존했다. `configs/versions/0.1.14.json`에는 diff가 없으며 확인 시점 SHA-256은
`97160fa78246ac49d0d51a821de67250bd5aaa3a063b37eee0da56966861a325`이다.

## 최종 결론

이번 데이터만으로 detector의 고정 회귀 FP/FN 0 가능성은 확인했지만 운영 가능한 단일-pass 성능은
확인하지 못했다. classifier의 99% 정답 승인율은 넘었으나 오인 승인 12건을 source-only 전역
정책으로 안전하게 `UNKNOWN`/`SEGMENT_RECAPTURE`로 분리하지 못했다. 따라서 목표는 완전 충족되지
않았고 결과를 `0.1.14`에 반영하지 않은 채 실험을 종료한다.
