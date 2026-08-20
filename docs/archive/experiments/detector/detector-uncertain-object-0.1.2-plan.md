# 단품 `DETECTOR_UNCERTAIN_OBJECT` 개선 및 DETR 0.1.2 승격 계획

## 문서 상태

- 상태: `PENDING_USER_APPROVAL`
- 운영 모델: package `0.1.1` 유지
- 후보 모델: package `0.1.2` 승격 거절
- 학습 승인: 미승인

이 문서는 BIXOLON Scanner 운영 로그와 `bread_project_2` 진단 결과를 바탕으로 단품 촬영의 `DETECTOR_UNCERTAIN_OBJECT` 오탐을 줄이면서 기존 미검출 차단 성능을 유지하기 위한 실행 계약이다. 사용자가 bbox/class overlay 검수표를 명시적으로 승인하기 전에는 학습, fine-tuning 또는 임계값 탐색을 실행하지 않는다.

## 확인된 원인

운영 detector는 score `0.56` 이상 후보를 정상 객체로 채택한다. score `0.20` 이상 `0.56` 미만 후보 중 면적 비율이 `0.039` 이상이고, 채택된 모든 bbox와의 IoU가 `0.5` 미만이면 독립 uncertainty 후보로 계산한다. 후보가 하나라도 있으면 classifier를 호출하기 전에 `DETECTOR_UNCERTAIN_OBJECT`로 `RECAPTURE`한다.

운영 `RECAPTURE` 78건 중 production draft에서 정상 bbox가 정확히 하나인 사례는 30건이며, 그중 29건에는 별도의 낮은 점수 후보가 있었다. overlay상 이 후보는 주로 받침, 종이 경계, 테이블 또는 반사 영역이었다. 따라서 단품 문제는 count mismatch가 아니라 배경 query가 독립 객체처럼 남는 문제다.

다음 우회는 허용하지 않는다.

- detector score 또는 uncertainty 임계값 하향
- 단품이라는 이유로 uncertainty hard gate 무시
- classifier 결과를 사용한 detector hard gate 사후 취소
- 받침 위치를 하드코딩한 bbox 제거
- test 결과에 맞춘 threshold 조정

## 로그 분리와 overlay 승인

검수표는 다음 세 묶음을 독립적으로 관리한다.

1. 정상 로그: `APPROVED`, `UNKNOWN`, legacy no-status
2. 운영 `RECAPTURE` 로그
3. 사람이 신고한 박스 미검출

각 행에는 이미지 식별자, 요청 상태, reason code, GT bbox/class, accepted bbox, uncertainty bbox, score, 촬영 세션, 물리 대상, 중복 군집과 split 적합성을 기록한다. overlay 색상은 GT 초록, accepted 파랑, uncertainty 주황, 실제 미검출 빨강으로 고정한다.

검수 판정은 다음 값만 사용한다.

- `TRUE_OBJECT`
- `BACKGROUND_FP`
- `MISSED_BOX`
- `CLASS_ERROR`
- `QUALITY_RECAPTURE`
- `EXCLUDE`

사용자가 bbox/class와 배경 오탐 판정을 승인하면 승인본으로 manifest를 생성한다. 수정 요청이 있으면 overlay와 검수표를 먼저 고친 뒤 다시 승인받는다.

## 중복 및 split 누수 방지

SHA-256, perceptual hash, 촬영 세션과 물리 대상 키를 함께 사용한다. 같은 군집의 원본, 파생본과 재촬영본은 하나의 split에만 들어간다. validation으로 checkpoint와 정책을 선택하고 test는 최종 평가에 한 번만 사용한다.

`bread_project_2` 300장 중 299장은 기존 policy-fit 데이터와 byte-identical하며 perceptual hash로 기존 `bread-v1` detector 이미지 전체와 대응한다. 이들은 기존 development/test 소속을 상속하고 test 대응 이미지는 학습에 사용하지 않는다. 남은 생성 이미지 `M/M_100.jpg`는 promotion 평가에서 제외한다.

현재 운영 로그는 물리 대상과 세션 다양성이 부족하므로 진단 및 development hard-example 후보로만 사용할 수 있다. 고정 validation 또는 promotion test의 독립 근거로 사용하지 않는다.

## 필요한 독립 데이터

여러 매장과 배경에서 동작하는 일반 모델을 목표로 최소 3개 독립 설치 환경에서 다음 데이터를 확보한다.

- 정상 100장 이상: 단품 50장 이상, 나머지는 2개 및 3개 이상 구성
- 재촬영 100장 이상: 무검출·밀집, 가림, 흐림, border/crop, 노출 문제 각각 20장 이상
- 각 조건별 3개 이상의 독립 촬영 세션
- 정답 class가 있는 `UNKNOWN` item 20개 이상
- 기존 로그, 촬영 세션 및 물리 대상과 겹치지 않는 고정 test

수량 또는 독립성이 충족되지 않으면 모델 성능을 보고하거나 승격하지 않고 부족한 환경, 조건과 수량만 보고한다.

## 학습 후보

### 1차: RT-DETRv2-R18 data-first 개선

현재 RT-DETRv2-R18 구조, 300 query와 ONNX 입출력 계약을 유지한다. 승인된 development 데이터에서 현 모델을 실행해 배경 uncertainty가 발생한 전체 프레임을 hard-negative 또는 hard-positive로 채굴한다. crop이나 배경 지우기 합성본은 사용하지 않는다.

- batch 8: positive 6장, 실제 배경 negative 2장
- 배경 uncertainty가 승인된 positive 프레임: sampler weight 2
- positive sampling의 절반 이상: 단품 프레임
- epoch 1~70: bbox-safe flip, 소규모 affine, brightness/contrast
- epoch 71~100: 기하 증강 해제
- blur·과노출을 정상 positive로 만드는 증강, mosaic, mixup: 비활성화

기존 Variable Focal Loss가 높은 점수의 unmatched query에 이미 더 큰 negative penalty를 부여하므로, 우선 custom loss보다 실제 hard-negative 노출과 sampling을 개선한다.

checkpoint는 validation loss만으로 고르지 않는다. 고정 임계값을 사용하는 운영 evaluator에서 아래 순서로 선택한다.

1. 미검출 및 재촬영 대상 recall
2. 단품 정상 이미지의 `DETECTOR_UNCERTAIN_OBJECT` 오탐
3. 전체 정상 이미지의 false `RECAPTURE`
4. `APPROVED` precision
5. validation loss

### 2차: DEIM challenger

1차 후보가 안전 KPI를 만족하지만 단품 오탐 개선 기준을 충족하지 못할 때만 DEIM의 Dense O2O matching과 Matchability-Aware Loss를 challenger로 평가한다. 공개 API, 전처리 크기, 고정 임계값과 ONNX 출력 의미는 변경하지 않는다. PyTorch/ONNX 및 CPU/CUDA parity와 p95를 별도로 통과해야 한다.

고정 배경 embedding memory 방식은 여러 매장·받침 환경의 일반화, 상태 의존성, p95와 provider parity 위험 때문에 0.1.2 범위에서 제외한다.

## 승격 게이트

모든 평가는 기존 운영 임계값으로 수행한다.

- 독립 단품 정상 test의 `DETECTOR_UNCERTAIN_OBJECT` false `RECAPTURE`: 0.1.1 대비 50% 이상 감소, 관측 비율 5% 이하
- 각 매장 하위 집합: 0.1.1보다 악화되지 않음
- 사람이 신고한 고정 박스 미검출 사례: 전부 차단
- 전체 재촬영 대상 recall: 99% 이상
- `APPROVED` precision: 99.5% 이상
- 정답 class가 있는 `UNKNOWN` Top-3 accuracy: 95% 이상
- PyTorch/ONNX: bbox·logit 허용오차 및 최종 status/reason/class 순위 parity
- ONNX CPU/CUDA: 최종 판정 parity
- RTX 5080, concurrency 1, warm-up 완료 full path p95: 100ms 이하
- API schema, detector 조기 종료, classifier 미호출, 오류 매핑과 model version null 규칙: 회귀 없음

고정 test가 실패하면 해당 결과에 맞춰 재학습하거나 임계값을 변경하지 않는다. 후보를 거절하고 새 development 데이터를 수집한다.

## 현재 0.1.2 후보 판정

현재 후보는 다음 게이트를 통과하지 못해 production 승격 대상이 아니다.

- 운영 정상 continue: 20/34
- 운영 재촬영 recall: 7/8
- 기존 미검출 차단: 2/4
- `UNKNOWN` Top-3: 17/20
- RTX 5080 full-path p95: 100.46089ms

따라서 앱과 package `0.1.1`을 유지한다. 새 후보가 모든 게이트를 통과할 때만 package `0.1.2`를 production 앱에 통합한다.

## API와 저장소 계약

- `/v1/scan`, status enum, reason code, bbox 좌표, classifier 조기 종료 순서와 임계값을 유지한다.
- 원본·증강 이미지, checkpoint와 ONNX를 Git에 추가하지 않는다.
- Git에는 manifest, 설정, 코드, 테스트와 개인정보 및 로컬 경로를 제거한 요약 보고서만 기록한다.
- 학습·평가 산출물은 ignored `artifacts/`, `checkpoints/`, `models/`에 둔다.

## 연구 근거

- RT-DETRv2, dynamic data augmentation과 scale-adaptive hyperparameter: <https://arxiv.org/abs/2407.17140>
- RT-DETRv3, training-only hierarchical dense positive supervision: <https://arxiv.org/abs/2409.08475>
- DEIM, Dense O2O matching과 Matchability-Aware Loss: <https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.pdf>
- DEIM 공식 PyTorch/ONNX 구현: <https://github.com/Intellindust-AI-Lab/DEIM>
- D-FINE, fine-grained box distribution refinement: <https://arxiv.org/abs/2410.13842>
- Focal Loss, foreground/background imbalance의 hard-example weighting: <https://arxiv.org/abs/1708.02002>
