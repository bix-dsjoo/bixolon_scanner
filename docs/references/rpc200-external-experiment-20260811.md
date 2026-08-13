# RPC200 외부 실험 이미지 분석

## 문서 성격

이 문서는 제공된 단일 이미지의 내용을 현재 저장소에서 참고할 수 있도록 전사하고 해석한 기록이다. 원본 데이터, manifest, 설정, checkpoint, 로그와 실행 보고서는 제공되지 않았으므로 재현 결과나 모델 승격 증거로 사용하지 않는다. 이미지 자체도 원본 학습 이미지나 대형 바이너리를 Git에 넣지 않는 저장소 원칙에 따라 커밋하지 않는다.

- 자료 제목: `RPC200 학습·인식 알고리즘 및 인식률 결과`
- 평가일: `2026-08-11`
- 데이터 범위: `RPC Subset2`, 200개 제품
- 표기 환경: RTX 5080, CUDA 13.0, TensorRT 11.2.1.2 FP16

## 학습 구성 전사

| 구분 | 이미지에 기록된 내용 |
|---|---|
| 원본 학습 데이터 | 200종 제품 crop 53,799장, class-balanced, 온라인 증강 |
| 합성 데이터 | 원본 cutout을 tray/box/crop에 합성하고 회전·가림·조명·블러 적용 |
| Validation 2차 학습 | easy+medium+hard 6,000장, GT 73,602개, 역전파/튜닝에 사용 |
| 독립 Test | 24,000장, GT 294,333개, 학습·튜닝에 사용하지 않은 독립 홀드아웃 |
| GPU 최적화 | FP16 AMP, TF32, cuDNN benchmark, pinned memory, AutoBatch |

모델 구성은 다음과 같다.

| 구성요소 | 역할 | 표기된 실행 방식 |
|---|---|---|
| YOLO26x-cls | 384px 입력, 200종 분류 | TensorRT FP16 |
| YOLO26n | 960px 입력, 제품 검출 | TensorRT FP16 |
| SigLIP 384 | 학습 crop 53,799개의 특징 DB와 cosine 검색 | 자료에 별도 런타임 표기 없음 |
| Proposal Rejector | 614개 결함 특징을 이용한 거절 | TensorRT FP16 |

## 실시간 추론 흐름 전사

1. 여러 제품이 놓인 tray 이미지를 GUI batch 1로 입력한다.
2. YOLO26n이 `confidence=0.282`, `NMS=0.50`으로 제품을 검출한다.
3. bbox에 7% padding과 letterbox를 적용하여 384px crop을 만들며 최대 20개를 처리한다.
4. 각 crop에 대해 YOLO26x softmax 200종 분류 점수 80%와 SigLIP 특징 DB cosine 유사도 20%를 결합한다.
5. 앙상블 결과와 614개 결함 특징 기반 Proposal Rejector의 클래스 거리로 저신뢰·오검출을 제거한다.
6. 통과 segment는 제품명 초록 bbox, 거절 segment는 `미인식` 빨간 bbox로 표시한다.

## 이미지에 보고된 평가 결과

자료는 class와 IoU 0.50을 함께 만족하는 일대일 매칭을 사용한다. `인식률=TP/전체 GT`, `오인식률=FP/전체 예측`이며, 평균 인식시간은 디코딩을 포함한 실제 GUI batch 1 E2E 시간이다.

| 평가 세트 | 성격 | 이미지 | GT | TP | FP | FN | 인식률 | 오인식률 | 평균 인식시간 | 장바구니 일치 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation easy | 학습·튜닝 재평가 | 2,000 | 14,281 | 14,218 | 0 | 63 | 99.559% | 0.000% | 151.691ms | 96.950% |
| Validation medium | 학습·튜닝 재평가 | 2,000 | 24,732 | 24,653 | 0 | 79 | 99.681% | 0.000% | 225.535ms | 96.400% |
| Validation hard | 학습·튜닝 재평가 | 2,000 | 34,589 | 34,396 | 0 | 193 | 99.442% | 0.000% | 307.174ms | 91.450% |
| Test | 독립 홀드아웃 | 24,000 | 294,333 | 292,066 | 1,140 | 2,267 | 99.230% | 0.389% | 236.336ms | 91.642% |

Test FP 1,140개는 잘못된 클래스 365개, 중복 31개, 배경 744개로 분해되어 있으며 합계가 일치한다. Validation은 split별 fresh 200종 표본, Test는 24,000장 전수로 측정했다고 표기되어 있다.

## 해석 시 주의점

- 독립 성능으로 해석할 수 있는 것은 Test뿐이다. easy/medium/hard Validation은 역전파와 임계값 튜닝에 사용되었으므로 독립 일반화 근거가 아니다.
- Test `인식률 99.230%`는 객체 단위 `TP/GT`이다. 이 저장소의 `APPROVED precision`, `RECAPTURE recall`, `UNKNOWN Top-3 accuracy`와 분모 및 의미가 달라 직접 비교할 수 없다.
- `오인식률 0.389%`는 `FP/전체 예측`이다. `APPROVED` 중 오승인 비율이나 이미지 단위 위험률이 아니다.
- 객체 단위 인식률이 99%대여도 Test 장바구니 완전 일치는 91.642%이다. 다중 객체 시스템에서는 작은 객체 오류가 이미지 전체 실패로 누적되므로 장바구니 일치를 보조 KPI로 볼 가치가 있다.
- 속도는 평균만 제공되고 p50/p95/p99, warm-up 횟수, 표본별 객체 수 분포가 없다. 현재 저장소의 warm-up 완료 CUDA full-path `p95 <= 100ms` 기준과 직접 비교하거나 통과로 판정할 수 없다.
- `인식/미인식` 두 출력은 현재 공개 상태 `APPROVED/UNKNOWN/RECAPTURE/ERROR` 및 item 상태 계약과 다르다. Proposal Rejector의 거절을 현재 상태 하나로 임의 매핑하면 안 된다.
- confidence 0.282, NMS 0.50, crop padding 7%, 80:20 앙상블, 최대 crop 20개는 이 실험의 설정값이다. 현재 코드에 상수로 복사하지 않고, 채택 시 버전 관리 설정과 모델 패키지 메타데이터로 관리해야 한다.
- 자료의 TensorRT FP16 실행 경로를 운영 Worker에 그대로 도입할 수 없다. 현재 Worker는 ONNX Runtime을 사용하며, 새로운 최적화 경로는 기준 ONNX 결과와 상태·클래스 순위 parity 및 KPI를 먼저 입증해야 한다.

## 현재 프로젝트에 참고할 항목

| 외부 실험 아이디어 | 현재 프로젝트에서의 활용 | 선행 조건 |
|---|---|---|
| 장바구니 완전 일치 | 다중 item 최상위 판정의 보조 평가 지표로 추가 | 기존 객체 KPI를 대체하지 않고 난이도·객체 수별로 함께 보고 |
| FP 원인 분해 | detector 오류를 배경·중복·오분류로 나누어 개선 우선순위 결정 | 일대일 matching 규칙과 reason taxonomy 고정 |
| 분류기 80% + 특징 검색 20% | classifier-only `experiment_only` ablation 후보 | development에서 가중치·threshold 잠금 후 독립 test, CPU/CUDA parity 검증 |
| 614개 결함 특징 Rejector | 오검출 또는 저신뢰 후보 분석용 보조 점수 후보 | detector 오류는 classifier 호출 전 gate, 저신뢰 분류는 `UNKNOWN`, 촬영 품질은 `RECAPTURE`로 도메인 분리 |
| cutout 합성 및 class balance | 촬영 구성과 가림 다양성을 늘리는 학습 실험 | 물리 대상·촬영 세션·원본 cutout 계보 기반 group-aware split과 합성 출처 기록 |
| crop batch 최대 20개 | 객체 수 증가에 따른 지연·메모리 회귀 시나리오 | query 포화 계약, 실제 운영 최대 개수, p50/p95/p99를 함께 검증 |

가장 낮은 위험으로 바로 참고할 수 있는 항목은 모델 변경이 없는 평가 확장이다. 독립 test에서 장바구니 완전 일치와 FP 원인 분해를 추가하고, 객체 수별 full-path 지연 백분위를 기록하면 현재 계약을 바꾸지 않고도 이 자료의 장점을 흡수할 수 있다.

SigLIP 검색 앙상블과 Proposal Rejector는 별도 실험으로 격리한다. 후보를 평가할 때는 동일한 잠긴 split에서 기존 단일 classifier와 비교하고, `APPROVED precision >= 99.5%`, 전체 `RECAPTURE recall >= 99%`, 정답 class가 있는 `UNKNOWN Top-3 accuracy >= 95%`, RTX 5080 full-path p95 100ms 이하, PyTorch/CPU ONNX/CUDA ONNX 상태·순위 parity를 모두 다시 검증한다. 이 증거가 없으면 운영 package나 API 판정 계약을 변경하지 않는다.
