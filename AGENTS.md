# AGENTS.md

이 파일은 저장소 전체에 적용됩니다. 이 프로젝트를 수정하는 모든 에이전트와 기여자는 아래 계약을 따라야 합니다.

## 프로젝트 목표

이 저장소는 이미지 판정 시스템의 데이터 계약, PyTorch 학습·평가, ONNX export, 그리고 Windows에서 장기 실행되는 Python 추론 Worker를 관리합니다.

운영 Worker는 단일 JPEG/PNG 이미지에서 여러 객체를 판정하고 다음 최상위 상태 중 정확히 하나를 반환합니다.

- `APPROVED`: 검출된 모든 객체가 승인 임계값을 통과한 결과
- `UNKNOWN`: 하나 이상의 객체가 승인 임계값을 통과하지 못한 결과와 객체별 Top-3 후보
- `RECAPTURE`: detector 또는 classifier가 촬영 부적합을 판정한 결과
- `ERROR`: 입력, 구성, 모델 또는 시스템 오류

`ERROR`를 `RECAPTURE`로 변환하지 마십시오. 모델 판정과 시스템 장애는 서로 다른 도메인입니다.

## 권장 코드 영역

새 코드를 추가할 때는 책임을 다음 영역으로 분리하십시오. 실제 디렉터리 이름이 정해지면 이 목록과 README를 함께 갱신하십시오.

- Worker API: multipart 입력 검증, 응답 직렬화, HTTP 오류 매핑
- Pipeline: detector → 조기 종료 → ROI → classifier → 최종 상태 결정
- Inference adapter: ONNX Runtime session, CPU/CUDA provider 선택, tensor 변환
- Training: detector 및 classifier PyTorch 학습
- Evaluation/benchmark: 정확도, PyTorch/ONNX parity, 지연 및 회귀 측정
- Configuration/model package: 전처리, 라벨, 임계값, 버전, checksum
- Tests: 상태 계약, 오류 처리, provider 호환성, 성능 회귀

API, 파이프라인 정책, 추론 엔진과 학습 코드 사이에 순환 의존성을 만들지 마십시오. 상태 결정은 한 곳에서 수행하고 HTTP 계층이나 모델 어댑터에 중복 구현하지 마십시오.

## 변경할 수 없는 파이프라인 계약

다음 순서와 조기 종료 규칙은 공개 동작입니다.

1. 입력 이미지를 검증하고 디코딩합니다.
2. Detector가 모든 객체 위치 및 프레임 전체 촬영 품질을 판단합니다.
3. Detector의 hard 품질 조건(무검출, query 포화, 최소 크기, 불확실 독립 후보, 활성화된 count/blur/exposure gate)이 재촬영을 요구하면 classifier를 호출하지 않고 `RECAPTURE`를 반환합니다.
4. 정상 ROI와 `classifier_confidence` 정책의 경계 접촉 ROI 전체를 하나의 batch로 classifier에 전달합니다.
5. Classifier가 명시적 품질 클래스를 예측하면 전역 `RECAPTURE`를 반환합니다.
6. `classifier_confidence` 정책에서 경계 접촉 item의 Top-1 신뢰도가 승인 임계값 미만이면 `DETECTOR_BORDER_CLIPPED` 전역 `RECAPTURE`를 반환합니다. 이 경로는 classifier가 실행된 경로입니다.
7. 각 일반 객체가 승인 임계값 이상이면 해당 item을 `APPROVED`로 반환합니다.
8. 임계값 미만 item은 `UNKNOWN`과 점수 내림차순 Top-3를 반환합니다. 모든 item이 승인일 때만 최상위 `APPROVED`, 그 외에는 최상위 `UNKNOWN`입니다.

이 순서, 상태 우선순위 또는 조기 종료 조건을 변경하려면 README의 계약과 관련 테스트를 같은 변경에서 갱신해야 합니다. 조용한 fallback이나 임의의 기본 승인 결과를 추가하지 마십시오.

## API 계약

- 기본 endpoint는 `POST /v1/scan`입니다.
- 요청은 `image` 필드에 JPEG/PNG 한 장을 담은 multipart 형식입니다.
- 최상위 응답 필드는 `request_id`, `status`, `reason_codes`, `items`, `processing_time_ms`, `model_versions`입니다.
- 각 `items[]`는 `item_id`, 원본 픽셀 기준 `bbox`, `status`, `reason_codes`, `prediction`, `top3`, `confidence`를 포함합니다.
- `status`는 `APPROVED`, `UNKNOWN`, `RECAPTURE`, `ERROR` 외 값을 반환할 수 없습니다.
- item `status`는 `APPROVED`, `UNKNOWN` 외 값을 반환할 수 없습니다.
- `UNKNOWN` item의 Top-3는 점수 내림차순이어야 합니다.
- Detector hard gate 조기 종료 시 classifier를 로드했더라도 해당 요청의 `model_versions.classifier`는 실행되지 않았음을 나타내도록 `null`로 반환합니다. `classifier_confidence` 경계 정책에서 분류 후 `DETECTOR_BORDER_CLIPPED`가 반환된 경우에는 실행된 classifier 버전을 반환합니다.
- `RECAPTURE`와 `ERROR`는 `items`를 빈 배열로 반환합니다.
- 입력 오류는 4xx, Worker 또는 모델 장애는 5xx로 매핑하고 가능한 경우 공통 `ERROR` 응답 형식을 유지합니다.
- API에 raw tensor, 전체 logits, 내부 예외 메시지, 스택 트레이스나 로컬 경로를 노출하지 마십시오.

공개 필드, enum, reason code 또는 의미를 변경하면 API 계약 변경으로 취급하십시오. README 예시, 스키마, 소비자 테스트와 버전 정책을 함께 갱신해야 합니다.

## 모델 및 추론 규칙

- 학습에는 PyTorch를 사용합니다.
- 운영 Worker는 ONNX Runtime만 사용합니다. PyTorch를 Worker 런타임 의존성으로 가져오지 마십시오.
- CPU와 CUDA Execution Provider는 같은 ONNX 모델과 같은 전처리·후처리 계약을 사용해야 합니다.
- provider별 출력 차이는 정해진 수치 허용 오차 내에 있어야 하며 최종 상태와 클래스 순위가 달라지지 않는지 테스트하십시오.
- 입력 크기, 색상 순서, 정규화, 라벨, NMS, crop 규칙, 승인 및 재촬영 임계값을 소스 코드에 하드코딩하지 마십시오.
- 위 값은 모델 패키지 메타데이터 또는 버전 관리되는 설정에서 로드하십시오.
- 필수 메타데이터 누락, 지원하지 않는 schema version, checksum 불일치는 시작 오류로 처리하십시오.
- 모델 버전은 semantic version을 사용하고 모든 응답과 평가 보고서에 기록하십시오.

최적화로 인해 판정 결과가 바뀌지 않도록 하십시오. 양자화, graph optimization, TensorRT provider 등 새로운 최적화 경로는 기준 ONNX 결과와 parity 및 KPI를 입증한 뒤 도입할 수 있습니다.

## 학습 및 데이터 규칙

- 원본 이미지, 증강 이미지, checkpoint, ONNX 파일과 기타 대형 바이너리를 Git에 커밋하지 마십시오.
- 데이터는 외부에 보관하고 JSONL 또는 CSV manifest로 참조하십시오.
- Manifest에는 이미지 식별자, 라벨/annotation, split, 대상 및 촬영 세션 메타데이터를 포함하십시오.
- 동일 물리 대상 또는 동일 촬영 세션의 파생 이미지가 `train`, `validation`, `test`에 분산되지 않도록 group-aware split을 적용하십시오.
- Validation split으로 모델과 임계값을 선택하고 test split은 최종 평가에만 사용하십시오.
- 데이터 수정, 라벨 정책 변경, split 재생성은 데이터셋 버전을 올리고 재현 가능한 기록을 남기십시오.
- Detector 재촬영 조건과 classifier의 명시적 품질 클래스는 서로 구분되는 라벨과 reason code를 사용하십시오.

모델 승격 순서는 `PyTorch checkpoint → 고정 평가셋 KPI 검증 → ONNX export → PyTorch/ONNX parity → 패키징 → Worker benchmark`입니다. 중간 검증을 생략한 모델을 운영 패키지로 표시하지 마십시오.

## 성능 및 품질 게이트

기준 GPU 환경은 Windows 11, Core Ultra 9 285K, RAM 64GB, RTX 5080 16GB입니다.

- 입력 부하: 초당 1장 이하, 요청 동시성 1
- 성능 범위: API 내부 디코딩, 전처리, detector, classifier, 후처리
- 측정 조건: ONNX Runtime CUDA Execution Provider 및 session warm-up 완료
- 성능 기준: detector와 classifier가 모두 실행되는 full-path p95 ≤ 100ms
- 정확도 기준: 전체 재촬영 대상 recall ≥ 99%
- 정확도 기준: `APPROVED` precision ≥ 99.5%
- 정확도 기준: 정답 클래스가 있는 `UNKNOWN` 샘플 Top-3 accuracy ≥ 95%
- CPU 기준: 결과 계약과 기능 호환 필수, 지연 기준 없음

평균 지연만 보고하지 마십시오. p50, p95, p99와 표본 수를 함께 기록하고 detector 조기 종료와 full-path를 구분하십시오. 벤치마크 결과에는 모델·데이터셋·ONNX Runtime·CUDA·드라이버 버전, 하드웨어와 warm-up 조건을 포함하십시오.

## 테스트 요구사항

새 기능이나 동작 변경에는 영향 범위에 맞는 테스트를 추가하십시오.

- Detector 조기 종료 시 classifier가 호출되지 않는지 검증
- `classifier_confidence` 경계 정책에서 높은 신뢰도는 계속 진행하고 낮은 신뢰도만 `DETECTOR_BORDER_CLIPPED`로 재촬영하는지 검증
- detector 불확실 독립 후보 및 선택적 count verifier의 불일치·저신뢰가 classifier 실행 전 `RECAPTURE`가 되는지 검증
- Classifier 품질 클래스가 `RECAPTURE`로 변환되는지 검증
- 임계값 경계의 `APPROVED`/`UNKNOWN` 판정 검증
- 다중 item 정렬, 최상위 상태 집계, `UNKNOWN` Top-3 정렬과 후보 수 검증
- 네 상태의 응답 필드 및 null/빈 배열 규칙 검증
- 손상 이미지, 미지원 형식, 누락 multipart 필드의 4xx `ERROR` 검증
- 모델 로딩, checksum, provider 실행 장애의 5xx `ERROR` 검증
- CPU/CUDA 전처리, 출력 허용 오차와 최종 판정 parity 검증
- 고정 데이터셋 정확도 KPI와 RTX 5080 full-path p95 성능 회귀 검증

성능 테스트의 일시적인 통과를 위해 정확도 임계값을 낮추거나, 정확도 테스트를 통과시키기 위해 test split에 맞춰 임계값을 조정하지 마십시오.

## 로그 및 개인정보 보호

- 구조화 로그에 `request_id`, 각 단계의 처리시간, 최종 상태, reason code, 실행된 모델 버전을 기록하십시오.
- 요청 간 추적을 위해 동일한 `request_id`를 API 응답과 모든 관련 로그에서 사용하십시오.
- 이미지 원본/바이트, 로컬 파일 경로, 전체 모델 출력과 민감한 메타데이터는 기본 로그에 기록하지 마십시오.
- 디버그 이미지 저장은 명시적으로 활성화된 개발 환경에서만 허용하고 보존 기간과 삭제 방법을 문서화하십시오.
- 사용자 응답에는 내부 오류 세부사항을 노출하지 말고, 상세 원인은 접근 제어된 내부 로그에만 남기십시오.

## 문서 및 변경 원칙

- README는 사용자와 운영자가 보는 공개 계약이고, 이 파일은 구현자가 지키는 저장소 규칙입니다.
- API, 모델 패키지, 데이터 manifest, 상태 또는 KPI를 변경할 때 README와 테스트를 같은 변경에서 갱신하십시오.
- 실제 설치, 실행, 학습, export, 평가 명령이 생기면 검증한 명령만 README에 추가하십시오.
- 문서는 한국어를 기본으로 작성하되 코드 식별자, 상태, API 필드와 reason code는 영어를 유지하십시오.
- `RECAPTURE` 철자를 일관되게 사용하십시오.
