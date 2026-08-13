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

## 코드 소유권과 의존 방향

Python canonical 코드는 `src/bixolon_scanner` 아래의 다음 영역에 둡니다.

- `contracts`: API schema, 오류, 이미지와 모델 package 계약
- `pipeline`: detector → 조기 종료 → ROI batch → classifier → 최종 상태를 결정하는 단일 정책
- `runtime`: 이미지 decode, 전처리·후처리, ONNX Runtime session과 CPU/CUDA provider
- `worker`: FastAPI, 환경 설정, 구조화 로그와 실행 조립
- `training`: 재사용 가능한 데이터셋, 모델, trainer와 calibration
- `evaluation`: 정확도, PyTorch/ONNX parity, 지연과 회귀 측정
- `experiments`: bread, detector, RPC200 버전별 orchestration
- `operations`: 운영 로그 수집과 검수 export
- `cli.py`: 통합 `bixolon` 명령과 기존 console command 호환 alias

운영 요청 경로의 의존 방향은 `worker → pipeline/runtime → contracts`입니다. `worker`는 `training`, `evaluation`, `experiments`, PyTorch를 import할 수 없습니다. `runtime`에도 PyTorch를 추가하지 마십시오. 실험은 `training`과 `evaluation`을 조립할 수 있지만 반대 방향 import는 허용하지 않습니다.

상태 결정은 `pipeline` 한 곳에서 수행하고 HTTP 계층이나 모델 adapter에 복제하지 마십시오. 여러 영역에서 사용하는 NMS, IoU, RGB 변환 등의 함수는 `runtime`의 명시적인 내부 공용 API로 승격하며 다른 모듈의 private 함수를 import하지 마십시오.

루트의 `api.py`, `pipeline.py`, `inference.py`, `package.py` 등과 과거 `training.*` 실험 경로는 호환 계층입니다. 새 구현을 호환 파일에 추가하지 마십시오. 이 경로와 기존 `bixolon-*` 명령은 Python `0.3.x`까지 유지하며 제거는 `0.4.0` 이상의 migration을 포함한 변경에서만 허용합니다.

Flutter canonical 코드는 `apps/product_scanner/lib`의 `core/design_system`, `shared`, `features/scanner`, `features/activity`에 둡니다. 과거 `screens`, `services`, `models`, `widgets` 파일은 import 호환 export 계층입니다. feature 사이에서 화면 구현을 직접 참조하지 말고 공용 데이터 계약은 `shared`, 재사용 UI는 `core/design_system`이 소유합니다.

## 버전 구분

다음 버전은 서로 독립적인 계약입니다.

- Python 배포 버전: `pyproject.toml`과 `bixolon_scanner.__version__`
- 모델 package 버전: detector, classifier, metadata와 checksum의 운영 단위
- 데이터셋 버전: manifest, label, split과 촬영 provenance
- Flutter 앱 버전: 작업자 UI와 Windows bundle

한 축의 버전 변경이나 배포가 다른 축의 자동 승격을 의미하지 않습니다. 현재 Python은 `0.2.0`, 운영 모델 package는 `bread-worker-0.1.1`, Flutter 앱은 `1.0.0+1`입니다. Detector `0.2.5`는 `experiment_only`이며 운영 기본값으로 사용하지 마십시오.

## 실험 수명주기

모든 모델 실험은 `proposal → active → promoted/rejected → archive` 수명주기를 따릅니다.

- `proposal`: 가설, 데이터, KPI, 중단 조건을 문서화합니다.
- `active`: 재현 가능한 설정과 canonical CLI 경로가 있어야 합니다.
- `promoted`: 잠긴 정확도·parity·성능 gate를 모두 통과하고 운영 package 채택을 명시합니다.
- `rejected`: 실패 gate와 근거를 보고서에 남깁니다.
- `archive`: 완료, 거절 또는 prototype 설정을 `configs/archive`와 실험 문서에 보존합니다.

통합 CLI에는 최신 active 실험과 재현 가능한 공용 파이프라인만 노출하십시오. `experiment_only`, rejected 또는 archive 결과를 운영 설정에 자동 반영하지 마십시오.

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

모델 승격 순서는 `PyTorch checkpoint → 고정 validation KPI → ONNX export → PyTorch/CPU/CUDA parity → 패키징 → 잠긴 test KPI → Worker benchmark → 승격 결정`입니다. 중간 검증을 생략한 모델을 운영 패키지로 표시하지 마십시오.

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

## 변경 단계 완료 조건

구조 변경은 가능한 한 다음 경계를 독립 커밋으로 유지하고 각 단계에서 관련 검증을 완료하십시오.

1. 저장소 기준선: ignore, 버전, Ruff, 검증 스크립트와 CI가 일치해야 합니다.
2. 운영 계층: `contracts`, `pipeline`, `runtime`, `worker` 경계와 기존 import 호환 테스트가 통과해야 합니다.
3. ML·도구 계층: `training`, `evaluation`, `experiments`, `operations`, 통합 CLI와 설정 redirect가 검증되어야 합니다.
4. Flutter: feature 경계, analyze, 전체 test와 Golden 무변경을 확인해야 합니다.
5. 문서: README, 이 파일, 문서 인덱스, 내부 링크와 실제 버전·실험 상태가 일치해야 합니다.

최종 변경은 `ruff check`, `ruff format --check`, 전체 Python 테스트, `flutter analyze`, 전체 Flutter 테스트와 `git diff --check`를 통과해야 합니다. CUDA parity, RTX 5080 benchmark와 원본 데이터 KPI가 필요한 변경은 결과를 문서화한 수동 gate가 끝나기 전까지 운영 승격으로 표시하지 마십시오.
