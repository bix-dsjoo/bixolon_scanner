# AGENTS.md

이 파일은 저장소 전체에 적용됩니다. 이 프로젝트를 수정하는 모든 에이전트와 기여자는 아래 계약을
따라야 합니다.

## 프로젝트와 상태 계약

이 저장소는 이미지 판정 시스템의 데이터 계약, PyTorch 학습·평가, ONNX export, Windows에서
장기 실행되는 Python Worker와 Flutter 앱을 관리합니다. Worker는 JPEG/PNG 한 장에 대해 다음 중
정확히 하나를 반환합니다.

- `SEGMENTATION`: 하나 이상의 segmentation과 객체별 판정
- `IMAGE_RECAPTURE`: detector가 이미지 전체의 촬영 부적합을 판정
- `ERROR`: 입력, 구성, 모델 또는 시스템 오류

각 segmentation은 `APPROVED`, `UNKNOWN`+Top-3 또는 `SEGMENT_RECAPTURE`입니다. `ERROR`를
재촬영으로 변환하지 마십시오. 모델 판정과 시스템 장애는 서로 다른 도메인입니다.

## 코드 소유권과 의존 방향

Python canonical 코드는 `src/bixolon_scanner` 아래에 둡니다.

- `contracts`: API schema, 오류, 이미지·Runtime·Catalog 계약
- `pipeline`: detector → 조기 종료 → ROI batch → classifier → 최종 상태의 단일 정책
- `runtime`: decode, 전처리·후처리, ONNX Runtime CPU/CUDA provider
- `worker`: FastAPI, 설정, 구조화 로그, 실행 조립
- `training`: 재사용 학습 데이터·모델·trainer·calibration
- `evaluation`: 정확도, parity, 지연과 회귀 진단
- `experiments`: 학습·평가 orchestration
- `operations`: 버전 번들, 로그 수집과 검수 export
- `configuration.py`: redirect-aware JSON 설정 로딩의 단일 진입점
- `command_registry.py`: 활성·진단·호환 CLI 명령의 지원 수준
- `cli.py`: 통합 `bixolon` 명령

요청 경로는 `worker → pipeline/runtime → contracts` 방향입니다. `worker`는 `training`,
`evaluation`, `experiments` 또는 PyTorch를 import하지 않으며 `runtime`에도 PyTorch를 추가하지
마십시오. 상태 결정은 `pipeline`에서만 수행합니다. 공용 NMS, IoU, RGB 변환은 `runtime`의
명시적 내부 공용 API를 사용하며 다른 모듈의 private 함수를 import하지 마십시오.

Flutter canonical 코드는 `apps/product_scanner/lib`의 `core/design_system`, `shared`,
`features/scanner`, `features/activity`에 둡니다. feature가 다른 feature의 화면 구현을 직접 참조하지
않고 공용 계약은 `shared`, 재사용 UI는 `core/design_system`이 소유합니다. 과거 Python·Flutter
경로의 re-export는 호환 계층이며 새 구현을 추가하지 마십시오.
`experiments/archive`는 소스에서 재현 테스트를 위해 보존하지만 runtime wheel에는 포함하지 않습니다.
활성 설정을 읽는 코드는 직접 `json.loads(path.read_text(...))`하지 않고 `load_json_config`를 사용합니다.

## 단일 제품 버전

배포 가능한 앱·Worker·Runtime·Catalog 조합은 하나의 semantic version으로 식별합니다. 현재
기준은 `configs/versions/0.0.1.json`이며 Python, Worker, Detector, Embedder, Detector policy,
Classifier policy, Catalog와 사용자 표시 버전은 모두 `0.0.1`입니다. Flutter 내부 빌드만
`0.0.1+1`을 사용합니다.

- development, demo, production 환경 버전을 만들지 않습니다.
- 활성 설정과 CLI에 promotion, waiver, certification 또는 release-lock 수명주기를 추가하지
  않습니다.
- 앱, Worker, Runtime, Catalog 중 배포할 내용이 바뀌어 새 EXE를 만들 때만 patch 버전을 올립니다.
- 학습·실험·평가를 반복하는 동안 제품 버전을 올리지 않습니다.
- 데이터셋명, capture session, source candidate와 원본 SHA-256은 제품 버전이 아닌 provenance로
  기록합니다.
- 정확도·parity·성능·신뢰성 평가는 선택 진단이며 번들 생성을 차단하는 배포 단계가 아닙니다.
- 과거 설정과 평가 판단은 `configs/archive`, `docs/archive`에 원문으로 보존하고 활성 기본값으로
  해석하지 않습니다.

새 버전 설정에는 제품 버전, 앱 build 번호, 입력 Runtime/Catalog/CUDA와 평가 증빙의 경로·고정
해시만 둡니다. lifecycle 또는 gate 필드를 생성하지 마십시오. 기존 metadata의 lifecycle 필드는
archive 검사를 위한 호환 reader만 읽을 수 있습니다.

## 버전 번들 계약

`scripts/build_app.ps1 -Version <version>`은 해당 `configs/versions/<version>.json`에서 Worker,
Runtime, Catalog, CUDA와 Flutter를 자체 포함 번들로 구성합니다.

- source manifest와 평가 증빙 SHA-256을 복사 전에 검증합니다.
- source model graph, weight, adapter, support, prototype payload를 변경하지 않습니다.
- Runtime/Catalog 메타데이터의 실행 구성요소 버전만 단일 제품 버전으로 바꿉니다.
- Catalog는 키·HMAC·`signature.json` 없이 `authentication=CHECKSUM-SHA256`을 사용합니다.
- Runtime/Catalog의 모든 파일 checksum, source manifest, shape와 runtime compatibility를 시작 시
  검증하며 불일치는 시작 오류입니다.
- checksum은 손상·파일 변경을 탐지하지만 발행자 진위 인증을 제공하지 않는다는 제한을 숨기지
  않습니다.
- 최종 번들에는 `version.json`, `provenance.json`, 전체 파일 `bundle-manifest.json`, DINOv3와
  Apache 라이선스를 포함합니다.
- 활성 CLI에는 `bixolon bundle verify`만 노출합니다. promotion, waiver, independent certification
  또는 release-lock 명령을 다시 추가하지 마십시오.

## 변경할 수 없는 파이프라인 계약

1. 입력 이미지를 검증하고 디코딩합니다.
2. Detector가 모든 객체 위치와 프레임 전체 촬영 품질을 판단합니다.
3. hard 품질 조건이 재촬영을 요구하면 classifier를 호출하지 않고 `IMAGE_RECAPTURE`를 반환합니다.
4. 정상 ROI와 `classifier_confidence` 경계 ROI 전체를 한 batch로 classifier에 전달합니다.
5. Classifier 품질 클래스는 해당 segmentation을 `SEGMENT_RECAPTURE`로 만듭니다.
6. 경계 접촉 ROI의 Top-1이 승인 임계값 미만이면 `DETECTOR_BORDER_CLIPPED`
   `SEGMENT_RECAPTURE`로 반환합니다.
7. 포함 중복 검토 정책이 활성화된 경우 거의 완전히 포함되고 같은 Top-1인 ROI 쌍에서 detector
   점수가 낮은 고신뢰 ROI를 삭제하거나 재촬영하지 않고 `DETECTOR_CONTAINED_DUPLICATE`
   `UNKNOWN`+Top-3로 반환합니다.
8. 나머지 객체가 승인 임계값 이상이면 `APPROVED`입니다.
9. 임계값 미만은 `BELOW_APPROVAL_THRESHOLD` `UNKNOWN`과 점수 내림차순 Top-3입니다. 하나 이상의
   segmentation이 있으면 최상위 상태는 `SEGMENTATION`입니다.

순서, 우선순위 또는 조기 종료를 바꾸면 README·API 계약·관련 테스트를 함께 갱신하십시오. 조용한
fallback이나 임의의 기본 승인 결과를 추가하지 마십시오.

## API 계약

- 기본 endpoint는 `POST /v1/scan`, multipart 필드는 `image`입니다.
- 최상위 필드는 `request_id`, `status`, `reason_codes`, `segmentations`, `processing_time_ms`,
  `worker_version`, `detector_version`, `classifier_version`입니다.
- 실행한 2.x 구성에는 `embedder_version`, `detector_policy_version`, `classifier_policy_version`,
  `catalog_version`도 반환합니다.
- 공개된 non-null 버전은 모두 같은 제품 버전이어야 합니다.
- Detector 조기 종료는 실행하지 않은 `classifier_version`, `embedder_version`,
  `classifier_policy_version`, `catalog_version`을 `null`로 반환합니다.
- `IMAGE_RECAPTURE`와 `ERROR`는 빈 `segmentations`를 반환합니다.
- 입력 오류는 4xx, Worker·모델 장애는 5xx이며 가능한 경우 공통 `ERROR` 형식을 유지합니다.
- raw tensor, 전체 logits, 내부 예외, stack trace 또는 로컬 경로를 노출하지 마십시오.

공개 필드, enum, reason code나 의미를 바꾸면 schema, README와 소비자 테스트를 같은 변경에서
갱신하십시오.

## 모델, 데이터와 평가 규칙

- 학습은 PyTorch, 실행 Worker는 ONNX Runtime만 사용합니다.
- CPU와 CUDA는 같은 ONNX, 전처리·후처리·metadata 계약을 사용하고 상태와 class rank parity를
  진단합니다.
- 입력 크기, 색 순서, 정규화, label, NMS, crop, threshold를 코드에 하드코딩하지 않고 버전 관리
  metadata에서 읽습니다.
- 필수 metadata 누락, 미지원 schema, checksum 불일치는 시작 오류입니다.
- 원본 이미지, 증강 이미지, checkpoint, ONNX와 대형 binary는 Git에 커밋하지 않습니다.
- 외부 데이터는 JSONL/CSV manifest로 참조하고 label, split, 촬영 provenance를 기록합니다.
- 같은 물리 대상이나 촬영 session 파생 이미지는 group-aware split으로 분리합니다.
- validation으로 모델·threshold를 선택하고 test는 최종 진단에만 사용합니다. test에 맞춘 threshold
  변경을 하지 않습니다.
- p50, p95, p99와 표본 수를 함께 기록하고 detector 조기 종료와 full-path를 구분합니다.

과거 KPI, 평가 결과, 예외와 제한은 `docs/archive/version-history.md` 및 그 링크 문서에 남아
있습니다. 현재 `0.0.1`을 독립 일반화 성능, 인증 또는 SLA 달성으로 표현하지 마십시오.

## 테스트 요구사항

동작 변경에 영향 범위 테스트를 추가합니다. 특히 다음 계약을 유지합니다.

- Detector 조기 종료 시 classifier가 호출되지 않는지 검증
- `classifier_confidence` 경계 정책에서 높은 신뢰도는 계속 진행하고 낮은 신뢰도만 `DETECTOR_BORDER_CLIPPED`로 재촬영하는지 검증
- detector 불확실 독립 후보 및 선택적 count verifier의 불일치·저신뢰가 classifier 실행 전 `RECAPTURE`가 되는지 검증
- Classifier 품질 클래스가 해당 `SEGMENT_RECAPTURE`로 변환되는지 검증
- 임계값 경계의 `APPROVED`/`UNKNOWN` 판정 검증
- Top-1/Top-2 pair probability가 낮거나 Ridge·retrieval이 안전 경계 아래에서 불일치하면
  `UNKNOWN`+Top-3이고, retrieval OOD 경계 미만은 `SEGMENT_RECAPTURE`인지 검증
- 포함 중복 검토 정책이 segmentation을 삭제하거나 `RECAPTURE`하지 않고 낮은 detector 점수의 고신뢰 동종 ROI만 `UNKNOWN`으로 만드는지 검증
- 다중 segmentation 정렬, `UNKNOWN` Top-3 정렬과 후보 수 검증
- 네 상태의 응답 필드 및 null/빈 배열 규칙 검증
- 손상 이미지, 미지원 형식, 누락 multipart 필드의 4xx `ERROR` 검증
- 모델 로딩, checksum, provider 실행 장애의 5xx `ERROR` 검증
- CPU/CUDA 전처리, 출력 허용 오차와 최종 판정 parity 검증
- 고정 데이터셋 정확도 KPI와 RTX 5080 full-path p95 성능 회귀 검증
- 단일 제품 버전 일치와 앱 readiness 대기 검증
- 새 metadata에 lifecycle 필드와 `signature.json`이 생성되지 않는지 검증
- source와 변환 번들의 binary SHA-256 동일성 및 checksum 손상 시 시작 실패 검증

## 로그와 개인정보 보호

구조화 로그에는 `request_id`, 단계별 처리시간, 최종 상태, reason code와 실행 버전을 기록하고 API
응답과 동일한 `request_id`를 사용합니다. 이미지 bytes, 로컬 경로, 전체 모델 출력과 민감한
metadata를 기본 로그에 기록하지 마십시오. 디버그 이미지는 명시적으로 활성화된 개발 환경에서만
보존 기간·삭제 방법과 함께 허용합니다. 사용자 응답에는 내부 오류를 노출하지 않습니다.

## 문서와 완료 조건

문서는 한국어를 기본으로 하되 코드 식별자, 상태, API 필드와 reason code는 영어를 유지합니다.
`RECAPTURE` 철자를 일관되게 사용합니다. API, Runtime/Catalog metadata, 데이터 manifest, 상태 또는
제품 버전을 바꾸면 README와 테스트도 같은 변경에서 갱신합니다.

최종 변경은 `ruff check`, `ruff format --check`, 전체 Python 테스트, `flutter analyze`, 전체 Flutter
테스트와 `git diff --check`를 통과해야 합니다. Windows 번들 변경은 실제 build, version/checksum
검증과 가능한 범위의 packaged Worker smoke를 기록합니다. 진단이 미실행됐다는 이유로 제품 버전
번들 생성을 별도 수명주기로 바꾸지는 말고, 실행 여부와 한계를 정확히 문서화하십시오.
