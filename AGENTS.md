# AGENTS.md

이 파일은 저장소 전체에 적용됩니다. 이 프로젝트를 수정하는 모든 에이전트와 기여자는 아래 계약을 따라야 합니다.

## 프로젝트 목표

이 저장소는 이미지 판정 시스템의 데이터 계약, PyTorch 학습·평가, ONNX export, 그리고 Windows에서 장기 실행되는 Python 추론 Worker를 관리합니다.

운영 Worker는 단일 JPEG/PNG 이미지에서 여러 객체를 판정하고 다음 이미지 상태 중 정확히 하나를 반환합니다.

- `SEGMENTATION`: 하나 이상의 segmentation과 객체별 판정 결과
- `IMAGE_RECAPTURE`: detector가 이미지 전체의 촬영 부적합을 판정한 결과
- `ERROR`: 입력, 구성, 모델 또는 시스템 오류

각 segmentation은 `APPROVED`, `UNKNOWN`+Top-3 또는 `SEGMENT_RECAPTURE`입니다. `ERROR`를 재촬영으로 변환하지 마십시오. 모델 판정과 시스템 장애는 서로 다른 도메인입니다.

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

한 축의 버전 변경이나 배포가 다른 축의 자동 승격을 의미하지 않습니다. 정식 Worker, Detector, Classifier 버전은 각각 `1.0.0`부터 시작하고 독립적으로 관리합니다. 이전 `0.x` package와 Detector `0.2.5`는 테스트 계열이며 운영 기본값으로 승격하지 마십시오.

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
3. Detector의 hard 품질 조건(무검출, query 포화, 최소 크기, 불확실 독립 후보, 활성화된 count/blur/exposure gate)이 재촬영을 요구하면 classifier를 호출하지 않고 `IMAGE_RECAPTURE`를 반환합니다.
4. 정상 ROI와 `classifier_confidence` 정책의 경계 접촉 ROI 전체를 하나의 batch로 classifier에 전달합니다.
5. Classifier가 명시적 품질 클래스를 예측하면 해당 segmentation을 `SEGMENT_RECAPTURE`로 반환합니다.
6. `classifier_confidence` 정책에서 경계 접촉 segmentation의 Top-1 신뢰도가 승인 임계값 미만이면 해당 segmentation을 `DETECTOR_BORDER_CLIPPED` `SEGMENT_RECAPTURE`로 반환합니다. 이 경로는 classifier가 실행된 경로입니다.
7. 패키지 메타데이터의 포함 중복 검토 정책이 활성화된 경우, 거의 완전히 포함되고 같은 Top-1을 가진 ROI 쌍에서 detector 점수가 낮은 고신뢰 ROI는 삭제하거나 재촬영하지 않고 `DETECTOR_CONTAINED_DUPLICATE` `UNKNOWN`과 점수 내림차순 Top-3로 반환합니다.
8. 각 나머지 일반 객체가 승인 임계값 이상이면 해당 segmentation을 `APPROVED`로 반환합니다.
9. 임계값 미만 segmentation은 `BELOW_APPROVAL_THRESHOLD` `UNKNOWN`과 점수 내림차순 Top-3를 반환합니다. 하나 이상의 segmentation이 있으면 최상위 상태는 `SEGMENTATION`입니다.

이 순서, 상태 우선순위 또는 조기 종료 조건을 변경하려면 README의 계약과 관련 테스트를 같은 변경에서 갱신해야 합니다. 조용한 fallback이나 임의의 기본 승인 결과를 추가하지 마십시오.

## API 계약

- 기본 endpoint는 `POST /v1/scan`입니다.
- 요청은 `image` 필드에 JPEG/PNG 한 장을 담은 multipart 형식입니다.
- 최상위 응답 필드는 `request_id`, `status`, `reason_codes`, `segmentations`, `processing_time_ms`, `worker_version`, `detector_version`, `classifier_version`입니다.
- 각 `segmentations[]`는 `segmentation_id`, 원본 픽셀 기준 `bbox`, `status`, `reason_codes`, `prediction`, `top3`, `confidence`를 포함합니다.
- 이미지 `status`는 `SEGMENTATION`, `IMAGE_RECAPTURE`, `ERROR` 외 값을 반환할 수 없습니다.
- segmentation `status`는 `APPROVED`, `UNKNOWN`, `SEGMENT_RECAPTURE` 외 값을 반환할 수 없습니다.
- `UNKNOWN` segmentation의 Top-3는 점수 내림차순이어야 합니다.
- Detector hard gate 조기 종료 시 classifier를 로드했더라도 해당 요청의 `classifier_version`은 실행되지 않았음을 나타내도록 `null`로 반환합니다.
- `IMAGE_RECAPTURE`와 `ERROR`는 `segmentations`를 빈 배열로 반환합니다.
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
- 성능 기준: detector와 classifier가 모두 실행되는 full-path 평균 및 p95 ≤ 100ms
- 정확도 기준: 전체 재촬영 대상 recall ≥ 99%
- 정확도 기준: 전체 인식률 ≥ 99%
- 정확도 기준: `APPROVED` 오인율 및 단측 95% 상한 ≤ 0.1%
- 정확도 기준: segmentation IoU@0.5 recall/precision ≥ 99%
- 정확도 기준: 정답 클래스가 있는 `UNKNOWN` 샘플 Top-3 accuracy ≥ 95%
- CPU 기준: 결과 계약과 기능 호환 필수, 지연 기준 없음

### Bread 1.1 전용 최종 목표와 운영 기준

`bread-zero-error-1.1.0`과 그 Worker·Detector·Classifier `1.1.0` 후보에는 프로젝트 소유자가
2026-08-18 지정한 다음 end-to-end 최종 목표와 운영 gate를 적용합니다. 이 기준은 위 1.0 정확도 gate를
1.1 후보에 그대로 중복 적용하지 않는 version-specific override입니다. 단, 승격 순서,
group-aware 분할, 잠긴 test, parity, 성능 및 package 검증 의무는 그대로 유지합니다.

- 장기 개선 최종 목표: 전체 판정 가능 GT 객체 대비 최종 `APPROVED` 비율 ≥ 99%

운영 승격 기준은 다음 여섯 가지입니다.

- `SEGMENTATION` 이미지 비율 ≥ 90%
- 전체 판정 가능 GT 객체 대비 최종 `APPROVED` 비율 ≥ 90%
- `SEGMENTATION` 이미지 중 IoU@0.5 FN 포함 이미지 비율 ≤ 0.1%
- `SEGMENTATION` 이미지 중 IoU@0.5 FP 포함 이미지 비율 ≤ 0.1%
- 전체 판정 가능 GT 객체 대비 `APPROVED` 객체 오인율 ≤ 0.1%
- 전체 판정 가능 GT 객체 대비 `UNKNOWN` Top-3 Candidate out 비율 ≤ 0.1%

`판정 가능 GT 객체`는 정답 상태가 객체 판정 대상인 이미지의 모든 GT를 뜻합니다. 예측 결과가
`IMAGE_RECAPTURE`이거나 detector가 GT를 놓쳤더라도 분모에서 제외하지 않습니다. 정답상 이미지
전체 재촬영 대상은 이 객체 분모에서 제외하고 별도의 recapture recall로 평가합니다. 따라서
`APPROVED` 99% 목표를 detector 오류나 불필요한 이미지 재촬영으로 우회할 수 없습니다.

모든 비교는 경계를 포함합니다. 99% 최종 목표의 달성 여부와 여섯 운영 gate의 통과 여부를
분리해 보고하십시오. 99% 미달만으로 운영 gate 실패로 판정하지 말고, 운영 gate 통과만으로
장기 개선 목표를 달성했다고 선언하지 마십시오. `UNKNOWN + Top-3` 비율과
`SEGMENT_RECAPTURE` 비율은 원인
분석용으로 계속 집계하지만 1.1 승격 합격 여부에는 사용하지 않습니다. unmatched detector FP는
객체 분모에서 제외한 뒤 이미지 FP gate로 검출합니다.
`SEGMENTATION` 최소 비율과 기존 `IMAGE_RECAPTURE` 비회귀를 함께 확인하여 detector 오류를
이미지 재촬영으로 우회하지 마십시오. 공식 정의와 현재 재판정은
`docs/experiments/bread/bread-zero-error-1.1.0.md`를 따릅니다.

### Bread 1.1.0 소유자 승인 예외와 1.1.1 이후 계약

프로젝트 소유자는 2026-08-19 `bread-zero-error-1.1.0-domain-lda-fixed-four-v3`를 Worker·
Detector·Classifier `1.1.0` 운영 기준선으로 명시적으로 승인했습니다. 이 결정은 최종 LDA head가
`single_objects` 200장 외의 E/M/H ROI 1,410개로 fit됐고 새 독립 잠금 test가 없다는 두 실패를
감사 가능한 `manual_waiver`로 남기는 1.1.0 한정 예외입니다. E/M/H와 반려 운영 115장을 독립
증거로 재분류하거나, 1.1.0 개발 수치를 독립 일반화 성능으로 표현하지 마십시오.

이 예외는 `1.1.1` 이상에 상속되지 않습니다. 1.1.1부터 Classifier의 모든 학습 가능한 파라미터,
feature/head fitting, normalization statistic, calibration, 승인 threshold, Top-3 ranking과 TTA
선택은 `manifests/bread-zero-error-1.1/classifier_manifest.jsonl`의 `single_objects` 200장과 그
결정적 파생 입력만 사용해야 합니다. E/M/H, scan log, operational collection, GT crop과 detector
ROI는 Classifier 학습·fitting에 사용할 수 없습니다. successor의 독립 승격에는 새 촬영 세션의
잠금 test가 다시 필요합니다. 버전별 반복과 종료 조건은
`docs/experiments/bread/bread-classifier-200-only-1.1.1-plan.md`를 따릅니다.

### Scanner 2.0.0 RC.7 반려와 RC.8 개발 계보

프로젝트 소유자는 2026-08-19 `bread_project_2`와 2026-08-18 운영 source의 415개 원본
장면에서 만든 Bread Project 5 증강 3,000장을 파생 provenance 제한에도 불구하고 Scanner
`2.0.0-rc.7` 승격 판정에 사용하도록 지시했습니다. 이 지시는 독립 촬영 provenance와 최소 독립
certification group 수만 `manual_waiver`로 허용하며 point metric gate를 면제하지 않습니다.
3,000장을 독립 trial 또는 독립 일반화 성능으로 표현하지 마십시오.

고정 RC.7 평가 결과 `SEGMENTATION` 이미지 FN 0.9278%, FP 0.8591%, 전체 GT 대비 승인 오인
0.1371%로 세 point gate가 실패했습니다. 따라서 RC.7은 `rejected`이며 `2.0.0` production으로
승격하거나 운영 기본값으로 설정할 수 없습니다. 이 반려를 threshold tuning이나 같은 파생 세트의
재사용으로 뒤집지 마십시오. 공식 근거는
`docs/experiments/bread/scanner-2.0.0-bread-project-5-promotion.md`를 따릅니다.

프로젝트 소유자는 2026-08-19부터 `bread_project_2` 300장과 2026-08-18 운영 수집본 115장을
Scanner `2.0.0-rc.8`의 개발·ambiguity/OOD calibration과 회귀에 함께 사용하도록 결정했습니다.
따라서 이 415장과 Bread Project 5 파생본은 더 이상 독립 승격 test가 아니며 독립 일반화 성능으로
표현할 수 없습니다. 당시 RC.8 기본 승격 계약은 이 전체 개발 identity와 exact/dHash≤2로 겹치지 않는
새 owner-private locked 촬영 세션을 요구했습니다. private 결과를 본 뒤 model, threshold, crop,
ranking 또는 decision policy를 바꾸면 그 private bundle도 개발 계보로 전환해야 합니다.

### Scanner 2.0.0 소유자 승인 예외와 영구 무키 Catalog 계약

프로젝트 소유자는 2026-08-20 새 owner-private locked test를 제공하지 않고, 고정
`2.0.0-rc.8`을 Scanner `2.0.0` production으로 승격하도록 명시적으로 승인했습니다. 이 예외는
`configs/releases/scanner_2.0.0_owner_waiver.json`과
`artifacts/releases/scanner-2.0.0-production/promotion-attestation.json`에 고정합니다. 300장과
운영 115장은 계속 개발 계보이며 독립 일반화 성능으로 표현할 수 없고,
`independent_certified=false`를 유지해야 합니다. 장기 `APPROVED ≥99%` 목표도 달성한 것으로
표현하지 마십시오.

Scanner 2.x Store Catalog는 지속적으로 signing key, key ID, HMAC과 `signature.json`을 사용하지
않습니다. `authentication=CHECKSUM-SHA256`을 기록하고 Worker 시작 시 모든 Catalog file SHA-256,
source manifest, shape와 Runtime compatibility를 검증하십시오. checksum 불일치는 시작 오류입니다.
이 계약은 손상·파일 변경 탐지이며 발행자 진위 인증은 제공하지 않는다는 제한을 숨기지 마십시오.
향후 Catalog compiler와 운영 예시는 기본적으로 무키 package를 생성·로딩해야 하며 사용자에게 키
생성, 보관 또는 회전을 요구하지 마십시오. 기존 HMAC package 읽기 호환은 유지할 수 있지만 Scanner
2.x production 기본값으로 되돌리지 마십시오.

평균 지연만 보고하지 마십시오. p50, p95, p99와 표본 수를 함께 기록하고 detector 조기 종료와 full-path를 구분하십시오. 벤치마크 결과에는 모델·데이터셋·ONNX Runtime·CUDA·드라이버 버전, 하드웨어와 warm-up 조건을 포함하십시오.

## 테스트 요구사항

새 기능이나 동작 변경에는 영향 범위에 맞는 테스트를 추가하십시오.

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
