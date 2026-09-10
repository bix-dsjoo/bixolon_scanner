# N100 구조 실험 R4

현재 개발 중인 0.2.1 기반 실험이며 공식 설치본을 대체하지 않는다. 실행 조건은
`configs/experiments/bread/n100_structural.json`, 산출물은 `artifacts/n100/structural-r4`에 보관한다.
구조 후보의 이름과 학습 provenance는 제품 버전과 구분한다.

## 순서와 데이터

1. RepViT-M0.9와 FastViT-T8 각각 일반 학습·증류·관계 증류를 같은 seed/20 epoch로 비교한다.
2. D-FINE-N 640을 GT 일반 학습과 GT 대응 teacher 증류로 비교한다. GT에 대응되지 않은 추가 검출은 증류하지 않는다.
3. 기존 detector의 stride 8 FPN 특징에서 ROIAlign과 작은 분류 head를 학습한다. 검출 graph/weight는 고정한다.
4. ONNX static INT8와 NNCF accuracy restoration을 수행한다. source 입력별 class·승인 경계·품질 증거를 보존한다.

현재 source 원본은 302장/649객체이며 같은 촬영 세션·물리 상품 집합이다. synthetic 파생 이미지도
같은 그룹이다. 이를 임의로 train/validation으로 나누지 않는다. source-only 고정 마지막 epoch
모델을 만들고 로그132와 final300은 회귀 진단으로만 사용한다. 독립 validation이 없으므로 threshold
재보정·독립 일반화 성능·배포 모델 선정이 완료됐다고 주장하지 않는다. NNCF의 validation callback
역시 이 실험에서는 source 진단이며 독립 검증을 의미하지 않는다.

## primary와 detail의 별도 Catalog

Runtime의 선택적 `classifier_resolution_fallback.catalog_directory`는 primary Catalog 루트 안의
하위 Catalog를 가리킨다. `catalog_checksums_sha256`를 반드시 함께 지정한다. 경로 탈출, 누락,
payload checksum 손상, 다른 feature ID/dimension, 상품 순서나 버전 불일치는 시작 오류다.
명시적 Catalog가 있을 때만 primary와 detail의 `embedder_id`가 달라도 허용한다. 기존 입력 크기,
전처리·정책·조기 종료·CPU fallback 계약은 유지한다.

detail에는 자기 support/prototype/adapter 및 rotation Catalog를 적용한다. 독립 verifier는 양쪽에서
같은 metadata·activation·payload를 사용해야 한다. JSON 공백 등 직렬화 차이는 의미 변화로 간주하지
않지만 각각의 checksum은 별도로 검증한다. 새 student의 support/prototype은 원본 support에서 다시
계산한다. 현재 margin threshold는 재보정 없이 고정한 진단 조건이며, 보정 완료로 표현하지 않는다.

## 재현과 판단

release worktree의 `src`가 import되도록 `PYTHONPATH`를 지정한다. 주요 실행 모듈:

- `training.structural_student --action cache`: provenance를 검증한 crop/teacher cache.
- `experiments.bread.structural_students`: 구조 export/parity, tensor timing, 6개 고정 학습.
- `experiments.bread.structural_regression`: 132/302/300장 최종 상태와 객체별 GT 대조.
- `experiments.bread.structural_cpu`: 4 logical CPU, AB/BA 교대, 132장×2회 전체 scan.
- `training.structural_detector`: 작은 detector 학습.
- `training.structural_shared_roi`: 고정 detector 특징 재사용 학습과 GT-box identity probe.
- `training.structural_quantization`: NNCF 3.3 ONNX static INT8와 입력별 증거 복원.
- `experiments.bread.structural_detector_quantization`: GT 대응과 추가 검출을 분리한 detector INT8 진단.
- `training.structural_qat`: PTQ가 FP32로 모두 복원된 뒤 실행하는 고정 5 epoch QAT+증류 대조군.

모듈 접두사는 `bixolon_scanner.`이다. 각 모듈의 `--help`와 고정 config를 사용한다.
학습 중 CPU 진단 시간은 제어된 성능 측정으로 사용하지 않는다. 단일 연산 benchmark, in-process scan,
HTTP 시간은 별도로 기록하고, 개발 PC의 수치를 N100 수치로 바꾸어 표현하지 않는다.
Windows 연구 도구 실행 시 `PYTHONUTF8=1`을 지정한다. NNCF는 별도 환경의 3.3.0을 사용하며
실험용 OpenVINO 2026.3.1을 제품 Worker에 섞지 않는다. 배포 Worker는 ORT-OpenVINO 1.24.1과
OpenVINO 2025.4.1을 유지한다.

USB 공간이 부족한 경우 manifest의 `benchmark_dependencies`로 명시한 기존 하위 폴더만 checksum
검증 후 로컬 캐시에 복사한다. 경로 탈출과 루트 전체 지정은 거부한다. 설정별 `runtime`에 overlay를
적용하므로 검증된 공통 모델 파일을 재사용할 수 있다. 새 EXE·모델과 이전 파일은 구분해 보관한다.

기준은 로그132의 오승인·미검출 0, 정답 승인 ≥1078, 재촬영 최소화다. 최종 보고에는 p50/p95/p99/max,
표본 수, 1초 초과 수, 전체 상태, 정답 승인 손실/개선 및 새 오승인을 별도로 기록한다. GT-box probe는
최종 승인율이나 미검출 0의 증거가 아니다. 원본 설치본과 악화 후보의 진단 기록은 보존하되, 악화 모델을
활성 구성에 적용하지 않는다. 디버그 이미지는 실험 산출물에만 저장하며 해당 실험 폴더 삭제로 정리한다.
