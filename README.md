# Bixolon Scanner

여러 상품이 있는 JPEG/PNG 한 장을 판정하는 Windows 시스템입니다. ONNX Runtime Worker,
PyTorch 학습·평가 도구와 Flutter 작업자 앱을 한 저장소에서 관리합니다.

## 현재 버전

현재 배포 가능한 실행 조합은 `0.1.5` 하나입니다.

| 구성 | 버전 |
|---|---|
| Python 패키지·Worker | `0.1.5` |
| Detector·Embedder·판정 정책 | `0.1.5` |
| Store Catalog | `0.1.5` |
| Flutter 앱 | `0.1.5+8` |
| 사용자 표시·Windows ProductVersion | `0.1.5` |

`0.1.5`는 인증이나 SLA 상태가 아니라 함께 실행되는 파일 조합의 식별자입니다. 앱, Worker,
Runtime과 Catalog 내용이 바뀌어 새 EXE를 배포할 때만 patch version을 올립니다. 학습과
평가를 반복하는 동안에는 제품 버전을 바꾸지 않습니다.

기준 설정은 [`configs/versions/0.1.5.json`](configs/versions/0.1.5.json)입니다. Detector는
상품 label이 없는 YOLO26 objectness 모델이고, Classifier는 동일 조건의 소스 비교에서 선택한
`single_objects_3`만 사용한 DINOv3 ConvNeXt-Tiny soup와 DINOv3 ViT-B/16 선택 검증기입니다.
multi-object 상품 label은 Classifier 학습·보정에 사용하지 않았습니다. 클래스·상품쌍·객체 수·
난이도·이미지별 예외 규칙은 없습니다. SKU 증분 Catalog는 기존 adapter 열을 그대로 보존하고
새 출력 열만 독립적으로 학습하며, 주·회전·독립 검증기 모두가 새 SKU에 동의할 때만 새 출력을
사용합니다. 20개 class-holdout 개발 회귀의 기존 클래스 판단 36,366건은 등록 전후 변화 0건입니다.
`0.1.5`는 큰 detector proposal을 크기만으로 거부하지 않고, proposal 내부의 raw query surplus 또는
복수 detection 중심점이 보강할 때만 기존 crowding hard gate를 적용합니다. 기존 raw query
근접·중복과 선택적 90°·180° 회전 입력의 객체 수 합의도 함께 검사합니다. 회전 검사는 기본 검출이
정확히 5개이고 bbox 면적·간격 조건이 맞을 때만 실행합니다. 회전 입력에서 객체가 하나 이상 추가
복구되고 기존 bbox가 모두 일대일로 대응하거나 보강된 raw query hard 조건이 충족될 때만
`DETECTOR_UNCERTAIN_OBJECT`로 판정해 classifier 실행 전에 `IMAGE_RECAPTURE`로 종료합니다.
큰 빵 하나이거나 detector 결과가 완전하면 그대로 분류합니다. 현재 개발
회귀 결과와 데이터 한계는 [0.1.5 개발 검증 보고서](docs/evaluation/scanner-0.1.5.md), 과거 버전과 당시 판단은
[`docs/archive/version-history.md`](docs/archive/version-history.md)에 보존합니다.

## 판정 계약

Worker는 이미지마다 다음 중 정확히 하나를 반환합니다.

- `SEGMENTATION`: 하나 이상의 객체와 객체별 `APPROVED`, `UNKNOWN`+Top-3 또는
  `SEGMENT_RECAPTURE`
- `IMAGE_RECAPTURE`: detector hard gate가 이미지 전체 재촬영을 요구함
- `ERROR`: 입력, 구성, 모델 또는 시스템 오류

`ERROR`를 재촬영으로 변환하지 않습니다. Detector hard gate로 classifier를 실행하지 않은 경우
`classifier_version`과 Catalog 계열 버전은 `null`입니다. 그 밖의 실행된 버전 필드는 모두
`0.1.5`입니다. `IMAGE_RECAPTURE`와 `SEGMENT_RECAPTURE`는 각각 하나의 공개 공통 reason만
반환합니다. 공개 필드와 판정 순서는 [API 계약](docs/contracts/api.md)을 따릅니다.

## Windows 번들 만들기

Python 3.11 이상 3.14 미만, Flutter stable과 Visual Studio Windows C++ build tools가 필요합니다.

```powershell
.\scripts\build_app.ps1 -Version 0.1.5
```

이 명령은 다음을 한 번에 수행합니다.

1. 선택한 Runtime/Catalog와 평가 증빙의 고정 해시를 검증합니다.
2. 모든 실행 구성요소를 `0.1.5`로 맞추고 Catalog를 무키 `CHECKSUM-SHA256` 형식으로 구성합니다.
3. PyInstaller Worker, 지정된 CUDA DLL과 Flutter 앱을 자체 포함 번들로 만듭니다.
4. `version.json`, `provenance.json`과 전체 파일 `bundle-manifest.json`을 생성합니다.
5. 모든 구성요소 버전, 필수 파일과 Windows EXE ProductVersion을 검증합니다.

결과는 `artifacts/versions/0.1.5/bixolon-scanner-0.1.5`에 생성됩니다. 설치 PC의 Python이나
signing key 환경 변수에 의존하지 않습니다. Catalog와 Runtime의 모든 파일 checksum이 시작 시
검증되며 손상되면 Worker는 준비 상태가 되지 않습니다. `CHECKSUM-SHA256`은 손상 탐지이며 발행자
진위 인증은 제공하지 않습니다.

### Intel N100 PC용 Windows 인스톨러

NVIDIA GPU가 없는 64비트 Windows N100 PC에는 CPU Detector, Intel GPU Embedder와 Flutter 앱을
포함한 Setup EXE를 사용합니다. Intel GPU가 없거나 초기화되지 않으면 Worker가 경고 로그를 남기고
CPU Embedder로 명시적으로 fallback합니다. 빌드 PC에는 Inno Setup 6이 필요합니다.

```powershell
.\scripts\build_app.ps1 -Version 0.1.5
.\scripts\build_worker_handoff.ps1 -Version 0.1.5 -ReuseBuildEnvironment
.\scripts\build_n100_test_candidate.ps1 -Version 0.1.5 -Force
.\scripts\build_n100_gpu_test.ps1 -Version 0.1.5 -ReuseBuildEnvironment -Force
```

`artifacts/handoff/n100-0.1.5-candidate-*.zip`은 CPU-only 기준 진단에 사용합니다.

CPU와 Intel 내장 GPU를 직접 비교하려면
`artifacts/handoff/n100-0.1.5-openvino-cpu-gpu-test.zip`을 N100에 풀고
`RUN-N100-GPU-TEST.cmd`를 실행합니다. 하나의 고정 Worker가 먼저 Detector와 모든 Embedder를
OpenVINO CPU로 실행하고, 이어 Detector만 CPU에 유지한 채 주·회전·독립 검증 Embedder를 Intel
GPU로 실행합니다. 두 실행은 같은 이미지 순서, Runtime, Catalog와 판정 정책을 사용하며 결과는
`n100-0.1.5-openvino-device-matrix.json` 하나에 응답 parity, 평균·p50·p95·p99, 단계별 지연,
시작시간과 peak working set을 함께 기록합니다. 비교 실험 중 GPU graph가 CPU로 조용히
fallback하는 것은 허용하지 않습니다. 최종 Worker의 startup fallback은 이 비교가 끝난 뒤 별도의
명시적 CPU profile 재조립으로 수행되며 `/health/ready`의 `provider`에 실제 실행 provider가 표시됩니다.

```powershell
.\scripts\build_n100_installer.ps1 -Version 0.1.5 `
  -N100DeviceMatrixPath D:\n100-0.1.5-openvino-device-matrix.json -Force
```

N100 device matrix에서 하이브리드 profile 초기화, 오류 0건, semantic parity와 CPU-only 대비 평균·p95
개선이 확인되어야 합니다. 앞으로 평균과 p95 `500ms 이하`를 현재 N100 운영 진단 기준으로 사용하되,
설치 생성 조건이나 SLA로 사용하지 않습니다.
활성 `0.1.5`와 같은 모델·Catalog·판정 정책으로 측정한 N100 결과만 참조 진단으로 사용할 수
있습니다. `0.1.5+8` N100 결과는 아직 없으며 기존 `0.1.4+7` 결과는 참고 기준일 뿐 새 정책의
성능 증빙이 아닙니다. provenance에는 측정 환경과 원본 결과 SHA-256을 기록합니다. 빌드가 완료되면
`artifacts/installers/0.1.5/BixolonScanner-N100-0.1.5-Setup.exe`와 외부 `.sha256`,
`installer-manifest.json`, 그리고 독립 실행용
`BixolonScanner-N100-0.1.5-Worker.zip`과 `.sha256`이 생성됩니다. 두 산출물은 같은 GPU 포함 Worker,
Runtime과 Catalog를 사용합니다. 인스톨러는 Microsoft Visual C++ x64 Runtime도 포함하므로 대상 PC에
Python, Flutter 또는 CUDA가 필요하지 않습니다.

앱은 실행 직후 카메라 초기화와 Worker warm-up을 동시에 시작하고, OpenVINO 컴파일 결과를
사용자별 version 캐시에 보존합니다. 촬영 뒤에는 Flutter 이미지 디코딩과 HTTP 요청을 병렬로
진행합니다. 캐시는 두 번째 실행부터의 준비 시간을 줄이지만 첫 실행 시간 자체를 보장하지 않습니다.

Setup EXE는 현재 Authenticode 서명이 없으므로 Windows가 알 수 없는 게시자 경고를 표시할 수
있습니다. 전달할 때 `.sha256` 파일을 함께 보내고 대상 PC에서 SHA-256을 확인하십시오. 이 값은
전송 중 손상을 탐지하지만 발행자 진위 인증을 제공하지 않습니다. 지원 OS, 설치·제거와 데이터
보존 범위는 [N100 설치 안내](installer/n100/INSTALL-N100-KO.txt)에 정리돼 있습니다. 어느 측정치도
보장된 지연시간이나 SLA로 표현하지 않습니다.

### Flutter 개발자 전달용 CPU Worker

같은 PC에서 `127.0.0.1:8000`으로 연결하는 Windows x64 CPU 패키지는 Python 3.11 격리 환경과
고정 `onnxruntime-openvino` 배포 lock으로 생성하고 `OpenVINOExecutionProvider`의 CPU device로 실행합니다.
N100 결과가 아직 없으면 provenance에
`PENDING_FIELD_MEASUREMENT`를 기록하고, 결과가 있으면 선택 profile과 측정치를 포함합니다.

```powershell
.\scripts\build_worker_handoff.ps1 -Version 0.1.5
```

결과는 `artifacts/handoff/0.1.5/bixolon-worker-0.1.5-windows-x64-openvino.zip`과 외부
`.zip.sha256` 파일입니다. 압축에는 실행 스크립트, 고정 CPU N100 측정 스크립트, 한국어
`API.md`, JSON Schema, Dart client·DTO 및 상태별 JSON 예시가 포함됩니다. 측정치는 진단 결과로만
취급합니다.

준비된 메타데이터와 생성된 번들을 다시 검증하려면 다음 명령을 사용합니다.

```powershell
bixolon bundle verify --config configs/versions/0.1.5.json
```

## 개발과 검증

```powershell
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest

cd apps\product_scanner
flutter analyze
flutter test
```

정확도, CPU/CUDA parity, latency와 장기 신뢰성 평가는 `evaluation` 명령으로 계속 실행할 수
있습니다. 결과는 진단 기록이며 버전 번들 생성을 차단하거나 별도의 배포 단계를 만들지 않습니다.
기본 `bixolon --help`에는 현재 버전 명령만 표시하며 진단 도구는
`bixolon --help-diagnostics`, 과거 재현 명령은 `bixolon --help-legacy`에서 분리해 확인합니다.

## 저장소 구조

```text
apps/product_scanner/           Flutter Windows 앱
configs/versions/               활성 제품 `0.1.5` 기준 설정 하나
configs/experiments/            재현 가능한 실험 설정
configs/archive/                과거 제품·실험 설정 원문
docs/evaluation/                활성 제품의 평가 요약과 한계
docs/archive/                   과거 계약·평가·판단·가이드
src/bixolon_scanner/
  command_registry.py           활성·진단·과거 호환 CLI 경계
  configuration.py              redirect-aware JSON 설정 로더
  contracts/                    API·오류·패키지 계약
  pipeline/                     단일 판정 정책
  runtime/                      이미지 decode·ONNX session/provider·adapter
  worker/                       FastAPI·설정·로그·조립
  training/                     학습과 calibration
  evaluation/                   정확도·parity·성능 진단
  experiments/                  실험 orchestration; archive는 source-only
  operations/                   버전 번들·로그·검수 export
tests/                          계약과 회귀 테스트
artifacts/, datasets/, runs/    Git 밖의 모델·데이터·실험 산출물
```

구현 경계와 변경 규칙은 [AGENTS.md](AGENTS.md), 문서 목록과 현재 해시는
[`docs/README.md`](docs/README.md), 로컬 산출물 보존·정리 기준은
[`docs/maintenance/repository-layout.md`](docs/maintenance/repository-layout.md)에서 확인할 수
있습니다. 대형 모델·데이터·빌드 산출물은 Git에 커밋하지 않습니다.
