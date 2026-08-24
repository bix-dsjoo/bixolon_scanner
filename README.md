# Bixolon Scanner

여러 상품이 있는 JPEG/PNG 한 장을 판정하는 Windows 시스템입니다. ONNX Runtime Worker,
PyTorch 학습·평가 도구와 Flutter 작업자 앱을 한 저장소에서 관리합니다.

## 현재 버전

현재 배포 가능한 실행 조합은 `0.1.2` 하나입니다.

| 구성 | 버전 |
|---|---|
| Python 패키지·Worker | `0.1.2` |
| Detector·Embedder·판정 정책 | `0.1.2` |
| Store Catalog | `0.1.2` |
| Flutter 앱 | `0.1.2+4` |
| 사용자 표시·Windows ProductVersion | `0.1.2` |

`0.1.2`는 인증이나 SLA 상태가 아니라 함께 실행되는 파일 조합의 식별자입니다. 앱, Worker,
Runtime과 Catalog 내용이 바뀌어 새 EXE를 배포할 때만 `0.1.2`, `0.1.3`처럼 올립니다. 학습과
평가를 반복하는 동안에는 제품 버전을 바꾸지 않습니다.

기준 설정은 [`configs/versions/0.1.2.json`](configs/versions/0.1.2.json)입니다. 이 번들은
N100 CPU 지연을 줄이기 위해 D-FINE Detector 한 개, 192 입력의 DINOv3 객체-presence verifier와
ONNX Runtime `CPUExecutionProvider` 실행 경로를 사용합니다. Detector는 `multi_object_scenes` 300장과
`operational_collections/2026-08-18` 115장만 사용했습니다. Classifier는
`single_objects_3` 240장만 사용해 last-stage 미세조정한 DINOv3 ConvNeXt-Tiny embedder이며,
Ridge/retrieval 불일치와 저유사도에 대한 전역 안전 정책을 사용합니다. 주변 객체 ownership mask에는
모든 ROI에 같은 거리 bias를 적용하고, detector가 classifier Top-2를 전역 confidence 조건으로
보강할 때만 순위를 교정합니다. 클래스·상품쌍·객체수·난이도별 예외 규칙은 포함하지 않았습니다.
원본·체크포인트·평가 자료의 경로와 SHA-256은
`provenance.json`에 남습니다. 과거 버전과 당시 판단은
[`docs/archive/version-history.md`](docs/archive/version-history.md)에 보존합니다.

## 판정 계약

Worker는 이미지마다 다음 중 정확히 하나를 반환합니다.

- `SEGMENTATION`: 하나 이상의 객체와 객체별 `APPROVED`, `UNKNOWN`+Top-3 또는
  `SEGMENT_RECAPTURE`
- `IMAGE_RECAPTURE`: detector hard gate가 이미지 전체 재촬영을 요구함
- `ERROR`: 입력, 구성, 모델 또는 시스템 오류

`ERROR`를 재촬영으로 변환하지 않습니다. Detector hard gate로 classifier를 실행하지 않은 경우
`classifier_version`과 Catalog 계열 버전은 `null`입니다. 그 밖의 실행된 버전 필드는 모두
`0.1.2`입니다. 공개 필드와 판정 순서는 [API 계약](docs/contracts/api.md)을 따릅니다.

## Windows 번들 만들기

Python 3.11 이상 3.14 미만, Flutter stable과 Visual Studio Windows C++ build tools가 필요합니다.

```powershell
.\scripts\build_app.ps1 -Version 0.1.2
```

이 명령은 다음을 한 번에 수행합니다.

1. 선택한 Runtime/Catalog와 평가 증빙의 고정 해시를 검증합니다.
2. 모든 실행 구성요소를 `0.1.2`로 맞추고 Catalog를 무키 `CHECKSUM-SHA256` 형식으로 구성합니다.
3. PyInstaller Worker, 지정된 CUDA DLL과 Flutter 앱을 자체 포함 번들로 만듭니다.
4. `version.json`, `provenance.json`과 전체 파일 `bundle-manifest.json`을 생성합니다.
5. 모든 구성요소 버전, 필수 파일과 Windows EXE ProductVersion을 검증합니다.

결과는 `artifacts/versions/0.1.2/bixolon-scanner-0.1.2`에 생성됩니다. 설치 PC의 Python이나
signing key 환경 변수에 의존하지 않습니다. Catalog와 Runtime의 모든 파일 checksum이 시작 시
검증되며 손상되면 Worker는 준비 상태가 되지 않습니다. `CHECKSUM-SHA256`은 손상 탐지이며 발행자
진위 인증은 제공하지 않습니다.

### Intel N100 PC용 Windows 인스톨러

NVIDIA GPU가 없는 64비트 Windows N100 PC에는 CPU Worker와 Flutter 앱을 포함한 Setup
EXE를 사용합니다. 빌드 PC에는 Inno Setup 6이 필요합니다. 먼저 CPU handoff와 N100 실측
후보를 만듭니다.

```powershell
.\scripts\build_app.ps1 -Version 0.1.2
.\scripts\build_worker_handoff.ps1 -Version 0.1.2 -ReuseBuildEnvironment
.\scripts\build_n100_test_candidate.ps1 -Version 0.1.2 -Force
```

`artifacts/handoff/n100-0.1.2-candidate-20260824.zip`을 N100 로컬 디스크에 풀고
`RUN-N100-TEST.cmd`를 실행합니다. 반환된 `n100-0.1.2-result.json`을 검토해 N100 장비, 응답 계약,
오류와 평균·p95 1초 목표가 모두 통과하면 `docs/diagnostics/n100-worker-0.1.2.json`으로
보존한 뒤 최종 인스톨러를 만듭니다.

```powershell
.\scripts\build_n100_installer.ps1 -Version 0.1.2
```

N100 결과가 없거나 평균 또는 p95가 1초를 넘으면 인스톨러 빌드는 중단됩니다. 통과한 결과가 있으면
`artifacts/installers/0.1.2/BixolonScanner-N100-0.1.2-Setup.exe`와 외부 `.sha256`,
`installer-manifest.json`이 생성됩니다. 인스톨러는 앱, ONNX Runtime, Runtime/Catalog, Microsoft
Visual C++ x64 Runtime을 포함하므로 대상 PC에 Python, Flutter 또는 CUDA가 필요하지 않습니다.
N100 후보는 이미 비교가 끝난 thread 조합을 다시 반복하지 않고 detector `1×4`, embedder
`4 threads`의 고정 CPU 설정을 측정합니다. 최종 Setup도 같은 CPU `1×4`이며 N100 감지, 응답 계약,
오류 0건과 현장 평균·p95 1초 이내를 모두 확인한 뒤에만 생성합니다.

Setup EXE는 현재 Authenticode 서명이 없으므로 Windows가 알 수 없는 게시자 경고를 표시할 수
있습니다. 전달할 때 `.sha256` 파일을 함께 보내고 대상 PC에서 SHA-256을 확인하십시오. 이 값은
전송 중 손상을 탐지하지만 발행자 진위 인증을 제공하지 않습니다. 지원 OS, 설치·제거와 데이터
보존 범위는 [N100 설치 안내](installer/n100/INSTALL-N100-KO.txt)에 정리돼 있습니다. 기존
`0.0.2` N100 평균은 약 2.05초였고, `0.1.2`의 1초 목표는 새 후보 ZIP의 N100 실측 JSON으로
확인합니다. 어느 수치도 보장된 지연시간이나 SLA로 표현하지 않습니다.

### Flutter 개발자 전달용 CPU Worker

같은 PC에서 `127.0.0.1:8000`으로 연결하는 Windows x64 CPU 패키지는 Python 3.11 격리 환경과
고정 `onnxruntime-openvino` 배포 lock으로 생성하지만 실제 실행 provider는 `CPUExecutionProvider`입니다.
N100 결과가 아직 없으면 provenance에
`PENDING_FIELD_MEASUREMENT`를 기록하고, 결과가 있으면 선택 profile과 측정치를 포함합니다.

```powershell
.\scripts\build_worker_handoff.ps1 -Version 0.1.2
```

결과는 `artifacts/handoff/0.1.2/bixolon-worker-0.1.2-windows-x64-openvino.zip`과 외부
`.zip.sha256` 파일입니다. 압축에는 실행 스크립트, 고정 CPU N100 측정 스크립트, 한국어
`API.md`, JSON Schema, Dart client·DTO 및 상태별 JSON 예시가 포함됩니다. 측정치는 진단 결과로만
취급합니다.

준비된 메타데이터와 생성된 번들을 다시 검증하려면 다음 명령을 사용합니다.

```powershell
bixolon bundle verify --config configs/versions/0.1.2.json
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
configs/versions/               단일 제품 버전 기준
configs/archive/                과거 설정 기록
docs/archive/                   과거 평가·판단·가이드
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
```

구현 경계와 변경 규칙은 [AGENTS.md](AGENTS.md), 문서 목록과 현재 해시는
[`docs/README.md`](docs/README.md)에서 확인할 수 있습니다. 대형 모델·데이터·빌드 산출물은 Git에
커밋하지 않습니다.
