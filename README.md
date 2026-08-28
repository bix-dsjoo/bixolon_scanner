# BIXOLON Bakery AI Scanner

여러 상품이 있는 JPEG/PNG 한 장을 판정하는 Windows 시스템입니다. ONNX Runtime Worker,
PyTorch 학습·평가 도구와 Flutter 작업자 앱을 한 저장소에서 관리합니다.

## 현재 버전

현재 배포 가능한 실행 조합은 `0.1.7` 하나입니다.

| 구성 | 버전 |
|---|---|
| Python 패키지·Worker | `0.1.7` |
| Detector·Embedder·판정 정책 | `0.1.7` |
| Store Catalog | `0.1.7` |
| Flutter 앱 | `0.1.7+10` |
| 사용자 표시·Windows ProductVersion | `0.1.7` |

기준 설정은 [`configs/versions/0.1.7.json`](configs/versions/0.1.7.json)입니다. 모델 graph·weight와
Catalog payload는 고정하고 Runtime·Catalog metadata의 공개 실행 버전만 하나의 제품 버전으로
맞춥니다. 과거 운영 `0.1.5` 계약은 `configs/archive`와 `docs/archive`에 보존합니다.

`0.1.7`은 `0.1.6`의 모델·threshold·판정 순서와 공개 API를 유지하면서 Python 판정/Runtime/Worker
조립과 Flutter 검수/로그 상태의 책임을 분리한 유지보수 patch입니다. Windows OpenVINO 구성과
GPU 초기화 실패 시의 명시적 CPU fallback도 그대로 유지합니다.

2026-08-27 주석 69장과 기존 개발 회귀 415장의 최종 packaged 응답은 `0.1.6`과 status, reason,
bbox, prediction, Top-3, confidence와 version null pattern이 모두 동일했습니다. 이 자료는
개발·비열화 방지 회귀이며 독립 일반화 성능이나 인증이 아닙니다. 상세 내용은
[0.1.7 개발 검증 보고서](docs/evaluation/scanner-0.1.7.md)를 참조하십시오.

## 판정 계약

Worker는 이미지마다 다음 중 정확히 하나를 반환합니다.

- `SEGMENTATION`: 하나 이상의 객체와 객체별 `APPROVED`, `UNKNOWN`+Top-3 또는
  `SEGMENT_RECAPTURE`
- `IMAGE_RECAPTURE`: detector hard gate가 이미지 전체 재촬영을 요구함
- `ERROR`: 입력, 구성, 모델 또는 시스템 오류

`ERROR`를 재촬영으로 변환하지 않습니다. Detector 조기 종료로 classifier를 실행하지 않은 경우
classifier·Catalog 계열 버전은 `null`입니다. 나머지 공개 non-null 버전은 모두 `0.1.7`입니다.
공개 필드와 판정 순서는 [API 계약](docs/contracts/api.md)을 따릅니다.

## Windows 배포물 만들기

Python 3.11, Flutter stable, Visual Studio Windows C++ build tools와 Inno Setup 6이 필요합니다.

```powershell
.\scripts\build_app.ps1 -Version 0.1.7
.\scripts\build_n100_gpu_test.ps1 -Version 0.1.7 -ReuseBuildEnvironment -Force
.\scripts\build_windows_installer.ps1 -Version 0.1.7 `
  -DeviceMatrixPath D:\n100-0.1.5-openvino-device-matrix.json -Force
```

첫 명령은 source manifest와 평가 증빙의 고정 해시를 검증하고 CUDA 포함 Flutter 앱 번들을
`artifacts/versions/0.1.7/bixolon-bakery-ai-scanner-0.1.7`에 만듭니다. 두 번째 명령은 Intel GPU
plugin과 CPU fallback을 포함한 OpenVINO Worker를 구성하며, N100 이름은 장비 진단 패키지에만
사용됩니다. 마지막 명령은 일반 Windows 제품 배포물을 생성합니다.

- `artifacts/installers/0.1.7/BixolonBakeryAIScanner-0.1.7-Setup.exe`
- `artifacts/installers/0.1.7/BixolonBakeryAIScanner-0.1.7-Worker.zip`
- 각 배포물의 `.sha256`과 `installer-manifest.json`

설치 PC에는 Python, Flutter 또는 CUDA가 필요하지 않습니다. Intel GPU가 없거나 초기화에 실패하면
Worker가 경고를 기록하고 CPU verifier·Embedder로 시작합니다. `/health/ready`의 `provider`가
`openvino+openvino_gpu`이면 Intel GPU 구성, `openvino`이면 CPU fallback입니다.

제품명, 설치 표시명과 배포 파일명은 `BIXOLON Bakery AI Scanner`를 사용합니다. `N100`은 성능
측정의 장비명과 provenance에서만 사용하며 프로그램명이나 배포 target 이름으로 사용하지 않습니다.

첨부 N100 100장 진단에서 하이브리드 worker 평균/p95는 `440.767/729.284ms`, CPU-only는
`637.597/1003.835ms`로 각각 `1.443/1.375배` 개선됐습니다. semantic mismatch와 오류는 0건이지만
최대 confidence 차이는 `0.0085055232`이고 p95는 500ms 진단 기준을 넘으므로 `passes=false`입니다.
이 수치는 SLA, 인증 또는 배포 승인으로 해석하지 않습니다.

Setup EXE는 Authenticode 서명이 없습니다. `.sha256`은 전송 중 손상을 확인하지만 발행자 진위를
인증하지 않습니다. 설치·제거와 데이터 보존 범위는
[Windows 설치 안내](installer/windows/INSTALL-KO.txt)에 정리돼 있습니다.

### CPU Worker 전달 패키지

CPU 전용 Flutter 개발자 전달 패키지는 다음 명령으로 별도 생성할 수 있습니다.

```powershell
.\scripts\build_worker_handoff.ps1 -Version 0.1.7
```

결과는 `artifacts/handoff/0.1.7/bixolon-worker-0.1.7-windows-x64-openvino.zip`입니다. 고정 N100
측정 스크립트는 제품명이 아니라 하드웨어 진단 도구로만 포함됩니다.

준비된 metadata와 번들을 다시 검증하려면 다음 명령을 사용합니다.

```powershell
bixolon bundle verify --config configs/versions/0.1.7.json
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

정확도, provider parity, latency와 장기 신뢰성 평가는 선택 진단입니다. 결과가 미실행됐거나 목표를
넘었다는 이유로 별도 promotion·waiver·release-lock 수명주기를 만들지 않습니다.

## 저장소 구조

```text
apps/product_scanner/           Flutter Windows 앱
configs/versions/               활성 제품 0.1.7 설정 하나
configs/archive/                과거 제품·실험 설정 원문
docs/evaluation/                활성 제품 평가와 한계
docs/archive/                   과거 계약·평가·판단·가이드
installer/windows/              일반 Windows Setup·Worker 배포 소스
src/bixolon_scanner/            Python canonical 코드
tests/                          계약과 회귀 테스트
artifacts/, datasets/, runs/    Git 밖의 모델·데이터·빌드 산출물
```

구현 경계와 변경 규칙은 [AGENTS.md](AGENTS.md), 문서 목록은
[docs/README.md](docs/README.md), 로컬 산출물 보존 기준은
[repository-layout.md](docs/maintenance/repository-layout.md)에서 확인할 수 있습니다.
