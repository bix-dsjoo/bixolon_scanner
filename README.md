# Bixolon Scanner

여러 상품이 있는 JPEG/PNG 한 장을 판정하는 Windows 시스템입니다. ONNX Runtime Worker,
PyTorch 학습·평가 도구와 Flutter 작업자 앱을 한 저장소에서 관리합니다.

## 현재 버전

현재 배포 가능한 실행 조합은 `0.0.2` 하나입니다.

| 구성 | 버전 |
|---|---|
| Python 패키지·Worker | `0.0.2` |
| Detector·Embedder·판정 정책 | `0.0.2` |
| Store Catalog | `0.0.2` |
| Flutter 앱 | `0.0.2+2` |
| 사용자 표시·Windows ProductVersion | `0.0.2` |

`0.0.2`는 인증이나 SLA 상태가 아니라 함께 실행되는 파일 조합의 식별자입니다. 앱, Worker,
Runtime과 Catalog 내용이 바뀌어 새 EXE를 배포할 때만 `0.0.2`, `0.0.3`처럼 올립니다. 학습과
평가를 반복하는 동안에는 제품 버전을 바꾸지 않습니다.

기준 설정은 [`configs/versions/0.0.2.json`](configs/versions/0.0.2.json)입니다. 이 번들은
`2.0.1-rc.3`의 ONNX·adapter·support·prototype payload와 판정 정책을 변경하지 않고 모든 실행
구성요소의 표시 버전만 `0.0.2`로 다시 패키징합니다. 원본과 평가 자료의 경로·SHA-256은
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
`0.0.2`입니다. 공개 필드와 판정 순서는 [API 계약](docs/contracts/api.md)을 따릅니다.

## Windows 번들 만들기

Python 3.11 이상 3.14 미만, Flutter stable과 Visual Studio Windows C++ build tools가 필요합니다.

```powershell
.\scripts\build_app.ps1 -Version 0.0.2
```

이 명령은 다음을 한 번에 수행합니다.

1. rc.3 원본 Runtime/Catalog와 평가 증빙의 고정 해시를 검증합니다.
2. 메타데이터 버전만 `0.0.2`로 바꾸고 Catalog를 무키 `CHECKSUM-SHA256` 형식으로 구성합니다.
3. PyInstaller Worker, 지정된 CUDA DLL과 Flutter 앱을 자체 포함 번들로 만듭니다.
4. `version.json`, `provenance.json`과 전체 파일 `bundle-manifest.json`을 생성합니다.
5. 모든 구성요소 버전, 필수 파일과 Windows EXE ProductVersion을 검증합니다.

결과는 `artifacts/versions/0.0.2/bixolon-scanner-0.0.2`에 생성됩니다. 설치 PC의 Python이나
signing key 환경 변수에 의존하지 않습니다. Catalog와 Runtime의 모든 파일 checksum이 시작 시
검증되며 손상되면 Worker는 준비 상태가 되지 않습니다. `CHECKSUM-SHA256`은 손상 탐지이며 발행자
진위 인증은 제공하지 않습니다.

### Flutter 개발자 전달용 CPU Worker

같은 PC에서 `127.0.0.1:8000`으로 연결하는 Windows x64 CPU 전용 패키지는 Python 3.11 격리
환경과 고정 lock으로 생성합니다.

```powershell
.\scripts\build_worker_handoff.ps1 -Version 0.0.2
```

결과는 `artifacts/handoff/0.0.2/bixolon-worker-0.0.2-windows-x64-cpu.zip`과 외부
`.zip.sha256` 파일입니다. 압축에는 실행 스크립트, N100 profile 비교 스크립트, 한국어 `API.md`,
JSON Schema, Dart client·DTO 및 상태별 JSON 예시가 포함됩니다. 수신 측 N100 실측 전 기본 profile은
`1 worker × 4 threads`이며 측정치는 진단 결과로만 취급합니다.

준비된 메타데이터와 생성된 번들을 다시 검증하려면 다음 명령을 사용합니다.

```powershell
bixolon bundle verify --config configs/versions/0.0.2.json
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
