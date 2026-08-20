# 개발 가이드

## Python

Python `3.11` 이상 `3.14` 미만을 지원합니다.

```powershell
python -m pip install -e ".[cpu,test,dev]"
bixolon --help
bixolon worker
```

새 작업은 통합 CLI의 `worker`, `data`, `train`, `evaluate`, `model`, `experiment`, `operations`, `tools` 그룹을 사용합니다. 기존 `bixolon-*` 명령은 호환 alias입니다.

설정의 canonical 위치는 `configs/runtime`, `training`, `operations`, `experiments`입니다. 루트 JSON은 `$redirect`만 가지며 공통 loader가 대상 누락과 순환을 거부합니다. `.env.example`은 환경 변수 도구가 그대로 읽을 수 있도록 redirect가 아닌 일반 env 형식을 유지합니다.

## Flutter

```powershell
cd apps\product_scanner
flutter pub get
flutter run -d windows
```

Flutter `3.44.9`, Dart `3.12.2`를 기준으로 합니다. Golden은 의도한 시각 변경을 검수한 경우에만 갱신합니다.

## 통합 검증

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify.ps1
```

스크립트는 Ruff check와 format check, 전체 Python 테스트, Flutter analyze와 test, `git diff --check`를 실행합니다. CUDA parity, RTX 5080 benchmark, 원본 데이터 KPI는 CI가 아닌 수동 승격 gate입니다.

구조를 바꿀 때는 공개 API·판정 우선순위·기존 CLI와 import 호환성을 먼저 회귀 테스트하고, 모델 정책 변경을 같은 커밋에 섞지 않습니다.
