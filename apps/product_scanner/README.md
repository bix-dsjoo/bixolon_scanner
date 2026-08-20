# BIXOLON Scanner Flutter 앱

Windows 작업자가 카메라 또는 JPEG/PNG 이미지에서 검출된 빵을 빠르게 확인하고 확정하는 앱입니다. `스캔` 작업대와 검색 가능한 `활동` 로그 화면을 제공하며 기존 Python Worker의 `POST /v1/scan` 계약을 그대로 사용합니다.

## 개발 환경

- Flutter stable 3.44.9
- Dart 3.12.2
- Visual Studio 2022 Build Tools와 Desktop development with C++ 구성 요소
- `bixolon-worker` 명령이 설치된 Python Worker 환경

## 코드 구조

앱은 feature-first 경계를 사용합니다.

```text
lib/
├─ core/design_system/       토큰, 테마, 카피, 공통 컴포넌트
├─ shared/                   API 모델과 상품 카탈로그
├─ features/scanner/         스캔 domain/application/data/presentation
└─ features/activity/        활동 domain/data/presentation
```

과거 `screens`, `services`, `models`, `widgets` 경로는 기존 import 호환을 위한 export 계층입니다. 새 구현은 canonical feature 경로에 추가합니다.

## 실행

기본 Worker 주소는 `http://127.0.0.1:8000`입니다. Windows 앱을 실행하면 Worker도 숨김 자식 프로세스로 자동 실행되며, 모델 warm-up이 끝나기 전에 분석을 눌러도 readiness가 열릴 때까지 기다립니다. 앱이 직접 시작한 Worker는 앱 종료 시 함께 종료됩니다. 이미 같은 주소에 Worker가 실행 중이면 `/health/ready`의 `worker_version`이 앱 계약 `1.0.0`과 일치할 때만 사용하고, 구버전 Worker에는 스캔을 전송하지 않습니다.

```powershell
cd apps\product_scanner
flutter pub get
flutter run -d windows
```

Worker 주소를 변경하려면 빌드 시 지정합니다.

```powershell
flutter run -d windows --dart-define=SCANNER_API_BASE_URL=http://192.168.0.20:8000
```

원격 Worker를 사용할 때 로컬 자동 실행을 끄려면 앱을 실행하기 전에 다음 환경 변수를 설정합니다.

```powershell
$env:SCANNER_AUTO_START_WORKER = "0"
flutter run -d windows --dart-define=SCANNER_API_BASE_URL=http://192.168.0.20:8000
```

`BIXOLON_PACKAGE_DIR`, `BIXOLON_CATALOG_DIR`, `BIXOLON_PROVIDER`, `BIXOLON_CUDA_DLL_DIR`가 이미 설정돼 있으면 자동 실행 Worker는 그 값을 우선 사용합니다. 설정이 없으면 release 폴더의 `worker\model-package`와 `worker\store-catalog`를 자동으로 사용합니다. Store Catalog는 key·key ID·서명 없이 파일별 SHA-256을 검증합니다. `worker\cuda-runtime`이 포함된 bundle은 `cuda` provider를 강제해 불완전한 GPU 배포가 CPU로 조용히 전환되지 않게 합니다. CUDA runtime이 없는 개발 빌드는 기존처럼 `auto` provider를 사용합니다.

## 검증과 빌드

```powershell
flutter analyze
flutter test
cd ..\..
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_worker.ps1
$cudaRuntime = "C:\path\to\CUDA-13-cuDNN-9-runtime"
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build_app_release.ps1 `
  -Composition configs\releases\scanner_2.0.0.json `
  -CudaRuntimeDirectory $cudaRuntime
```

`build_worker.ps1`은 학습용 PyTorch를 제외하고 FastAPI와 ONNX Runtime만 포함한
`artifacts\worker\bixolon-worker` 독립 실행본을 만듭니다. Scanner 2.0 Release는 승격 레지스트리에
잠긴 `artifacts\releases\scanner-2.0.0-production\worker-build\bixolon-worker`를
`worker` 폴더에 복사하므로 설치 PC의 Python 또는 전역 `bixolon-worker` 설치에 의존하지
않습니다.

CUDA 13·cuDNN 9 기반 ONNX Runtime GPU Release를 만들 때는 재배포 가능한 DLL 디렉터리를 CMake cache 또는 환경 변수로 지정한 뒤 빌드합니다. 해당 디렉터리는 `cublas64_13.dll`, `cublasLt64_13.dll`, `cudart64_13.dll`, cuDNN 9 component DLL, `cufft64_12.dll`, `nvJitLink_130_0.dll`, `nvrtc64_130_0.dll`, `nvrtc-builtins64_130.dll`, `zlibwapi.dll`을 포함해야 합니다.

```powershell
$cudaRuntime = "C:\path\to\CUDA-13-cuDNN-9-runtime"
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build_app_release.ps1 `
  -CudaRuntimeDirectory $cudaRuntime
```

성공한 Release는 `worker\bixolon-worker.exe`, `worker\model-package`, `worker\store-catalog`, `worker\cuda-runtime`을 포함하고 `/health/ready`의 `provider`가 `cuda`여야 합니다. CUDA DLL이 불완전하면 readiness를 열지 않으며 CPU full-path로 조용히 실행하지 않습니다.

대표 운영 상태는 `test/goldens` 이미지로도 회귀 검증합니다. 골든은 시각 비평 후 의도한 변경일 때만 `flutter test test/scanner_screen_test.dart --update-goldens`로 갱신합니다.

Flutter 원본 Release 결과는 `build\windows\x64\runner\Release`, 전체 파일 manifest로 잠긴 배포본은
`artifacts\releases\bixolon-scanner-2.0.0+4`에 생성됩니다.
운영 Runtime·Catalog와 기대 버전은 release composition manifest에서 읽는다. Release 빌드는 composition, Worker, Runtime, checksum-only Catalog 또는 CUDA runtime 파일이 하나라도 없으면 실패하며 전체 bundle file manifest를 생성한다. 앱 `2.0.0+4`는 `/health/ready`에서 Worker·Detector·Classifier `2.0.0`을 모두 검사하고, `DETECTOR_CONTAINED_DUPLICATE`를 재촬영으로 바꾸지 않은 채 중복 박스와 Top-3 검토로 안내한다. 모델 바이너리는 Git에 커밋하지 않는다.
재현 입력은 `configs\releases\scanner_2.0.0.json`, production 근거는 `artifacts\releases\scanner-2.0.0-production\promotion-attestation.json`이다.

## 로컬 데이터

MVP에는 별도 검색·저장 서버 API가 없으므로 다음 기능을 앱 내부에서 처리합니다.

- 상품 표시·검색: `assets/catalog/bread_ko.json`의 버전 관리되는 20개 카탈로그에서 사용자에게는 한국어 `display_name_ko`를 표시하고, 한국어 표시명·모델 원본 영문 `class_name`·`class_id`를 모두 검색
- Scan Log: Windows application support 디렉터리의 `ProductScanner\scan_logs`에 원본 이미지와 JSON을 함께 저장
- 박스 미검출 피드백: `ProductScanner\feedback_logs\missed_object`에 원본 이미지와 검수 대기 JSON을 일반 Activity와 분리해 저장

Scan Log v3는 Worker 상태와 최상위 `reason_codes`에 더해 각 detection의 `reason_codes`와 Worker/Detector/Classifier 버전을 보존합니다. 일반 `APPROVED`·`UNKNOWN` 기록은 최종 확정 때 자동 저장하고, `RECAPTURE`는 결과 화면의 Outlined `재촬영 기록 저장`을 실행한 경우에만 빈 `detections`와 함께 저장합니다. 저장 중에는 중복 입력을 막고 성공 뒤에도 현재 이미지와 재촬영 안내를 유지합니다. 같은 Scan ID는 한 세션에서 한 번만 저장하며 실패하면 판정과 이미지를 보존한 채 `다시 저장`할 수 있습니다. v1·v2 로그는 기존 필드만으로 계속 호환 로드하며, 현재 로그 폴더 안의 안전한 `original_image` 파일만 해석합니다.

Worker가 `APPROVED` 또는 `UNKNOWN`을 반환했지만 실제 상품 박스가 빠진 경우 결과 하단의 `박스 미검출 기록`을 실행합니다. 앱은 Worker의 `worker_status`와 `reason_codes`, 검출된 bbox를 변경하지 않고 `operator_feedback.type=MISSED_OBJECT`, `expected_status=RECAPTURE`, `expected_reason=DETECTOR_MISSED_OBJECT`, `annotation_status=PENDING_BBOX_CLASS_REVIEW`를 별도로 기록합니다. 앱 실행마다 생성되는 `capture_session_id`도 모든 새 로그에 포함되므로 촬영 조건을 바꿀 때 앱을 재시작하면 세션 단위 분리가 가능합니다. 이 피드백은 Activity 확정 기록으로 취급하지 않으며, 학습에 넣기 전에 누락 bbox와 class overlay 검수·승인을 받아야 합니다.

상단 `활동`에서는 최근 100건을 한국어 상품명·기존 영문 상품명·class ID·Scan ID·reason code로 검색하고 입력원·기간·정렬 필터와 상세 판정을 확인할 수 있습니다. 각 행은 저장 이미지의 44px 지연 디코딩 썸네일을 표시하고, 상세는 최대 240px 미리보기를 제공합니다. 이미지가 없거나 손상되었거나 로그 폴더 밖 경로를 가리키면 경로를 노출하지 않고 같은 크기의 대체 상태를 유지합니다. `RECAPTURE` 행은 상품 요약 대신 공통 reason-code 표현기의 한국어 원인과 재촬영 배지를 표시합니다. 이전 앱에서 영문 상품명으로 저장된 로그도 현재 카탈로그와 일치하는 항목만 화면에서 한국어로 표시하며, 제거된 과거 상품명은 원문을 유지합니다. 상품 또는 확정 방식이 없는 구버전 기록은 내부 `Unknown` 값을 노출하지 않고 `상품 정보 없음`·`확정 방식 확인 불가`로 안내합니다. 여러 상품 중 영문명이나 class ID로 검색해도 실제 일치한 한국어 상품을 목록 요약의 첫 항목으로 표시합니다. 목록과 상세는 상품·수정 여부 또는 재촬영 원인을 먼저 보여주고 처리시간·item ID·confidence·확정 방식·raw reason code는 접힌 진단 정보에서 제공합니다. 진단 정보는 60px 헤더와 접힘 상태 Semantics를 제공하며 Tab 포커스 후 Enter·Space로 열 수 있습니다.

상품 결과 헤더에는 `n/n개 확인 · 분석 72.1 ms`, `RECAPTURE` 헤더에는 `분석 72.1 ms` 형식으로 Worker 응답의 `processing_time_ms`를 표시합니다. 이 값은 클라이언트 왕복시간이 아니라 Worker 내부의 이미지 디코딩·전처리·detector·필요한 classifier·후처리를 포함한 처리시간입니다. 응답이 없는 `ERROR`에는 시간을 임의로 만들지 않습니다. 이번 앱 변경은 `DETECTOR_UNCERTAIN_OBJECT` 가드나 모델 임계값을 완화하지 않으며, 수동 저장한 1~4개 실제 장면을 라벨링·평가한 뒤 별도 모델 버전과 KPI 검증으로 보정합니다.

Windows 카메라는 Flutter의 `camera_windows` 구현을 사용합니다. 플러그인의 미러 프리뷰만 수평 보정하고 저장 이미지와 Bounding Box는 원본 좌표를 유지합니다. 카메라가 없거나 권한이 거부되어도 이미지 파일 분석은 계속 사용할 수 있습니다.

UI는 번들된 Pretendard Variable을 사용합니다. 글꼴은 `assets/fonts/OFL.txt`의 SIL Open Font License 1.1 조건을 따릅니다.

## UI 기준

- 상세 토큰, 컴포넌트, 접근성 계약은 [`DESIGN_SYSTEM.md`](DESIGN_SYSTEM.md)를 따릅니다.
- 누적 검수와 변경 과정은 [디자인 시스템 개선 이력](docs/DESIGN_SYSTEM_HISTORY.md)에 보존하며 현재 계약을 대체하지 않습니다.
- 토스플레이스의 [POS](https://tossplace.com/product/pos), [키오스크](https://tossplace.com/product/kiosk), [Front Template API](https://docs.tossplace.com/reference/plugin-sdk/front/template.html)에서 확인한 단일 핵심 행동, 큰 선택 영역, 표준 선택 화면, 단계적 정보 공개 패턴을 운영 검수 흐름에 맞게 적용합니다.
- BIXOLON Orange `#EE7203`을 선택과 Primary action에 사용하고 Orange 배경의 텍스트는 Ink `#171717`을 사용합니다.
- Pretendard의 화면 위계는 `AppTypography`의 `12/18·13/20·14/20·15/20·16/24·18/26·24/32`, `400·500·600·700` 굵기와 역할별 tracking만 사용하며 화면에서 글자 값을 직접 만들지 않습니다.
- 선택 가능한 객체·상품·활동은 왼쪽 띠 대신 전체 Brand Soft 면과 사방 Brand 외곽선을 사용합니다. 확정된 값은 radio/check, 현재 상세 대상은 화살표로 구분합니다. Activity의 컴팩트 필터는 폭을 흔드는 표식 대신 전체 Brand 외곽선과 Bold 레이블을 함께 사용합니다.
- Windows 마우스 hover는 공통 `rowHover` 면을 사용하며 Selected 면을 덮지 않습니다. 키보드 포커스는 선택 행·Activity 필터·정렬 드롭다운과 새로고침·이전/다음·검색 지우기·검색 닫기 같은 48px 아이콘 행동까지 Surface에서 3:1 이상인 Deep Orange `#D96500` 사방 2px 링으로 표시합니다. 버튼은 면을 바꾸지 않고 Outlined·Text에 Focus Ring, Orange Filled Primary에는 대비가 명확한 Ink 2px 경계를 사용합니다.
- 다중 검수의 고정 CTA는 `남은 상품 확인 필요 → n개 상품 최종 확정 → 저장 중`으로 같은 위치에서 전환되며, 모든 상품이 확인된 뒤에만 활성화됩니다. 저장 중에는 재실행을 막고 보조기기에 `저장 중. 완료될 때까지 기다려 주세요`를 live region으로 한 번 안내합니다. 로컬 저장 실패 시에는 확인 결과와 키보드 포커스를 유지하고 오류 문구를 live region으로 알린 뒤 같은 위치의 단일 `다시 저장` Primary로 복구합니다. Enter 재시도 성공 후에는 실패 CTA를 제거하고 완료 live region을 한 번 안내하며 다음 Tab 탐색을 새 작업에서 계속할 수 있습니다.
- 미완료 검수에서는 현재 `n번 상품을 확인해 주세요` 안내 하나만 발화합니다. 자동 승인 또는 마지막 상품 확인으로 모든 검수가 끝나면 화면 공간이나 포커스 정지점을 추가하지 않고 `n개 상품 확인 완료. 최종 확정할 수 있어요.`를 단일 live region으로 알려 다음 행동을 연결합니다.
- 저장 완료 토스트는 상단 셸 아래의 전역 레이어에서 3초 동안 표시되어 스캔과 Activity 사이를 이동해도 유지됩니다. pointer 입력을 가로막지 않고 단일 live region으로 완료를 안내하며 새 완료가 생기면 유지 시간이 다시 시작됩니다.
- 저장에 성공하면 완료한 입력원을 그대로 유지합니다. 이미지 작업은 카메라 연결 단계로 돌아가지 않고 이미지 전용 빈 미리보기와 `다음 이미지를 선택해 주세요` 안내, 단일 `이미지 선택` Primary를 제공하며 `Ctrl+O`로 다음 파일을 바로 고를 수 있습니다. 키보드로 최종 확정 또는 저장 재시도를 완료했다면 포커스도 `이미지 선택`으로 이어져 Enter로 다음 파일을 열 수 있고, 저장 중 다른 컨트롤로 이동한 경우에는 포커스를 빼앗지 않습니다. 카메라 작업 완료와 명시적인 세션 초기화는 기존 카메라 입력 대기 상태를 유지합니다.
- UNKNOWN의 Top-3와 전체 상품 검색 결과는 같은 60px 전체 선택 행으로 표시합니다. 상품 검색은 상품명 또는 class ID를 지원하고 결과 없음에는 검색어 변경 방법을 안내합니다. 검색 inspector는 `n번 상품 검색/변경`과 이미 확정된 `현재 상품 이름 · class ID`를 유지하고, 현재 상품이 검색 결과에 있으면 선택 radio·Brand 면·외곽선·Semantics `selected`로 표시합니다. 검색 결과는 `n개` 또는 `상위 8 / n개` 수량과 상시 스크롤바를 제공해 화면 밖 결과를 알 수 있게 합니다. Top-3와 검색 결과는 그룹마다 현재 선택 결과를 우선한 탐색 행 하나만 Tab 순서에 두고 `↑/↓/←/→`로 포커스만 순환하며, Enter·Space 또는 클릭으로만 상품을 확정합니다. 화면 밖 결과로 이동하면 포커스 카드 전체가 자동으로 나타나고 검색 입력으로 돌아갔다가 Tab을 눌러도 같은 탐색 행을 복원합니다. 객체·Top-3/검색 결과·Activity 목록은 왼쪽 띠 없이 전체 면·사방 외곽선·radio/check 또는 화살표로 선택을 표시하고, 보조기기에는 상호 배타적인 단일 선택 그룹과 `selected` 상태로 제공합니다. 다중 객체 목록은 현재 선택 행 하나만 Tab 순서에 두며, 행 포커스의 `↑/↓`는 선택·검수 상세·포커스를 함께 이동합니다. 확인 필요 객체에서 Tab은 첫 Top-3 후보로 이어지고 Shift+Tab은 같은 객체 행으로 돌아오며, 검색 결과의 Shift+Tab은 검색 입력으로 돌아옵니다. 후보 선택·목록 클릭·방향키 이동 뒤에는 현재 검수 행이 결과 목록의 보이는 영역으로 자동 이동합니다. 선택된 다중 객체가 있으면 결과 헤더의 `이전 | n/전체 | 다음` 단계 탐색기로 위치를 함께 표시하고, 모든 상품 확인 후에는 숨겨 최종 확정 CTA에 집중시킵니다.
- 후보 또는 전체 상품 검색 결과를 Enter·Space로 확정하면 다음 미확정 상품의 첫 후보로 포커스를 이어가고, 마지막 상품이면 `최종 확정` CTA로 이동합니다. 이미 확정한 동일 상품을 다시 선택하면 상품과 기존 `confirmation_method`를 변경하지 않아 실제 수정이 아닌 기록을 만들지 않습니다. 검색 입력의 `Esc`와 뒤로 아이콘의 Enter·Space는 현재 선택을 보존한 채 검색을 닫고 `다른 상품 검색/상품 변경` 행동으로 포커스를 복원합니다. Top-3가 없는 상품의 뒤로 아이콘은 `검색 닫기`로 안내하며, 마우스 클릭은 포커스를 강제로 옮기지 않아 pointer와 키보드의 기대를 분리합니다.
- 저장 중에는 검수 상세 선택지를 숨기고 객체 목록·프리뷰 box·이전/다음 이동을 disabled 표면과 Semantics로 잠급니다. 클릭·방향키가 저장 snapshot 뒤의 상품이나 선택 객체를 바꾸지 않으며, 진행 `저장 중`만 현재 핵심 행동으로 남습니다.
- 확인한 상품 선택을 버릴 수 있는 이미지 교체·재촬영은 400px 확인 대화상자에서 손실 대상을 설명하고 `취소 | 실행 동사`를 같은 48px 행동으로 제공합니다. 안전한 취소가 첫 포커스이며 Tab·Enter로 결정하고 Esc로 변경 없이 닫을 수 있습니다.
- 선택 객체의 Top-3 또는 상품 검색은 해당 객체 행 바로 다음에 이어집니다. 객체가 많으면 검수 영역 `340px`를 먼저 확보하고 객체 목록만 스크롤하여 현재 행·선택지·`다른 상품 검색`을 최소 작업대에서도 함께 볼 수 있게 합니다.
- 프리뷰의 선택 box는 항상 최상단의 `n 현재 검수` Brand 표식으로 표시합니다. 작은 box도 44px 선택 영역을 보장하며, 밀집된 비선택 box는 번호와 `?`로 축약해 이미지 가림을 줄입니다. 모든 활성 box는 직접 포커스와 Enter·Space 선택을 지원하고, 선택 Orange와 구분되는 Deep Orange 2px 사방 포커스 링을 표시합니다. 일반 Tab 순서에는 현재 선택 Box 하나만 포함해 객체 수만큼 정지점이 반복되지 않습니다. Preview Box에 포커스한 상태에서 `↑/↓`로 객체를 이동하면 포커스와 Tab 대상도 새 `현재 검수` Box를 따라가며, 다른 화면 영역의 포커스는 강제로 가져오지 않습니다.
- 프리뷰 좌상단 칩은 단계 문구를 반복하지 않고 `라이브 카메라·촬영 이미지·선택한 이미지·카메라 미리보기` 중 입력 출처를 표시합니다. 패널 헤더는 현재 단계, 본문은 안내, 고정 CTA는 다음 행동만 담당하며 분석 프리뷰의 진행 표시는 중복 Semantics를 만들지 않습니다.
- 라이브 카메라와 촬영·선택 이미지 프리뷰에는 흰색 촬영 범위, 중앙 X, BIXOLON 대각선 가이드를 올리지 않습니다. 촬영·분석 진행 피드백과 결과 bounding box만 기능적 오버레이로 유지합니다.
- 첫 진입은 `카메라 확인 중 → 입력 준비 → 촬영 준비`로 전환되며, 카메라 연결 성공 시 `이미지 선택` Secondary와 `촬영하기` 단일 Primary를 제공합니다. 초기화·재연결 실패는 `카메라를 사용할 수 없어요`, 연결 뒤 촬영 호출 실패는 `촬영하지 못했어요`로 원인을 구분하고 Primary를 `다시 연결`로 교체합니다. 이전 `RECAPTURE` 또는 `ERROR`에서 재촬영하다 실패해도 현재 카메라 복구 상태를 우선합니다. 재연결 진행 중에는 전 표면을 `카메라 확인 중`으로 유지하고 진행 `연결 확인 중` CTA와 live region으로 중복 실행을 막으며, 이때 실제로 사용할 수 있는 `이미지 선택`은 유지합니다. 성공 후에는 보존된 판정과 `다시 촬영` 행동으로 돌아갑니다. 이 로컬 복구 상태는 Worker의 `ERROR` 또는 모델의 `RECAPTURE`로 변환하지 않습니다. 이미지 선택 후에는 `분석 준비 → 분석 중`으로 헤더·본문·고정 CTA가 함께 전환되고, 분석 중에는 실행할 수 없는 이미지 교체를 숨겨 진행 `분석 중` 하나만 표시합니다.
- 카메라 연결 성공은 상단 `카메라 연결됨` 배지를 단일 live region으로 안내합니다. 확인 중·미연결·확인 필요 배지는 자동 발화하지 않으며 실패 시에는 본문의 구체적인 원인과 복구 행동 하나만 안내해 상태 중복을 막습니다.
- `Ctrl+O`는 스캔과 Activity에서 화면의 이미지 선택 행동과 같은 상태 계약을 사용합니다. 분석·저장 중에는 파일 선택과 변경 확인 대화상자를 열거나 키를 처리한 것으로 가장하지 않고, 실제 대체 입력이 가능한 카메라 연결 확인 중에는 계속 사용할 수 있습니다.
- Activity는 목록 헤더에서만 결과 수를 표시하며, 검색어 지우기와 `모두 초기화`를 제공합니다. 검색이 비어 있으면 실제 `/` 키캡으로 단축키를 안내하고 검색 필드를 첫 Tab 위치로 유지합니다. 조건에 맞는 기록이 없으면 검색·필터 편집은 유지하되 상단의 중복 초기화를 숨기고 중앙 48px `모두 초기화` 하나만 Primary로 제공합니다. 입력원·기간 필터는 그룹마다 현재 선택값 하나만 Tab 순서에 두고 `←/→/↑/↓`로 값을 순환 선택해 7개 칩을 반복 탐색하지 않으며, 선택·포커스·결과를 함께 갱신합니다. 필터의 중복 값은 보조기기에서 `입력원, 전체`·`기간, 전체`처럼 그룹 문맥과 함께 읽습니다. 여러 상품 기록은 `첫 상품 외 n개`로 요약하고 검색 중에는 일치한 상품을 먼저 보여줍니다. 목록의 마지막 열은 기술적인 처리시간 대신 `자동 확정/n개 수정` 검수 결과를 표시합니다.
- Activity의 이미지 증거는 목록에서 44px 썸네일, 상세에서 최대 240px 미리보기로 단계적으로 공개합니다. 썸네일은 제한된 decode width로 지연 디코딩해 최근 100건 원본을 한꺼번에 메모리에 올리지 않습니다. `RECAPTURE`는 공통 한국어 원인과 재촬영 배지를 사용하고 Scan ID·상품명뿐 아니라 raw reason code와 한국어 원인으로도 검색할 수 있습니다.
- Activity 기록 목록은 현재 선택 행 하나만 Tab 순서에 둡니다. 행에 진입하면 `↑/↓`로 이전·다음, `Home/End`로 처음·끝, `Page Up/Page Down`으로 현재 viewport 한 페이지씩 이동하며 선택·상세·포커스와 다음 Tab 정류장을 함께 갱신합니다. 최대 100건에서도 선택 행을 60px 경계에 맞춰 자동 스크롤하고 한 행의 문맥을 겹쳐 보여줍니다. 선택 행의 Tab은 오른쪽 상세 `진단 정보`로 이동하고 Shift+Tab은 같은 선택 행으로 돌아와 작업 순서를 대칭으로 유지합니다. 검색 입력 중에는 목록 이동키를 텍스트 편집에만 사용합니다.
- Activity 저장소는 앱 시작과 카메라 준비를 방해하지 않도록 첫 Activity 방문에서만 읽고, 한 번 생성한 화면은 `IndexedStack`에 유지합니다. 최초 로드는 검색·필터·정렬을 숨기고 `활동 기록을 불러오는 중이에요` 진행 상태만 표시하며 데이터가 도착한 뒤 도구를 공개합니다. 장기 로드 중 스캔 화면으로 이동해도 같은 요청을 한 번만 완료하고 재진입 시 결과와 `F5` 범위를 즉시 복원합니다. 비활성 중 실패했다면 이전 오류를 한 프레임 노출하지 않고 재진입 시 바로 다시 불러옵니다. 저장된 Activity가 없으면 `스캔 화면으로 이동`만 Primary로 제공합니다. 저장 맥락이 없는 최초 로드 오류는 Error Red의 `활동 기록을 불러오지 못했어요`와 `새로고침`만 노출하고, 저장 직후 자동 동기화 실패의 Attention 안내와 혼용하지 않습니다. 기존 기록이 있는 비차단 새로고침 오류는 현재 목록을 유지합니다.
- Activity 새로고침은 기존 목록과 상세를 유지한 채 48px 아이콘 위치에서 진행 상태를 표시합니다. Tab 포커스에는 사방 2px Focus Ring을 제공하고, 진행 중에는 `활동 기록 새로고침 중` live region과 disabled 상태로 중복 실행을 막습니다. 기존 기록이 있는 새로고침 실패는 비차단 알림의 `새로고침` 하나로 복구하며 같은 툴바 아이콘은 오류가 유지되는 동안 숨깁니다. 키보드 실행이면 실패 시 인라인 재시도, 성공 시 원래 툴바 아이콘으로 포커스를 이어가되 진행 중 사용자가 다른 컨트롤로 이동하면 포커스를 되가져오지 않습니다. `F5`는 계속 사용할 수 있고 재시도 시작 시 알림을 닫은 뒤 원래 툴바 위치의 진행 상태로 전환합니다. 정렬 후에도 선택 행을 자동으로 화면 안에 유지하고 긴 목록의 자동 이동은 60px 행 경계에 맞춰 고정 헤더 아래에 잘린 기록 조각을 남기지 않습니다.
- 새 스캔 저장에 성공하면 Activity 데이터 revision과 최신 저장 Scan ID가 갱신됩니다. Activity를 이미 열어 둔 세션에서도 다음 진입 시 최근 100건을 한 번만 자동 갱신하고 방금 저장한 기록을 선택해 목록과 상세를 연결하며, 단순 화면 전환만으로는 다시 로드하지 않습니다. 저장 뒤 Activity를 처음 방문하거나 최초 로드 중 저장이 완료된 경우에도 현재 요청을 중복하지 않고 완료한 뒤 최신 revision을 한 번 재동기화해 최신 Scan ID를 선택합니다. 이때 화면의 로딩 레이블은 유지하되 저장 완료 토스트와 경쟁하는 live 발표는 만들지 않습니다. 자동 갱신 중에도 검색어·입력원·기간·정렬은 유지되고 검색 조건이 새 기록을 제외하면 조건을 임의로 바꾸지 않습니다. 자동 동기화가 실패하면 저장 성공과 화면 갱신 실패를 분리한 주의 안내를 표시하고 기존 목록을 유지하며, 재시도 성공 시 보존한 최신 저장 Scan ID를 선택합니다. 수동 `F5` 새로고침은 기존 선택과 진행 live region을 유지합니다.
- 상단 셸과 패널 헤더는 `60px`입니다. 상단 셸은 Workspace 배경과 Orange 기준선 위에 `105×30px` 공식 BIXOLON SVG, 아이콘 없는 `112×52px` `스캔·활동` 돌출 탭, 카메라 상태 배지를 두고 선택·포커스 Semantics을 유지합니다. 섹션·표 헤더는 `40px`, 일반 컨트롤은 최소 `44px`, Primary action은 `48px`, 목록 행은 `60px`입니다. 확인 대화상자 `400px`, 빈 상태 최대 폭 `360px`, 진단 레이블 `88px`, 포커스 링 `2px`, 선택 외곽선 `1.5px`까지 `AppDesignTokens`를 직접 사용합니다.
- 공통 컴포넌트와 화면의 패딩·간격은 `AppSpacing`의 `4·8·12·16·24·32px` 역할만 사용하며 Theme 버튼·입력 패딩도 같은 토큰에 연결합니다.
- 로고·탭·상태 배지, 프리뷰 출처 칩 최소 높이, inline·preview·page 진행 표시 크기는 `AppDesignTokens`로 관리합니다. 상태 배지와 출처 칩은 고정 높이에 텍스트를 가두지 않고 큰 글자에서 최소 높이 위로 확장합니다.
- 상태·패널 전환은 `AppMotion`의 `160ms·240ms`와 `easeOutCubic` 곡선을 사용합니다. 시스템의 reduced motion 설정에서는 전환 시간을 제거하고 자동 스크롤을 점프 방식으로 바꾸며, 회전형 진행 표시는 같은 크기의 정적 대기 아이콘으로 대체하되 상태 문구와 live region은 유지합니다.
- 스캔 작업대는 프리뷰와 결과 패널을 약 `64:36`으로 분할하며 결과 패널은 `440~520px`를 유지합니다. 비율·최소/최대 폭, 검수 영역 높이, Activity 검색창 폭과 적층 전환점은 `AppDesignTokens`·`AppBreakpoints`에서 일관되게 관리합니다.
- Windows 창은 현재 모니터 DPI를 반영한 최소 `1280×720` Flutter 작업 영역 아래로 축소되지 않습니다. 기본 크기와 주요 검증 작업 영역은 `1440×900`입니다.
- 화면 헤더는 제공된 공식 BIXOLON SVG 워드마크를 그대로 사용합니다. Windows 제목 표시줄·작업 표시줄·실행 파일은 기존 Orange 포커스 마크를 유지하며 `python tool/generate_windows_icon.py --check`로 16–256px ICO 리소스의 최신 상태를 검증합니다.
- `1280×720`에서 125%·150% 큰 글자와 Windows 150% 표시 배율을 회귀 검증합니다.
- `1280×720`과 `1440×900` 각각에서 준비, 분석 중, `APPROVED`, `UNKNOWN`, `RECAPTURE`, `ERROR`, 저장 중, 완료, Activity 목록·상세·빈·오류 상태의 레이아웃 매트릭스를 실행합니다. 각 화면은 오버플로가 없어야 하고 Filled Primary는 최대 하나이며 실제 높이는 48px 이상이어야 합니다.
- 같은 운영 상태 매트릭스는 현재 Semantics 트리의 모든 활성 tap 노드를 순회해 폭과 높이가 각각 최소 44px이고, 작업대 안에 완전히 포함되며, `label` 또는 `hint`로 접근 가능한 이름과 키보드 포커스 상태를 제공하는지 확인합니다. 시각 크기만 큰 장식 영역이나 tap action이 제거된 진행·비활성 상태를 조작 가능 영역으로 계산하지 않습니다.
- 색상에만 의존하지 않고 아이콘과 문구로 `APPROVED`, 확인 필요, `RECAPTURE`, `ERROR` 상태를 구분합니다. `ERROR`·`RECAPTURE`·카메라 문제·Activity 빈/오류 상태는 제목과 해결 설명을 하나의 live region으로 알리고, 인접한 복구 CTA는 별도 버튼 노드로 유지합니다.
- 입력 이미지 `ERROR`는 다른 입력 선택, 연결·서버 `ERROR`는 현재 이미지 재분석을 Primary로 제공하며 모델 판정인 `RECAPTURE`와 서로 변환하지 않습니다.
- 오류·재촬영 본문과 CTA는 `AppActionCopy`의 같은 실제 행동 동사를 사용합니다. 서버·시간 초과는 `다시 분석`, Activity 로드는 `새로고침`, 카메라 재촬영은 `다시 촬영`, 이미지 재촬영 판정은 `다른 이미지 선택`, 로컬 저장 실패는 `다시 저장`으로 안내하며 제출 진행 CTA는 `저장 중`으로 표시합니다.
