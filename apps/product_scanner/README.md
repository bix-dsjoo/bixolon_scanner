# Product Scanner Flutter 앱

Windows 작업자가 카메라 또는 JPEG/PNG 이미지에서 검출된 빵을 빠르게 확인하고 확정하는 앱입니다. `Scan` 작업대와 검색 가능한 `Activity` 로그 화면을 제공하며 기존 Python Worker의 `POST /v1/scan` 계약을 그대로 사용합니다.

## 개발 환경

- Flutter stable 3.44.9
- Dart 3.12.2
- Visual Studio 2022 Build Tools와 Desktop development with C++ 구성 요소
- 실행 중인 Bixolon Scanner Worker

## 실행

기본 Worker 주소는 `http://127.0.0.1:8000`입니다.

```powershell
cd apps\product_scanner
flutter pub get
flutter run -d windows
```

Worker 주소를 변경하려면 빌드 시 지정합니다.

```powershell
flutter run -d windows --dart-define=SCANNER_API_BASE_URL=http://192.168.0.20:8000
```

## 검증과 빌드

```powershell
flutter analyze
flutter test
flutter build windows --release
```

Release 결과는 `build\windows\x64\runner\Release`에 생성됩니다.

## 로컬 데이터

MVP에는 별도 검색·저장 서버 API가 없으므로 다음 기능을 앱 내부에서 처리합니다.

- 상품 검색: `assets/catalog/bread_ko.json`의 버전 관리되는 20개 카탈로그에서 모델의 원본 영문 `class_name`과 `class_id`를 검색
- Scan Log: Windows application support 디렉터리의 `ProductScanner\scan_logs`에 원본 이미지와 JSON을 함께 저장

Scan Log는 최초 AI 판정, Top-3와 confidence, 최종 상품, 사용자 수정 여부 및 확정 방식을 분리해 보존합니다. 상단 `Activity`에서 최근 100건을 상품명 또는 Scan ID로 검색하고 상세 판정을 확인할 수 있습니다. 저장에 실패하면 화면을 초기화하지 않습니다.

Windows 카메라는 Flutter의 `camera_windows` 구현을 사용합니다. 플러그인의 미러 프리뷰만 수평 보정하고 저장 이미지와 Bounding Box는 원본 좌표를 유지합니다. 카메라가 없거나 권한이 거부되어도 이미지 파일 분석은 계속 사용할 수 있습니다.

UI는 번들된 Pretendard Variable을 사용합니다. 글꼴은 `assets/fonts/OFL.txt`의 SIL Open Font License 1.1 조건을 따릅니다.

## UI 기준

- 기본 모서리 반경은 `4px`, 경계선은 `1px`로 통일합니다.
- 상단 셸은 `52px`, 일반 컨트롤은 `36px`, 목록 행은 `48~52px`를 사용합니다.
- Scan 작업대는 카메라/이미지와 결과 패널을 약 `66:34`로 분할하고, Activity는 로그 목록과 상세 인스펙터를 `7:5`로 분할합니다.
- 선택과 Primary action에만 파란색을 사용하며, 초록색·주황색·빨간색은 각각 성공·확인 필요·오류 상태에만 사용합니다.
- 카드 중첩과 장식용 배지를 사용하지 않고, 툴바 → 목록 → 상세의 3단계 계층을 1px 경계로 구분합니다.
