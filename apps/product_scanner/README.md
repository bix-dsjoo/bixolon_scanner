# Product Scanner Flutter 앱

Windows 작업자가 카메라 또는 JPEG/PNG 이미지에서 검출된 빵을 빠르게 확인하고 확정하는 단일 화면 앱입니다. 기존 Python Worker의 `POST /v1/scan` 계약을 그대로 사용합니다.

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

- 상품 검색: `assets/catalog/bread_ko.json`의 버전 관리되는 20개 한글 상품 카탈로그를 `class_id` 기준으로 검색
- Scan Log: Windows application support 디렉터리의 `ProductScanner\scan_logs`에 원본 이미지와 JSON을 함께 저장

Scan Log는 최초 AI 판정, Top-3와 confidence, 최종 상품, 사용자 수정 여부 및 확정 방식을 분리해 보존합니다. 저장에 실패하면 화면을 초기화하지 않습니다.

Windows 카메라는 Flutter의 `camera_windows` 구현을 사용합니다. 카메라가 없거나 권한이 거부되어도 이미지 파일 분석은 계속 사용할 수 있습니다.
