# Flutter 연동 예제

전달 ZIP의 `flutter_example/lib`에는 현재 앱에서 사용하는 Worker HTTP client와 응답 DTO가
들어 있습니다. 프로젝트에 `http: ^1.6.0`을 추가하고 필요한 파일을 복사한 뒤 다음처럼 생성합니다.

```dart
final api = WorkerScannerApi(
  baseUrl: 'http://127.0.0.1:8000',
  waitForReady: true,
  expectedVersion: '0.0.2',
  timeout: const Duration(seconds: 65),
  readinessTimeout: const Duration(seconds: 180),
);
```

`scan`에는 JPEG 또는 PNG bytes와 파일명을 전달합니다.

```dart
final response = await api.scan(
  imageBytes: bytes,
  fileName: 'scan.jpg',
);
```

처리 원칙은 다음과 같습니다.

- `SEGMENTATION`: `items`를 순서대로 표시합니다.
- `APPROVED`: `prediction`을 자동 승인 결과로 사용합니다.
- `UNKNOWN`: `top3`를 사용자 선택 후보로 표시합니다.
- `SEGMENT_RECAPTURE`: 해당 영역의 재촬영을 안내합니다.
- `IMAGE_RECAPTURE`: 이미지 전체 재촬영을 안내합니다.
- `ERROR` 또는 4xx/5xx: 입력·Worker 오류로 처리하며 재촬영 판정으로 바꾸지 않습니다.

Windows 앱에서 Worker를 자동 실행할 경우 Worker process handle을 보관하고 앱 종료 시 같은
process만 종료하십시오. API 호출은 readiness가 `ready`이고 모든 non-null version이 `0.0.2`인
경우에만 시작합니다.
