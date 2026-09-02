# Mock Worker

모델과 OpenVINO 없이 Flutter 연동 화면과 상태 분기를 개발할 수 있는 순수 Dart 서버입니다.

```powershell
dart run bin/mock_worker.dart --port 8000 --status approved
```

`--status`는 `approved`, `unknown`, `segment-recapture`, `image-recapture`, `error` 중 하나입니다.
실행 중인 서버에는 다음 endpoint가 열립니다.

- `GET /health/live`
- `GET /health/ready`
- `POST /v1/scan` (`multipart/form-data`의 실제 image 내용은 판정하지 않음)

Mock 결과는 UI 개발용이며 모델 정확도나 Runtime 동작을 증명하지 않습니다.
