# BIXOLON Scanner 외부 개발 SDK

이 배포물은 외부 Flutter 개발사가 저장소 접근 없이 Windows용 Scanner를 연동하기 위한 자료입니다.
실행 provider는 호환성을 우선한 **ONNX Runtime CPU**입니다.

## 제공 파일의 역할

두 ZIP은 수명주기가 다릅니다.

1. `BIXOLON-Scanner-SDK-Windows-x64-<sdk-version>.zip`
   - Flutter SDK 소스와 Windows plugin
   - 공통 ONNX Runtime CPU Worker 실행 파일
   - API 계약, 상태별 JSON, Mock Worker
   - 설치·수동 실행·매장 번들 활성화 예제
2. `BIXOLON-Store-Model-<store>-<model-version>.zip`
   - 해당 매장의 detector/embedder ONNX와 metadata
   - 상품 Catalog, prototype/support/adapter payload
   - 파일별 SHA-256 manifest

Flutter 앱 최초 설치 프로그램에는 첫 번째 ZIP의 `runtime-payload/bixolon_runtime`과 해당 매장의
두 번째 ZIP을 같이 포함하십시오. 매장 PC 사용자는 Flutter 앱 하나만 설치하고 실행하면 됩니다.
별도 `Runtime Setup.exe`를 실행하게 만들 필요가 없습니다.

이후 상품이나 모델만 바뀌고 Worker/API 호환성이 유지되는 경우 새 Store Model Bundle만 설치하고
Worker를 재시작합니다. Flutter 앱과 installer를 다시 빌드하지 않습니다. 단일 ONNX만 덮어쓰면
metadata/checksum/Catalog가 불일치하므로 반드시 전체 Store Model Bundle 단위로 교체해야 합니다.
외부 앱에 Scanner 제품 version을 상수로 넣지 말고 `/health/ready`에서 읽으십시오. 배포 version을
고정해야 한다면 앱 재빌드 없이 바꿀 수 있는 외부 설정으로 주입합니다.

SDK와 Store Model의 version은 독립적입니다. SDK `1.1.1`은 `worker_runtime_schema=2.0`과
`catalog_schema=2.0`을 만족하는 Store Model `0.1.16`, `0.1.17` 등을 선택해서 실행할 수 있습니다.
재학습·Catalog 변경은 Store Model version만 올리고, Flutter SDK·Worker 또는 API 호환성이 바뀔
때만 SDK version을 올립니다. 숫자가 같은지를 호환 조건으로 사용하지 마십시오.

기본 출력 경로는 다음처럼 분리됩니다.

```text
artifacts/external-sdk/1.1.1/BIXOLON-Scanner-SDK-Windows-x64-1.1.1.zip
artifacts/store-models/three_bakery/0.1.17/BIXOLON-Store-Model-three_bakery-0.1.17.zip
```

## 권장 설치 레이아웃

```text
<Flutter 설치 폴더>/
  MyStoreApp.exe
  bixolon_runtime/
    worker/
      bixolon-worker.exe
      _internal/...

%ProgramData%/BIXOLON/Scanner/
  active-bundle.json
  bundles/
    three_bakery/
      0.1.17/
        model-package/...
        store-catalog/...
        store-bundle.json
        bundle-manifest.json
```

앱 설치 폴더는 관리자 권한 없이는 수정하기 어려우므로 공통 Runtime은 앱과 같이 설치하고, 자주
바뀌는 매장 번들은 `%ProgramData%` 아래 version directory에 둡니다. 새 번들은 새 directory로
검증·설치한 후 `active-bundle.json`만 원자적으로 교체합니다. 실행 중인 Worker는 기존 파일을 계속
사용하므로 적용 시점은 앱 또는 운영 도구가 Worker를 재시작하는 시점입니다.

여러 version을 설치한 경우 Flutter에서 선택한 directory를 명시할 수 있습니다.

```dart
final layout = BixolonRuntimeLayout.discover(
  storeBundleRoot:
      r'C:\ProgramData\BIXOLON\Scanner\bundles\three_bakery\0.1.17',
);
final session = await BixolonScannerSession.startLocal(layout: layout);
```

전역 기본값은 `install-store-bundle.ps1 -Activate`가 원자적으로 갱신하는
`active-bundle.json`을 사용합니다.

## Flutter 연동 순서

1. `flutter-sdk/bixolon_scanner_sdk`를 외부 앱 저장소에 복사하거나 private package로 관리합니다.
2. SDK [README](flutter-sdk/bixolon_scanner_sdk/README.md)의 세 가지 실행 방식 중 제품에 맞는 방식을
   선택합니다.
3. `/health/ready` 성공과 non-null version 일치를 확인한 뒤 `POST /v1/scan`을 호출합니다.
4. Worker의 `status`와 segmentation `status`를 최종 판정으로 사용합니다.

SDK는 Worker 자동 실행을 강제하지 않습니다.

- HTTP-only: 이미 실행된 local/remote Worker에 연결
- Explicit process control: 로그인, 매장 선택, 첫 scan 등 원하는 시점에 실행
- Managed local convenience: 앱 시작과 동시에 실행하고 앱 종료 시 같이 종료
- External supervisor: Windows Service나 별도 launcher가 Worker 수명을 관리

## Mock으로 먼저 개발

Flutter SDK가 포함된 Flutter의 Dart를 사용하면 별도 Python 환경 없이 실행할 수 있습니다.

```powershell
cd mock-worker
dart run bin/mock_worker.dart --status approved
```

실제 CPU Worker로 바꿀 때 client 코드는 그대로 두고 Worker 주소 또는 실행 정책만 변경합니다.

## 보안과 검증 한계

Runtime과 Store Model Bundle의 manifest/checksum은 전송 손상이나 파일 변경을 검출합니다. 발행자
진위를 증명하는 전자서명은 아닙니다. 전달 채널의 접근 통제와 ZIP SHA-256 확인은 별도로 수행해야
합니다. Worker는 시작할 때 Runtime/Catalog 내부 checksum을 다시 검증하며 불일치 시 시작하지
않습니다.

Store Model 0.1.17의 `worker-profile.json`에는 CPU Detector 8 / Embedder 12 threads를 기록합니다. SDK Core 1.1.1은 CPU 시작 시 이를 읽으며, 명시한 `extraEnvironment`가 우선합니다. 이전 번들에 파일이 없으면 기존 Worker 기본값을 유지합니다.
