# BIXOLON Scanner Flutter SDK

Windows용 BIXOLON Scanner Worker의 HTTP 계약과 선택적 프로세스 제어를 제공하는 Flutter
package `1.1.0`입니다. SDK가 앱 구조나 Worker 실행 시점 또는 Store Model version을 강제하지
않습니다.

## 설치

배포 ZIP에서 `flutter-sdk/bixolon_scanner_sdk`를 프로젝트의 `packages` 폴더로 복사한 뒤
`pubspec.yaml`에 추가합니다.

```yaml
dependencies:
  bixolon_scanner_sdk:
    path: packages/bixolon_scanner_sdk
```

## 방법 1: 이미 실행 중인 Worker에 연결

로컬 Worker를 직접 실행하거나 Windows Service, 별도 supervisor, 원격 PC를 사용하는 경우입니다.
이 경로는 native process API를 호출하지 않습니다.

```dart
final session = await BixolonScannerSession.connect(
  baseUri: Uri.parse('http://127.0.0.1:8000'),
);
```

## 방법 2: 원하는 시점에 Worker를 명시적으로 실행

앱 로그인 완료, 매장 선택 완료, 첫 스캔 직전 등 원하는 시점에 호출할 수 있습니다.

```dart
final layout = BixolonRuntimeLayout.discover(
  runtimeRoot: r'C:\Program Files\MyApp\bixolon_runtime',
  storeBundleRoot: r'C:\ProgramData\MyCompany\Scanner\bundles\bread-dev\0.1.15',
);
final controller = BixolonWorkerController();
await controller.start(
  BixolonWorkerLaunchConfiguration.cpu(
    layout: layout,
    terminateWithApp: true,
  ),
);
final session = await BixolonScannerSession.connect();
```

`terminateWithApp: false`로 실행하면 Flutter 창이 닫혀도 Worker를 유지할 수 있습니다. 이 경우
별도 supervisor가 Worker 종료와 중복 실행을 관리하도록 구성하십시오.

## 방법 3: 로컬 Worker 실행과 연결을 한 번에 처리

앱 시작과 Worker 수명을 같이 관리하는 경우에만 편의 API를 사용합니다.

```dart
final session = await BixolonScannerSession.startLocal();
```

기본 탐색 순서는 다음과 같습니다.

- Runtime: `BIXOLON_RUNTIME_ROOT` 또는 앱 EXE 옆 `bixolon_runtime`
- Store bundle: `BIXOLON_STORE_BUNDLE_DIR`, ProgramData의 `active-bundle.json`, Runtime 안
  `store-bundle` 순서

여러 Store Model version을 동시에 설치한 경우 앱에서 선택한 directory를 명시하면 됩니다. SDK
`1.1.0`과 Store Model `0.1.15`처럼 두 version은 독립적이며, SDK는 Store Model의
`worker_runtime_schema`와 `catalog_schema` 호환성을 전제로 재사용합니다.

## 스캔

```dart
final bytes = await File(imagePath).readAsBytes();
final result = await session.client.scanBytes(bytes, filename: 'capture.jpg');

switch (result.status) {
  case ScanStatus.segmentation:
    // 각 segmentation.status를 사용합니다.
  case ScanStatus.imageRecapture:
    // 이미지 전체를 다시 촬영합니다.
  case ScanStatus.error:
    // 시스템/입력 오류입니다. 재촬영 상태로 변환하지 않습니다.
}
```

KIOSK/POS가 촬영 파일 경로를 이미 관리한다면 같은 직렬 요청 큐를 사용하는 편의 API를 쓸 수
있습니다.

```dart
final result = await session.client.scanFile(imagePath);
```

SDK는 Python Worker와 같은 응답 불변식을 검증합니다. 예를 들어 정렬되지 않은 `UNKNOWN` Top-3,
prediction이 없는 `APPROVED`, segmentation이 포함된 `ERROR`는 정상 결과로 전달하지 않고
`BixolonContractException`으로 실패합니다. HTTP 4xx/5xx라도 Worker가 공통 `ERROR` body를 반환하면
그 body는 `ScanResponse`로 전달되므로 입력 오류와 통신 장애를 구분할 수 있습니다.

SDK는 응답의 `status`, `reason_codes`, `class_id`를 그대로 전달합니다. 앱에서 confidence 임계값을
다시 계산하거나 상품 class 목록을 하드코딩하지 마십시오. 상품 추가는 Store Model Bundle 교체로
처리하고 Flutter를 다시 빌드하지 않습니다.

`expectedModelVersion`은 생략해도 SDK가 응답의 모든 non-null 구성요소 version이 서로 같은지
검증합니다. 모델 번들만 업데이트할 제품은 Flutter 소스에 `0.1.15` 같은 모델 version을 상수로
넣지 마십시오. 특정 version 고정이 운영상 필요하다면 앱 재빌드 없이 바꿀 수 있는 설치 설정이나
서버 설정에서 `expectedModelVersion`을 주입하십시오. 현재 Scanner 제품 version은
`session.readiness.versions.worker`로 표시할 수 있습니다.

## 프로세스 제어 범위

`BixolonWorkerController`는 SDK가 직접 시작한 로컬 프로세스만 추적합니다. Windows Service나
다른 프로그램이 실행한 Worker에는 `BixolonScannerSession.connect`를 사용하십시오. 같은 PC에서
복수 Worker를 실행하려면 서로 다른 port를 구성하고 각 client의 `baseUri`를 맞추십시오.

SDK가 시작한 Worker까지 함께 종료하려면 비동기 종료 API를 사용합니다.

```dart
await session.shutdown();
```

별도 supervisor가 Worker를 소유한다면 `session.close()`로 HTTP client만 닫거나
`await session.shutdown(stopLocalWorker: false)`를 사용하십시오.
