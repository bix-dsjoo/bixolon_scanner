import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/controllers/scanner_controller.dart';
import 'package:product_scanner/models/scan_models.dart';
import 'package:product_scanner/services/image_input.dart';
import 'package:product_scanner/services/scan_log_repository.dart';
import 'package:product_scanner/services/scanner_api.dart';

import 'support/test_catalog.dart';

void main() {
  test('동일 상품 재선택은 기존 자동·사용자 확정 근거를 변경하지 않는다', () {
    final response = _reviewResponse();
    final controller =
        ScannerController(
            _FakeScannerApi(response),
            _FakeCameraGateway(),
            _FakeImageFileGateway(_testImage),
            _MemoryScanLogRepository(),
            testCatalog,
          )
          ..detections = response.items
              .map(ReviewDetection.fromScanItem)
              .toList(growable: false);

    controller.showSearch('item_001');
    controller.confirmSearchProduct(
      'item_001',
      testCatalog.search('bread_06').single,
    );
    expect(
      controller.detections.first.confirmationMethod,
      ConfirmationMethod.autoApproved,
    );
    expect(controller.detections.first.wasUserChanged, isFalse);

    controller.confirmCandidate(
      'item_002',
      controller.detections[1].source.top3.first,
    );
    controller.selectedItemId = 'item_002';
    controller.showSearch('item_002');
    controller.confirmSearchProduct(
      'item_002',
      testCatalog.search('bread_13').single,
    );
    expect(
      controller.detections[1].confirmationMethod,
      ConfirmationMethod.top3Selected,
    );
    expect(controller.detections[1].wasUserChanged, isTrue);
    controller.dispose();
  });

  test('후보 확정 후 다음 미확정 상품으로 이동하고 최종 로그를 저장한다', () async {
    final api = _FakeScannerApi(_reviewResponse());
    final logs = _MemoryScanLogRepository();
    final controller = ScannerController(
      api,
      _FakeCameraGateway(),
      _FakeImageFileGateway(_testImage),
      logs,
      testCatalog,
    );

    await controller.chooseImage();
    expect(controller.inputMode, InputMode.image);
    await controller.analyze();

    expect(controller.processState, ProcessState.reviewing);
    expect(controller.selectedItemId, 'item_002');
    expect(controller.confirmedCount, 1);
    expect(controller.detections.first.finalProduct?.displayName, '크루아상');

    controller.confirmCandidate(
      'item_002',
      controller.detections[1].source.top3.first,
    );
    expect(controller.allConfirmed, isTrue);
    expect(controller.detections[1].finalProduct?.displayName, '머핀');
    expect(controller.selectedItemId, isNull);

    await controller.submit();
    expect(logs.saved, hasLength(1));
    expect(logs.saved.single.scanId, 'request_12345678');
    expect(
      logs.saved.single.detections[1].confirmationMethod,
      ConfirmationMethod.top3Selected,
    );
    expect(controller.processState, ProcessState.ready);
    expect(controller.inputMode, InputMode.image);
    expect(controller.activityDataRevision, 1);
    expect(controller.latestSavedScanId, 'request_12345678');
    expect(controller.completionMessage, '2개 상품을 확정했어요');
    controller.dispose();
  });

  test('카메라 작업 완료 후에는 카메라 입력 대기 상태를 유지한다', () async {
    final response = _reviewResponse();
    final controller =
        ScannerController(
            _FakeScannerApi(response),
            _FakeCameraGateway(),
            _FakeImageFileGateway(_testImage),
            _MemoryScanLogRepository(),
            testCatalog,
          )
          ..inputMode = InputMode.camera
          ..imageBytes = _testImage.bytes
          ..imageFileName = _testImage.fileName
          ..processState = ProcessState.reviewing
          ..response = response
          ..detections = response.items
              .map(ReviewDetection.fromScanItem)
              .toList(growable: false);
    controller.confirmCandidate('item_002', response.items[1].top3.first);

    await controller.submit();

    expect(controller.processState, ProcessState.ready);
    expect(controller.inputMode, InputMode.camera);
    expect(controller.imageBytes, isNull);
    expect(controller.completionMessage, '2개 상품을 확정했어요');
    controller.dispose();
  });

  test('기존 Activity 로그는 저장 스키마 변경 없이 한국어로 표시한다', () async {
    final storedLog = ScanLogSummary(
      scanId: 'legacy_log',
      analyzedAt: DateTime.utc(2026, 8, 10),
      confirmedAt: DateTime.utc(2026, 8, 10, 0, 1),
      inputMode: InputMode.image,
      processingTimeMs: 42,
      modelVersions: const ModelVersions(
        detector: '0.1.1',
        classifier: '0.1.1',
      ),
      items: const [
        ScanLogItemSummary(
          itemId: 'item_001',
          productName: 'Muffin',
          confidence: .9,
          userModified: false,
          confirmationMethod: 'AUTO_APPROVED',
          classId: 'bread_13',
          className: 'Muffin',
        ),
        ScanLogItemSummary(
          itemId: 'item_002',
          productName: 'Retired product',
          confidence: .8,
          userModified: true,
          confirmationMethod: 'SEARCH_SELECTED',
          classId: 'retired_01',
          className: 'Retired product',
        ),
      ],
    );
    final logs = _StaticScanLogRepository([storedLog]);
    final controller = ScannerController(
      _FakeScannerApi(_reviewResponse()),
      _FakeCameraGateway(),
      _FakeImageFileGateway(_testImage),
      logs,
      testCatalog,
    );

    final localized = await controller.loadScanLogs();

    expect(localized.single.items.first.productName, '머핀');
    expect(localized.single.items.last.productName, 'Retired product');
    expect(logs.logs.single.items.first.productName, 'Muffin');
    controller.dispose();
  });

  test('저장 실패 후 확인 결과를 유지하고 같은 요청을 다시 저장할 수 있다', () async {
    final logs = _FailOnceScanLogRepository();
    final response = _reviewResponse();
    final controller =
        ScannerController(
            _FakeScannerApi(response),
            _FakeCameraGateway(),
            _FakeImageFileGateway(_testImage),
            logs,
            testCatalog,
          )
          ..inputMode = InputMode.image
          ..imageBytes = _testImage.bytes
          ..imageFileName = _testImage.fileName
          ..processState = ProcessState.reviewing
          ..response = response
          ..detections = response.items
              .map(ReviewDetection.fromScanItem)
              .toList(growable: false);
    controller.confirmCandidate('item_002', response.items[1].top3.first);

    await controller.submit();

    expect(logs.saveCalls, 1);
    expect(controller.processState, ProcessState.reviewing);
    expect(controller.activityDataRevision, 0);
    expect(controller.latestSavedScanId, isNull);
    expect(controller.errorMessage, contains('저장하지 못했어요'));
    expect(controller.allConfirmed, isTrue);
    expect(controller.response?.requestId, response.requestId);
    expect(controller.imageBytes, isNotNull);

    await controller.submit();

    expect(logs.saveCalls, 2);
    expect(logs.saved, hasLength(1));
    expect(logs.saved.single.scanId, response.requestId);
    expect(controller.processState, ProcessState.ready);
    expect(controller.inputMode, InputMode.image);
    expect(controller.activityDataRevision, 1);
    expect(controller.latestSavedScanId, response.requestId);
    expect(controller.completionMessage, '2개 상품을 확정했어요');
    controller.dispose();
  });

  test('RECAPTURE는 객체 결과 없이 재촬영 안내 상태로 유지한다', () async {
    final controller = ScannerController(
      _FakeScannerApi(_recaptureResponse()),
      _FakeCameraGateway(),
      _FakeImageFileGateway(_testImage),
      _MemoryScanLogRepository(),
      testCatalog,
    );

    await controller.chooseImage();
    await controller.analyze();

    expect(controller.processState, ProcessState.reviewing);
    expect(controller.isRecapture, isTrue);
    expect(controller.detections, isEmpty);
    expect(controller.recaptureTitle, '상품을 찾지 못했어요');
    controller.dispose();
  });

  test('수동 세션 초기화는 이미지 작업 중에도 기본 카메라 입력으로 돌아간다', () {
    final controller = ScannerController(
      _FakeScannerApi(_reviewResponse()),
      _FakeCameraGateway(),
      _FakeImageFileGateway(_testImage),
      _MemoryScanLogRepository(),
      testCatalog,
    )..inputMode = InputMode.image;

    controller.resetSession();

    expect(controller.inputMode, InputMode.camera);
    expect(controller.processState, ProcessState.ready);
    controller.dispose();
  });

  test('일반 RECAPTURE 안내는 입력원별 실제 복구 행동을 명시한다', () {
    final controller = ScannerController(
      _FakeScannerApi(_genericRecaptureResponse()),
      _FakeCameraGateway(),
      _FakeImageFileGateway(_testImage),
      _MemoryScanLogRepository(),
      testCatalog,
    )..response = _genericRecaptureResponse();

    controller.inputMode = InputMode.camera;
    expect(controller.recaptureDetail, contains('다시 촬영해 주세요'));
    expect(controller.recaptureDetail, isNot(contains('다시 시도')));

    controller.inputMode = InputMode.image;
    expect(controller.recaptureDetail, '상품이 잘 보이는 다른 이미지를 선택해 주세요.');
    controller.dispose();
  });

  test('Worker 장애는 RECAPTURE가 아닌 ERROR UI 상태가 된다', () async {
    final controller = ScannerController(
      _ThrowingScannerApi(),
      _FakeCameraGateway(),
      _FakeImageFileGateway(_testImage),
      _MemoryScanLogRepository(),
      testCatalog,
    );

    await controller.chooseImage();
    await controller.analyze();

    expect(controller.processState, ProcessState.error);
    expect(controller.isRecapture, isFalse);
    expect(controller.errorMessage, contains('서버'));
    expect(controller.errorRecovery, ScannerErrorRecovery.retryAnalysis);
    controller.dispose();
  });

  test('입력 이미지 오류는 ERROR를 유지하고 입력 교체 복구를 요청한다', () async {
    final controller = ScannerController(
      const _ThrowingScannerApi(
        message: 'JPEG 또는 PNG 이미지를 선택해 주세요.',
        recovery: ScannerErrorRecovery.replaceInput,
      ),
      _FakeCameraGateway(),
      _FakeImageFileGateway(_testImage),
      _MemoryScanLogRepository(),
      testCatalog,
    );

    await controller.chooseImage();
    await controller.analyze();

    expect(controller.processState, ProcessState.error);
    expect(controller.isRecapture, isFalse);
    expect(controller.errorRecovery, ScannerErrorRecovery.replaceInput);
    expect(controller.errorMessage, contains('JPEG'));
    controller.dispose();
  });

  test('카메라 초기화 실패는 촬영 실패와 다른 내부 복구 원인을 유지한다', () async {
    final controller = ScannerController(
      _FakeScannerApi(_reviewResponse()),
      _InitializationFailingCameraGateway(),
      _FakeImageFileGateway(_testImage),
      _MemoryScanLogRepository(),
      testCatalog,
    );

    await controller.initialize();

    expect(controller.processState, ProcessState.ready);
    expect(controller.response, isNull);
    expect(controller.isRecapture, isFalse);
    expect(controller.hasActiveCameraIssue, isTrue);
    expect(controller.cameraIssueType, CameraIssueType.unavailable);
    expect(controller.cameraIssueTitle, '카메라를 사용할 수 없어요');
    expect(controller.cameraMessage, '카메라를 사용할 수 없어요. 연결 상태를 확인해 주세요.');
    controller.dispose();
  });

  test('카메라 촬영 실패는 Worker ERROR로 바꾸지 않고 재연결로 복구한다', () async {
    final camera = _CaptureFailingCameraGateway();
    final controller = ScannerController(
      _FakeScannerApi(_reviewResponse()),
      camera,
      _FakeImageFileGateway(_testImage),
      _MemoryScanLogRepository(),
      testCatalog,
    )..cameraInitializing = false;

    await controller.captureAndAnalyze();

    expect(controller.processState, ProcessState.ready);
    expect(controller.response, isNull);
    expect(controller.isRecapture, isFalse);
    expect(controller.hasActiveCameraIssue, isTrue);
    expect(controller.cameraIssueType, CameraIssueType.captureFailed);
    expect(controller.cameraIssueTitle, '촬영하지 못했어요');
    expect(controller.cameraMessage, '카메라 응답을 받지 못했어요. 다시 연결해 주세요.');

    await controller.reconnectCamera();
    expect(camera.initializeCalls, 1);
    expect(controller.cameraMessage, isNull);
    expect(controller.cameraIssueType, isNull);
    expect(controller.hasActiveCameraIssue, isFalse);
    expect(controller.processState, ProcessState.ready);
    controller.dispose();
  });

  test('이전·다음 상품 이동은 목록을 순환하고 검색 상태를 닫는다', () {
    final response = _reviewResponse();
    final controller =
        ScannerController(
            _FakeScannerApi(response),
            _FakeCameraGateway(),
            _FakeImageFileGateway(_testImage),
            _MemoryScanLogRepository(),
            testCatalog,
          )
          ..processState = ProcessState.reviewing
          ..response = response
          ..detections = response.items
              .map(ReviewDetection.fromScanItem)
              .toList(growable: false)
          ..selectedItemId = 'item_002'
          ..searchItemId = 'item_002'
          ..searchQuery = 'Muffin';

    controller.selectPreviousDetection();
    expect(controller.selectedItemId, 'item_001');
    expect(controller.searchItemId, isNull);
    expect(controller.searchQuery, isEmpty);

    controller.selectNextDetection();
    expect(controller.selectedItemId, 'item_002');
    controller.selectNextDetection();
    expect(controller.selectedItemId, 'item_001');
    controller.dispose();
  });
}

final InputImage _testImage = InputImage(
  bytes: base64Decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  ),
  fileName: 'test.png',
);

ScanResponse _reviewResponse() => const ScanResponse(
  requestId: 'request_12345678',
  status: ScanStatus.unknown,
  reasonCodes: ['ITEM_BELOW_APPROVAL_THRESHOLD'],
  items: [
    ScanItem(
      itemId: 'item_001',
      bbox: BoundingBox(x: 10, y: 20, width: 100, height: 80),
      status: ItemStatus.approved,
      reasonCodes: [],
      prediction: Product(
        classId: 'bread_06',
        className: 'Croissant',
        displayName: 'Croissant',
      ),
      top3: [],
      confidence: .98,
    ),
    ScanItem(
      itemId: 'item_002',
      bbox: BoundingBox(x: 140, y: 70, width: 90, height: 100),
      status: ItemStatus.unknown,
      reasonCodes: ['BELOW_APPROVAL_THRESHOLD'],
      prediction: null,
      top3: [
        Candidate(
          classId: 'bread_13',
          className: 'Muffin',
          displayName: 'Muffin',
          confidence: .72,
        ),
        Candidate(
          classId: 'bread_06',
          className: 'Croissant',
          displayName: 'Croissant',
          confidence: .18,
        ),
      ],
      confidence: .72,
    ),
  ],
  processingTimeMs: 84.2,
  modelVersions: ModelVersions(detector: '0.1.1', classifier: '0.1.1'),
);

ScanResponse _recaptureResponse() => const ScanResponse(
  requestId: 'request_87654321',
  status: ScanStatus.recapture,
  reasonCodes: ['DETECTOR_NO_OBJECT'],
  items: [],
  processingTimeMs: 42,
  modelVersions: ModelVersions(detector: '0.1.1'),
);

ScanResponse _genericRecaptureResponse() => const ScanResponse(
  requestId: 'request_generic_recapture',
  status: ScanStatus.recapture,
  reasonCodes: ['CLASSIFIER_QUALITY_REJECTED'],
  items: [],
  processingTimeMs: 43,
  modelVersions: ModelVersions(detector: '0.1.1', classifier: '0.1.1'),
);

class _FakeScannerApi implements ScannerApi {
  const _FakeScannerApi(this.response);
  final ScanResponse response;

  @override
  Future<ScanResponse> scan({
    required Uint8List imageBytes,
    required String fileName,
  }) async => response;
}

class _ThrowingScannerApi implements ScannerApi {
  const _ThrowingScannerApi({
    this.message = '분석 서버에 연결할 수 없어요.',
    this.recovery = ScannerErrorRecovery.retryAnalysis,
  });

  final String message;
  final ScannerErrorRecovery recovery;

  @override
  Future<ScanResponse> scan({
    required Uint8List imageBytes,
    required String fileName,
  }) => throw ScannerApiException(message, recovery: recovery);
}

class _FakeCameraGateway implements CameraGateway {
  @override
  CameraController? get controller => null;

  @override
  bool get isReady => true;

  @override
  Future<InputImage> capture() async => _testImage;

  @override
  Future<void> dispose() async {}

  @override
  Future<void> initialize() async {}
}

class _CaptureFailingCameraGateway implements CameraGateway {
  int initializeCalls = 0;

  @override
  CameraController? get controller => null;

  @override
  bool get isReady => true;

  @override
  Future<InputImage> capture() => throw StateError('capture failed');

  @override
  Future<void> dispose() async {}

  @override
  Future<void> initialize() async => initializeCalls += 1;
}

class _InitializationFailingCameraGateway implements CameraGateway {
  @override
  CameraController? get controller => null;

  @override
  bool get isReady => false;

  @override
  Future<InputImage> capture() => throw UnimplementedError();

  @override
  Future<void> dispose() async {}

  @override
  Future<void> initialize() => throw StateError('camera unavailable');
}

class _FakeImageFileGateway implements ImageFileGateway {
  const _FakeImageFileGateway(this.image);
  final InputImage image;

  @override
  Future<InputImage?> pick() async => image;
}

class _MemoryScanLogRepository implements ScanLogRepository {
  final List<ScanLogRecord> saved = [];

  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async => const [];

  @override
  Future<void> save(ScanLogRecord record) async => saved.add(record);
}

class _StaticScanLogRepository implements ScanLogRepository {
  const _StaticScanLogRepository(this.logs);

  final List<ScanLogSummary> logs;

  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async => logs;

  @override
  Future<void> save(ScanLogRecord record) async {}
}

class _FailOnceScanLogRepository implements ScanLogRepository {
  final List<ScanLogRecord> saved = [];
  int saveCalls = 0;

  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async => const [];

  @override
  Future<void> save(ScanLogRecord record) async {
    saveCalls += 1;
    if (saveCalls == 1) throw StateError('disk unavailable');
    saved.add(record);
  }
}
