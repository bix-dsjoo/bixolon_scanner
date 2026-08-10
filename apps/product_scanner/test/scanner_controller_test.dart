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
  @override
  Future<ScanResponse> scan({
    required Uint8List imageBytes,
    required String fileName,
  }) => throw const ScannerApiException('분석 서버에 연결할 수 없어요.');
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

class _FakeImageFileGateway implements ImageFileGateway {
  const _FakeImageFileGateway(this.image);
  final InputImage image;

  @override
  Future<InputImage?> pick() async => image;
}

class _MemoryScanLogRepository implements ScanLogRepository {
  final List<ScanLogRecord> saved = [];

  @override
  Future<void> save(ScanLogRecord record) async => saved.add(record);
}
