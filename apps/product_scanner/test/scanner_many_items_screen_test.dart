import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/app.dart';
import 'package:product_scanner/controllers/scanner_controller.dart';
import 'package:product_scanner/models/scan_models.dart';
import 'package:product_scanner/services/image_input.dart';
import 'package:product_scanner/services/scan_log_repository.dart';
import 'package:product_scanner/services/scanner_api.dart';

import 'support/test_catalog.dart';

void main() {
  testWidgets('8개 상품 연속 확정에서도 다음 검수 행과 Top-3를 함께 표시한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final response = _manyItemResponse();
    final controller =
        ScannerController(
            _UnusedApi(),
            _EmptyCameraGateway(),
            _EmptyFileGateway(),
            _MemoryLogRepository(),
            testCatalog,
          )
          ..cameraInitializing = false
          ..inputMode = InputMode.image
          ..imageBytes = _testInputImage.bytes
          ..imageFileName = _testInputImage.fileName
          ..imageSize = const Size(400, 400)
          ..processState = ProcessState.reviewing
          ..response = response
          ..detections = testReviewDetections(response)
          ..selectedItemId = 'item_006';
    controller
      ..confirmCandidate('item_006', response.items[5].top3.first)
      ..confirmCandidate('item_007', response.items[6].top3.first);

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();

    expect(controller.selectedItemId, 'item_008');
    expect(find.text('8번 상품을 확인해 주세요'), findsOneWidget);
    expect(find.text('8 / 8'), findsOneWidget);
    expect(find.text('머핀'), findsNWidgets(3));
    expect(find.text('스콘'), findsOneWidget);
    expect(find.text('베이글'), findsOneWidget);
    final listRect = tester.getRect(
      find.byKey(const ValueKey('detection-list')),
    );
    final selectedRect = tester.getRect(
      find.byKey(const ValueKey('detection-row-item_008')),
    );
    final inspectorRect = tester.getRect(
      find.byKey(const ValueKey('review-inspector')),
    );
    expect(selectedRect.top, greaterThanOrEqualTo(listRect.top));
    expect(selectedRect.bottom, lessThanOrEqualTo(listRect.bottom + 0.5));
    expect(
      tester.getRect(find.text('다른 상품 검색')).bottom,
      lessThanOrEqualTo(inspectorRect.bottom),
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_many_items_1280x720.png'),
    );

    controller.dispose();
  });
}

ScanResponse _manyItemResponse() {
  final items = <ScanItem>[
    for (var index = 1; index <= 5; index++)
      ScanItem(
        itemId: 'item_${index.toString().padLeft(3, '0')}',
        bbox: BoundingBox(
          x: ((index - 1) % 4) * 80,
          y: ((index - 1) ~/ 4) * 80,
          width: 64,
          height: 64,
        ),
        status: ItemStatus.approved,
        reasonCodes: const [],
        prediction: const Product(
          classId: 'bread_06',
          className: 'Croissant',
          displayName: 'Croissant',
        ),
        top3: const [],
        confidence: .99,
      ),
    for (var index = 6; index <= 8; index++)
      ScanItem(
        itemId: 'item_${index.toString().padLeft(3, '0')}',
        bbox: BoundingBox(
          x: ((index - 1) % 4) * 80,
          y: ((index - 1) ~/ 4) * 80,
          width: 64,
          height: 64,
        ),
        status: ItemStatus.unknown,
        reasonCodes: const ['BELOW_APPROVAL_THRESHOLD'],
        prediction: null,
        top3: const [
          Candidate(
            classId: 'bread_13',
            className: 'Muffin',
            displayName: 'Muffin',
            confidence: .75,
          ),
          Candidate(
            classId: 'bread_04',
            className: 'Scon',
            displayName: 'Scon',
            confidence: .16,
          ),
          Candidate(
            classId: 'bread_11',
            className: 'Bagel',
            displayName: 'Bagel',
            confidence: .09,
          ),
        ],
        confidence: .75,
      ),
  ];
  return ScanResponse(
    requestId: 'request_many_items',
    status: ScanStatus.unknown,
    reasonCodes: const ['ITEM_BELOW_APPROVAL_THRESHOLD'],
    items: items,
    processingTimeMs: 76,
    modelVersions: const ModelVersions(detector: '0.1.1', classifier: '0.1.1'),
  );
}

final InputImage _testInputImage = InputImage(
  bytes: base64Decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  ),
  fileName: 'test.png',
);

class _UnusedApi implements ScannerApi {
  @override
  Future<ScanResponse> scan({
    required Uint8List imageBytes,
    required String fileName,
  }) => throw UnimplementedError();
}

class _EmptyCameraGateway implements CameraGateway {
  @override
  CameraController? get controller => null;

  @override
  bool get isReady => false;

  @override
  Future<InputImage> capture() => throw UnimplementedError();

  @override
  Future<void> dispose() async {}

  @override
  Future<void> initialize() async {}
}

class _EmptyFileGateway implements ImageFileGateway {
  @override
  Future<InputImage?> pick() async => null;
}

class _MemoryLogRepository implements ScanLogRepository {
  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async => const [];

  @override
  Future<void> save(ScanLogRecord record) async {}
}
