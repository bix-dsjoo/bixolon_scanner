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
  testWidgets('125% 골든과 150% 큰 글자에서 최소 검수·Activity가 넘치지 않는다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    tester.platformDispatcher.textScaleFactorTestValue = 1.25;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    addTearDown(tester.platformDispatcher.clearTextScaleFactorTestValue);

    final controller =
        ScannerController(
            _UnusedApi(),
            _EmptyCameraGateway(),
            _EmptyFileGateway(),
            _MemoryLogRepository(logs: [_logSummary]),
            testCatalog,
          )
          ..cameraInitializing = false
          ..inputMode = InputMode.image
          ..imageBytes = _testInputImage.bytes
          ..imageFileName = _testInputImage.fileName
          ..imageSize = const Size(400, 400)
          ..processState = ProcessState.reviewing
          ..response = _response
          ..detections = testReviewDetections(_response)
          ..selectedItemId = 'item_002';

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_unknown_text_125_1280x720.png'),
    );

    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_text_125_1280x720.png'),
    );

    final search = find.widgetWithText(TextField, '상품명 또는 Scan ID');
    await tester.enterText(search, '스콘');
    await tester.pumpAndSettle();
    expect(find.text('검색 결과'), findsOneWidget);
    expect(find.text('스콘 외 2개'), findsOneWidget);
    expect(find.text('머핀 외 2개'), findsNothing);
    expect(find.byTooltip('검색어 지우기'), findsOneWidget);
    expect(find.widgetWithText(TextButton, '모두 초기화'), findsOneWidget);
    expect(tester.takeException(), isNull);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_filtered_text_125_1280x720.png'),
    );

    await tester.tap(find.widgetWithText(TextButton, '모두 초기화'));
    await tester.pumpAndSettle();
    expect(tester.widget<TextField>(search).controller?.text, isEmpty);

    tester.platformDispatcher.textScaleFactorTestValue = 1.5;
    await tester.pumpAndSettle();
    expect(find.text('활동 기록'), findsOneWidget);
    expect(tester.takeException(), isNull);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_text_150_1280x720.png'),
    );

    await tester.tap(find.text('스캔'));
    await tester.pumpAndSettle();
    expect(find.text('2번 상품을 확인해 주세요'), findsOneWidget);
    final confidence = tester.widget<Text>(find.text('75%'));
    expect(confidence.maxLines, 1);
    expect(confidence.softWrap, isFalse);
    expect(tester.takeException(), isNull);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_unknown_text_150_1280x720.png'),
    );

    controller.dispose();
  });
}

final InputImage _testInputImage = InputImage(
  bytes: base64Decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  ),
  fileName: 'test.png',
);

final _logSummary = ScanLogSummary(
  scanId: 'request_activity_1234',
  analyzedAt: DateTime.utc(2026, 8, 10, 1),
  confirmedAt: DateTime.utc(2026, 8, 10, 1, 1),
  inputMode: InputMode.camera,
  processingTimeMs: 71.2,
  modelVersions: const ModelVersions(detector: '0.1.1', classifier: '0.1.1'),
  items: const [
    ScanLogItemSummary(
      itemId: 'item_001',
      productName: '머핀',
      confidence: .92,
      userModified: false,
      confirmationMethod: 'AUTO_APPROVED',
    ),
    ScanLogItemSummary(
      itemId: 'item_002',
      productName: '스콘',
      confidence: .84,
      userModified: true,
      confirmationMethod: 'TOP3_SELECTED',
    ),
    ScanLogItemSummary(
      itemId: 'item_003',
      productName: '베이글',
      confidence: .78,
      userModified: true,
      confirmationMethod: 'SEARCH_SELECTED',
    ),
  ],
);

const _response = ScanResponse(
  requestId: 'request_widget_1234',
  status: ScanStatus.unknown,
  reasonCodes: ['ITEM_BELOW_APPROVAL_THRESHOLD'],
  items: [
    ScanItem(
      itemId: 'item_001',
      bbox: BoundingBox(x: 0, y: 0, width: 100, height: 100),
      status: ItemStatus.approved,
      reasonCodes: [],
      prediction: Product(
        classId: 'bread_06',
        className: 'Croissant',
        displayName: 'Croissant',
      ),
      top3: [],
      confidence: .99,
    ),
    ScanItem(
      itemId: 'item_002',
      bbox: BoundingBox(x: 100, y: 100, width: 100, height: 100),
      status: ItemStatus.unknown,
      reasonCodes: ['BELOW_APPROVAL_THRESHOLD'],
      prediction: null,
      top3: [
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
  ],
  processingTimeMs: 70,
  modelVersions: ModelVersions(detector: '0.1.1', classifier: '0.1.1'),
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
  _MemoryLogRepository({this.logs = const []});

  final List<ScanLogSummary> logs;

  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async => logs;

  @override
  Future<void> save(ScanLogRecord record) async {}
}
