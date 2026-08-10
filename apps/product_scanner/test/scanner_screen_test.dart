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
  testWidgets('1440x900 작업대에서 확인 필요 상품과 Top-3를 표시한다', (tester) async {
    tester.view.physicalSize = const Size(1440, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final controller =
        ScannerController(
            _UnusedApi(),
            _EmptyCameraGateway(),
            _EmptyFileGateway(),
            _MemoryLogRepository(),
            testCatalog,
          )
          ..cameraInitializing = false
          ..processState = ProcessState.reviewing
          ..response = _response
          ..detections = _response.items
              .map(ReviewDetection.fromScanItem)
              .toList(growable: false)
          ..selectedItemId = 'item_002';

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Product Scanner'), findsOneWidget);
    expect(find.text('상품 확인이 필요해요'), findsOneWidget);
    expect(find.text('어떤 상품인가요?'), findsOneWidget);
    expect(find.text('Muffin'), findsOneWidget);
    expect(find.text('1 / 2 상품 확인 완료'), findsOneWidget);

    await tester.tap(find.text('Muffin'));
    await tester.pumpAndSettle();
    expect(find.text('2개 상품을 모두 확인했어요'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '최종 확정'), findsOneWidget);
    controller.dispose();
  });

  testWidgets('업로드 이미지 RECAPTURE는 다른 이미지 선택만 Primary로 제공한다', (tester) async {
    tester.view.physicalSize = const Size(1200, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

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
          ..processState = ProcessState.reviewing
          ..response = const ScanResponse(
            requestId: 'request_recapture_1',
            status: ScanStatus.recapture,
            reasonCodes: ['DETECTOR_BORDER_CLIPPED'],
            items: [],
            processingTimeMs: 40,
            modelVersions: ModelVersions(detector: '0.1.1'),
          );

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pump();

    expect(find.text('다른 이미지를 선택해 주세요'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '다른 이미지 선택'), findsOneWidget);
    expect(find.text('분석하기'), findsNothing);
    controller.dispose();
  });

  testWidgets('Activity에서 저장된 스캔 로그와 상세 정보를 확인한다', (tester) async {
    tester.view.physicalSize = const Size(1440, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final repository = _MemoryLogRepository(logs: [_logSummary]);
    final controller = ScannerController(
      _UnusedApi(),
      _EmptyCameraGateway(),
      _EmptyFileGateway(),
      repository,
      testCatalog,
    )..cameraInitializing = false;

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('Activity'));
    await tester.pumpAndSettle();

    expect(find.text('확정된 스캔 1건'), findsOneWidget);
    expect(find.text('Muffin'), findsWidgets);
    expect(find.text('request_activity_1234'), findsOneWidget);
    expect(find.text('Confirmed'), findsWidgets);
    controller.dispose();
  });

  testWidgets('후보에 없는 상품을 영어 원본명으로 검색해 확정한다', (tester) async {
    tester.view.physicalSize = const Size(1440, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final controller =
        ScannerController(
            _UnusedApi(),
            _EmptyCameraGateway(),
            _EmptyFileGateway(),
            _MemoryLogRepository(),
            testCatalog,
          )
          ..cameraInitializing = false
          ..processState = ProcessState.reviewing
          ..response = _response
          ..detections = _response.items
              .map(ReviewDetection.fromScanItem)
              .toList(growable: false)
          ..selectedItemId = 'item_002';

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('다른 상품 검색'));
    await tester.pumpAndSettle();

    final field = find.widgetWithText(TextField, '상품명 검색');
    expect(field, findsOneWidget);
    await tester.enterText(field, 'Egg');
    await tester.pump();
    expect(find.text('Egg Tart'), findsOneWidget);

    await tester.tap(find.text('Egg Tart'));
    await tester.pumpAndSettle();
    expect(find.text('2개 상품을 모두 확인했어요'), findsOneWidget);
    controller.dispose();
  });

  testWidgets('분석 중과 Worker 오류를 서로 다른 상태로 표시한다', (tester) async {
    tester.view.physicalSize = const Size(1200, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final controller = ScannerController(
      _UnusedApi(),
      _EmptyCameraGateway(),
      _EmptyFileGateway(),
      _MemoryLogRepository(),
      testCatalog,
    )..cameraInitializing = false;

    controller.processState = ProcessState.analyzing;
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    expect(find.text('분석하고 있어요'), findsOneWidget);

    controller
      ..processState = ProcessState.error
      ..errorMessage = 'Worker에 연결할 수 없습니다.'
      ..notifyListeners();
    await tester.pump();
    expect(find.text('분석하지 못했어요'), findsOneWidget);
    expect(find.text('Worker에 연결할 수 없습니다.'), findsOneWidget);
    controller.dispose();
  });

  testWidgets('Activity 빈 상태에서 기록 생성 조건을 안내한다', (tester) async {
    tester.view.physicalSize = const Size(1200, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final controller = ScannerController(
      _UnusedApi(),
      _EmptyCameraGateway(),
      _EmptyFileGateway(),
      _MemoryLogRepository(),
      testCatalog,
    )..cameraInitializing = false;

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('Activity'));
    await tester.pumpAndSettle();

    expect(find.text('저장된 활동이 없습니다'), findsOneWidget);
    expect(find.text('상품을 최종 확정하면 이곳에 기록됩니다.'), findsOneWidget);
    controller.dispose();
  });

  testWidgets('800px 좁은 작업대에서도 검수 패널이 넘치지 않는다', (tester) async {
    tester.view.physicalSize = const Size(800, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final controller =
        ScannerController(
            _UnusedApi(),
            _EmptyCameraGateway(),
            _EmptyFileGateway(),
            _MemoryLogRepository(),
            testCatalog,
          )
          ..cameraInitializing = false
          ..processState = ProcessState.reviewing
          ..response = _response
          ..detections = _response.items
              .map(ReviewDetection.fromScanItem)
              .toList(growable: false)
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
    expect(find.text('어떤 상품인가요?'), findsOneWidget);
    expect(find.text('1 / 2 상품 확인 완료'), findsOneWidget);
    controller.dispose();
  });
}

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
      productName: 'Muffin',
      confidence: .92,
      userModified: false,
      confirmationMethod: 'AUTO_APPROVED',
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
  Future<List<ScanLogSummary>> list({int limit = 100}) async =>
      logs.take(limit).toList();

  @override
  Future<void> save(ScanLogRecord record) async {}
}
