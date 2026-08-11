import 'dart:async';
import 'dart:convert';
import 'dart:ui' show SemanticsAction, Tristate;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/app.dart';
import 'package:product_scanner/controllers/scanner_controller.dart';
import 'package:product_scanner/models/scan_models.dart';
import 'package:product_scanner/services/image_input.dart';
import 'package:product_scanner/services/scan_log_repository.dart';
import 'package:product_scanner/services/scanner_api.dart';

import 'support/test_catalog.dart';

void main() {
  testWidgets('APPROVED와 저장 중에는 Primary action이 하나만 유지된다', (tester) async {
    final semantics = tester.ensureSemantics();
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final fileGateway = _CountingFileGateway();
    final controller =
        ScannerController(
            _UnusedApi(),
            _EmptyCameraGateway(),
            fileGateway,
            _MemoryLogRepository(),
            testCatalog,
          )
          ..cameraInitializing = false
          ..inputMode = InputMode.image
          ..imageBytes = _testInputImage.bytes
          ..imageFileName = _testInputImage.fileName
          ..imageSize = const Size(400, 400)
          ..processState = ProcessState.reviewing
          ..response = _approvedResponse
          ..detections = testReviewDetections(_approvedResponse);

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();

    final reviewComplete = find.bySemanticsLabel(
      '검수 상태. 1개 상품 확인 완료. 최종 확정할 수 있어요.',
    );
    expect(reviewComplete, findsOneWidget);
    expect(
      tester
          .getSemantics(reviewComplete)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_approved_1280x720.png'),
    );

    expect(find.widgetWithText(FilledButton, '1개 상품 최종 확정'), findsOneWidget);
    expect(find.widgetWithText(OutlinedButton, '카메라로 돌아가기'), findsOneWidget);
    expect(find.byType(FilledButton), findsOneWidget);
    expect(find.byKey(const ValueKey('step-navigator')), findsNothing);

    controller
      ..processState = ProcessState.submitting
      ..notifyListeners();
    await tester.pump(const Duration(milliseconds: 600));

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_submitting_1280x720.png'),
    );

    expect(find.text('저장이 끝나면 다음 이미지를 준비할 수 있어요'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '저장 중'), findsOneWidget);
    expect(find.byType(FilledButton), findsOneWidget);
    final progress = find.bySemanticsLabel('저장 중. 완료될 때까지 기다려 주세요');
    expect(progress, findsOneWidget);
    final progressData = tester.getSemantics(progress).getSemanticsData();
    expect(progressData.flagsCollection.isLiveRegion, isTrue);
    expect(progressData.flagsCollection.isButton, isTrue);
    expect(progressData.hasAction(SemanticsAction.tap), isFalse);
    expect(
      tester.widget<FilledButton>(find.byType(FilledButton)).onPressed,
      isNull,
    );
    controller.detections.single.confirmationMethod =
        ConfirmationMethod.userCorrected;
    expect(controller.hasUserChanges, isTrue);
    await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
    await tester.sendKeyDownEvent(LogicalKeyboardKey.keyO);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.keyO);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
    await tester.pump();
    expect(fileGateway.pickCalls, 0);
    expect(find.byType(AlertDialog), findsNothing);
    expect(controller.processState, ProcessState.submitting);
    expect(tester.takeException(), isNull);
    controller.dispose();
    semantics.dispose();
  });

  testWidgets('저장 중 다른 화면으로 이동하면 완료 포커스를 강제로 되돌리지 않는다', (tester) async {
    final semantics = tester.ensureSemantics();
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final logs = _DeferredLogRepository();
    final controller =
        ScannerController(
            _UnusedApi(),
            _EmptyCameraGateway(),
            _CountingFileGateway(),
            logs,
            testCatalog,
          )
          ..cameraInitializing = false
          ..inputMode = InputMode.image
          ..imageBytes = _testInputImage.bytes
          ..imageFileName = _testInputImage.fileName
          ..imageSize = const Size(400, 400)
          ..processState = ProcessState.reviewing
          ..response = _approvedResponse
          ..detections = testReviewDetections(_approvedResponse);
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();

    for (
      var index = 0;
      index < 12 &&
          FocusManager.instance.primaryFocus?.debugLabel !=
              'final-confirmation-action';
      index += 1
    ) {
      await tester.sendKeyEvent(LogicalKeyboardKey.tab);
      await tester.pump();
    }
    expect(
      FocusManager.instance.primaryFocus?.debugLabel,
      'final-confirmation-action',
    );
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pump();
    expect(controller.processState, ProcessState.submitting);

    await tester.tap(find.text('활동'));
    await tester.pump();
    logs.completeSave();
    await tester.pumpAndSettle();

    expect(controller.processState, ProcessState.ready);
    expect(controller.inputMode, InputMode.image);
    expect(
      FocusManager.instance.primaryFocus?.debugLabel,
      isNot('next-image-action'),
    );
    expect(find.text('저장된 활동이 없어요'), findsOneWidget);
    controller.dispose();
    semantics.dispose();
  });

  testWidgets('저장 실패는 확인 결과와 단일 다시 저장 행동을 유지한다', (tester) async {
    final semantics = tester.ensureSemantics();
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final logs = _FailOnceLogRepository();
    final fileGateway = _CountingFileGateway();
    final controller =
        ScannerController(
            _UnusedApi(),
            _EmptyCameraGateway(),
            fileGateway,
            logs,
            testCatalog,
          )
          ..cameraInitializing = false
          ..inputMode = InputMode.image
          ..imageBytes = _testInputImage.bytes
          ..imageFileName = _testInputImage.fileName
          ..imageSize = const Size(400, 400)
          ..processState = ProcessState.reviewing
          ..response = _approvedResponse
          ..detections = testReviewDetections(_approvedResponse);
    await controller.submit();

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('저장하지 못했어요. 확인 결과는 유지됐어요.'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '다시 저장'), findsOneWidget);
    expect(find.byType(FilledButton), findsOneWidget);
    expect(controller.detections.single.isConfirmed, isTrue);
    final error = find.bySemanticsLabel('저장하지 못했어요. 확인 결과는 유지됐어요.');
    expect(error, findsOneWidget);
    expect(
      tester
          .getSemantics(error)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_save_error_1280x720.png'),
    );

    tester.platformDispatcher.textScaleFactorTestValue = 1.5;
    addTearDown(tester.platformDispatcher.clearTextScaleFactorTestValue);
    await tester.pumpAndSettle();
    expect(find.text('저장하지 못했어요. 확인 결과는 유지됐어요.'), findsOneWidget);
    expect(
      tester.getSize(find.widgetWithText(FilledButton, '다시 저장')).height,
      greaterThanOrEqualTo(48),
    );
    expect(tester.takeException(), isNull);

    await tester.tap(find.widgetWithText(FilledButton, '다시 저장'));
    await tester.pumpAndSettle();

    expect(logs.saveCalls, 2);
    expect(controller.processState, ProcessState.ready);
    expect(find.text('1개 상품을 확정했어요'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '다시 저장'), findsNothing);
    controller.dispose();
    semantics.dispose();
  });

  testWidgets('키보드 저장 실패는 같은 CTA에서 재시도와 완료까지 이어진다', (tester) async {
    final semantics = tester.ensureSemantics();
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final logs = _FailOnceLogRepository();
    final fileGateway = _CountingFileGateway();
    final controller =
        ScannerController(
            _UnusedApi(),
            _EmptyCameraGateway(),
            fileGateway,
            logs,
            testCatalog,
          )
          ..cameraInitializing = false
          ..inputMode = InputMode.image
          ..imageBytes = _testInputImage.bytes
          ..imageFileName = _testInputImage.fileName
          ..imageSize = const Size(400, 400)
          ..processState = ProcessState.reviewing
          ..response = _approvedResponse
          ..detections = testReviewDetections(_approvedResponse);

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();

    final finalAction = find.bySemanticsLabel('1개 상품 최종 확정');
    var finalActionFocused = false;
    for (var index = 0; index < 12 && !finalActionFocused; index += 1) {
      await tester.sendKeyEvent(LogicalKeyboardKey.tab);
      await tester.pump();
      finalActionFocused =
          tester
              .getSemantics(finalAction)
              .getSemanticsData()
              .flagsCollection
              .isFocused ==
          Tristate.isTrue;
    }
    expect(finalActionFocused, isTrue);

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();

    expect(logs.saveCalls, 1);
    expect(controller.processState, ProcessState.reviewing);
    final retry = find.bySemanticsLabel('다시 저장');
    expect(retry, findsOneWidget);
    expect(
      tester.getSemantics(retry).getSemanticsData().flagsCollection.isFocused,
      Tristate.isTrue,
    );
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('저장하지 못했어요. 확인 결과는 유지됐어요.'))
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_save_error_focus_1280x720.png'),
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();

    expect(logs.saveCalls, 2);
    expect(controller.processState, ProcessState.ready);
    expect(controller.inputMode, InputMode.image);
    expect(find.bySemanticsLabel('다시 저장'), findsNothing);
    final completion = find.bySemanticsLabel('1개 상품을 확정했어요');
    expect(completion, findsOneWidget);
    expect(
      tester
          .getSemantics(completion)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    expect(FocusManager.instance.primaryFocus?.debugLabel, 'next-image-action');
    final nextImageAction = find.widgetWithText(FilledButton, '이미지 선택');
    expect(nextImageAction, findsOneWidget);
    expect(
      tester
          .getSemantics(nextImageAction)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    expect(find.widgetWithText(FilledButton, '다시 연결'), findsNothing);
    expect(find.widgetWithText(OutlinedButton, '카메라로 돌아가기'), findsOneWidget);
    expect(find.text('다음 이미지를 선택해 주세요'), findsOneWidget);
    expect(find.bySemanticsLabel('이미지 미리보기 영역, 선택된 이미지 없음'), findsOneWidget);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_completed_focus_1280x720.png'),
    );
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pump();
    expect(fileGateway.pickCalls, 1);
    await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
    await tester.sendKeyDownEvent(LogicalKeyboardKey.keyO);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.keyO);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
    await tester.pump();
    expect(fileGateway.pickCalls, 2);
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pump();
    expect(FocusManager.instance.primaryFocus, isNotNull);

    controller.dispose();
    semantics.dispose();
  });
}

final InputImage _testInputImage = InputImage(
  bytes: base64Decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  ),
  fileName: 'test.png',
);

const _approvedResponse = ScanResponse(
  requestId: 'request_approved_1234',
  status: ScanStatus.approved,
  reasonCodes: [],
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
  ],
  processingTimeMs: 51,
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

class _CountingFileGateway implements ImageFileGateway {
  int pickCalls = 0;

  @override
  Future<InputImage?> pick() async {
    pickCalls += 1;
    return null;
  }
}

class _MemoryLogRepository implements ScanLogRepository {
  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async => const [];

  @override
  Future<void> save(ScanLogRecord record) async {}
}

class _DeferredLogRepository implements ScanLogRepository {
  final Completer<void> _saveCompleter = Completer<void>();

  void completeSave() => _saveCompleter.complete();

  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async => const [];

  @override
  Future<void> save(ScanLogRecord record) => _saveCompleter.future;
}

class _FailOnceLogRepository implements ScanLogRepository {
  int saveCalls = 0;

  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async => const [];

  @override
  Future<void> save(ScanLogRecord record) async {
    saveCalls += 1;
    if (saveCalls == 1) throw StateError('disk unavailable');
  }
}
