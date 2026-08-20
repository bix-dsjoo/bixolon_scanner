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
import 'package:product_scanner/theme/app_tokens.dart';
import 'package:product_scanner/widgets/app_components.dart';

import 'support/test_catalog.dart';

void main() {
  testWidgets('폐기 확인은 이미지 선택과 다시 촬영의 대상·동사를 일치시킨다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    addTearDown(tester.platformDispatcher.clearTextScaleFactorTestValue);
    final semantics = tester.ensureSemantics();

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
          ..response = _response
          ..detections = testReviewDetections(_response)
          ..selectedItemId = 'item_002';
    controller.confirmCandidate('item_002', _response.items[1].top3.first);

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();
    expect(
      tester.widget<MaterialApp>(find.byType(MaterialApp)).title,
      'BIXOLON Scanner v0.0.1',
    );

    await tester.tap(find.widgetWithText(OutlinedButton, '다른 이미지 선택'));
    await tester.pumpAndSettle();
    expect(find.text('다른 이미지를 선택할까요?'), findsOneWidget);
    expect(find.text('지금까지 확인한 상품 선택이 사라져요.'), findsOneWidget);
    expect(find.byType(AppConfirmDialog), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '다른 이미지 선택'), findsOneWidget);
    expect(find.text('다시 촬영할까요?'), findsNothing);
    final dialog = tester.widget<AlertDialog>(find.byType(AlertDialog));
    expect(dialog.constraints?.minWidth, AppDesignTokens.standard.dialogWidth);
    expect(dialog.constraints?.maxWidth, AppDesignTokens.standard.dialogWidth);
    final cancel = find.widgetWithText(OutlinedButton, '취소');
    final confirm = find.widgetWithText(FilledButton, '다른 이미지 선택');
    expect(tester.getSize(cancel).height, 48);
    expect(tester.getSize(confirm).height, 48);
    expect(tester.widget<OutlinedButton>(cancel).focusNode?.hasFocus, isTrue);
    final cancelSemantics = tester.getSemantics(cancel).getSemanticsData();
    expect(cancelSemantics.flagsCollection.isButton, isTrue);
    expect(cancelSemantics.flagsCollection.isFocused, Tristate.isTrue);
    expect(cancelSemantics.hasAction(SemanticsAction.tap), isTrue);
    await expectLater(
      find.byType(Overlay).first,
      matchesGoldenFile('goldens/discard_image_dialog_1280x720.png'),
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    expect(tester.widget<FilledButton>(confirm).focusNode?.hasFocus, isTrue);
    final confirmSemantics = tester.getSemantics(confirm).getSemanticsData();
    expect(confirmSemantics.flagsCollection.isButton, isTrue);
    expect(confirmSemantics.flagsCollection.isFocused, Tristate.isTrue);
    expect(confirmSemantics.hasAction(SemanticsAction.tap), isTrue);
    await expectLater(
      find.byType(Overlay).first,
      matchesGoldenFile('goldens/discard_image_confirm_focus_1280x720.png'),
    );
    await tester.sendKeyEvent(LogicalKeyboardKey.escape);
    await tester.pumpAndSettle();
    expect(find.byType(AppConfirmDialog), findsNothing);
    expect(controller.hasUserChanges, isTrue);

    controller
      ..inputMode = InputMode.camera
      ..notifyListeners();
    tester.platformDispatcher.textScaleFactorTestValue = 1.5;
    await tester.pumpAndSettle();
    await tester.tap(find.widgetWithText(OutlinedButton, '다시 촬영'));
    await tester.pumpAndSettle();
    expect(find.text('다시 촬영할까요?'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '다시 촬영'), findsOneWidget);
    expect(tester.takeException(), isNull);
    expect(
      tester.getSize(find.widgetWithText(OutlinedButton, '취소')).height,
      48,
    );
    expect(
      tester.getSize(find.widgetWithText(FilledButton, '다시 촬영')).height,
      48,
    );
    await expectLater(
      find.byType(Overlay).first,
      matchesGoldenFile('goldens/discard_camera_dialog_text_150_1280x720.png'),
    );
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();
    expect(find.byType(AppConfirmDialog), findsNothing);
    expect(controller.hasUserChanges, isTrue);

    tester.platformDispatcher.textScaleFactorTestValue = 1;
    controller
      ..inputMode = InputMode.image
      ..notifyListeners();
    await tester.pumpAndSettle();
    await tester.tap(find.widgetWithText(OutlinedButton, '다른 이미지 선택'));
    await tester.pumpAndSettle();
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    expect(
      tester
          .widget<FilledButton>(find.widgetWithText(FilledButton, '다른 이미지 선택'))
          .focusNode
          ?.hasFocus,
      isTrue,
    );
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();
    expect(find.byType(AppConfirmDialog), findsNothing);
    expect(controller.hasUserChanges, isTrue);

    await tester.tap(find.widgetWithText(OutlinedButton, '카메라로 돌아가기'));
    await tester.pumpAndSettle();
    expect(find.text('카메라로 돌아갈까요?'), findsOneWidget);
    final returnToCamera = find.widgetWithText(FilledButton, '카메라로 돌아가기');
    expect(returnToCamera, findsOneWidget);
    await tester.tap(returnToCamera);
    await tester.pumpAndSettle();
    expect(controller.inputMode, InputMode.camera);
    expect(controller.processState, ProcessState.ready);
    expect(controller.imageBytes, isNull);
    expect(controller.hasUserChanges, isFalse);

    semantics.dispose();
    controller.dispose();
  });
}

final InputImage _testInputImage = InputImage(
  bytes: base64Decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  ),
  fileName: 'test.png',
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
  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async => const [];

  @override
  Future<void> save(ScanLogRecord record) async {}
}
