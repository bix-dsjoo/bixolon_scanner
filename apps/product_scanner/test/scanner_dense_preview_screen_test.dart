import 'dart:convert';
import 'dart:ui' show Tristate;

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
import 'package:product_scanner/theme/app_copy.dart';
import 'package:product_scanner/theme/app_tokens.dart';

import 'support/test_catalog.dart';

void main() {
  testWidgets('밀집 객체에서도 현재 검수 box와 44px 선택 영역을 유지한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
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
          ..response = _denseResponse
          ..detections = testReviewDetections(_denseResponse)
          ..selectedItemId = 'item_002';

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text(AppPreviewCopy.selectedImage), findsOneWidget);
    expect(find.text('2  현재 검수'), findsOneWidget);
    expect(find.text('2 / 6'), findsOneWidget);
    expect(find.text('3  ?'), findsOneWidget);
    for (final item in _denseResponse.items) {
      final box = find.byKey(ValueKey('detection-box-${item.itemId}'));
      expect(box, findsOneWidget);
      final size = tester.getSize(box);
      expect(size.width, greaterThanOrEqualTo(44));
      expect(size.height, greaterThanOrEqualTo(44));
    }
    FocusNode previewFocusNode(String itemId) {
      final box = find.byKey(ValueKey('detection-box-$itemId'));
      final inkWell = find.descendant(of: box, matching: find.byType(InkWell));
      return tester.widget<InkWell>(inkWell).focusNode!;
    }

    final previewFocusNodes = <String, FocusNode>{
      for (final item in _denseResponse.items)
        item.itemId: previewFocusNode(item.itemId),
    };

    FocusNode rowFocusNode(String itemId) {
      final row = find.byKey(ValueKey('detection-row-$itemId'));
      final inkWell = find.descendant(of: row, matching: find.byType(InkWell));
      return tester.widget<InkWell>(inkWell).focusNode!;
    }

    final selectedRowFocusNode = rowFocusNode('item_002');
    final nextRowFocusNode = rowFocusNode('object_alpha');
    expect(selectedRowFocusNode.skipTraversal, isFalse);
    expect(nextRowFocusNode.skipTraversal, isTrue);
    selectedRowFocusNode.requestFocus();
    await tester.pumpAndSettle();
    await tester.sendKeyEvent(LogicalKeyboardKey.arrowDown);
    await tester.pumpAndSettle();
    expect(controller.selectedItemId, 'object_alpha');
    expect(selectedRowFocusNode.skipTraversal, isTrue);
    expect(nextRowFocusNode.skipTraversal, isFalse);
    expect(nextRowFocusNode.hasFocus, isTrue);

    await tester.sendKeyEvent(LogicalKeyboardKey.arrowUp);
    await tester.pumpAndSettle();
    expect(controller.selectedItemId, 'item_002');
    expect(selectedRowFocusNode.skipTraversal, isFalse);
    expect(nextRowFocusNode.skipTraversal, isTrue);
    expect(selectedRowFocusNode.hasFocus, isTrue);

    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    final firstCandidate = find.byKey(
      const ValueKey('candidate-item_002-bread_13'),
    );
    final firstCandidateInkWell = find.descendant(
      of: firstCandidate,
      matching: find.byType(InkWell),
    );
    final firstCandidateFocusNode = tester
        .widget<InkWell>(firstCandidateInkWell)
        .focusNode!;
    expect(firstCandidateFocusNode.hasFocus, isTrue);
    final secondCandidate = find.byKey(
      const ValueKey('candidate-item_002-bread_04'),
    );
    final secondCandidateFocusNode = tester
        .widget<InkWell>(
          find.descendant(of: secondCandidate, matching: find.byType(InkWell)),
        )
        .focusNode!;
    expect(firstCandidateFocusNode.skipTraversal, isFalse);
    expect(secondCandidateFocusNode.skipTraversal, isTrue);
    await tester.sendKeyEvent(LogicalKeyboardKey.arrowDown);
    await tester.pumpAndSettle();
    expect(controller.selectedItemId, 'item_002');
    expect(controller.selectedDetection?.finalProduct, isNull);
    expect(firstCandidateFocusNode.skipTraversal, isTrue);
    expect(secondCandidateFocusNode.skipTraversal, isFalse);
    expect(secondCandidateFocusNode.hasFocus, isTrue);
    await tester.sendKeyEvent(LogicalKeyboardKey.arrowUp);
    await tester.pumpAndSettle();
    expect(firstCandidateFocusNode.hasFocus, isTrue);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile(
        'goldens/scanner_dense_object_candidate_focus_1280x720.png',
      ),
    );

    await tester.sendKeyDownEvent(LogicalKeyboardKey.shiftLeft);
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.shiftLeft);
    await tester.pumpAndSettle();
    expect(selectedRowFocusNode.hasFocus, isTrue);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_dense_object_focus_1280x720.png'),
    );

    expect(previewFocusNodes['item_002']!.skipTraversal, isFalse);
    for (final entry in previewFocusNodes.entries) {
      if (entry.key != 'item_002') expect(entry.value.skipTraversal, isTrue);
    }
    previewFocusNodes['item_002']!.requestFocus();
    await tester.pumpAndSettle();
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    expect(previewFocusNodes.values.any((node) => node.hasFocus), isFalse);
    FocusManager.instance.primaryFocus?.unfocus();
    await tester.pumpAndSettle();
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('2번 현재 검수, 확인 필요 상품 영역'))
          .getSemanticsData()
          .flagsCollection
          .isSelected,
      Tristate.isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_dense_preview_1280x720.png'),
    );

    final firstBox = find.byKey(const ValueKey('detection-box-item_001'));
    final firstInkWell = find.descendant(
      of: firstBox,
      matching: find.byType(InkWell),
    );
    tester.widget<InkWell>(firstInkWell).focusNode!.requestFocus();
    await tester.pumpAndSettle();

    final firstSemantics = find.bySemanticsLabel(RegExp(r'^1번 .*확정 상품 영역$'));
    expect(
      tester
          .getSemantics(firstSemantics)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    final focusSurface = tester.widget<AnimatedContainer>(
      find.byKey(const ValueKey('detection-focus-item_001')),
    );
    final focusBorder =
        (focusSurface.decoration! as BoxDecoration).border! as Border;
    expect(focusBorder.top.color, AppComponentColors.light.focusRing);
    expect(focusBorder.top.width, 2);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_dense_preview_focus_1280x720.png'),
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();
    expect(controller.selectedItemId, 'item_001');
    expect(previewFocusNodes['item_001']!.skipTraversal, isFalse);
    expect(previewFocusNodes['item_002']!.skipTraversal, isTrue);

    await tester.sendKeyEvent(LogicalKeyboardKey.arrowDown);
    await tester.pumpAndSettle();
    expect(controller.selectedItemId, 'item_002');
    expect(previewFocusNodes['item_001']!.skipTraversal, isTrue);
    expect(previewFocusNodes['item_002']!.skipTraversal, isFalse);
    final movedSelection = find.bySemanticsLabel('2번 현재 검수, 확인 필요 상품 영역');
    expect(
      tester
          .getSemantics(movedSelection)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    expect(
      tester
          .getSemantics(find.bySemanticsLabel(RegExp(r'^1번 .*확정 상품 영역$')))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isFalse,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile(
        'goldens/scanner_dense_preview_direction_focus_1280x720.png',
      ),
    );

    final thirdBox = find.byKey(const ValueKey('detection-box-object_alpha'));
    final thirdInkWell = find.descendant(
      of: thirdBox,
      matching: find.byType(InkWell),
    );
    tester.widget<InkWell>(thirdInkWell).focusNode!.requestFocus();
    await tester.pumpAndSettle();
    await tester.sendKeyEvent(LogicalKeyboardKey.space);
    await tester.pumpAndSettle();
    expect(controller.selectedItemId, 'object_alpha');

    final candidate = find.byKey(
      const ValueKey('candidate-object_alpha-bread_13'),
    );
    final candidateInkWell = find.descendant(
      of: candidate,
      matching: find.byType(InkWell),
    );
    final candidateFocusNode = tester
        .widget<InkWell>(candidateInkWell)
        .focusNode!;
    candidateFocusNode.requestFocus();
    await tester.pumpAndSettle();
    await tester.sendKeyEvent(LogicalKeyboardKey.arrowUp);
    await tester.pumpAndSettle();
    expect(controller.selectedItemId, 'object_alpha');
    expect(controller.selectedDetection?.finalProduct, isNull);
    final lastCandidateFocusNode = tester
        .widget<InkWell>(
          find.descendant(
            of: find.byKey(const ValueKey('candidate-object_alpha-bread_11')),
            matching: find.byType(InkWell),
          ),
        )
        .focusNode!;
    expect(FocusManager.instance.primaryFocus, same(lastCandidateFocusNode));
    expect(candidateFocusNode.skipTraversal, isTrue);
    expect(lastCandidateFocusNode.skipTraversal, isFalse);

    controller.resetSession();
    await tester.pumpAndSettle();
    expect(find.byKey(const ValueKey('detection-box-item_001')), findsNothing);
    expect(tester.takeException(), isNull);

    semantics.dispose();
    controller.dispose();
  });
}

final InputImage _testInputImage = InputImage(
  bytes: base64Decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  ),
  fileName: 'dense.png',
);

const _candidates = [
  Candidate(
    classId: 'bread_13',
    className: 'Muffin',
    displayName: 'Muffin',
    confidence: .64,
  ),
  Candidate(
    classId: 'bread_04',
    className: 'Scon',
    displayName: 'Scon',
    confidence: .22,
  ),
  Candidate(
    classId: 'bread_11',
    className: 'Bagel',
    displayName: 'Bagel',
    confidence: .14,
  ),
];

const _denseResponse = ScanResponse(
  requestId: 'request_dense_preview',
  status: ScanStatus.unknown,
  reasonCodes: ['ITEM_BELOW_APPROVAL_THRESHOLD'],
  items: [
    ScanItem(
      itemId: 'item_001',
      bbox: BoundingBox(x: 130, y: 130, width: 30, height: 30),
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
      bbox: BoundingBox(x: 145, y: 145, width: 18, height: 18),
      status: ItemStatus.unknown,
      reasonCodes: ['BELOW_APPROVAL_THRESHOLD'],
      prediction: null,
      top3: _candidates,
      confidence: .64,
    ),
    ScanItem(
      itemId: 'object_alpha',
      bbox: BoundingBox(x: 164, y: 132, width: 22, height: 22),
      status: ItemStatus.unknown,
      reasonCodes: ['BELOW_APPROVAL_THRESHOLD'],
      prediction: null,
      top3: _candidates,
      confidence: .61,
    ),
    ScanItem(
      itemId: 'item_004',
      bbox: BoundingBox(x: 136, y: 170, width: 16, height: 16),
      status: ItemStatus.approved,
      reasonCodes: [],
      prediction: Product(
        classId: 'bread_03',
        className: 'Waffle',
        displayName: 'Waffle',
      ),
      top3: [],
      confidence: .96,
    ),
    ScanItem(
      itemId: 'item_005',
      bbox: BoundingBox(x: 140, y: 140, width: 55, height: 55),
      status: ItemStatus.approved,
      reasonCodes: [],
      prediction: Product(
        classId: 'bread_12',
        className: 'Egg Tart',
        displayName: 'Egg Tart',
      ),
      top3: [],
      confidence: .95,
    ),
    ScanItem(
      itemId: 'item_006',
      bbox: BoundingBox(x: 188, y: 166, width: 20, height: 20),
      status: ItemStatus.approved,
      reasonCodes: [],
      prediction: Product(
        classId: 'bread_11',
        className: 'Bagel',
        displayName: 'Bagel',
      ),
      top3: [],
      confidence: .94,
    ),
  ],
  processingTimeMs: 82,
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
