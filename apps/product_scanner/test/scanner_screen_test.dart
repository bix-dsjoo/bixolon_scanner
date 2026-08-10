import 'dart:async';
import 'dart:convert';
import 'dart:ui' show SemanticsAction, Tristate;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/activity/activity_filters.dart';
import 'package:product_scanner/app.dart';
import 'package:product_scanner/controllers/scanner_controller.dart';
import 'package:product_scanner/models/scan_models.dart';
import 'package:product_scanner/services/image_input.dart';
import 'package:product_scanner/services/scan_log_repository.dart';
import 'package:product_scanner/services/scanner_api.dart';
import 'package:product_scanner/theme/app_copy.dart';
import 'package:product_scanner/theme/app_tokens.dart';
import 'package:product_scanner/widgets/app_scan_guide.dart';

import 'support/test_catalog.dart';

void main() {
  testWidgets('1440x900 작업대에서 확인 필요 상품과 Top-3를 표시한다', (tester) async {
    tester.view.physicalSize = const Size(1440, 900);
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

    expect(find.byType(AppScanGuide), findsNothing);
    expect(find.text('이미지 입력 · 카메라 미연결'), findsOneWidget);
    expect(find.text('카메라 확인 필요'), findsNothing);
    expect(tester.getSize(find.byType(Scaffold).first), const Size(1440, 900));
    expect(
      tester.getSize(find.byKey(const ValueKey('scan-result-panel'))).width,
      closeTo(1440 * AppDesignTokens.standard.scanResultPanelFraction, .01),
    );
    expect(
      tester.getSize(find.byKey(const ValueKey('review-inspector'))).height,
      lessThanOrEqualTo(AppDesignTokens.standard.reviewInspectorMaxHeight),
    );
    expect(
      tester.getSize(find.byKey(const ValueKey('scan-result-header'))).height,
      AppDesignTokens.standard.headerHeight,
    );
    final reviewListRect = tester.getRect(
      find.byKey(const ValueKey('review-object-list-frame')),
    );
    final inspectorRect = tester.getRect(
      find.byKey(const ValueKey('review-inspector')),
    );
    expect(
      reviewListRect.height,
      closeTo(AppDesignTokens.standard.rowHeight * 2 + 1, .01),
    );
    expect(inspectorRect.top, closeTo(reviewListRect.bottom, .01));
    expect(find.text('2 / 2'), findsOneWidget);
    expect(find.bySemanticsLabel('현재 2번 상품, 전체 2개'), findsOneWidget);
    expect(find.bySemanticsLabel('2번 상품을 확인해 주세요'), findsOneWidget);
    expect(find.bySemanticsLabel(RegExp(r'^검수 상태\.')), findsNothing);

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_unknown_1440x900.png'),
    );

    final previousAction = find.bySemanticsLabel('이전 상품 (↑)');
    for (var index = 0; index < 20; index += 1) {
      await tester.sendKeyEvent(LogicalKeyboardKey.tab);
      await tester.pump();
      if (tester
              .getSemantics(previousAction)
              .getSemanticsData()
              .flagsCollection
              .isFocused ==
          Tristate.isTrue) {
        break;
      }
    }
    expect(
      tester
          .getSemantics(previousAction)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    await tester.pumpAndSettle();
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_step_navigation_focus_1440x900.png'),
    );
    FocusManager.instance.primaryFocus?.unfocus();
    await tester.pumpAndSettle();

    expect(find.text('BIXOLON Scanner'), findsOneWidget);
    expect(find.text('상품 확인이 필요해요'), findsOneWidget);
    expect(find.text('2번 상품을 확인해 주세요'), findsOneWidget);
    expect(find.text('선택하면 다음 확인 항목으로 이동해요.'), findsOneWidget);
    final currentDetection = find.bySemanticsLabel('2번 상품, 확인 필요');
    final currentDetectionData = tester
        .getSemantics(currentDetection)
        .getSemanticsData();
    expect(
      currentDetectionData.flagsCollection.isInMutuallyExclusiveGroup,
      isTrue,
    );
    expect(currentDetectionData.flagsCollection.isSelected, Tristate.isTrue);
    expect(find.text('머핀'), findsOneWidget);
    expect(find.text('베이글'), findsOneWidget);
    expect(
      tester
          .getSize(find.byKey(const ValueKey('candidate-item_002-bread_13')))
          .height,
      greaterThanOrEqualTo(60),
    );
    expect(find.byIcon(Icons.arrow_forward_rounded), findsOneWidget);
    expect(find.text('1 / 2 상품 확인 완료'), findsOneWidget);
    final incompleteAction = find.widgetWithText(FilledButton, '1개 상품 확인 필요');
    expect(incompleteAction, findsOneWidget);
    expect(tester.widget<FilledButton>(incompleteAction).onPressed, isNull);

    await tester.tap(find.text('머핀'));
    await tester.pumpAndSettle();
    expect(find.text('2개 상품 확인 완료'), findsOneWidget);
    expect(find.byKey(const ValueKey('step-navigator')), findsNothing);
    final submitAction = find.widgetWithText(FilledButton, '2개 상품 최종 확정');
    expect(submitAction, findsOneWidget);
    expect(tester.widget<FilledButton>(submitAction).onPressed, isNotNull);
    expect(tester.getSize(submitAction).height, greaterThanOrEqualTo(48));
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('2개 상품 최종 확정'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isFalse,
    );
    expect(
      find.bySemanticsLabel('검수 상태. 2개 상품 확인 완료. 최종 확정할 수 있어요.'),
      findsOneWidget,
    );

    controller.selectDetection('item_002');
    await tester.pumpAndSettle();
    expect(find.text('2번 상품을 변경할까요?'), findsOneWidget);
    final selectedCandidate = find.bySemanticsLabel('머핀, 신뢰도 75%');
    final alternateCandidate = find.bySemanticsLabel('스콘, 신뢰도 16%');
    expect(selectedCandidate, findsOneWidget);
    expect(alternateCandidate, findsOneWidget);
    final selectedCandidateData = tester
        .getSemantics(selectedCandidate)
        .getSemanticsData();
    final alternateCandidateData = tester
        .getSemantics(alternateCandidate)
        .getSemanticsData();
    expect(selectedCandidateData.flagsCollection.isButton, isTrue);
    expect(
      selectedCandidateData.flagsCollection.isInMutuallyExclusiveGroup,
      isTrue,
    );
    expect(selectedCandidateData.flagsCollection.isSelected, Tristate.isTrue);
    expect(selectedCandidateData.hasAction(SemanticsAction.tap), isTrue);
    expect(
      alternateCandidateData.flagsCollection.isInMutuallyExclusiveGroup,
      isTrue,
    );
    expect(alternateCandidateData.flagsCollection.isSelected, Tristate.isFalse);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_candidate_selected_1440x900.png'),
    );
    semantics.dispose();
    controller.dispose();
  });

  testWidgets('저장 중에는 객체와 상품 선택을 시각·의미·입력 모두 잠근다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final semantics = tester.ensureSemantics();
    final repository = _DeferredSaveLogRepository();
    final controller =
        ScannerController(
            _UnusedApi(),
            _EmptyCameraGateway(),
            _EmptyFileGateway(),
            repository,
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
    controller.selectDetection('item_002');
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();
    await tester.runAsync(
      () => precacheImage(
        MemoryImage(_testInputImage.bytes),
        tester.element(find.byType(Scaffold).first),
      ),
    );
    await tester.pump();

    final submission = controller.submit();
    await tester.pump(const Duration(milliseconds: 600));
    expect(controller.processState, ProcessState.submitting);

    expect(find.bySemanticsLabel('머핀, 신뢰도 75%'), findsNothing);
    expect(find.text('다른 상품 검색'), findsNothing);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('2번 상품, 머핀, 확정, 신뢰도 75%'))
          .getSemanticsData()
          .flagsCollection
          .isEnabled,
      Tristate.isFalse,
    );
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('2번 현재 검수, 머핀, 확정 상품 영역'))
          .getSemanticsData()
          .flagsCollection
          .isEnabled,
      Tristate.isFalse,
    );
    final previous = tester
        .getSemantics(find.bySemanticsLabel('이전 상품 (↑)'))
        .getSemanticsData();
    expect(previous.flagsCollection.isEnabled, Tristate.isFalse);
    expect(previous.hasAction(SemanticsAction.tap), isFalse);
    await tester.tap(find.text('크루아상'));
    await tester.sendKeyEvent(LogicalKeyboardKey.arrowUp);
    await tester.pump();
    expect(controller.selectedItemId, 'item_002');
    expect(controller.detections[1].finalProduct?.classId, 'bread_13');
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile(
        'goldens/scanner_submitting_review_locked_1280x720.png',
      ),
    );

    repository.completeSave();
    await submission;
    await tester.pumpAndSettle();
    semantics.dispose();
    controller.dispose();
  });

  testWidgets('키보드 후보 확정은 다음 후보와 최종 확정으로 포커스를 이어간다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final semantics = tester.ensureSemantics();

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

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();

    final firstChoice = find.bySemanticsLabel('머핀, 신뢰도 75%');
    for (var index = 0; index < 24; index += 1) {
      await tester.sendKeyEvent(LogicalKeyboardKey.tab);
      await tester.pump();
      if (tester
              .getSemantics(firstChoice)
              .getSemanticsData()
              .flagsCollection
              .isFocused ==
          Tristate.isTrue) {
        break;
      }
    }
    expect(
      tester
          .getSemantics(firstChoice)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );

    for (final expectedItemId in ['item_007', 'item_008']) {
      await tester.sendKeyEvent(LogicalKeyboardKey.enter);
      await tester.pumpAndSettle();
      expect(controller.selectedItemId, expectedItemId);
      expect(
        tester
            .getSemantics(firstChoice)
            .getSemanticsData()
            .flagsCollection
            .isFocused,
        Tristate.isTrue,
      );
    }

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();
    expect(controller.allConfirmed, isTrue);
    expect(controller.selectedItemId, isNull);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('8개 상품 최종 확정'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );

    semantics.dispose();
    controller.dispose();
  });

  testWidgets('상품이 많아도 후보 확정 뒤 다음 검수 행을 목록 안에 유지한다', (tester) async {
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

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();

    controller.confirmCandidate('item_006', response.items[5].top3.first);
    await tester.pumpAndSettle();
    expect(controller.selectedItemId, 'item_007');

    controller.confirmCandidate('item_007', response.items[6].top3.first);
    await tester.pumpAndSettle();
    expect(controller.selectedItemId, 'item_008');

    final listRect = tester.getRect(
      find.byKey(const ValueKey('detection-list')),
    );
    final selectedRect = tester.getRect(
      find.byKey(const ValueKey('detection-row-item_008')),
    );
    expect(selectedRect.top, greaterThanOrEqualTo(listRect.top));
    expect(selectedRect.bottom, lessThanOrEqualTo(listRect.bottom + 0.5));
    final listFrameRect = tester.getRect(
      find.byKey(const ValueKey('review-object-list-frame')),
    );
    final inspectorRect = tester.getRect(
      find.byKey(const ValueKey('review-inspector')),
    );
    expect(inspectorRect.top, closeTo(listFrameRect.bottom, .01));
    expect(
      inspectorRect.height,
      lessThanOrEqualTo(AppDesignTokens.standard.reviewInspectorMaxHeight),
    );
    final searchActionRect = tester.getRect(find.text('다른 상품 검색'));
    expect(searchActionRect.top, greaterThanOrEqualTo(inspectorRect.top));
    expect(searchActionRect.bottom, lessThanOrEqualTo(inspectorRect.bottom));
    expect(find.text('8번 상품을 확인해 주세요'), findsOneWidget);
    expect(tester.takeException(), isNull);
    controller.dispose();
  });

  testWidgets('Tab 순서로 상단 탐색을 이동하고 Enter로 실행한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final semantics = tester.ensureSemantics();

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
    await tester.pumpAndSettle();

    final scan = find.bySemanticsLabel('스캔');
    final activity = find.bySemanticsLabel('활동');
    expect(
      tester.getSemantics(scan).getSemanticsData().flagsCollection.isSelected,
      Tristate.isTrue,
    );
    expect(
      tester
          .getSemantics(activity)
          .getSemanticsData()
          .flagsCollection
          .isSelected,
      Tristate.isFalse,
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    expect(
      tester.getSemantics(scan).getSemanticsData().flagsCollection.isFocused,
      Tristate.isTrue,
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    expect(
      tester
          .getSemantics(activity)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/navigation_focus_1280x720.png'),
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();
    expect(find.text('활동 기록'), findsOneWidget);
    expect(
      tester
          .getSemantics(activity)
          .getSemanticsData()
          .flagsCollection
          .isSelected,
      Tristate.isTrue,
    );

    semantics.dispose();
    controller.dispose();
  });

  testWidgets('업로드 이미지 RECAPTURE는 다른 이미지 선택만 Primary로 제공한다', (tester) async {
    final semantics = tester.ensureSemantics();
    tester.view.physicalSize = const Size(1280, 720);
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
          ..imageBytes = _testInputImage.bytes
          ..imageFileName = _testInputImage.fileName
          ..imageSize = const Size(400, 400)
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

    expect(find.byType(AppScanGuide), findsOneWidget);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_recapture_1280x720.png'),
    );

    expect(find.text('다른 이미지를 선택해 주세요'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '다른 이미지 선택'), findsOneWidget);
    expect(find.text('분석하기'), findsNothing);
    final recaptureMessage = find.bySemanticsLabel(
      '다른 이미지를 선택해 주세요. 일부 상품이 이미지 밖으로 잘려 있어요.',
    );
    expect(recaptureMessage, findsOneWidget);
    final recaptureData = tester
        .getSemantics(recaptureMessage)
        .getSemanticsData();
    expect(recaptureData.flagsCollection.isLiveRegion, isTrue);
    expect(recaptureData.hasAction(SemanticsAction.tap), isFalse);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('다른 이미지 선택'))
          .getSemanticsData()
          .hasAction(SemanticsAction.tap),
      isTrue,
    );
    controller.dispose();
    semantics.dispose();
  });

  testWidgets('활동에서 저장된 스캔 로그와 접힌 진단 정보를 확인한다', (tester) async {
    final semantics = tester.ensureSemantics();
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
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    expect(
      tester
          .getSize(find.byKey(const ValueKey('activity-table-header')))
          .height,
      AppDesignTokens.standard.sectionHeaderHeight,
    );

    expect(
      tester.getSize(find.widgetWithText(TextField, '상품명 또는 Scan ID')).width,
      AppDesignTokens.standard.activitySearchWidth,
    );

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_list_1440x900.png'),
    );

    expect(find.text('머핀 외 2개'), findsOneWidget);
    expect(find.text('3개'), findsOneWidget);
    expect(find.text('머핀'), findsWidgets);
    expect(find.text('확정 상품'), findsOneWidget);
    expect(find.text('2개 수정'), findsOneWidget);
    expect(find.text('카메라 미연결'), findsOneWidget);
    expect(find.text('카메라 확인 필요'), findsNothing);
    expect(find.byIcon(Icons.arrow_forward_rounded), findsOneWidget);
    expect(find.text('request_activity_1234'), findsNothing);
    expect(find.textContaining('item_001'), findsNothing);
    expect(find.textContaining('92.0%'), findsNothing);
    final disclosure = find.bySemanticsLabel('진단 정보');
    expect(
      tester
          .getSemantics(disclosure)
          .getSemanticsData()
          .flagsCollection
          .isExpanded,
      Tristate.isFalse,
    );
    var disclosureFocused = false;
    for (var index = 0; index < 24 && !disclosureFocused; index += 1) {
      await tester.sendKeyEvent(LogicalKeyboardKey.tab);
      await tester.pump();
      disclosureFocused =
          tester
              .getSemantics(disclosure)
              .getSemanticsData()
              .flagsCollection
              .isFocused ==
          Tristate.isTrue;
    }
    await tester.pumpAndSettle();
    expect(disclosureFocused, isTrue);
    final disclosureSurface = tester.widget<AnimatedContainer>(
      find.byKey(const ValueKey('disclosure-surface-진단 정보')),
    );
    final disclosureBorder =
        (disclosureSurface.decoration! as BoxDecoration).border! as Border;
    expect(disclosureBorder.top.color, AppComponentColors.light.focusRing);
    expect(disclosureBorder.top.width, 2);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_diagnostics_focus_1440x900.png'),
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();
    expect(
      tester
          .getSemantics(disclosure)
          .getSemanticsData()
          .flagsCollection
          .isExpanded,
      Tristate.isTrue,
    );
    expect(find.text('스캔·모델·객체 판정 정보'), findsOneWidget);
    expect(find.text('모델 버전'), findsOneWidget);
    expect(find.text('request_activity_1234'), findsOneWidget);
    expect(find.text('객체별 판정'), findsOneWidget);
    expect(find.textContaining('item_001'), findsOneWidget);
    expect(find.textContaining('item_002'), findsOneWidget);
    expect(find.textContaining('item_003'), findsOneWidget);
    FocusManager.instance.primaryFocus?.unfocus();
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.textContaining('item_003'));
    await tester.pumpAndSettle();
    final diagnosticRect = tester.getRect(find.textContaining('item_003'));
    expect(diagnosticRect.top, greaterThanOrEqualTo(0));
    expect(diagnosticRect.bottom, lessThanOrEqualTo(900));
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_diagnostics_1440x900.png'),
    );
    semantics.dispose();
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
          ..detections = testReviewDetections(_response)
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

    final field = find.widgetWithText(TextField, '상품명 또는 class ID');
    expect(field, findsOneWidget);

    await tester.enterText(field, '없는상품');
    await tester.pump();
    expect(find.text('일치하는 상품이 없어요'), findsOneWidget);
    expect(find.text('검색어를 바꾸거나 class ID를 입력해 보세요.'), findsOneWidget);
    expect(
      tester
          .getSemantics(
            find.bySemanticsLabel('일치하는 상품이 없어요. 검색어를 바꾸거나 class ID를 입력해 보세요.'),
          )
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_product_search_empty_1440x900.png'),
    );

    await tester.enterText(field, 'Egg');
    await tester.pump();
    expect(find.text('에그 타르트'), findsOneWidget);
    final result = find.bySemanticsLabel('에그 타르트');
    expect(result, findsOneWidget);
    expect(tester.getSize(result).height, greaterThanOrEqualTo(60));
    final resultSemantics = tester.getSemantics(result).getSemanticsData();
    expect(resultSemantics.flagsCollection.isButton, isTrue);
    expect(resultSemantics.flagsCollection.isInMutuallyExclusiveGroup, isTrue);
    expect(resultSemantics.flagsCollection.isSelected, Tristate.isFalse);
    expect(resultSemantics.hasAction(SemanticsAction.tap), isTrue);

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_product_search_1440x900.png'),
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    expect(
      tester.getSemantics(result).getSemanticsData().flagsCollection.isFocused,
      Tristate.isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_product_search_focus_1440x900.png'),
    );
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();
    expect(find.text('2개 상품 확인 완료'), findsOneWidget);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('2개 상품 최종 확정'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    controller.dispose();
  });

  testWidgets('확정 상품을 다시 검색하면 현재 선택과 키보드 진입점을 유지한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
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
          ..detections = testReviewDetections(_response)
          ..selectedItemId = 'item_002';
    controller.confirmCandidate(
      'item_002',
      controller.detections[1].source.top3.first,
    );
    controller.selectedItemId = 'item_002';

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('다른 상품 검색'));
    await tester.pumpAndSettle();

    expect(find.text('2번 상품 변경'), findsOneWidget);
    expect(find.text('현재 상품  머핀 · bread_13'), findsOneWidget);

    final firstResult = find.byKey(const ValueKey('search-product-bread_03'));
    final currentResult = find.byKey(const ValueKey('search-product-bread_13'));
    FocusNode resultFocusNode(Finder result) => tester
        .widget<InkWell>(
          find.descendant(of: result, matching: find.byType(InkWell)),
        )
        .focusNode!;
    final firstFocusNode = resultFocusNode(firstResult);
    final currentFocusNode = resultFocusNode(currentResult);
    expect(firstFocusNode.skipTraversal, isTrue);
    expect(currentFocusNode.skipTraversal, isFalse);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('머핀'))
          .getSemanticsData()
          .flagsCollection
          .isSelected,
      Tristate.isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile(
        'goldens/scanner_product_search_current_context_1280x720.png',
      ),
    );

    final field = find.widgetWithText(TextField, '상품명 또는 class ID');
    await tester.tap(field);
    await tester.pumpAndSettle();
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    expect(currentFocusNode.hasFocus, isTrue);
    final inspectorRect = tester.getRect(
      find.byKey(const ValueKey('review-inspector')),
    );
    final currentResultRect = tester.getRect(
      find.descendant(of: currentResult, matching: find.byType(InkWell)),
    );
    expect(currentResultRect.top, greaterThanOrEqualTo(inspectorRect.top));
    expect(currentResultRect.bottom, lessThanOrEqualTo(inspectorRect.bottom));
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile(
        'goldens/scanner_product_search_current_selection_1280x720.png',
      ),
    );
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();
    expect(controller.searchItemId, isNull);
    expect(
      controller.detections[1].confirmationMethod,
      ConfirmationMethod.top3Selected,
    );
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('2개 상품 최종 확정'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    controller.dispose();
  });

  testWidgets('상품 검색 결과는 Tab 한 번으로 진입하고 방향키만으로 확정하지 않는다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
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
          ..detections = testReviewDetections(_response)
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

    final field = find.widgetWithText(TextField, '상품명 또는 class ID');
    final firstResult = find.byKey(const ValueKey('search-product-bread_03'));
    final secondResult = find.byKey(const ValueKey('search-product-bread_04'));
    final fifthResult = find.byKey(const ValueKey('search-product-bread_12'));
    FocusNode resultFocusNode(Finder result) => tester
        .widget<InkWell>(
          find.descendant(of: result, matching: find.byType(InkWell)),
        )
        .focusNode!;
    final firstFocusNode = resultFocusNode(firstResult);
    final secondFocusNode = resultFocusNode(secondResult);
    final fifthFocusNode = resultFocusNode(fifthResult);
    await tester.tap(field);
    await tester.pumpAndSettle();
    final fieldFocusNode = tester
        .widget<EditableText>(
          find.descendant(of: field, matching: find.byType(EditableText)),
        )
        .focusNode;

    expect(fieldFocusNode.hasFocus, isTrue);
    expect(find.bySemanticsLabel('검색 결과 6개'), findsOneWidget);
    expect(find.text('검색 결과'), findsOneWidget);
    expect(find.text('6개'), findsOneWidget);
    final inspector = find.byKey(const ValueKey('review-inspector'));
    final inspectorScrollbar = find.descendant(
      of: inspector,
      matching: find.byType(Scrollbar),
    );
    expect(
      tester.widget<Scrollbar>(inspectorScrollbar).thumbVisibility,
      isTrue,
    );
    expect(firstFocusNode.skipTraversal, isFalse);
    expect(secondFocusNode.skipTraversal, isTrue);
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    expect(firstFocusNode.hasFocus, isTrue);

    await tester.sendKeyEvent(LogicalKeyboardKey.arrowDown);
    await tester.pumpAndSettle();
    expect(controller.selectedItemId, 'item_002');
    expect(controller.selectedDetection?.finalProduct, isNull);
    expect(controller.searchItemId, 'item_002');
    expect(firstFocusNode.skipTraversal, isTrue);
    expect(secondFocusNode.skipTraversal, isFalse);
    expect(secondFocusNode.hasFocus, isTrue);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile(
        'goldens/scanner_product_search_roving_focus_1280x720.png',
      ),
    );

    for (var index = 0; index < 3; index += 1) {
      await tester.sendKeyEvent(LogicalKeyboardKey.arrowDown);
      await tester.pumpAndSettle();
    }
    expect(fifthFocusNode.hasFocus, isTrue);
    expect(controller.selectedDetection?.finalProduct, isNull);
    final inspectorRect = tester.getRect(inspector);
    final fifthResultRect = tester.getRect(
      find.descendant(of: fifthResult, matching: find.byType(InkWell)),
    );
    expect(fifthResultRect.top, greaterThanOrEqualTo(inspectorRect.top));
    expect(fifthResultRect.bottom, lessThanOrEqualTo(inspectorRect.bottom));
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile(
        'goldens/scanner_product_search_offscreen_focus_1280x720.png',
      ),
    );

    await tester.sendKeyDownEvent(LogicalKeyboardKey.shiftLeft);
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.shiftLeft);
    await tester.pumpAndSettle();
    expect(fieldFocusNode.hasFocus, isTrue);
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    expect(fifthFocusNode.hasFocus, isTrue);

    await tester.sendKeyEvent(LogicalKeyboardKey.space);
    await tester.pumpAndSettle();
    expect(controller.detections[1].finalProduct?.classId, 'bread_12');
    expect(controller.searchItemId, isNull);
    expect(controller.allConfirmed, isTrue);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('2개 상품 최종 확정'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    controller.dispose();
  });

  testWidgets('분석 중과 복구 유형별 ERROR를 서로 다른 행동으로 표시한다', (tester) async {
    final semantics = tester.ensureSemantics();
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final fileGateway = _StaticFileGateway(_testInputImage);
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
          ..imageSize = const Size(400, 400);

    controller.processState = ProcessState.analyzing;
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.runAsync(
      () => precacheImage(
        MemoryImage(_testInputImage.bytes),
        tester.element(find.byType(Scaffold).first),
      ),
    );
    await tester.pump();
    expect(find.text('상품을 찾고 있어요'), findsOneWidget);
    expect(find.text('이미지 분석 중'), findsNothing);
    expect(find.text('분석 중'), findsNWidgets(2));
    expect(find.bySemanticsLabel('입력 미리보기, 선택한 이미지'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '분석 중'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '분석하기'), findsNothing);
    expect(find.widgetWithText(OutlinedButton, '다른 이미지 선택'), findsNothing);
    expect(find.byTooltip('이미지 선택 (Ctrl+O)'), findsNothing);
    final progress = find.bySemanticsLabel('분석 중. 완료될 때까지 기다려 주세요');
    expect(progress, findsOneWidget);
    final progressData = tester.getSemantics(progress).getSemanticsData();
    expect(progressData.flagsCollection.isLiveRegion, isTrue);
    expect(progressData.flagsCollection.isButton, isTrue);
    expect(progressData.hasAction(SemanticsAction.tap), isFalse);
    await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
    await tester.sendKeyDownEvent(LogicalKeyboardKey.keyO);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.keyO);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
    await tester.pump();
    expect(fileGateway.pickCalls, 0);
    expect(controller.processState, ProcessState.analyzing);
    await tester.pump(const Duration(milliseconds: 600));
    await tester.pump();

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_analyzing_1280x720.png'),
    );

    controller
      ..processState = ProcessState.error
      ..errorMessage = '분석 서버에 연결할 수 없어요.'
      ..notifyListeners();
    await tester.pump();

    expect(find.byType(AppScanGuide), findsNothing);
    expect(find.text('이미지 입력 · 카메라 미연결'), findsOneWidget);

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_error_1280x720.png'),
    );

    expect(find.text('분석하지 못했어요'), findsOneWidget);
    expect(find.text('분석 서버에 연결할 수 없어요.'), findsOneWidget);
    expect(find.byIcon(Icons.cloud_off_outlined), findsOneWidget);
    expect(find.text('분석 오류'), findsWidgets);
    expect(find.widgetWithText(FilledButton, '다시 분석'), findsOneWidget);
    final serverError = find.bySemanticsLabel('분석하지 못했어요. 분석 서버에 연결할 수 없어요.');
    expect(serverError, findsOneWidget);
    expect(
      tester
          .getSemantics(serverError)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );

    controller
      ..errorRecovery = ScannerErrorRecovery.replaceInput
      ..errorMessage = 'JPEG 또는 PNG 이미지를 선택해 주세요.'
      ..notifyListeners();
    await tester.pump();

    expect(find.byType(AppScanGuide), findsNothing);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_input_error_1280x720.png'),
    );

    expect(find.text('이미지를 분석할 수 없어요'), findsOneWidget);
    expect(find.byIcon(Icons.broken_image_outlined), findsOneWidget);
    expect(find.byIcon(Icons.cloud_off_outlined), findsNothing);
    expect(find.widgetWithText(FilledButton, '다른 이미지 선택'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '다시 분석'), findsNothing);
    expect(find.text('재촬영 필요'), findsNothing);
    final inputError = find.bySemanticsLabel(
      '이미지를 분석할 수 없어요. JPEG 또는 PNG 이미지를 선택해 주세요.',
    );
    expect(inputError, findsOneWidget);
    expect(
      tester
          .getSemantics(inputError)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    controller.dispose();
    semantics.dispose();
  });

  testWidgets('활동 빈 상태에서 기록 생성 조건을 안내한다', (tester) async {
    final semantics = tester.ensureSemantics();
    tester.view.physicalSize = const Size(1280, 720);
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
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_empty_1280x720.png'),
    );

    expect(find.text('저장된 활동이 없어요'), findsOneWidget);
    final emptyMessage = find.bySemanticsLabel(
      '저장된 활동이 없어요. 상품을 최종 확정하면 이곳에서 확인할 수 있어요.',
    );
    expect(emptyMessage, findsOneWidget);
    expect(
      tester
          .getSemantics(emptyMessage)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    expect(find.text('상품을 최종 확정하면 이곳에서 확인할 수 있어요.'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '스캔 화면으로 이동'), findsOneWidget);
    expect(find.byType(TextField), findsNothing);
    expect(find.byType(ChoiceChip), findsNothing);

    await tester.tap(find.widgetWithText(FilledButton, '스캔 화면으로 이동'));
    await tester.pumpAndSettle();
    expect(find.text('저장된 활동이 없어요'), findsNothing);
    expect(find.text('입력 준비'), findsWidgets);
    controller.dispose();
    semantics.dispose();
  });

  testWidgets('1280x720 최소 작업대에서 검수 패널이 넘치지 않는다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
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
    expect(
      tester.getSize(find.byKey(const ValueKey('scan-result-panel'))).width,
      closeTo(1280 * AppDesignTokens.standard.scanResultPanelFraction, .01),
    );
    expect(find.text('2번 상품을 확인해 주세요'), findsOneWidget);
    expect(find.text('1 / 2 상품 확인 완료'), findsOneWidget);
    controller.dispose();
  });

  testWidgets('Windows 150% 표시 배율에서 1280x720 논리 작업대를 유지한다', (tester) async {
    tester.view.physicalSize = const Size(1920, 1080);
    tester.view.devicePixelRatio = 1.5;
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

    expect(
      tester.view.physicalSize / tester.view.devicePixelRatio,
      const Size(1280, 720),
    );
    expect(find.text('2번 상품을 확인해 주세요'), findsOneWidget);
    expect(tester.takeException(), isNull);
    controller.dispose();
  });

  testWidgets('방향키와 검색·닫기 단축키로 선택 상품을 탐색한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
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

    await tester.sendKeyEvent(LogicalKeyboardKey.arrowUp);
    await tester.pump();
    expect(controller.selectedItemId, 'item_001');
    expect(find.text('1 / 2'), findsOneWidget);

    await tester.sendKeyEvent(LogicalKeyboardKey.slash);
    await tester.pumpAndSettle();
    expect(controller.searchItemId, 'item_001');
    expect(find.widgetWithText(TextField, '상품명 또는 class ID'), findsOneWidget);
    expect(find.bySemanticsLabel('검색 닫기'), findsOneWidget);

    await tester.sendKeyEvent(LogicalKeyboardKey.escape);
    await tester.pumpAndSettle();
    expect(controller.searchItemId, isNull);
    expect(
      controller.detections.first.confirmationMethod,
      ConfirmationMethod.autoApproved,
    );
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('상품 변경'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    controller.dispose();
  });

  testWidgets('활동 필터와 검색 상태를 화면 전환 후에도 유지한다', (tester) async {
    tester.view.physicalSize = const Size(1440, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final repository = _MemoryLogRepository(
      logs: [_logSummary, _imageLogSummary],
    );
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
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    await tester.tap(find.widgetWithText(ChoiceChip, '이미지'));
    await tester.pumpAndSettle();
    expect(find.text('에그 타르트'), findsWidgets);
    expect(find.text('머핀'), findsNothing);

    final search = find.widgetWithText(TextField, '상품명 또는 Scan ID');
    await tester.enterText(search, '에그');
    await tester.pump();
    await tester.tap(find.text('스캔'));
    await tester.pump();
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    expect(tester.widget<TextField>(search).controller?.text, '에그');
    expect(find.text('에그 타르트'), findsWidgets);
    expect(find.text('검색 결과'), findsOneWidget);
    expect(find.text('1 / 2건'), findsOneWidget);
    expect(find.widgetWithText(TextButton, '모두 초기화'), findsOneWidget);
    controller.dispose();
  });

  testWidgets('Activity 검색어와 모든 조건을 화면에서 바로 초기화한다', (tester) async {
    tester.view.physicalSize = const Size(1440, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final semantics = tester.ensureSemantics();

    final controller = ScannerController(
      _UnusedApi(),
      _EmptyCameraGateway(),
      _EmptyFileGateway(),
      _MemoryLogRepository(logs: [_logSummary, _imageLogSummary]),
      testCatalog,
    )..cameraInitializing = false;

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    final search = find.widgetWithText(TextField, '상품명 또는 Scan ID');
    await tester.enterText(search, '에그');
    await tester.pumpAndSettle();
    expect(find.text('검색 결과'), findsOneWidget);
    expect(find.text('1 / 2건'), findsOneWidget);
    final clearSearch = find.bySemanticsLabel('검색어 지우기');
    expect(clearSearch, findsOneWidget);
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    expect(
      tester
          .getSemantics(clearSearch)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    final clearSurface = tester.widget<AnimatedContainer>(
      find.byKey(const ValueKey('icon-action-surface-검색어 지우기')),
    );
    final clearBorder =
        (clearSurface.decoration! as BoxDecoration).border! as Border;
    expect(clearBorder.top.color, AppComponentColors.light.focusRing);
    expect(clearBorder.top.width, 2);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_search_clear_focus_1440x900.png'),
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.space);
    await tester.pumpAndSettle();
    expect(tester.widget<TextField>(search).controller?.text, isEmpty);
    expect(find.text('활동 목록'), findsOneWidget);
    expect(find.text('2건'), findsOneWidget);

    await tester.tap(find.widgetWithText(ChoiceChip, '카메라'));
    await tester.pumpAndSettle();
    expect(find.text('1 / 2건'), findsOneWidget);
    final reset = find.widgetWithText(TextButton, '모두 초기화');
    expect(reset, findsOneWidget);

    await tester.tap(reset);
    await tester.pumpAndSettle();
    expect(find.text('활동 목록'), findsOneWidget);
    expect(find.text('2건'), findsOneWidget);
    expect(
      tester
          .widget<ChoiceChip>(find.widgetWithText(ChoiceChip, '전체').first)
          .selected,
      isTrue,
    );
    expect(find.widgetWithText(TextButton, '모두 초기화'), findsNothing);

    await tester.enterText(search, '없는 상품');
    tester.view.physicalSize = const Size(1280, 720);
    await tester.pumpAndSettle();
    expect(find.text('조건에 맞는 기록이 없어요'), findsOneWidget);
    expect(find.text('검색어나 필터를 바꾸거나 모두 초기화해 보세요.'), findsOneWidget);
    expect(find.widgetWithText(TextButton, '모두 초기화'), findsNothing);
    final emptyReset = find.widgetWithText(FilledButton, '모두 초기화');
    expect(emptyReset, findsOneWidget);
    expect(find.byType(FilledButton), findsOneWidget);
    expect(tester.getSize(emptyReset).height, greaterThanOrEqualTo(48));
    final noResults = find.bySemanticsLabel(
      '조건에 맞는 기록이 없어요. 검색어나 필터를 바꾸거나 모두 초기화해 보세요.',
    );
    expect(noResults, findsOneWidget);
    expect(
      tester
          .getSemantics(noResults)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('모두 초기화'))
          .getSemanticsData()
          .hasAction(SemanticsAction.tap),
      isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_no_results_1280x720.png'),
    );

    await tester.tap(emptyReset);
    await tester.pumpAndSettle();
    expect(tester.widget<TextField>(search).controller?.text, isEmpty);
    expect(find.text('활동 목록'), findsOneWidget);
    expect(find.text('2건'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '모두 초기화'), findsNothing);
    semantics.dispose();
    controller.dispose();
  });

  testWidgets('Activity는 Tab으로 필터 그룹에 진입하고 방향키로 값을 선택한다', (tester) async {
    tester.view.physicalSize = const Size(1440, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final semantics = tester.ensureSemantics();

    final controller = ScannerController(
      _UnusedApi(),
      _EmptyCameraGateway(),
      _EmptyFileGateway(),
      _MemoryLogRepository(logs: [_logSummary, _imageLogSummary]),
      testCatalog,
    )..cameraInitializing = false;

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    expect(
      find.bySemanticsLabel(RegExp(r'머핀 외 2개.*카메라.*2개 수정 활동 기록')),
      findsOneWidget,
    );
    expect(find.bySemanticsLabel('입력원, 전체'), findsOneWidget);
    expect(find.bySemanticsLabel('기간, 전체'), findsOneWidget);
    expect(find.bySemanticsLabel('전체'), findsNothing);

    final search = find.widgetWithText(TextField, '상품명 또는 Scan ID');
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pump();
    expect(tester.widget<TextField>(search).focusNode?.hasFocus, isTrue);

    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    final refresh = find.bySemanticsLabel('활동 기록 새로고침');
    expect(
      tester.getSemantics(refresh).getSemanticsData().flagsCollection.isFocused,
      Tristate.isTrue,
    );
    final refreshSurface = tester.widget<AnimatedContainer>(
      find.byKey(const ValueKey('icon-action-surface-활동 기록 새로고침')),
    );
    final refreshBorder =
        (refreshSurface.decoration! as BoxDecoration).border! as Border;
    expect(refreshBorder.top.color, AppComponentColors.light.focusRing);
    expect(refreshBorder.top.width, 2);
    expect(
      tester.getSize(refresh).height,
      greaterThanOrEqualTo(AppDesignTokens.standard.controlHeight),
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_refresh_focus_1440x900.png'),
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pump();
    final allInput = find.widgetWithText(ChoiceChip, '전체').first;
    expect(
      tester
          .getSemantics(allInput)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.arrowRight);
    await tester.pumpAndSettle();
    final cameraFilter = find.widgetWithText(ChoiceChip, '카메라');
    expect(find.bySemanticsLabel('입력원, 카메라'), findsOneWidget);
    expect(
      tester
          .getSemantics(cameraFilter)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    final focusedFilterChip = tester.widget<ChoiceChip>(cameraFilter);
    expect(focusedFilterChip.selected, isTrue);
    expect(focusedFilterChip.focusNode?.skipTraversal, isFalse);
    expect(
      tester.widget<ChoiceChip>(allInput).focusNode?.skipTraversal,
      isTrue,
    );
    expect(focusedFilterChip.side?.color, AppComponentColors.light.focusRing);
    expect(focusedFilterChip.side?.width, 2);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_filter_focus_1440x900.png'),
    );

    expect(find.text('머핀'), findsWidgets);
    expect(find.text('에그 타르트'), findsNothing);

    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pump();
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('기간, 전체'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );

    final sortDropdown = find.byType(DropdownButton<ActivitySortOrder>);
    var sortFocused = false;
    for (var index = 0; index < 12 && !sortFocused; index += 1) {
      await tester.sendKeyEvent(LogicalKeyboardKey.tab);
      await tester.pump();
      sortFocused =
          tester
              .getSemantics(sortDropdown)
              .getSemanticsData()
              .flagsCollection
              .isFocused ==
          Tristate.isTrue;
    }
    await tester.pumpAndSettle();
    expect(sortFocused, isTrue);
    final sortSurface = tester.widget<AnimatedContainer>(
      find.byKey(const ValueKey('dropdown-surface-활동 정렬')),
    );
    final sortBorder =
        (sortSurface.decoration! as BoxDecoration).border! as Border;
    expect(sortBorder.top.color, AppComponentColors.light.focusRing);
    expect(sortBorder.top.width, 2);
    expect(
      tester
          .getSemantics(sortDropdown)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_sort_focus_1440x900.png'),
    );

    semantics.dispose();
    controller.dispose();
  });

  testWidgets('Activity의 F5는 기록을 새로고침하고 검색 입력 중 Esc는 검색만 닫는다', (tester) async {
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
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();
    expect(repository.listCalls, 1);

    await tester.sendKeyEvent(LogicalKeyboardKey.f5);
    await tester.pumpAndSettle();
    expect(repository.listCalls, 2);

    await tester.sendKeyEvent(LogicalKeyboardKey.slash);
    await tester.pump();
    final search = find.widgetWithText(TextField, '상품명 또는 Scan ID');
    expect(tester.widget<TextField>(search).focusNode?.hasFocus, isTrue);
    await tester.enterText(search, '머핀');
    await tester.sendKeyEvent(LogicalKeyboardKey.escape);
    await tester.pump();
    expect(tester.widget<TextField>(search).focusNode?.hasFocus, isFalse);
    expect(tester.widget<TextField>(search).controller?.text, '머핀');
    controller.dispose();
  });

  testWidgets('상품 검색 입력 중 전역 방향키가 객체 선택을 바꾸지 않는다', (tester) async {
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
    await tester.tap(find.text('다른 상품 검색'));
    await tester.pumpAndSettle();
    final search = find.widgetWithText(TextField, '상품명 또는 class ID');
    expect(search, findsOneWidget);
    await tester.tap(search);
    await tester.enterText(search, 'M');

    await tester.sendKeyEvent(LogicalKeyboardKey.arrowUp);
    await tester.pump();
    expect(controller.selectedItemId, 'item_002');
    expect(controller.searchItemId, 'item_002');

    await tester.sendKeyEvent(LogicalKeyboardKey.escape);
    await tester.pumpAndSettle();
    expect(controller.searchItemId, isNull);
    expect(controller.selectedDetection?.finalProduct, isNull);
    final searchAction = find.bySemanticsLabel('다른 상품 검색');
    expect(
      tester
          .getSemantics(searchAction)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();
    expect(controller.searchItemId, 'item_002');
    await tester.tap(search);
    await tester.enterText(search, 'M');

    await tester.sendKeyDownEvent(LogicalKeyboardKey.shiftLeft);
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.shiftLeft);
    await tester.pumpAndSettle();
    final back = find.bySemanticsLabel('후보로 돌아가기');
    expect(
      tester.getSemantics(back).getSemanticsData().flagsCollection.isFocused,
      Tristate.isTrue,
    );
    final backSurface = tester.widget<AnimatedContainer>(
      find.byKey(const ValueKey('icon-action-surface-후보로 돌아가기')),
    );
    final backBorder =
        (backSurface.decoration! as BoxDecoration).border! as Border;
    expect(backBorder.top.color, AppComponentColors.light.focusRing);
    expect(backBorder.top.width, 2);
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();
    expect(find.widgetWithText(TextField, '상품명 또는 class ID'), findsNothing);
    expect(controller.selectedDetection?.finalProduct, isNull);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('다른 상품 검색'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile(
        'goldens/scanner_product_search_keyboard_cancel_1280x720.png',
      ),
    );
    semantics.dispose();
    controller.dispose();
  });

  testWidgets('완료 토스트와 Activity 오류 상태가 다음 행동을 안내한다', (tester) async {
    tester.view.physicalSize = const Size(1440, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final semantics = tester.ensureSemantics();

    final controller =
        ScannerController(
            _UnusedApi(),
            _EmptyCameraGateway(),
            _EmptyFileGateway(),
            _ThrowingLogRepository(),
            testCatalog,
            completionFeedbackDuration: AppMotion.feedbackHold,
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
    await controller.submit();

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pump();
    expect(find.text('2개 상품을 확정했어요'), findsOneWidget);
    final completion = find.bySemanticsLabel('2개 상품을 확정했어요');
    expect(
      tester
          .getSemantics(completion)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    expect(
      tester
          .widget<IgnorePointer>(
            find.byKey(const ValueKey('global-completion-feedback')),
          )
          .ignoring,
      isTrue,
    );

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_completed_1440x900.png'),
    );

    await tester.pump(const Duration(seconds: 2));
    controller
      ..inputMode = InputMode.image
      ..imageBytes = _testInputImage.bytes
      ..imageFileName = _testInputImage.fileName
      ..imageSize = const Size(400, 400)
      ..processState = ProcessState.reviewing
      ..response = _response
      ..detections = testReviewDetections(_response)
      ..selectedItemId = 'item_002';
    controller.confirmCandidate('item_002', _response.items[1].top3.first);
    await controller.submit();
    await tester.pump();

    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();
    await tester.pump(const Duration(seconds: 1));
    expect(find.text('2개 상품을 확정했어요'), findsOneWidget);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_completion_toast_1440x900.png'),
    );

    await tester.pump(const Duration(seconds: 2));
    await tester.pump();
    expect(find.text('2개 상품을 확정했어요'), findsNothing);

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_error_1440x900.png'),
    );

    expect(find.text('활동 화면을 갱신하지 못했어요'), findsOneWidget);
    expect(find.text('방금 확정한 기록은 저장됐어요. 잠시 후 새로고침해 주세요.'), findsOneWidget);
    expect(find.text('활동 기록을 불러오지 못했어요'), findsNothing);
    final activityError = find.bySemanticsLabel(
      '활동 화면을 갱신하지 못했어요. 방금 확정한 기록은 저장됐어요. 잠시 후 새로고침해 주세요.',
    );
    expect(activityError, findsOneWidget);
    final activityErrorData = tester
        .getSemantics(activityError)
        .getSemanticsData();
    expect(activityErrorData.flagsCollection.isLiveRegion, isTrue);
    expect(activityErrorData.hasAction(SemanticsAction.tap), isFalse);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('새로고침'))
          .getSemanticsData()
          .hasAction(SemanticsAction.tap),
      isTrue,
    );
    expect(find.textContaining('다시 시도'), findsNothing);
    expect(find.text('활동 기록'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '새로고침'), findsOneWidget);
    expect(find.byType(TextField), findsNothing);
    expect(find.byType(ChoiceChip), findsNothing);
    expect(tester.takeException(), isNull);
    semantics.dispose();
    controller.dispose();
  });

  testWidgets('Activity에서 Ctrl+O로 이미지를 선택하면 스캔 작업대로 이동한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final fileGateway = _StaticFileGateway(_testInputImage);
    final controller = ScannerController(
      _UnusedApi(),
      _EmptyCameraGateway(),
      fileGateway,
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
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    controller
      ..processState = ProcessState.analyzing
      ..notifyListeners();
    await tester.pump();
    await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
    final blocked = await tester.sendKeyDownEvent(LogicalKeyboardKey.keyO);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.keyO);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
    await tester.pump();

    expect(blocked, isFalse);
    expect(fileGateway.pickCalls, 0);
    expect(find.text('활동 기록'), findsOneWidget);

    controller
      ..processState = ProcessState.ready
      ..notifyListeners();
    await tester.pump();
    await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
    expect(HardwareKeyboard.instance.isControlPressed, isTrue);
    final handled = await tester.sendKeyDownEvent(LogicalKeyboardKey.keyO);
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 50)),
    );
    await tester.pumpAndSettle();
    await tester.sendKeyUpEvent(LogicalKeyboardKey.keyO);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);

    expect(handled, isTrue);
    expect(fileGateway.pickCalls, 1);
    expect(controller.inputMode, InputMode.image);
    expect(find.text(AppPreviewCopy.selectedImage), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '분석하기'), findsOneWidget);
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

final _imageLogSummary = ScanLogSummary(
  scanId: 'request_activity_5678',
  analyzedAt: DateTime.utc(2026, 8, 10, 2),
  confirmedAt: DateTime.utc(2026, 8, 10, 2, 1),
  inputMode: InputMode.image,
  processingTimeMs: 64.8,
  modelVersions: const ModelVersions(detector: '0.1.1', classifier: '0.1.1'),
  items: const [
    ScanLogItemSummary(
      itemId: 'item_001',
      productName: '에그 타르트',
      confidence: .89,
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

class _StaticFileGateway implements ImageFileGateway {
  _StaticFileGateway(this.image);

  final InputImage image;
  int pickCalls = 0;

  @override
  Future<InputImage?> pick() async {
    pickCalls += 1;
    return image;
  }
}

class _MemoryLogRepository implements ScanLogRepository {
  _MemoryLogRepository({this.logs = const []});

  final List<ScanLogSummary> logs;
  int listCalls = 0;

  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async {
    listCalls += 1;
    return logs.take(limit).toList();
  }

  @override
  Future<void> save(ScanLogRecord record) async {}
}

class _DeferredSaveLogRepository implements ScanLogRepository {
  final Completer<void> _save = Completer<void>();

  void completeSave() => _save.complete();

  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async => const [];

  @override
  Future<void> save(ScanLogRecord record) => _save.future;
}

class _ThrowingLogRepository implements ScanLogRepository {
  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) {
    throw StateError('unavailable');
  }

  @override
  Future<void> save(ScanLogRecord record) async {}
}
