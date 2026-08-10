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
import 'package:product_scanner/theme/app_tokens.dart';

import 'support/test_catalog.dart';

void main() {
  testWidgets('최초 로드는 작동하지 않는 도구를 숨기고 현재 상태를 안내한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final semantics = tester.ensureSemantics();
    final repository = _ControlledLogRepository(_manyLogs().take(3).toList())
      ..deferNext();
    final controller = _controller(repository);

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    expect(repository.listCalls, 0);
    await tester.tap(find.text('활동'));
    await tester.pump();

    expect(repository.listCalls, 1);
    expect(find.text('활동 기록'), findsOneWidget);
    expect(find.text('활동 기록을 불러오는 중이에요'), findsOneWidget);
    final loading = find.bySemanticsLabel('활동 기록을 불러오는 중이에요');
    expect(loading, findsOneWidget);
    expect(
      tester
          .getSemantics(loading)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    expect(find.byType(TextField), findsNothing);
    expect(find.byType(ChoiceChip), findsNothing);
    expect(find.byType(DropdownButton<ActivitySortOrder>), findsNothing);
    expect(find.byTooltip('새로고침 중'), findsNothing);
    await tester.pump(const Duration(milliseconds: 600));
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_initial_loading_1280x720.png'),
    );

    repository.completeDeferred();
    await tester.pumpAndSettle();
    expect(find.text('활동 기록을 불러오는 중이에요'), findsNothing);
    expect(find.byType(TextField), findsOneWidget);
    expect(find.byType(ChoiceChip), findsNWidgets(7));
    expect(find.byType(DropdownButton<ActivitySortOrder>), findsOneWidget);
    expect(find.text('상품 20'), findsWidgets);

    semantics.dispose();
    controller.dispose();
  });

  testWidgets('저장 맥락이 없는 최초 로드 오류는 일반 오류 계약으로 복구한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final semantics = tester.ensureSemantics();
    final repository = _ControlledLogRepository(_manyLogs().take(2).toList())
      ..failNext = true;
    final controller = _controller(repository);
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
    expect(find.text('활동 기록을 불러오지 못했어요'), findsOneWidget);
    expect(find.text('저장된 기록은 그대로 유지됩니다. 잠시 후 새로고침해 주세요.'), findsOneWidget);
    expect(find.text('활동 화면을 갱신하지 못했어요'), findsNothing);
    expect(find.byIcon(Icons.error_outline_rounded), findsOneWidget);
    expect(
      tester.widget<Icon>(find.byIcon(Icons.error_outline_rounded)).color,
      AppPalette.error,
    );
    final errorAnnouncement = find.bySemanticsLabel(
      '활동 기록을 불러오지 못했어요. 저장된 기록은 그대로 유지됩니다. 잠시 후 새로고침해 주세요.',
    );
    expect(
      tester
          .getSemantics(errorAnnouncement)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_initial_error_1280x720.png'),
    );

    await tester.tap(find.widgetWithText(FilledButton, '새로고침'));
    await tester.pumpAndSettle();
    expect(repository.listCalls, 2);
    expect(find.text('상품 20'), findsWidgets);
    expect(find.text('활동 기록을 불러오지 못했어요'), findsNothing);

    semantics.dispose();
    controller.dispose();
  });

  testWidgets('첫 Activity 방문을 지연하고 저장 직후에는 최신 기록을 조용히 동기화한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final semantics = tester.ensureSemantics();
    final repository =
        _ControlledLogRepository(_manyLogs().skip(1).take(2).toList())
          ..recordToAddOnSave = _manyLogs().first
          ..deferNext();
    final controller = _controller(repository);
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    expect(repository.listCalls, 0);

    _prepareApprovedReview(controller);
    await controller.submit();
    await tester.pump();
    expect(controller.activityDataRevision, 1);
    expect(repository.listCalls, 0);

    await tester.tap(find.text('활동'));
    await tester.pump();
    await tester.pump();

    expect(repository.listCalls, 1);
    final loading = find.bySemanticsLabel('활동 기록을 불러오는 중이에요');
    expect(loading, findsOneWidget);
    expect(
      tester
          .getSemantics(loading)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isFalse,
    );
    final completion = find.bySemanticsLabel('1개 상품을 확정했어요');
    expect(
      tester
          .getSemantics(completion)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );

    repository.completeDeferred();
    await tester.pumpAndSettle();
    expect(find.text('상품 20'), findsWidgets);
    final latestRow = find.byKey(const ValueKey('activity-log-log_020'));
    final latestSemantics = tester.getSemantics(
      find.descendant(of: latestRow, matching: find.byType(Semantics)).first,
    );
    expect(latestSemantics.flagsCollection.isSelected, Tristate.isTrue);
    expect(
      find.byKey(const ValueKey('activity-detail-log_020')),
      findsOneWidget,
    );

    await tester.tap(find.text('스캔'));
    await tester.pump();
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();
    expect(repository.listCalls, 1);

    semantics.dispose();
    controller.dispose();
  });

  testWidgets('Activity 최초 로드 중 저장이 끝나면 완료만 발표하고 최신 기록을 재동기화한다', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final semantics = tester.ensureSemantics();
    final repository =
        _ControlledLogRepository(_manyLogs().skip(1).take(2).toList())
          ..recordToAddOnSave = _manyLogs().first
          ..deferNext()
          ..deferSave();
    final controller = _controller(repository);
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    _prepareApprovedReview(controller);

    final submission = controller.submit();
    await tester.pump();
    expect(controller.processState, ProcessState.submitting);
    await tester.tap(find.text('활동'));
    await tester.pump();
    await tester.pump();

    expect(repository.listCalls, 1);
    var loading = find.bySemanticsLabel('활동 기록을 불러오는 중이에요');
    expect(
      tester
          .getSemantics(loading)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    expect(find.text('1개 상품을 확정했어요'), findsNothing);

    repository.completeSave();
    await submission;
    await tester.pump();

    expect(controller.activityDataRevision, 1);
    final completion = find.bySemanticsLabel('1개 상품을 확정했어요');
    expect(
      tester
          .getSemantics(completion)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    loading = find.bySemanticsLabel('활동 기록을 불러오는 중이에요');
    expect(
      tester
          .getSemantics(loading)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isFalse,
    );

    repository.completeDeferred();
    await tester.pumpAndSettle();
    expect(repository.listCalls, 2);
    expect(find.text('상품 20'), findsWidgets);
    expect(
      find.byKey(const ValueKey('activity-detail-log_020')),
      findsOneWidget,
    );

    semantics.dispose();
    controller.dispose();
  });

  testWidgets('장기 최초 로드는 화면을 벗어나도 한 번만 완료하고 재진입 F5를 유지한다', (tester) async {
    final repository = _ControlledLogRepository(_manyLogs().take(2).toList())
      ..deferNext();
    final controller = _controller(repository);
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('활동'));
    await tester.pump();
    expect(repository.listCalls, 1);
    expect(find.text('활동 기록을 불러오는 중이에요'), findsOneWidget);

    await tester.tap(find.text('스캔'));
    await tester.pump();
    repository.completeDeferred();
    await tester.pumpAndSettle();
    expect(find.text('상품 20'), findsNothing);

    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();
    expect(repository.listCalls, 1);
    expect(find.text('상품 20'), findsWidgets);

    repository.deferNext();
    await tester.sendKeyEvent(LogicalKeyboardKey.f5);
    await tester.pump();
    expect(repository.listCalls, 2);
    expect(find.byTooltip('새로고침 중'), findsOneWidget);
    repository.completeDeferred();
    await tester.pumpAndSettle();

    controller.dispose();
  });

  testWidgets('비활성 중 실패한 최초 로드는 재진입 시 오류를 깜박이지 않고 다시 불러온다', (tester) async {
    final semantics = tester.ensureSemantics();
    final repository = _ControlledLogRepository(_manyLogs().take(2).toList())
      ..deferNext();
    final controller = _controller(repository);
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('활동'));
    await tester.pump();
    expect(repository.listCalls, 1);

    await tester.tap(find.text('스캔'));
    await tester.pump();
    repository.completeDeferredError();
    await tester.pumpAndSettle();
    expect(find.text('활동 기록을 불러오지 못했어요'), findsNothing);

    repository.deferNext();
    await tester.tap(find.text('활동'));
    await tester.pump();

    expect(repository.listCalls, 2);
    expect(find.text('활동 기록을 불러오지 못했어요'), findsNothing);
    final loading = find.bySemanticsLabel('활동 기록을 불러오는 중이에요');
    expect(loading, findsOneWidget);
    expect(
      tester
          .getSemantics(loading)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );

    repository.completeDeferred();
    await tester.pumpAndSettle();
    expect(find.text('상품 20'), findsWidgets);
    expect(repository.listCalls, 2);

    semantics.dispose();
    controller.dispose();
  });

  testWidgets('새 저장 후 Activity로 돌아오면 필터를 유지하며 최신 기록을 불러온다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final newLog = _manyLogs().first;
    final repository = _ControlledLogRepository(
      _manyLogs().skip(1).take(2).toList(),
    )..recordToAddOnSave = newLog;
    final controller = _controller(repository);
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
    await tester.enterText(find.byType(TextField), '상품 20');
    await tester.pumpAndSettle();
    expect(find.text('조건에 맞는 기록이 없어요'), findsOneWidget);

    await tester.tap(find.text('스캔'));
    await tester.pump();
    _prepareApprovedReview(controller);
    await controller.submit();
    await tester.pump();
    expect(controller.activityDataRevision, 1);

    repository.deferNext();
    await tester.tap(find.text('활동'));
    await tester.pump();
    await tester.pump();

    expect(repository.listCalls, 2);
    final automaticRefresh = find.byTooltip('새로고침 중');
    expect(automaticRefresh, findsOneWidget);
    final automaticRefreshSemantics = tester
        .getSemantics(automaticRefresh)
        .getSemanticsData();
    expect(automaticRefreshSemantics.flagsCollection.isLiveRegion, isFalse);
    expect(automaticRefreshSemantics.label, contains('활동 기록 새로고침 중'));
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
    repository.completeDeferred();
    await tester.pumpAndSettle();

    expect(
      tester.widget<TextField>(find.byType(TextField)).controller?.text,
      '상품 20',
    );
    expect(find.text('상품 20'), findsWidgets);
    expect(find.text('조건에 맞는 기록이 없어요'), findsNothing);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_new_save_synced_1280x720.png'),
    );
    await tester.tap(find.bySemanticsLabel('검색어 지우기'));
    await tester.pumpAndSettle();
    final latestRow = find.byKey(const ValueKey('activity-log-log_020'));
    final latestSemantics = tester.getSemantics(
      find.descendant(of: latestRow, matching: find.byType(Semantics)).first,
    );
    expect(latestSemantics.flagsCollection.isSelected, Tristate.isTrue);
    expect(
      find.byKey(const ValueKey('activity-detail-log_020')),
      findsOneWidget,
    );
    await tester.tap(find.text('스캔'));
    await tester.pump();
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();
    expect(repository.listCalls, 2);
    expect(
      tester.widget<TextField>(find.byType(TextField)).controller?.text,
      isEmpty,
    );
    controller.dispose();
  });

  testWidgets('저장 성공 후 Activity 동기화 실패는 저장과 화면 오류를 분리하고 재시도한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final semantics = tester.ensureSemantics();
    final newLog = _manyLogs().first;
    final repository = _ControlledLogRepository(
      _manyLogs().skip(1).take(2).toList(),
    )..recordToAddOnSave = newLog;
    final controller = _controller(repository);
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
    expect(find.text('상품 19'), findsWidgets);

    await tester.tap(find.text('스캔'));
    await tester.pump();
    _prepareApprovedReview(controller);
    await controller.submit();
    await tester.pump();

    repository.failNext = true;
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    expect(repository.listCalls, 2);
    expect(find.text('1개 상품을 확정했어요'), findsOneWidget);
    const syncFailure = '방금 확정한 기록은 저장됐어요. 활동 화면만 갱신하지 못했어요.';
    expect(find.text(syncFailure), findsOneWidget);
    expect(find.text('새로고침하지 못했어요. 기존 활동을 표시하고 있어요.'), findsNothing);
    expect(find.text('상품 19'), findsWidgets);
    expect(find.text('상품 20'), findsNothing);
    final failureAnnouncement = find.bySemanticsLabel(syncFailure);
    expect(failureAnnouncement, findsOneWidget);
    expect(
      tester
          .getSemantics(failureAnnouncement)
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_saved_sync_error_1280x720.png'),
    );

    await tester.tap(find.widgetWithText(TextButton, '새로고침'));
    await tester.pumpAndSettle();

    expect(repository.listCalls, 3);
    expect(find.text(syncFailure), findsNothing);
    expect(find.text('상품 20'), findsWidgets);
    final latestRow = find.byKey(const ValueKey('activity-log-log_020'));
    final latestSemantics = tester.getSemantics(
      find.descendant(of: latestRow, matching: find.byType(Semantics)).first,
    );
    expect(latestSemantics.flagsCollection.isSelected, Tristate.isTrue);
    expect(
      find.byKey(const ValueKey('activity-detail-log_020')),
      findsOneWidget,
    );

    semantics.dispose();
    controller.dispose();
  });

  testWidgets('빈 Activity의 저장 후 동기화 실패도 저장 성공과 복구 행동을 안내한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final repository = _ControlledLogRepository([])
      ..recordToAddOnSave = _manyLogs().first;
    final controller = _controller(repository);
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();
    expect(find.text('저장된 활동이 없어요'), findsOneWidget);

    await tester.tap(find.text('스캔'));
    await tester.pump();
    _prepareApprovedReview(controller);
    await controller.submit();
    await tester.pump();

    repository.failNext = true;
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    expect(find.text('활동 화면을 갱신하지 못했어요'), findsOneWidget);
    expect(find.text('방금 확정한 기록은 저장됐어요. 잠시 후 새로고침해 주세요.'), findsOneWidget);
    expect(find.text('활동 기록을 불러오지 못했어요'), findsNothing);
    expect(find.widgetWithText(FilledButton, '새로고침'), findsOneWidget);

    await tester.tap(find.widgetWithText(FilledButton, '새로고침'));
    await tester.pumpAndSettle();
    expect(find.text('상품 20'), findsWidgets);
    expect(
      find.byKey(const ValueKey('activity-detail-log_020')),
      findsOneWidget,
    );

    controller.dispose();
  });

  testWidgets('오래된순으로 바꿔도 현재 활동 행을 긴 목록 안에 유지한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final repository = _ControlledLogRepository(_manyLogs());
    final controller = _controller(repository);
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    await tester.tap(find.text('최신순'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('오래된순').last);
    await tester.pumpAndSettle();

    final listRect = tester.getRect(
      find.byKey(const ValueKey('activity-log-list')),
    );
    final selectedRect = tester.getRect(
      find.byKey(const ValueKey('activity-log-log_020')),
    );
    final firstVisibleRect = tester.getRect(
      find.byKey(const ValueKey('activity-log-log_014')),
    );
    expect(firstVisibleRect.top, closeTo(listRect.top, 0.5));
    expect(selectedRect.top, greaterThanOrEqualTo(listRect.top));
    expect(selectedRect.bottom, lessThanOrEqualTo(listRect.bottom + 0.5));
    expect(
      listRect.bottom - selectedRect.bottom,
      lessThan(AppDesignTokens.standard.rowHeight),
    );
    expect(
      tester
          .widget<DropdownButton<ActivitySortOrder>>(
            find.byType(DropdownButton<ActivitySortOrder>),
          )
          .value,
      ActivitySortOrder.oldest,
    );
    expect(find.text('상품 20'), findsWidgets);
    expect(find.text('자동 확정'), findsWidgets);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_long_list_1280x720.png'),
    );

    controller.dispose();
  });

  testWidgets('Activity 행은 표준 목록 키로 선택·상세·포커스를 함께 이동한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final repository = _ControlledLogRepository(_manyLogs());
    final controller = _controller(repository);
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    final firstRow = find.byKey(const ValueKey('activity-log-log_020'));
    final secondRow = find.byKey(const ValueKey('activity-log-log_019'));
    final firstInkWell = find.descendant(
      of: firstRow,
      matching: find.byType(InkWell),
    );
    final secondInkWell = find.descendant(
      of: secondRow,
      matching: find.byType(InkWell),
    );
    expect(
      tester.widget<InkWell>(firstInkWell).focusNode!.skipTraversal,
      isFalse,
    );
    expect(
      tester.widget<InkWell>(secondInkWell).focusNode!.skipTraversal,
      isTrue,
    );
    tester.widget<InkWell>(firstInkWell).focusNode!.requestFocus();
    await tester.pumpAndSettle();
    expect(tester.widget<InkWell>(firstInkWell).focusNode!.hasFocus, isTrue);

    await tester.sendKeyEvent(LogicalKeyboardKey.arrowDown);
    await tester.pumpAndSettle();

    expect(tester.widget<InkWell>(secondInkWell).focusNode!.hasFocus, isTrue);
    expect(
      tester.widget<InkWell>(firstInkWell).focusNode!.skipTraversal,
      isTrue,
    );
    expect(
      tester.widget<InkWell>(secondInkWell).focusNode!.skipTraversal,
      isFalse,
    );
    final secondSemantics = tester.getSemantics(
      find.descendant(of: secondRow, matching: find.byType(Semantics)).first,
    );
    expect(secondSemantics.flagsCollection.isFocused, Tristate.isTrue);
    expect(secondSemantics.flagsCollection.isSelected, Tristate.isTrue);
    expect(
      find.byKey(const ValueKey('activity-detail-log_019')),
      findsOneWidget,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile(
        'goldens/activity_row_keyboard_navigation_1280x720.png',
      ),
    );

    for (var index = 0; index < 18; index++) {
      await tester.sendKeyEvent(LogicalKeyboardKey.arrowDown);
      await tester.pumpAndSettle();
    }
    final lastRow = find.byKey(const ValueKey('activity-log-log_001'));
    final lastInkWell = find.descendant(
      of: lastRow,
      matching: find.byType(InkWell),
    );
    expect(tester.widget<InkWell>(lastInkWell).focusNode!.hasFocus, isTrue);
    expect(
      find.byKey(const ValueKey('activity-detail-log_001')),
      findsOneWidget,
    );
    final listRect = tester.getRect(
      find.byKey(const ValueKey('activity-log-list')),
    );
    final lastRowRect = tester.getRect(lastRow);
    expect(lastRowRect.top, greaterThanOrEqualTo(listRect.top));
    expect(lastRowRect.bottom, lessThanOrEqualTo(listRect.bottom + .5));

    await tester.sendKeyEvent(LogicalKeyboardKey.home);
    await tester.pumpAndSettle();
    expect(
      find.byKey(const ValueKey('activity-detail-log_020')),
      findsOneWidget,
    );
    expect(tester.widget<InkWell>(firstInkWell).focusNode!.hasFocus, isTrue);

    final visibleRowCount =
        (listRect.height / AppDesignTokens.standard.rowHeight).floor();
    final pageStep = visibleRowCount > 1 ? visibleRowCount - 1 : 1;
    final pageTarget = 20 - pageStep;
    await tester.sendKeyEvent(LogicalKeyboardKey.pageDown);
    await tester.pumpAndSettle();
    expect(
      find.byKey(
        ValueKey(
          'activity-detail-log_${pageTarget.toString().padLeft(3, '0')}',
        ),
      ),
      findsOneWidget,
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.pageUp);
    await tester.pumpAndSettle();
    expect(
      find.byKey(const ValueKey('activity-detail-log_020')),
      findsOneWidget,
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.end);
    await tester.pumpAndSettle();
    expect(
      find.byKey(const ValueKey('activity-detail-log_001')),
      findsOneWidget,
    );
    expect(tester.widget<InkWell>(lastInkWell).focusNode!.hasFocus, isTrue);
    expect(
      tester.widget<InkWell>(lastInkWell).focusNode!.skipTraversal,
      isFalse,
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    expect(tester.widget<InkWell>(lastInkWell).focusNode!.hasFocus, isFalse);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('진단 정보'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_list_exit_focus_1280x720.png'),
    );

    await tester.sendKeyDownEvent(LogicalKeyboardKey.shiftLeft);
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.shiftLeft);
    await tester.pumpAndSettle();
    expect(tester.widget<InkWell>(lastInkWell).focusNode!.hasFocus, isTrue);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('진단 정보'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isFalse,
    );

    await tester.tap(find.byType(TextField));
    await tester.sendKeyEvent(LogicalKeyboardKey.home);
    await tester.pumpAndSettle();
    expect(
      find.byKey(const ValueKey('activity-detail-log_001')),
      findsOneWidget,
    );

    controller.dispose();
  });

  testWidgets('새로고침 중과 실패에도 기존 활동과 선택을 유지한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final semantics = tester.ensureSemantics();

    final repository = _ControlledLogRepository(_manyLogs().take(3).toList());
    final controller = _controller(repository);
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    repository.deferNext();
    await tester.tap(find.byTooltip('새로고침 (F5)'));
    await tester.pump();
    expect(repository.listCalls, 2);
    expect(find.byTooltip('새로고침 중'), findsOneWidget);
    final progress = find.bySemanticsLabel('활동 기록 새로고침 중. 완료될 때까지 기다려 주세요');
    expect(progress, findsOneWidget);
    final progressData = tester.getSemantics(progress).getSemanticsData();
    expect(progressData.flagsCollection.isLiveRegion, isTrue);
    expect(progressData.flagsCollection.isButton, isTrue);
    expect(progressData.hasAction(SemanticsAction.tap), isFalse);
    expect(find.text('상품 20'), findsWidgets);
    expect(find.text('확정 상품'), findsOneWidget);

    await tester.sendKeyEvent(LogicalKeyboardKey.f5);
    await tester.pump(const Duration(milliseconds: 600));
    expect(repository.listCalls, 2);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_refreshing_1280x720.png'),
    );

    repository.completeDeferred();
    await tester.pumpAndSettle();
    expect(find.byTooltip('새로고침 (F5)'), findsOneWidget);

    repository.failNext = true;
    await tester.tap(find.byTooltip('새로고침 (F5)'));
    await tester.pumpAndSettle();
    expect(repository.listCalls, 3);
    expect(find.text('새로고침하지 못했어요. 기존 활동을 표시하고 있어요.'), findsOneWidget);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('새로고침하지 못했어요. 기존 활동을 표시하고 있어요.'))
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    final refresh = find.widgetWithText(TextButton, '새로고침');
    final refreshSemantics = find.bySemanticsLabel('새로고침');
    expect(refresh, findsOneWidget);
    expect(refreshSemantics, findsOneWidget);
    expect(find.byTooltip('새로고침 (F5)'), findsNothing);
    expect(find.bySemanticsLabel('활동 기록 새로고침'), findsNothing);
    final refreshSemanticsData = tester
        .getSemantics(refreshSemantics)
        .getSemanticsData();
    expect(refreshSemanticsData.flagsCollection.isButton, isTrue);
    expect(refreshSemanticsData.hasAction(SemanticsAction.tap), isTrue);
    expect(find.widgetWithText(TextButton, '다시 시도'), findsNothing);
    expect(tester.getSize(refresh).height, greaterThanOrEqualTo(44));
    expect(find.text('상품 20'), findsWidgets);
    expect(find.text('활동 기록을 불러오지 못했어요'), findsNothing);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_refresh_error_1280x720.png'),
    );
    repository.deferNext();
    await tester.sendKeyEvent(LogicalKeyboardKey.f5);
    await tester.pump();
    expect(repository.listCalls, 4);
    expect(find.byTooltip('새로고침 중'), findsOneWidget);
    expect(
      find.bySemanticsLabel('활동 기록 새로고침 중. 완료될 때까지 기다려 주세요'),
      findsOneWidget,
    );
    expect(find.text('새로고침하지 못했어요. 기존 활동을 표시하고 있어요.'), findsNothing);
    repository.completeDeferred();
    await tester.pumpAndSettle();
    expect(repository.listCalls, 4);
    expect(find.text('새로고침하지 못했어요. 기존 활동을 표시하고 있어요.'), findsNothing);
    expect(find.text('상품 20'), findsWidgets);
    expect(find.byTooltip('새로고침 (F5)'), findsOneWidget);

    final toolbarRefresh = find.bySemanticsLabel('활동 기록 새로고침');
    var refreshFocused = false;
    for (var index = 0; index < 16 && !refreshFocused; index += 1) {
      await tester.sendKeyEvent(LogicalKeyboardKey.tab);
      await tester.pump();
      refreshFocused =
          tester
              .getSemantics(toolbarRefresh)
              .getSemanticsData()
              .flagsCollection
              .isFocused ==
          Tristate.isTrue;
    }
    expect(refreshFocused, isTrue);

    repository.failNext = true;
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pumpAndSettle();
    expect(repository.listCalls, 5);
    expect(find.byTooltip('새로고침 (F5)'), findsNothing);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('새로고침'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/activity_refresh_error_focus_1280x720.png'),
    );

    repository.deferNext();
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pump();
    expect(repository.listCalls, 6);
    expect(find.byTooltip('새로고침 중'), findsOneWidget);
    expect(find.text('새로고침하지 못했어요. 기존 활동을 표시하고 있어요.'), findsNothing);
    repository.completeDeferred();
    await tester.pumpAndSettle();
    expect(find.byTooltip('새로고침 (F5)'), findsOneWidget);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('활동 기록 새로고침'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );

    repository.deferNext();
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pump();
    expect(repository.listCalls, 7);
    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pump();
    final focusChosenDuringRefresh = FocusManager.instance.primaryFocus;
    expect(focusChosenDuringRefresh, isNotNull);
    repository.completeDeferred();
    await tester.pumpAndSettle();
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('활동 기록 새로고침'))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isFalse,
    );
    expect(FocusManager.instance.primaryFocus, same(focusChosenDuringRefresh));

    semantics.dispose();
    controller.dispose();
  });

  testWidgets('구버전 Activity는 한국어 fallback과 영문 검색 일치 상품을 표시한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final repository = _ControlledLogRepository([
      ScanLogSummary(
        scanId: 'legacy_log',
        analyzedAt: DateTime.utc(2026, 8, 10, 1),
        confirmedAt: DateTime.utc(2026, 8, 10, 1, 1),
        inputMode: InputMode.image,
        processingTimeMs: 50,
        modelVersions: const ModelVersions(
          detector: '0.1.1',
          classifier: '0.1.1',
        ),
        items: const [
          ScanLogItemSummary(
            itemId: 'item_001',
            productName: 'Unknown',
            confidence: .9,
            userModified: false,
            confirmationMethod: 'UNKNOWN',
          ),
          ScanLogItemSummary(
            itemId: 'item_002',
            productName: '머핀',
            confidence: .8,
            userModified: false,
            confirmationMethod: 'AUTO_APPROVED',
            classId: 'bread_13',
            className: 'Muffin',
          ),
        ],
      ),
    ]);
    final controller = _controller(repository);
    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.tap(find.text('활동'));
    await tester.pumpAndSettle();

    expect(find.text('상품 정보 없음 외 1개'), findsOneWidget);
    expect(find.text('상품 정보 없음'), findsOneWidget);
    expect(find.text('Unknown'), findsNothing);
    await tester.tap(find.text('진단 정보'));
    await tester.pumpAndSettle();
    expect(find.textContaining('확정 방식 확인 불가'), findsOneWidget);
    expect(find.textContaining('UNKNOWN'), findsNothing);

    await tester.enterText(find.byType(TextField), 'Muffin');
    await tester.pumpAndSettle();
    expect(find.text('머핀 외 1개'), findsOneWidget);

    controller.dispose();
  });
}

ScannerController _controller(ScanLogRepository repository) =>
    ScannerController(
      _UnusedApi(),
      _EmptyCameraGateway(),
      _EmptyFileGateway(),
      repository,
      testCatalog,
    )..cameraInitializing = false;

void _prepareApprovedReview(ScannerController controller) {
  controller
    ..inputMode = InputMode.image
    ..imageBytes = base64Decode(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
    )
    ..imageFileName = 'new-scan.png'
    ..processState = ProcessState.reviewing
    ..response = _activityApprovedResponse
    ..detections = _activityApprovedResponse.items
        .map(ReviewDetection.fromScanItem)
        .toList(growable: false);
}

const _activityApprovedResponse = ScanResponse(
  requestId: 'log_020',
  status: ScanStatus.approved,
  reasonCodes: [],
  items: [
    ScanItem(
      itemId: 'item_new',
      bbox: BoundingBox(x: 0, y: 0, width: 100, height: 100),
      status: ItemStatus.approved,
      reasonCodes: [],
      prediction: Product(
        classId: 'bread_06',
        className: 'Croissant',
        displayName: '크루아상',
      ),
      top3: [],
      confidence: .99,
    ),
  ],
  processingTimeMs: 42,
  modelVersions: ModelVersions(detector: '0.1.1', classifier: '0.1.1'),
);

List<ScanLogSummary> _manyLogs() => [
  for (var value = 20; value >= 1; value--)
    ScanLogSummary(
      scanId: 'log_${value.toString().padLeft(3, '0')}',
      analyzedAt: DateTime.utc(2026, 8, 10, 1, value),
      confirmedAt: DateTime.utc(2026, 8, 10, 1, value, 30),
      inputMode: value.isEven ? InputMode.camera : InputMode.image,
      processingTimeMs: 50 + value.toDouble(),
      modelVersions: const ModelVersions(
        detector: '0.1.1',
        classifier: '0.1.1',
      ),
      items: [
        ScanLogItemSummary(
          itemId: 'item_${value.toString().padLeft(3, '0')}',
          productName: '상품 $value',
          confidence: .9,
          userModified: false,
          confirmationMethod: 'AUTO_APPROVED',
        ),
      ],
    ),
];

class _ControlledLogRepository implements ScanLogRepository {
  _ControlledLogRepository(this.logs);

  final List<ScanLogSummary> logs;
  int listCalls = 0;
  bool failNext = false;
  ScanLogSummary? recordToAddOnSave;
  Completer<List<ScanLogSummary>>? _deferred;
  Completer<void>? _saveDeferred;

  void deferNext() {
    _deferred = Completer<List<ScanLogSummary>>();
  }

  void completeDeferred() {
    _deferred?.complete(logs);
    _deferred = null;
  }

  void completeDeferredError() {
    _deferred?.completeError(StateError('refresh failed'));
    _deferred = null;
  }

  void deferSave() {
    _saveDeferred = Completer<void>();
  }

  void completeSave() {
    _saveDeferred?.complete();
    _saveDeferred = null;
  }

  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async {
    listCalls += 1;
    if (failNext) {
      failNext = false;
      throw StateError('refresh failed');
    }
    final deferred = _deferred;
    if (deferred != null) return deferred.future;
    return List<ScanLogSummary>.unmodifiable(logs);
  }

  @override
  Future<void> save(ScanLogRecord record) async {
    final log = recordToAddOnSave;
    if (log != null) logs.insert(0, log);
    final deferred = _saveDeferred;
    if (deferred != null) await deferred.future;
  }
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
