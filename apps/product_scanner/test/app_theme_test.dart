import 'dart:io';
import 'dart:ui' show PointerDeviceKind, SemanticsAction, Tristate;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/theme/app_copy.dart';
import 'package:product_scanner/theme/app_theme.dart';
import 'package:product_scanner/theme/app_tokens.dart';
import 'package:product_scanner/widgets/app_components.dart';

void main() {
  test('핵심 행동 콘텐츠 토큰은 같은 작업에 하나의 이름을 사용한다', () {
    expect(AppActionCopy.capture, '촬영하기');
    expect(AppActionCopy.returnToCapture, '촬영 화면으로 돌아가기');
    expect(AppActionCopy.capturing, '촬영 중');
    expect(AppActionCopy.capturingAnnouncement, '촬영 중. 카메라 응답을 기다려 주세요');
    expect(AppActionCopy.recapture, '다시 촬영');
    expect(AppActionCopy.analyze, '분석하기');
    expect(AppActionCopy.analyzing, '분석 중');
    expect(AppActionCopy.analyzingAnnouncement, '분석 중. 완료될 때까지 기다려 주세요');
    expect(AppActionCopy.reanalyze, '다시 분석');
    expect(AppActionCopy.checkingConnection, '연결 확인 중');
    expect(
      AppActionCopy.checkingConnectionAnnouncement,
      '카메라 연결 확인 중. 완료될 때까지 기다려 주세요',
    );
    expect(AppActionCopy.refreshing, '새로고침 중');
    expect(
      AppActionCopy.refreshingAnnouncement,
      '활동 기록 새로고침 중. 완료될 때까지 기다려 주세요',
    );
    expect(AppActionCopy.saving, '저장 중');
    expect(AppActionCopy.savingAnnouncement, '저장 중. 완료될 때까지 기다려 주세요');
    expect(AppPreviewCopy.liveCamera, '라이브 카메라');
    expect(AppPreviewCopy.cameraPreview, '카메라 미리보기');
    expect(AppPreviewCopy.capturedImage, '촬영 이미지');
    expect(AppPreviewCopy.selectedImage, '선택한 이미지');
    expect(
      AppPreviewCopy.semanticLabel(AppPreviewCopy.selectedImage),
      '입력 미리보기, 선택한 이미지',
    );
  });

  test('BIXOLON Orange CTA는 Ink 전경색으로 본문 대비를 충족한다', () {
    expect(
      _contrast(AppColors.primary, AppColors.ink),
      greaterThanOrEqualTo(4.5),
    );
    expect(_contrast(AppColors.primary, Colors.white), lessThan(4.5));
  });

  test('버튼 포커스는 면을 바꾸지 않고 역할별 2px 경계로 구분한다', () {
    final theme = buildAppTheme();
    const focused = <WidgetState>{WidgetState.focused};
    const resting = <WidgetState>{};

    final filledStyle = theme.filledButtonTheme.style!;
    expect(
      filledStyle.side!.resolve(focused),
      const BorderSide(color: AppColors.ink, width: 2),
    );
    expect(filledStyle.side!.resolve(resting), BorderSide.none);
    expect(filledStyle.overlayColor!.resolve(focused), Colors.transparent);

    final outlinedStyle = theme.outlinedButtonTheme.style!;
    expect(
      outlinedStyle.side!.resolve(focused),
      const BorderSide(color: AppColors.focus, width: 2),
    );
    expect(
      outlinedStyle.side!.resolve(resting),
      const BorderSide(color: AppColors.divider),
    );
    expect(outlinedStyle.overlayColor!.resolve(focused), Colors.transparent);

    final textStyle = theme.textButtonTheme.style!;
    expect(
      textStyle.side!.resolve(focused),
      const BorderSide(color: AppColors.focus, width: 2),
    );
    expect(textStyle.side!.resolve(resting), BorderSide.none);
    expect(textStyle.overlayColor!.resolve(focused), Colors.transparent);
  });

  test('상태 배지 문구용 Ink는 모든 상태 배경에서 4.5:1을 충족한다', () {
    for (final background in [
      AppColors.successSoft,
      AppColors.attentionSoft,
      AppColors.errorSoft,
    ]) {
      expect(_contrast(AppColors.ink, background), greaterThanOrEqualTo(4.5));
    }
  });

  test('텍스트와 포커스 의미 색상은 실제 표면에서 접근성 대비를 충족한다', () {
    for (final surface in [
      AppColors.surface,
      AppColors.workspace,
      AppColors.primarySoft,
    ]) {
      expect(_contrast(AppColors.muted, surface), greaterThanOrEqualTo(4.5));
      expect(_contrast(AppColors.focus, surface), greaterThanOrEqualTo(3));
      expect(_contrast(AppColors.subtle, surface), greaterThanOrEqualTo(3));
    }
    expect(
      _contrast(
        AppComponentColors.light.disabledContent,
        AppComponentColors.light.disabledSurface,
      ),
      greaterThanOrEqualTo(4.5),
    );
    expect(AppColors.primary, const Color(0xFFEE7203));
    expect(AppColors.focus, const Color(0xFFD96500));
    expect(AppColors.attention, const Color(0xFFB45F06));
    expect(AppColors.attention, isNot(AppColors.primary));
  });

  test('운영형 토큰은 터치 최소 크기와 브랜드 값을 고정한다', () {
    final theme = buildAppTheme();
    expect(AppPalette.brand, const Color(0xFFEE7203));
    expect(AppDesignTokens.standard.controlHeight, greaterThanOrEqualTo(44));
    expect(AppDesignTokens.standard.actionHeight, greaterThanOrEqualTo(48));
    expect(AppDesignTokens.standard.headerHeight, 60);
    expect(AppDesignTokens.standard.sectionHeaderHeight, 40);
    expect(AppDesignTokens.standard.actionBarHeight, greaterThanOrEqualTo(72));
    expect(AppDesignTokens.standard.compactVisualSize, 32);
    expect(AppDesignTokens.standard.brandLogoWidth, 105);
    expect(AppDesignTokens.standard.brandLogoHeight, 30);
    expect(AppDesignTokens.standard.navigationItemWidth, 112);
    expect(AppDesignTokens.standard.navigationItemTopInset, 8);
    expect(AppDesignTokens.standard.stepNavigatorLabelWidth, 52);
    expect(AppDesignTokens.standard.previewLabelMinHeight, 36);
    expect(AppDesignTokens.standard.previewSourceMarkerSize, 6);
    expect(AppDesignTokens.standard.navigationIndicatorThickness, 3);
    expect(AppDesignTokens.standard.inlineProgressSize, 20);
    expect(AppDesignTokens.standard.previewProgressSize, 24);
    expect(AppDesignTokens.standard.pageProgressSize, 28);
    expect(AppDesignTokens.standard.scanResultPanelFraction, .36);
    expect(AppDesignTokens.standard.scanResultPanelMinWidth, 440);
    expect(AppDesignTokens.standard.scanResultPanelMaxWidth, 520);
    expect(AppDesignTokens.standard.reviewInspectorReservedHeight, 340);
    expect(AppDesignTokens.standard.reviewInspectorMaxHeight, 340);
    expect(AppDesignTokens.standard.activitySearchWidth, 320);
    expect(AppDesignTokens.standard.dialogWidth, 400);
    expect(AppDesignTokens.standard.emptyStateMaxWidth, 360);
    expect(AppDesignTokens.standard.metadataLabelWidth, 88);
    expect(AppDesignTokens.standard.focusRingWidth, 2);
    expect(AppDesignTokens.standard.selectionOutlineWidth, 1.5);
    expect(AppDesignTokens.standard.disabledContentOpacity, .56);
    expect(
      AppDesignTokens.standard.scanResultPanelMinWidth,
      lessThan(AppDesignTokens.standard.scanResultPanelMaxWidth),
    );
    expect(
      AppDesignTokens.standard.reviewInspectorReservedHeight,
      lessThanOrEqualTo(AppDesignTokens.standard.reviewInspectorMaxHeight),
    );
    expect(AppDesignTokens.standard.pillRadius, 999);
    expect(AppDesignTokens.standard.feedbackHold, const Duration(seconds: 3));
    expect(AppMotion.interactionCurve, Curves.easeOutCubic);
    expect(AppComponentColors.light.previewScrim, AppPalette.previewScrim);
    expect(AppBreakpoints.supportedMinimumWidth, 1280);
    expect(AppBreakpoints.supportedMinimumHeight, 720);
    expect(AppBreakpoints.scanStacked, 960);
    expect(AppBreakpoints.activityStacked, 1040);
    expect(theme.visualDensity, VisualDensity.standard);
    expect(theme.materialTapTargetSize, MaterialTapTargetSize.padded);
  });

  test('Windows 창 최소 크기는 Flutter 지원 작업대 토큰과 일치한다', () {
    final mainSource = File('windows/runner/main.cpp').readAsStringSync();
    final windowSource = File(
      'windows/runner/win32_window.cpp',
    ).readAsStringSync();

    expect(
      mainSource,
      contains(
        'kMinimumWindowWidth = '
        '${AppBreakpoints.supportedMinimumWidth.toInt()};',
      ),
    );
    expect(
      mainSource,
      contains(
        'kMinimumWindowHeight = '
        '${AppBreakpoints.supportedMinimumHeight.toInt()};',
      ),
    );
    expect(mainSource, contains('window.SetMinimumSize('));
    expect(windowSource, contains('case WM_GETMINMAXINFO:'));
    expect(windowSource, contains('FlutterDesktopGetDpiForHWND(hwnd)'));
  });

  test('간격 토큰과 Theme 컴포넌트 패딩은 4px 리듬을 유지한다', () {
    const base = AppSpacing.x1;
    for (final value in [
      AppSpacing.x1,
      AppSpacing.x2,
      AppSpacing.x3,
      AppSpacing.x4,
      AppSpacing.x6,
      AppSpacing.x8,
      AppDesignTokens.standard.headerHeight,
      AppDesignTokens.standard.sectionHeaderHeight,
      AppDesignTokens.standard.compactVisualSize,
      AppDesignTokens.standard.navigationItemWidth,
      AppDesignTokens.standard.navigationItemTopInset,
      AppDesignTokens.standard.previewLabelMinHeight,
      AppDesignTokens.standard.inlineProgressSize,
      AppDesignTokens.standard.previewProgressSize,
      AppDesignTokens.standard.pageProgressSize,
      AppDesignTokens.standard.scanResultPanelMinWidth,
      AppDesignTokens.standard.scanResultPanelMaxWidth,
      AppDesignTokens.standard.reviewInspectorReservedHeight,
      AppDesignTokens.standard.reviewInspectorMaxHeight,
      AppDesignTokens.standard.activitySearchWidth,
      AppDesignTokens.standard.dialogWidth,
      AppDesignTokens.standard.emptyStateMaxWidth,
      AppDesignTokens.standard.metadataLabelWidth,
    ]) {
      expect(value % base, 0);
    }

    final customized = AppDesignTokens.standard.copyWith(
      dialogWidth: 440,
      emptyStateMaxWidth: 400,
      metadataLabelWidth: 96,
      focusRingWidth: 4,
      selectionOutlineWidth: 2.5,
      brandLogoWidth: 125,
      brandLogoHeight: 34,
      navigationItemWidth: 128,
      navigationItemTopInset: 12,
      stepNavigatorLabelWidth: 60,
      previewSourceMarkerSize: 8,
      navigationIndicatorThickness: 5,
      disabledContentOpacity: .72,
    );
    final midpoint = AppDesignTokens.standard.lerp(customized, .5);
    expect(midpoint.dialogWidth, 420);
    expect(midpoint.emptyStateMaxWidth, 380);
    expect(midpoint.metadataLabelWidth, 92);
    expect(midpoint.focusRingWidth, 3);
    expect(midpoint.selectionOutlineWidth, 2);
    expect(midpoint.brandLogoWidth, 115);
    expect(midpoint.brandLogoHeight, 32);
    expect(midpoint.navigationItemWidth, 120);
    expect(midpoint.navigationItemTopInset, 10);
    expect(midpoint.stepNavigatorLabelWidth, 56);
    expect(midpoint.previewSourceMarkerSize, 7);
    expect(midpoint.navigationIndicatorThickness, 4);
    expect(midpoint.disabledContentOpacity, closeTo(.64, .000001));

    final theme = buildAppTheme();
    expect(
      theme.filledButtonTheme.style?.padding?.resolve(<WidgetState>{}),
      const EdgeInsets.symmetric(
        horizontal: AppSpacing.x4,
        vertical: AppSpacing.x3,
      ),
    );
    expect(
      theme.outlinedButtonTheme.style?.padding?.resolve(<WidgetState>{}),
      const EdgeInsets.symmetric(
        horizontal: AppSpacing.x4,
        vertical: AppSpacing.x2,
      ),
    );
    expect(
      theme.textButtonTheme.style?.padding?.resolve(<WidgetState>{}),
      const EdgeInsets.symmetric(
        horizontal: AppSpacing.x3,
        vertical: AppSpacing.x2,
      ),
    );
    expect(
      theme.inputDecorationTheme.contentPadding,
      const EdgeInsets.symmetric(
        horizontal: AppSpacing.x3,
        vertical: AppSpacing.x2,
      ),
    );
  });

  testWidgets('공통 패널과 섹션 헤더는 높이 토큰으로 렌더링한다', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: const Scaffold(
          body: Column(
            children: [
              AppPanelHeader(key: ValueKey('panel-header'), title: '상품 검수'),
              AppSectionLabel(key: ValueKey('section-header'), label: '활동 목록'),
            ],
          ),
        ),
      ),
    );

    expect(
      tester.getSize(find.byKey(const ValueKey('panel-header'))).height,
      AppDesignTokens.standard.headerHeight,
    );
    expect(
      tester.getSize(find.byKey(const ValueKey('section-header'))).height,
      AppDesignTokens.standard.sectionHeaderHeight,
    );
  });

  test('모든 핵심 타이포그래피 역할은 Pretendard 토큰을 유지한다', () {
    final theme = buildAppTheme();
    final styles = [
      theme.textTheme.headlineMedium,
      theme.textTheme.headlineSmall,
      theme.textTheme.titleLarge,
      theme.textTheme.titleMedium,
      theme.textTheme.bodyLarge,
      theme.textTheme.bodyMedium,
      theme.textTheme.bodySmall,
      theme.textTheme.labelLarge,
      theme.inputDecorationTheme.hintStyle,
      theme.chipTheme.labelStyle,
    ];

    expect(AppTypography.family, 'Pretendard');
    expect(AppTypography.titleSize, 24);
    expect(AppTypography.bodySize, 16);
    expect(AppTypography.regular, FontWeight.w400);
    expect(AppTypography.medium, FontWeight.w500);
    expect(AppTypography.semibold, FontWeight.w600);
    expect(AppTypography.bold, FontWeight.w700);
    expect(AppTypography.compactControlSize, 13);
    expect(AppTypography.compactControlLineHeight, 20);
    expect(
      theme.chipTheme.labelStyle?.fontSize,
      AppTypography.compactControlSize,
    );
    expect(
      theme.chipTheme.labelStyle?.height,
      AppTypography.height(
        AppTypography.compactControlLineHeight,
        AppTypography.compactControlSize,
      ),
    );
    expect(
      theme.textTheme.headlineMedium?.letterSpacing,
      AppTypography.titleTracking,
    );
    expect(
      theme.textTheme.headlineSmall?.letterSpacing,
      AppTypography.sectionTracking,
    );
    expect(AppTypography.brandTracking, -.2);
    expect(AppTypography.dataLabelTracking, .2);
    for (final style in styles) {
      expect(style?.fontFamily, AppTypography.family);
      expect(style?.fontFamilyFallback, AppTypography.fallbackFamilies);
    }
  });

  testWidgets('Windows에서도 주요 조작 영역은 44px 이상을 유지한다', (tester) async {
    debugDefaultTargetPlatformOverride = TargetPlatform.windows;
    try {
      await tester.pumpWidget(
        MaterialApp(
          theme: buildAppTheme(),
          home: Scaffold(
            body: Column(
              children: [
                FilledButton(onPressed: _noop, child: const Text('주요 행동')),
                OutlinedButton(onPressed: _noop, child: const Text('보조 행동')),
                IconButton(onPressed: _noop, icon: const Icon(Icons.refresh)),
                const TextField(decoration: InputDecoration(hintText: '검색')),
                ChoiceChip(
                  label: const Text('필터'),
                  selected: true,
                  onSelected: (_) {},
                ),
              ],
            ),
          ),
        ),
      );

      expect(
        tester.getSize(find.widgetWithText(FilledButton, '주요 행동')).height,
        greaterThanOrEqualTo(48),
      );
      for (final target in [
        find.widgetWithText(OutlinedButton, '보조 행동'),
        find.byType(IconButton),
        find.byType(TextField),
        find.widgetWithText(ChoiceChip, '필터'),
      ]) {
        expect(tester.getSize(target).height, greaterThanOrEqualTo(44));
      }
    } finally {
      debugDefaultTargetPlatformOverride = null;
    }
  });

  testWidgets('선택 표면은 왼쪽 띠가 아닌 전체 외곽선을 사용한다', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: Scaffold(
          body: AppSelectableSurface(
            selected: true,
            onTap: () {},
            child: const Text('선택 항목'),
          ),
        ),
      ),
    );

    final animated = tester.widget<AnimatedContainer>(
      find.descendant(
        of: find.byType(AppSelectableSurface),
        matching: find.byType(AnimatedContainer),
      ),
    );
    final decoration = animated.decoration! as BoxDecoration;
    final border = decoration.border! as Border;
    expect(decoration.color, AppComponentColors.light.selectionSurface);
    expect(border.left.color, AppPalette.brand);
    expect(border.top.color, AppPalette.brand);
    expect(border.right.color, AppPalette.brand);
    expect(border.bottom.color, AppPalette.brand);
    expect(border.left.width, border.top.width);
  });

  testWidgets('마우스 hover는 토큰 면을 사용하고 선택 면을 덮지 않는다', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: Scaffold(
          body: Column(
            children: [
              AppSelectableSurface(
                key: const ValueKey('resting-surface'),
                selected: false,
                onTap: _noop,
                child: const Text('대기 항목'),
              ),
              AppSelectableSurface(
                key: const ValueKey('selected-surface'),
                selected: true,
                onTap: _noop,
                child: const Text('선택 항목'),
              ),
            ],
          ),
        ),
      ),
    );

    final mouse = await tester.createGesture(kind: PointerDeviceKind.mouse);
    addTearDown(mouse.removePointer);
    await mouse.addPointer();
    await mouse.moveTo(
      tester.getCenter(find.byKey(const ValueKey('resting-surface'))),
    );
    await tester.pumpAndSettle();

    BoxDecoration decorationOf(Key key) {
      final animated = tester.widget<AnimatedContainer>(
        find.descendant(
          of: find.byKey(key),
          matching: find.byType(AnimatedContainer),
        ),
      );
      return animated.decoration! as BoxDecoration;
    }

    expect(
      decorationOf(const ValueKey('resting-surface')).color,
      AppComponentColors.light.rowHover,
    );

    await mouse.moveTo(
      tester.getCenter(find.byKey(const ValueKey('selected-surface'))),
    );
    await tester.pumpAndSettle();
    expect(
      decorationOf(const ValueKey('selected-surface')).color,
      AppComponentColors.light.selectionSurface,
    );
  });

  testWidgets('선택 표면의 기본·hover·selected·focus 상태를 시각 회귀한다', (tester) async {
    tester.view.physicalSize = const Size(720, 400);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final focusNode = FocusNode();
    addTearDown(focusNode.dispose);

    Widget stateRow({
      required Key key,
      required String label,
      bool selected = false,
      FocusNode? rowFocusNode,
    }) {
      return AppSelectableSurface(
        key: key,
        selected: selected,
        focusNode: rowFocusNode,
        onTap: _noop,
        minHeight: 52,
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.symmetric(horizontal: 16),
        borderRadius: BorderRadius.circular(8),
        restingBorder: Border.all(color: AppColors.divider),
        child: Row(
          children: [
            Expanded(
              child: Text(
                label,
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
            ),
            Icon(
              selected
                  ? Icons.radio_button_checked_rounded
                  : Icons.chevron_right_rounded,
              color: selected ? AppColors.primary : AppColors.subtle,
              size: 20,
            ),
          ],
        ),
      );
    }

    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: Scaffold(
          backgroundColor: AppColors.workspace,
          body: Center(
            child: RepaintBoundary(
              key: const ValueKey('selection-state-board'),
              child: Container(
                width: 560,
                padding: const EdgeInsets.all(24),
                color: AppColors.surface,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text(
                      '선택 표면 상태',
                      style: buildAppTheme().textTheme.titleLarge,
                    ),
                    const SizedBox(height: 16),
                    stateRow(key: const ValueKey('state-resting'), label: '기본'),
                    stateRow(
                      key: const ValueKey('state-hover'),
                      label: '마우스 hover',
                    ),
                    stateRow(
                      key: const ValueKey('state-selected'),
                      label: '선택됨',
                      selected: true,
                    ),
                    stateRow(
                      key: const ValueKey('state-focused'),
                      label: '키보드 포커스',
                      rowFocusNode: focusNode,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );

    focusNode.requestFocus();
    final mouse = await tester.createGesture(kind: PointerDeviceKind.mouse);
    addTearDown(mouse.removePointer);
    await mouse.addPointer();
    await mouse.moveTo(
      tester.getCenter(find.byKey(const ValueKey('state-hover'))),
    );
    await tester.pumpAndSettle();

    await expectLater(
      find.byKey(const ValueKey('selection-state-board')),
      matchesGoldenFile('goldens/selectable_surface_states.png'),
    );
  });

  testWidgets('비활성 Primary는 컴포넌트 토큰의 면과 문구색을 사용한다', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: const Scaffold(
          body: FilledButton(onPressed: null, child: Text('확인 필요')),
        ),
      ),
    );

    final context = tester.element(find.byType(FilledButton));
    final style = Theme.of(context).filledButtonTheme.style!;
    expect(
      style.backgroundColor!.resolve({WidgetState.disabled}),
      AppComponentColors.light.disabledSurface,
    );
    expect(
      style.foregroundColor!.resolve({WidgetState.disabled}),
      AppComponentColors.light.disabledContent,
    );
  });

  testWidgets('reduced motion에서는 선택 표면 전환 시간을 제거한다', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: const MediaQuery(
          data: MediaQueryData(disableAnimations: true),
          child: Scaffold(
            body: AppSelectableSurface(
              selected: true,
              onTap: _noop,
              child: Text('선택 항목'),
            ),
          ),
        ),
      ),
    );

    final animated = tester.widget<AnimatedContainer>(
      find.descendant(
        of: find.byType(AppSelectableSurface),
        matching: find.byType(AnimatedContainer),
      ),
    );
    expect(animated.duration, Duration.zero);
  });

  testWidgets('키보드 포커스는 선택 표면 사방에 2px Orange 링을 표시한다', (tester) async {
    final semantics = tester.ensureSemantics();
    final focusNode = FocusNode();
    addTearDown(focusNode.dispose);
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: Scaffold(
          body: AppSelectableSurface(
            selected: false,
            onTap: _noop,
            focusNode: focusNode,
            semanticLabel: '선택 항목',
            restingBorder: Border.all(color: AppColors.divider),
            child: const Text('선택 항목'),
          ),
        ),
      ),
    );

    focusNode.requestFocus();
    await tester.pump();
    final animated = tester.widget<AnimatedContainer>(
      find.descendant(
        of: find.byType(AppSelectableSurface),
        matching: find.byType(AnimatedContainer),
      ),
    );
    final border = (animated.decoration! as BoxDecoration).border! as Border;
    for (final side in [border.left, border.top, border.right, border.bottom]) {
      expect(side.color, AppComponentColors.light.focusRing);
      expect(side.width, 2);
    }
    final data = tester
        .getSemantics(find.bySemanticsLabel('선택 항목'))
        .getSemanticsData();
    expect(data.flagsCollection.isFocused, Tristate.isTrue);
    semantics.dispose();
  });

  testWidgets('선택 표면은 단일 의미 노드에 선택 상태와 탭 액션을 제공한다', (tester) async {
    final semantics = tester.ensureSemantics();
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: Scaffold(
          body: AppSelectableSurface(
            selected: true,
            inMutuallyExclusiveGroup: true,
            semanticLabel: 'Muffin 선택',
            onTap: _noop,
            child: const Row(
              children: [Icon(Icons.check), Text('Muffin'), Text('75%')],
            ),
          ),
        ),
      ),
    );

    expect(find.bySemanticsLabel('Muffin 선택'), findsOneWidget);
    final data = tester
        .getSemantics(find.bySemanticsLabel('Muffin 선택'))
        .getSemanticsData();
    expect(data.flagsCollection.isButton, isTrue);
    expect(data.flagsCollection.isInMutuallyExclusiveGroup, isTrue);
    expect(data.flagsCollection.isSelected, Tristate.isTrue);
    expect(data.hasAction(SemanticsAction.tap), isTrue);
    semantics.dispose();
  });

  testWidgets('비활성 선택 표면은 탭을 제거하고 disabled 토큰으로 약화한다', (tester) async {
    final semantics = tester.ensureSemantics();
    var activations = 0;
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: Scaffold(
          body: AppSelectableSurface(
            selected: false,
            enabled: false,
            semanticLabel: '저장 중 상품 선택',
            onTap: () => activations += 1,
            child: const Text('머핀'),
          ),
        ),
      ),
    );

    final data = tester
        .getSemantics(find.bySemanticsLabel('저장 중 상품 선택'))
        .getSemanticsData();
    expect(data.flagsCollection.isEnabled, Tristate.isFalse);
    expect(data.hasAction(SemanticsAction.tap), isFalse);
    final surface = tester.widget<AnimatedContainer>(
      find.descendant(
        of: find.byType(AppSelectableSurface),
        matching: find.byType(AnimatedContainer),
      ),
    );
    expect(
      (surface.decoration! as BoxDecoration).color,
      AppComponentColors.light.disabledSurface,
    );
    expect(
      tester
          .widget<Opacity>(
            find.descendant(
              of: find.byType(AppSelectableSurface),
              matching: find.byType(Opacity),
            ),
          )
          .opacity,
      AppDesignTokens.standard.disabledContentOpacity,
    );
    await tester.tap(find.text('머핀'));
    expect(activations, 0);
    semantics.dispose();
  });

  testWidgets('선택 표면은 Enter와 Space 키로 동일한 행동을 실행한다', (tester) async {
    final focusNode = FocusNode();
    addTearDown(focusNode.dispose);
    var activations = 0;
    var keyboardActivations = 0;
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: Scaffold(
          body: AppSelectableSurface(
            selected: false,
            focusNode: focusNode,
            onTap: () => activations += 1,
            onKeyboardTap: () => keyboardActivations += 1,
            child: const Text('검수 객체'),
          ),
        ),
      ),
    );

    focusNode.requestFocus();
    await tester.pump();
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.sendKeyEvent(LogicalKeyboardKey.space);
    expect(activations, 2);
    expect(keyboardActivations, 2);

    await tester.tap(find.text('검수 객체'));
    await tester.pump();
    expect(activations, 3);
    expect(keyboardActivations, 2);
  });

  testWidgets('상태 배지는 색상과 함께 아이콘·Ink 문구를 제공한다', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: const Scaffold(
          body: AppStatusBadge(
            label: '카메라 연결됨',
            icon: Icons.videocam_rounded,
            color: AppColors.success,
            backgroundColor: AppColors.successSoft,
          ),
        ),
      ),
    );

    expect(find.byIcon(Icons.videocam_rounded), findsOneWidget);
    final label = tester.widget<Text>(find.text('카메라 연결됨'));
    expect(label.style?.color, AppColors.ink);
    expect(
      tester.getSize(find.byType(AppStatusBadge)).height,
      greaterThanOrEqualTo(AppDesignTokens.standard.compactVisualSize),
    );
  });

  testWidgets('단계 탐색기는 현재 위치와 48px 포커스 가능한 이전·다음 행동을 제공한다', (tester) async {
    final semantics = tester.ensureSemantics();
    var previousCalls = 0;
    var nextCalls = 0;
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: Scaffold(
          body: AppStepNavigator(
            current: 2,
            total: 6,
            semanticUnit: '상품',
            previousTooltip: '이전 상품 (↑)',
            nextTooltip: '다음 상품 (↓)',
            onPrevious: () => previousCalls += 1,
            onNext: () => nextCalls += 1,
          ),
        ),
      ),
    );

    expect(find.text('2 / 6'), findsOneWidget);
    expect(find.bySemanticsLabel('현재 2번 상품, 전체 6개'), findsOneWidget);
    final buttons = find.byType(AppIconActionButton);
    expect(buttons, findsNWidgets(2));
    for (final button in buttons.evaluate()) {
      expect(
        tester.getSize(find.byWidget(button.widget)).height,
        AppDesignTokens.standard.actionHeight,
      );
    }

    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pump();
    final previous = find.bySemanticsLabel('이전 상품 (↑)');
    expect(
      tester
          .getSemantics(previous)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    final previousSurface = tester.widget<AnimatedContainer>(
      find.byKey(const ValueKey('icon-action-surface-이전 상품 (↑)')),
    );
    final previousBorder =
        (previousSurface.decoration! as BoxDecoration).border! as Border;
    expect(previousBorder.top.color, AppComponentColors.light.focusRing);
    expect(previousBorder.top.width, 2);
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);

    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pump();
    final next = find.bySemanticsLabel('다음 상품 (↓)');
    expect(
      tester.getSemantics(next).getSemanticsData().flagsCollection.isFocused,
      Tristate.isTrue,
    );
    await tester.sendKeyEvent(LogicalKeyboardKey.space);
    expect(previousCalls, 1);
    expect(nextCalls, 1);
    semantics.dispose();
  });

  testWidgets('키보드 단축키 힌트는 실제 키와 의미를 비조작 요소로 표시한다', (tester) async {
    final semantics = tester.ensureSemantics();
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: const Scaffold(
          body: AppKeyboardShortcutHint(
            shortcut: '/',
            semanticLabel: '검색 단축키: /',
          ),
        ),
      ),
    );

    expect(find.text('/'), findsOneWidget);
    expect(find.bySemanticsLabel('검색 단축키: /'), findsOneWidget);
    expect(find.byType(IconButton), findsNothing);
    final hintSize = tester.getSize(
      find.byKey(const ValueKey('shortcut-hint-/')),
    );
    expect(hintSize.width, greaterThanOrEqualTo(24));
    expect(hintSize.height, greaterThanOrEqualTo(24));
    semantics.dispose();
  });

  testWidgets('공통 필터 그룹은 선택 면과 값 변경을 일관되게 제공한다', (tester) async {
    final semantics = tester.ensureSemantics();
    var selected = 1;
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: StatefulBuilder(
          builder: (context, setState) => Scaffold(
            body: AppFilterGroup<int>(
              label: '입력원',
              value: selected,
              options: const [
                AppFilterOption(1, '전체'),
                AppFilterOption(2, '이미지'),
              ],
              onChanged: (value) => setState(() => selected = value),
            ),
          ),
        ),
      ),
    );

    final allFilter = find.byKey(const ValueKey('filter-입력원-전체'));
    final imageFilter = find.byKey(const ValueKey('filter-입력원-이미지'));
    var allChip = tester.widget<ChoiceChip>(allFilter);
    var imageChip = tester.widget<ChoiceChip>(imageFilter);
    expect(allChip.selected, isTrue);
    expect(allChip.focusNode?.skipTraversal, isFalse);
    expect(imageChip.focusNode?.skipTraversal, isTrue);
    expect(allChip.side?.color, AppComponentColors.light.selectionOutline);
    expect(allChip.side?.width, 1.5);
    expect(allChip.labelStyle?.fontWeight, AppTypography.bold);
    expect(imageChip.side?.color, AppSemanticColors.light.outline);
    expect(imageChip.side?.width, 1);
    expect(imageChip.labelStyle?.fontWeight, AppTypography.semibold);
    expect(find.bySemanticsLabel('입력원, 전체'), findsOneWidget);
    expect(find.bySemanticsLabel('입력원, 이미지'), findsOneWidget);
    final allSemantics = tester.getSemantics(allFilter).getSemanticsData();
    expect(allSemantics.flagsCollection.isButton, isTrue);
    expect(allSemantics.flagsCollection.isInMutuallyExclusiveGroup, isTrue);
    expect(allSemantics.flagsCollection.isSelected, Tristate.isTrue);
    expect(allSemantics.hasAction(SemanticsAction.tap), isTrue);
    await tester.tap(find.widgetWithText(ChoiceChip, '이미지'));
    await tester.pump();
    expect(selected, 2);
    allChip = tester.widget<ChoiceChip>(allFilter);
    imageChip = tester.widget<ChoiceChip>(imageFilter);
    expect(allChip.selected, isFalse);
    expect(allChip.focusNode?.skipTraversal, isTrue);
    expect(allChip.side?.width, 1);
    expect(imageChip.selected, isTrue);
    expect(imageChip.focusNode?.skipTraversal, isFalse);
    expect(imageChip.side?.color, AppComponentColors.light.selectionOutline);
    expect(imageChip.side?.width, 1.5);
    expect(imageChip.labelStyle?.fontWeight, AppTypography.bold);
    expect(
      tester
          .getSemantics(imageFilter)
          .getSemanticsData()
          .flagsCollection
          .isSelected,
      Tristate.isTrue,
    );

    imageChip.focusNode!.requestFocus();
    await tester.pump();
    await tester.sendKeyEvent(LogicalKeyboardKey.arrowLeft);
    await tester.pump();
    expect(selected, 1);
    allChip = tester.widget<ChoiceChip>(allFilter);
    imageChip = tester.widget<ChoiceChip>(imageFilter);
    expect(allChip.selected, isTrue);
    expect(allChip.focusNode?.hasFocus, isTrue);
    expect(allChip.focusNode?.skipTraversal, isFalse);
    expect(imageChip.focusNode?.skipTraversal, isTrue);
    expect(
      tester
          .getSemantics(allFilter)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    semantics.dispose();
  });

  testWidgets('공통 필터 칩은 선택 경계보다 키보드 포커스 링을 우선한다', (tester) async {
    final semantics = tester.ensureSemantics();
    final focusNode = FocusNode();
    addTearDown(focusNode.dispose);

    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: Scaffold(
          body: AppFilterChip(
            label: '카메라',
            semanticLabel: '입력원, 카메라',
            selected: false,
            focusNode: focusNode,
            choiceKey: const ValueKey('focused-filter-chip'),
            onSelected: _noop,
          ),
        ),
      ),
    );

    final chipFinder = find.byKey(const ValueKey('focused-filter-chip'));
    var chip = tester.widget<ChoiceChip>(chipFinder);
    expect(chip.side?.color, AppSemanticColors.light.outline);
    expect(chip.side?.width, 1);

    focusNode.requestFocus();
    await tester.pump();
    chip = tester.widget<ChoiceChip>(chipFinder);
    expect(chip.side?.color, AppComponentColors.light.focusRing);
    expect(chip.side?.width, 2);
    expect(find.bySemanticsLabel('입력원, 카메라'), findsOneWidget);
    expect(
      tester
          .getSemantics(chipFinder)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    semantics.dispose();
  });

  testWidgets('공통 드롭다운은 44px 제어면 전체에 키보드 포커스 링을 표시한다', (tester) async {
    final semantics = tester.ensureSemantics();
    final focusNode = FocusNode();
    addTearDown(focusNode.dispose);

    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: Scaffold(
          body: AppDropdownControl<int>(
            value: 1,
            semanticLabel: '활동 정렬',
            focusNode: focusNode,
            items: const [
              DropdownMenuItem(value: 1, child: Text('최신순')),
              DropdownMenuItem(value: 2, child: Text('오래된순')),
            ],
            onChanged: (_) {},
          ),
        ),
      ),
    );

    final surface = find.byKey(const ValueKey('dropdown-surface-활동 정렬'));
    BoxDecoration decoration() =>
        tester.widget<AnimatedContainer>(surface).decoration! as BoxDecoration;

    expect(
      tester.getSize(surface).height,
      AppDesignTokens.standard.controlHeight,
    );
    expect(
      (decoration().border! as Border).top.color,
      AppSemanticColors.light.outline,
    );
    expect((decoration().border! as Border).top.width, 1);
    expect(find.bySemanticsLabel(RegExp(r'활동 정렬\s+최신순')), findsOneWidget);

    focusNode.requestFocus();
    await tester.pumpAndSettle();
    final focusedBorder = decoration().border! as Border;
    expect(focusedBorder.top.color, AppComponentColors.light.focusRing);
    expect(focusedBorder.top.width, 2);
    expect(
      tester
          .getSemantics(find.byType(DropdownButton<int>))
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    semantics.dispose();
  });

  testWidgets('공통 드롭다운은 reduced motion에서 포커스 전환 시간을 제거한다', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: const MediaQuery(
          data: MediaQueryData(disableAnimations: true),
          child: Scaffold(
            body: AppDropdownControl<int>(
              value: 1,
              semanticLabel: '활동 정렬',
              items: [DropdownMenuItem(value: 1, child: Text('최신순'))],
              onChanged: _ignoreNullableInt,
            ),
          ),
        ),
      ),
    );

    final surface = tester.widget<AnimatedContainer>(
      find.byKey(const ValueKey('dropdown-surface-활동 정렬')),
    );
    expect(surface.duration, Duration.zero);
  });

  testWidgets('공통 아이콘 행동은 48px 조작 영역과 사방 포커스 링을 제공한다', (tester) async {
    final semantics = tester.ensureSemantics();
    final focusNode = FocusNode();
    addTearDown(focusNode.dispose);
    var refreshCalls = 0;

    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: MediaQuery(
          data: const MediaQueryData(disableAnimations: true),
          child: Scaffold(
            body: AppIconActionButton(
              semanticLabel: '새로고침',
              tooltip: '새로고침 (F5)',
              focusNode: focusNode,
              onPressed: () => refreshCalls += 1,
              icon: const Icon(Icons.refresh_rounded),
            ),
          ),
        ),
      ),
    );

    final surface = find.byKey(const ValueKey('icon-action-surface-새로고침'));
    final button = find.bySemanticsLabel('새로고침');
    expect(
      tester.getSize(surface),
      Size(
        AppDesignTokens.standard.actionHeight,
        AppDesignTokens.standard.actionHeight,
      ),
    );
    expect(find.byTooltip('새로고침 (F5)'), findsOneWidget);
    expect(tester.widget<AnimatedContainer>(surface).duration, Duration.zero);

    focusNode.requestFocus();
    await tester.pump();
    final decoration =
        tester.widget<AnimatedContainer>(surface).decoration! as BoxDecoration;
    final border = decoration.border! as Border;
    expect(border.top.color, AppComponentColors.light.focusRing);
    expect(border.top.width, 2);
    final data = tester.getSemantics(button).getSemanticsData();
    expect(data.flagsCollection.isButton, isTrue);
    expect(data.flagsCollection.isFocused, Tristate.isTrue);
    expect(data.hasAction(SemanticsAction.tap), isTrue);

    await tester.tap(button);
    await tester.pump();
    expect(refreshCalls, 1);
    semantics.dispose();
  });

  testWidgets('공통 disclosure는 접힘 상태·포커스·reduced motion을 함께 제공한다', (
    tester,
  ) async {
    final semantics = tester.ensureSemantics();
    final focusNode = FocusNode();
    addTearDown(focusNode.dispose);

    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: MediaQuery(
          data: const MediaQueryData(disableAnimations: true),
          child: Scaffold(
            body: AppDisclosure(
              title: '진단 정보',
              description: '스캔·모델·객체 판정 정보',
              icon: Icons.tune_rounded,
              focusNode: focusNode,
              children: const [Text('Scan ID')],
            ),
          ),
        ),
      ),
    );

    final disclosure = find.bySemanticsLabel('진단 정보');
    final surface = find.byKey(const ValueKey('disclosure-surface-진단 정보'));
    final content = find.byKey(const ValueKey('disclosure-content-진단 정보'));
    expect(
      tester.getSize(surface).height,
      AppDesignTokens.standard.headerHeight,
    );
    expect(find.text('Scan ID'), findsNothing);
    var data = tester.getSemantics(disclosure).getSemanticsData();
    expect(data.flagsCollection.isButton, isTrue);
    expect(data.flagsCollection.isExpanded, Tristate.isFalse);
    expect(data.hasAction(SemanticsAction.expand), isTrue);

    focusNode.requestFocus();
    await tester.pump();
    final decoration =
        tester.widget<AnimatedContainer>(surface).decoration! as BoxDecoration;
    final border = decoration.border! as Border;
    expect(border.top.color, AppComponentColors.light.focusRing);
    expect(border.top.width, 2);
    expect(
      tester
          .getSemantics(disclosure)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pump();
    expect(find.text('Scan ID'), findsOneWidget);
    data = tester.getSemantics(disclosure).getSemanticsData();
    expect(data.flagsCollection.isExpanded, Tristate.isTrue);
    expect(data.hasAction(SemanticsAction.collapse), isTrue);
    expect(content, findsOneWidget);
    expect(find.byType(AnimatedSize), findsNothing);

    await tester.sendKeyEvent(LogicalKeyboardKey.space);
    await tester.pump();
    expect(find.text('Scan ID'), findsNothing);
    expect(
      tester
          .getSemantics(disclosure)
          .getSemanticsData()
          .flagsCollection
          .isExpanded,
      Tristate.isFalse,
    );
    semantics.dispose();
  });

  testWidgets('공통 로딩 상태는 보이는 문구와 단일 live region을 함께 제공한다', (tester) async {
    final semantics = tester.ensureSemantics();
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: const Scaffold(
          body: AppLoadingState(message: '활동 기록을 불러오는 중이에요'),
        ),
      ),
    );

    expect(find.text('활동 기록을 불러오는 중이에요'), findsOneWidget);
    expect(find.byType(CircularProgressIndicator), findsOneWidget);
    expect(
      tester.getSize(find.byType(CircularProgressIndicator)),
      Size.square(AppDesignTokens.standard.pageProgressSize),
    );
    final loading = find.bySemanticsLabel('활동 기록을 불러오는 중이에요');
    expect(loading, findsOneWidget);
    final data = tester.getSemantics(loading).getSemanticsData();
    expect(data.flagsCollection.isLiveRegion, isTrue);
    expect(data.hasAction(SemanticsAction.tap), isFalse);
    semantics.dispose();
  });

  testWidgets('공통 상태 알림은 화면 공간과 조작 행동 없이 live region만 제공한다', (tester) async {
    final semantics = tester.ensureSemantics();
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: const Scaffold(
          body: AppStateAnnouncement(message: '검수 상태. 1개 상품 확인 완료.'),
        ),
      ),
    );

    final announcement = find.bySemanticsLabel('검수 상태. 1개 상품 확인 완료.');
    expect(announcement, findsOneWidget);
    expect(tester.getSize(find.byType(AppStateAnnouncement)), Size.zero);
    final data = tester.getSemantics(announcement).getSemanticsData();
    expect(data.flagsCollection.isLiveRegion, isTrue);
    expect(data.hasAction(SemanticsAction.tap), isFalse);
    semantics.dispose();
  });

  testWidgets('공통 진행 시각은 reduced motion에서 회전 대신 정적 대기 아이콘을 사용한다', (
    tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: const Scaffold(
          body: AppProgressVisual(
            size: 24,
            color: AppColors.ink,
            strokeWidth: 2,
          ),
        ),
      ),
    );

    expect(find.byType(CircularProgressIndicator), findsOneWidget);
    expect(find.byIcon(Icons.hourglass_top_rounded), findsNothing);
    expect(
      tester.getSize(find.byType(AppProgressVisual)),
      const Size.square(24),
    );

    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: const MediaQuery(
          data: MediaQueryData(disableAnimations: true),
          child: Scaffold(
            body: AppProgressVisual(
              size: 24,
              color: AppColors.ink,
              strokeWidth: 2,
            ),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.byType(CircularProgressIndicator), findsNothing);
    expect(find.byIcon(Icons.hourglass_top_rounded), findsOneWidget);
    expect(
      tester.getSize(find.byType(AppProgressVisual)),
      const Size.square(24),
    );
  });

  testWidgets('인라인 알림은 live region 메시지와 복구 행동을 별도 노드로 제공한다', (tester) async {
    final semantics = tester.ensureSemantics();
    var refreshCalls = 0;
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: Scaffold(
          body: AppInlineNotice(
            message: '새로고침하지 못했어요. 기존 활동을 표시하고 있어요.',
            icon: Icons.sync_problem_rounded,
            tone: AppColors.error,
            backgroundColor: AppColors.errorSoft,
            action: TextButton(
              onPressed: () => refreshCalls += 1,
              child: const Text('새로고침'),
            ),
          ),
        ),
      ),
    );

    final message = find.bySemanticsLabel('새로고침하지 못했어요. 기존 활동을 표시하고 있어요.');
    final action = find.bySemanticsLabel('새로고침');
    expect(message, findsOneWidget);
    expect(action, findsOneWidget);
    final messageData = tester.getSemantics(message).getSemanticsData();
    final actionData = tester.getSemantics(action).getSemanticsData();
    expect(messageData.flagsCollection.isLiveRegion, isTrue);
    expect(messageData.hasAction(SemanticsAction.tap), isFalse);
    expect(actionData.flagsCollection.isButton, isTrue);
    expect(actionData.hasAction(SemanticsAction.tap), isTrue);

    await tester.tap(action);
    await tester.pump();
    expect(refreshCalls, 1);
    semantics.dispose();
  });

  testWidgets('상태 배지와 토스트는 중복 없는 상태·live region 레이블을 제공한다', (tester) async {
    final semantics = tester.ensureSemantics();
    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: const Scaffold(
          body: Column(
            children: [
              AppStatusBadge(
                label: '카메라 연결됨',
                icon: Icons.videocam_rounded,
                color: AppColors.success,
                liveRegion: true,
              ),
              AppStatusBadge(
                label: '저장됨',
                icon: Icons.check_circle_outline_rounded,
                color: AppColors.success,
              ),
              AppToast(message: '2개 상품을 확정했어요', icon: Icons.check_circle),
            ],
          ),
        ),
      ),
    );

    expect(find.bySemanticsLabel('카메라 연결됨'), findsOneWidget);
    expect(find.bySemanticsLabel('저장됨'), findsOneWidget);
    expect(find.bySemanticsLabel('2개 상품을 확정했어요'), findsOneWidget);
    final cameraData = tester
        .getSemantics(find.bySemanticsLabel('카메라 연결됨'))
        .getSemanticsData();
    final savedData = tester
        .getSemantics(find.bySemanticsLabel('저장됨'))
        .getSemanticsData();
    final toastData = tester
        .getSemantics(find.bySemanticsLabel('2개 상품을 확정했어요'))
        .getSemanticsData();
    expect(cameraData.flagsCollection.isLiveRegion, isTrue);
    expect(savedData.flagsCollection.isLiveRegion, isFalse);
    expect(toastData.flagsCollection.isLiveRegion, isTrue);
    semantics.dispose();
  });
}

void _noop() {}

void _ignoreNullableInt(int? _) {}

double _contrast(Color a, Color b) {
  final lighter = a.computeLuminance() > b.computeLuminance() ? a : b;
  final darker = lighter == a ? b : a;
  return (lighter.computeLuminance() + .05) / (darker.computeLuminance() + .05);
}
