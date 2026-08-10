import 'package:flutter/material.dart';

abstract final class AppPalette {
  static const brand = Color(0xFFEE7203);
  static const brandTint = Color(0xFFF28E00);
  static const brandSoft = Color(0xFFFFF4E9);
  static const ink = Color(0xFF171717);
  static const inkSoft = Color(0xFF3F3D3E);
  static const muted = Color(0xFF696767);
  static const subtle = Color(0xFF878786);
  static const outline = Color(0xFFE3E3E1);
  static const hover = Color(0xFFF0F0EE);
  static const disabled = Color(0xFFE8E8E6);
  static const disabledInk = muted;
  static const focusRing = Color(0xFFD96500);
  static const workspace = Color(0xFFF5F5F5);
  static const surface = Color(0xFFFFFFFF);
  static const elevated = Color(0xFFFAFAF9);
  static const preview = Color(0xFF101010);
  static const previewLabel = Color(0xCC171717);
  static const previewScrim = Color(0x8A000000);
  static const onPreview = Color(0xFFFFFFFF);
  static const onPreviewMuted = Color(0xB3FFFFFF);
  static const previewGuideOutline = Color.from(
    alpha: .16,
    red: 1,
    green: 1,
    blue: 1,
  );
  static const previewGuideAccent = Color.from(
    alpha: .9,
    red: 238 / 255,
    green: 114 / 255,
    blue: 3 / 255,
  );
  static const success = Color(0xFF16865A);
  static const successSoft = Color(0xFFF0F8F4);
  static const attention = Color(0xFFB45F06);
  static const attentionSoft = Color(0xFFFFF6E8);
  static const error = Color(0xFFB42318);
  static const errorSoft = Color(0xFFFFF1F0);
}

abstract final class AppTypography {
  static const family = 'Pretendard';
  static const fallbackFamilies = ['Segoe UI', 'Malgun Gothic', 'Arial'];

  static const titleSize = 24.0;
  static const titleLineHeight = 32.0;
  static const sectionSize = 18.0;
  static const sectionLineHeight = 26.0;
  static const bodySize = 16.0;
  static const bodyLineHeight = 24.0;
  static const supportingSize = 14.0;
  static const supportingLineHeight = 20.0;
  static const captionSize = 12.0;
  static const captionLineHeight = 18.0;
  static const compactControlSize = 13.0;
  static const compactControlLineHeight = 20.0;
  static const actionSize = 15.0;
  static const actionLineHeight = 20.0;

  static const regular = FontWeight.w400;
  static const medium = FontWeight.w500;
  static const semibold = FontWeight.w600;
  static const bold = FontWeight.w700;

  static const titleTracking = -.4;
  static const sectionTracking = -.25;
  static const brandTracking = -.2;
  static const dataLabelTracking = .2;

  static double height(double lineHeight, double fontSize) =>
      lineHeight / fontSize;
}

abstract final class AppMotion {
  static const fast = Duration(milliseconds: 160);
  static const standard = Duration(milliseconds: 240);
  static const feedbackHold = Duration(seconds: 3);
  static const interactionCurve = Curves.easeOutCubic;
}

/// Component-level roles keep interaction styling independent from brand and
/// status colors. Screens should consume these roles instead of inventing a
/// selected, hover, focus, or toast color locally.
@immutable
class AppComponentColors extends ThemeExtension<AppComponentColors> {
  const AppComponentColors({
    required this.selectionSurface,
    required this.selectionOutline,
    required this.selectionContent,
    required this.rowHover,
    required this.actionBarSurface,
    required this.toastSurface,
    required this.onToast,
    required this.focusRing,
    required this.disabledSurface,
    required this.disabledContent,
    required this.previewLabelSurface,
    required this.previewScrim,
    required this.onPreview,
    required this.onPreviewMuted,
    required this.previewGuideOutline,
    required this.previewGuideAccent,
  });

  final Color selectionSurface;
  final Color selectionOutline;
  final Color selectionContent;
  final Color rowHover;
  final Color actionBarSurface;
  final Color toastSurface;
  final Color onToast;
  final Color focusRing;
  final Color disabledSurface;
  final Color disabledContent;
  final Color previewLabelSurface;
  final Color previewScrim;
  final Color onPreview;
  final Color onPreviewMuted;
  final Color previewGuideOutline;
  final Color previewGuideAccent;

  static const light = AppComponentColors(
    selectionSurface: AppPalette.brandSoft,
    selectionOutline: AppPalette.brand,
    selectionContent: AppPalette.ink,
    rowHover: AppPalette.hover,
    actionBarSurface: AppPalette.surface,
    toastSurface: AppPalette.ink,
    onToast: AppPalette.surface,
    focusRing: AppPalette.focusRing,
    disabledSurface: AppPalette.disabled,
    disabledContent: AppPalette.disabledInk,
    previewLabelSurface: AppPalette.previewLabel,
    previewScrim: AppPalette.previewScrim,
    onPreview: AppPalette.onPreview,
    onPreviewMuted: AppPalette.onPreviewMuted,
    previewGuideOutline: AppPalette.previewGuideOutline,
    previewGuideAccent: AppPalette.previewGuideAccent,
  );

  @override
  AppComponentColors copyWith({
    Color? selectionSurface,
    Color? selectionOutline,
    Color? selectionContent,
    Color? rowHover,
    Color? actionBarSurface,
    Color? toastSurface,
    Color? onToast,
    Color? focusRing,
    Color? disabledSurface,
    Color? disabledContent,
    Color? previewLabelSurface,
    Color? previewScrim,
    Color? onPreview,
    Color? onPreviewMuted,
    Color? previewGuideOutline,
    Color? previewGuideAccent,
  }) {
    return AppComponentColors(
      selectionSurface: selectionSurface ?? this.selectionSurface,
      selectionOutline: selectionOutline ?? this.selectionOutline,
      selectionContent: selectionContent ?? this.selectionContent,
      rowHover: rowHover ?? this.rowHover,
      actionBarSurface: actionBarSurface ?? this.actionBarSurface,
      toastSurface: toastSurface ?? this.toastSurface,
      onToast: onToast ?? this.onToast,
      focusRing: focusRing ?? this.focusRing,
      disabledSurface: disabledSurface ?? this.disabledSurface,
      disabledContent: disabledContent ?? this.disabledContent,
      previewLabelSurface: previewLabelSurface ?? this.previewLabelSurface,
      previewScrim: previewScrim ?? this.previewScrim,
      onPreview: onPreview ?? this.onPreview,
      onPreviewMuted: onPreviewMuted ?? this.onPreviewMuted,
      previewGuideOutline: previewGuideOutline ?? this.previewGuideOutline,
      previewGuideAccent: previewGuideAccent ?? this.previewGuideAccent,
    );
  }

  @override
  AppComponentColors lerp(AppComponentColors? other, double t) {
    if (other == null) return this;
    return AppComponentColors(
      selectionSurface: Color.lerp(
        selectionSurface,
        other.selectionSurface,
        t,
      )!,
      selectionOutline: Color.lerp(
        selectionOutline,
        other.selectionOutline,
        t,
      )!,
      selectionContent: Color.lerp(
        selectionContent,
        other.selectionContent,
        t,
      )!,
      rowHover: Color.lerp(rowHover, other.rowHover, t)!,
      actionBarSurface: Color.lerp(
        actionBarSurface,
        other.actionBarSurface,
        t,
      )!,
      toastSurface: Color.lerp(toastSurface, other.toastSurface, t)!,
      onToast: Color.lerp(onToast, other.onToast, t)!,
      focusRing: Color.lerp(focusRing, other.focusRing, t)!,
      disabledSurface: Color.lerp(disabledSurface, other.disabledSurface, t)!,
      disabledContent: Color.lerp(disabledContent, other.disabledContent, t)!,
      previewLabelSurface: Color.lerp(
        previewLabelSurface,
        other.previewLabelSurface,
        t,
      )!,
      previewScrim: Color.lerp(previewScrim, other.previewScrim, t)!,
      onPreview: Color.lerp(onPreview, other.onPreview, t)!,
      onPreviewMuted: Color.lerp(onPreviewMuted, other.onPreviewMuted, t)!,
      previewGuideOutline: Color.lerp(
        previewGuideOutline,
        other.previewGuideOutline,
        t,
      )!,
      previewGuideAccent: Color.lerp(
        previewGuideAccent,
        other.previewGuideAccent,
        t,
      )!,
    );
  }
}

@immutable
class AppSemanticColors extends ThemeExtension<AppSemanticColors> {
  const AppSemanticColors({
    required this.brand,
    required this.onBrand,
    required this.brandSoft,
    required this.success,
    required this.successSoft,
    required this.attention,
    required this.attentionSoft,
    required this.error,
    required this.errorSoft,
    required this.workspace,
    required this.surface,
    required this.elevated,
    required this.ink,
    required this.muted,
    required this.subtle,
    required this.outline,
  });

  final Color brand;
  final Color onBrand;
  final Color brandSoft;
  final Color success;
  final Color successSoft;
  final Color attention;
  final Color attentionSoft;
  final Color error;
  final Color errorSoft;
  final Color workspace;
  final Color surface;
  final Color elevated;
  final Color ink;
  final Color muted;
  final Color subtle;
  final Color outline;

  static const light = AppSemanticColors(
    brand: AppPalette.brand,
    onBrand: AppPalette.ink,
    brandSoft: AppPalette.brandSoft,
    success: AppPalette.success,
    successSoft: AppPalette.successSoft,
    attention: AppPalette.attention,
    attentionSoft: AppPalette.attentionSoft,
    error: AppPalette.error,
    errorSoft: AppPalette.errorSoft,
    workspace: AppPalette.workspace,
    surface: AppPalette.surface,
    elevated: AppPalette.elevated,
    ink: AppPalette.ink,
    muted: AppPalette.muted,
    subtle: AppPalette.subtle,
    outline: AppPalette.outline,
  );

  @override
  AppSemanticColors copyWith({
    Color? brand,
    Color? onBrand,
    Color? brandSoft,
    Color? success,
    Color? successSoft,
    Color? attention,
    Color? attentionSoft,
    Color? error,
    Color? errorSoft,
    Color? workspace,
    Color? surface,
    Color? elevated,
    Color? ink,
    Color? muted,
    Color? subtle,
    Color? outline,
  }) {
    return AppSemanticColors(
      brand: brand ?? this.brand,
      onBrand: onBrand ?? this.onBrand,
      brandSoft: brandSoft ?? this.brandSoft,
      success: success ?? this.success,
      successSoft: successSoft ?? this.successSoft,
      attention: attention ?? this.attention,
      attentionSoft: attentionSoft ?? this.attentionSoft,
      error: error ?? this.error,
      errorSoft: errorSoft ?? this.errorSoft,
      workspace: workspace ?? this.workspace,
      surface: surface ?? this.surface,
      elevated: elevated ?? this.elevated,
      ink: ink ?? this.ink,
      muted: muted ?? this.muted,
      subtle: subtle ?? this.subtle,
      outline: outline ?? this.outline,
    );
  }

  @override
  AppSemanticColors lerp(AppSemanticColors? other, double t) {
    if (other == null) return this;
    return AppSemanticColors(
      brand: Color.lerp(brand, other.brand, t)!,
      onBrand: Color.lerp(onBrand, other.onBrand, t)!,
      brandSoft: Color.lerp(brandSoft, other.brandSoft, t)!,
      success: Color.lerp(success, other.success, t)!,
      successSoft: Color.lerp(successSoft, other.successSoft, t)!,
      attention: Color.lerp(attention, other.attention, t)!,
      attentionSoft: Color.lerp(attentionSoft, other.attentionSoft, t)!,
      error: Color.lerp(error, other.error, t)!,
      errorSoft: Color.lerp(errorSoft, other.errorSoft, t)!,
      workspace: Color.lerp(workspace, other.workspace, t)!,
      surface: Color.lerp(surface, other.surface, t)!,
      elevated: Color.lerp(elevated, other.elevated, t)!,
      ink: Color.lerp(ink, other.ink, t)!,
      muted: Color.lerp(muted, other.muted, t)!,
      subtle: Color.lerp(subtle, other.subtle, t)!,
      outline: Color.lerp(outline, other.outline, t)!,
    );
  }
}

@immutable
class AppDesignTokens extends ThemeExtension<AppDesignTokens> {
  const AppDesignTokens({
    required this.space,
    required this.controlHeight,
    required this.actionHeight,
    required this.rowHeight,
    required this.headerHeight,
    required this.sectionHeaderHeight,
    required this.actionBarHeight,
    required this.compactVisualSize,
    required this.stepNavigatorLabelWidth,
    required this.previewLabelMinHeight,
    required this.previewSourceMarkerSize,
    required this.navigationIndicatorThickness,
    required this.inlineProgressSize,
    required this.previewProgressSize,
    required this.pageProgressSize,
    required this.scanGuideFraction,
    required this.scanGuideCornerLength,
    required this.scanGuideCenterMarkSize,
    required this.scanGuideStrokeWidth,
    required this.scanResultPanelFraction,
    required this.scanResultPanelMinWidth,
    required this.scanResultPanelMaxWidth,
    required this.reviewInspectorReservedHeight,
    required this.reviewInspectorMaxHeight,
    required this.activitySearchWidth,
    required this.dialogWidth,
    required this.emptyStateMaxWidth,
    required this.metadataLabelWidth,
    required this.focusRingWidth,
    required this.selectionOutlineWidth,
    required this.disabledContentOpacity,
    required this.controlRadius,
    required this.selectionRadius,
    required this.panelRadius,
    required this.pillRadius,
    required this.motionFast,
    required this.motionStandard,
    required this.feedbackHold,
  });

  final double space;
  final double controlHeight;
  final double actionHeight;
  final double rowHeight;
  final double headerHeight;
  final double sectionHeaderHeight;
  final double actionBarHeight;
  final double compactVisualSize;
  final double stepNavigatorLabelWidth;
  final double previewLabelMinHeight;
  final double previewSourceMarkerSize;
  final double navigationIndicatorThickness;
  final double inlineProgressSize;
  final double previewProgressSize;
  final double pageProgressSize;
  final double scanGuideFraction;
  final double scanGuideCornerLength;
  final double scanGuideCenterMarkSize;
  final double scanGuideStrokeWidth;
  final double scanResultPanelFraction;
  final double scanResultPanelMinWidth;
  final double scanResultPanelMaxWidth;
  final double reviewInspectorReservedHeight;
  final double reviewInspectorMaxHeight;
  final double activitySearchWidth;
  final double dialogWidth;
  final double emptyStateMaxWidth;
  final double metadataLabelWidth;
  final double focusRingWidth;
  final double selectionOutlineWidth;
  final double disabledContentOpacity;
  final double controlRadius;
  final double selectionRadius;
  final double panelRadius;
  final double pillRadius;
  final Duration motionFast;
  final Duration motionStandard;
  final Duration feedbackHold;

  static const standard = AppDesignTokens(
    space: 4,
    controlHeight: 44,
    actionHeight: 48,
    rowHeight: 60,
    headerHeight: 60,
    sectionHeaderHeight: 40,
    actionBarHeight: 72,
    compactVisualSize: 32,
    stepNavigatorLabelWidth: 52,
    previewLabelMinHeight: 36,
    previewSourceMarkerSize: 6,
    navigationIndicatorThickness: 3,
    inlineProgressSize: 20,
    previewProgressSize: 24,
    pageProgressSize: 28,
    scanGuideFraction: .64,
    scanGuideCornerLength: 34,
    scanGuideCenterMarkSize: 7,
    scanGuideStrokeWidth: 2.5,
    scanResultPanelFraction: .36,
    scanResultPanelMinWidth: 440,
    scanResultPanelMaxWidth: 520,
    reviewInspectorReservedHeight: 340,
    reviewInspectorMaxHeight: 340,
    activitySearchWidth: 320,
    dialogWidth: 400,
    emptyStateMaxWidth: 360,
    metadataLabelWidth: 88,
    focusRingWidth: 2,
    selectionOutlineWidth: 1.5,
    disabledContentOpacity: .56,
    controlRadius: 8,
    selectionRadius: 8,
    panelRadius: 12,
    pillRadius: 999,
    motionFast: AppMotion.fast,
    motionStandard: AppMotion.standard,
    feedbackHold: AppMotion.feedbackHold,
  );

  @override
  AppDesignTokens copyWith({
    double? space,
    double? controlHeight,
    double? actionHeight,
    double? rowHeight,
    double? headerHeight,
    double? sectionHeaderHeight,
    double? actionBarHeight,
    double? compactVisualSize,
    double? stepNavigatorLabelWidth,
    double? previewLabelMinHeight,
    double? previewSourceMarkerSize,
    double? navigationIndicatorThickness,
    double? inlineProgressSize,
    double? previewProgressSize,
    double? pageProgressSize,
    double? scanGuideFraction,
    double? scanGuideCornerLength,
    double? scanGuideCenterMarkSize,
    double? scanGuideStrokeWidth,
    double? scanResultPanelFraction,
    double? scanResultPanelMinWidth,
    double? scanResultPanelMaxWidth,
    double? reviewInspectorReservedHeight,
    double? reviewInspectorMaxHeight,
    double? activitySearchWidth,
    double? dialogWidth,
    double? emptyStateMaxWidth,
    double? metadataLabelWidth,
    double? focusRingWidth,
    double? selectionOutlineWidth,
    double? disabledContentOpacity,
    double? controlRadius,
    double? selectionRadius,
    double? panelRadius,
    double? pillRadius,
    Duration? motionFast,
    Duration? motionStandard,
    Duration? feedbackHold,
  }) {
    return AppDesignTokens(
      space: space ?? this.space,
      controlHeight: controlHeight ?? this.controlHeight,
      actionHeight: actionHeight ?? this.actionHeight,
      rowHeight: rowHeight ?? this.rowHeight,
      headerHeight: headerHeight ?? this.headerHeight,
      sectionHeaderHeight: sectionHeaderHeight ?? this.sectionHeaderHeight,
      actionBarHeight: actionBarHeight ?? this.actionBarHeight,
      compactVisualSize: compactVisualSize ?? this.compactVisualSize,
      stepNavigatorLabelWidth:
          stepNavigatorLabelWidth ?? this.stepNavigatorLabelWidth,
      previewLabelMinHeight:
          previewLabelMinHeight ?? this.previewLabelMinHeight,
      previewSourceMarkerSize:
          previewSourceMarkerSize ?? this.previewSourceMarkerSize,
      navigationIndicatorThickness:
          navigationIndicatorThickness ?? this.navigationIndicatorThickness,
      inlineProgressSize: inlineProgressSize ?? this.inlineProgressSize,
      previewProgressSize: previewProgressSize ?? this.previewProgressSize,
      pageProgressSize: pageProgressSize ?? this.pageProgressSize,
      scanGuideFraction: scanGuideFraction ?? this.scanGuideFraction,
      scanGuideCornerLength:
          scanGuideCornerLength ?? this.scanGuideCornerLength,
      scanGuideCenterMarkSize:
          scanGuideCenterMarkSize ?? this.scanGuideCenterMarkSize,
      scanGuideStrokeWidth: scanGuideStrokeWidth ?? this.scanGuideStrokeWidth,
      scanResultPanelFraction:
          scanResultPanelFraction ?? this.scanResultPanelFraction,
      scanResultPanelMinWidth:
          scanResultPanelMinWidth ?? this.scanResultPanelMinWidth,
      scanResultPanelMaxWidth:
          scanResultPanelMaxWidth ?? this.scanResultPanelMaxWidth,
      reviewInspectorReservedHeight:
          reviewInspectorReservedHeight ?? this.reviewInspectorReservedHeight,
      reviewInspectorMaxHeight:
          reviewInspectorMaxHeight ?? this.reviewInspectorMaxHeight,
      activitySearchWidth: activitySearchWidth ?? this.activitySearchWidth,
      dialogWidth: dialogWidth ?? this.dialogWidth,
      emptyStateMaxWidth: emptyStateMaxWidth ?? this.emptyStateMaxWidth,
      metadataLabelWidth: metadataLabelWidth ?? this.metadataLabelWidth,
      focusRingWidth: focusRingWidth ?? this.focusRingWidth,
      selectionOutlineWidth:
          selectionOutlineWidth ?? this.selectionOutlineWidth,
      disabledContentOpacity:
          disabledContentOpacity ?? this.disabledContentOpacity,
      controlRadius: controlRadius ?? this.controlRadius,
      selectionRadius: selectionRadius ?? this.selectionRadius,
      panelRadius: panelRadius ?? this.panelRadius,
      pillRadius: pillRadius ?? this.pillRadius,
      motionFast: motionFast ?? this.motionFast,
      motionStandard: motionStandard ?? this.motionStandard,
      feedbackHold: feedbackHold ?? this.feedbackHold,
    );
  }

  @override
  AppDesignTokens lerp(AppDesignTokens? other, double t) {
    if (other == null) return this;
    return AppDesignTokens(
      space: _lerpDouble(space, other.space, t),
      controlHeight: _lerpDouble(controlHeight, other.controlHeight, t),
      actionHeight: _lerpDouble(actionHeight, other.actionHeight, t),
      rowHeight: _lerpDouble(rowHeight, other.rowHeight, t),
      headerHeight: _lerpDouble(headerHeight, other.headerHeight, t),
      sectionHeaderHeight: _lerpDouble(
        sectionHeaderHeight,
        other.sectionHeaderHeight,
        t,
      ),
      actionBarHeight: _lerpDouble(actionBarHeight, other.actionBarHeight, t),
      compactVisualSize: _lerpDouble(
        compactVisualSize,
        other.compactVisualSize,
        t,
      ),
      stepNavigatorLabelWidth: _lerpDouble(
        stepNavigatorLabelWidth,
        other.stepNavigatorLabelWidth,
        t,
      ),
      previewLabelMinHeight: _lerpDouble(
        previewLabelMinHeight,
        other.previewLabelMinHeight,
        t,
      ),
      previewSourceMarkerSize: _lerpDouble(
        previewSourceMarkerSize,
        other.previewSourceMarkerSize,
        t,
      ),
      navigationIndicatorThickness: _lerpDouble(
        navigationIndicatorThickness,
        other.navigationIndicatorThickness,
        t,
      ),
      inlineProgressSize: _lerpDouble(
        inlineProgressSize,
        other.inlineProgressSize,
        t,
      ),
      previewProgressSize: _lerpDouble(
        previewProgressSize,
        other.previewProgressSize,
        t,
      ),
      pageProgressSize: _lerpDouble(
        pageProgressSize,
        other.pageProgressSize,
        t,
      ),
      scanGuideFraction: _lerpDouble(
        scanGuideFraction,
        other.scanGuideFraction,
        t,
      ),
      scanGuideCornerLength: _lerpDouble(
        scanGuideCornerLength,
        other.scanGuideCornerLength,
        t,
      ),
      scanGuideCenterMarkSize: _lerpDouble(
        scanGuideCenterMarkSize,
        other.scanGuideCenterMarkSize,
        t,
      ),
      scanGuideStrokeWidth: _lerpDouble(
        scanGuideStrokeWidth,
        other.scanGuideStrokeWidth,
        t,
      ),
      scanResultPanelFraction: _lerpDouble(
        scanResultPanelFraction,
        other.scanResultPanelFraction,
        t,
      ),
      scanResultPanelMinWidth: _lerpDouble(
        scanResultPanelMinWidth,
        other.scanResultPanelMinWidth,
        t,
      ),
      scanResultPanelMaxWidth: _lerpDouble(
        scanResultPanelMaxWidth,
        other.scanResultPanelMaxWidth,
        t,
      ),
      reviewInspectorReservedHeight: _lerpDouble(
        reviewInspectorReservedHeight,
        other.reviewInspectorReservedHeight,
        t,
      ),
      reviewInspectorMaxHeight: _lerpDouble(
        reviewInspectorMaxHeight,
        other.reviewInspectorMaxHeight,
        t,
      ),
      activitySearchWidth: _lerpDouble(
        activitySearchWidth,
        other.activitySearchWidth,
        t,
      ),
      dialogWidth: _lerpDouble(dialogWidth, other.dialogWidth, t),
      emptyStateMaxWidth: _lerpDouble(
        emptyStateMaxWidth,
        other.emptyStateMaxWidth,
        t,
      ),
      metadataLabelWidth: _lerpDouble(
        metadataLabelWidth,
        other.metadataLabelWidth,
        t,
      ),
      focusRingWidth: _lerpDouble(focusRingWidth, other.focusRingWidth, t),
      selectionOutlineWidth: _lerpDouble(
        selectionOutlineWidth,
        other.selectionOutlineWidth,
        t,
      ),
      disabledContentOpacity: _lerpDouble(
        disabledContentOpacity,
        other.disabledContentOpacity,
        t,
      ),
      controlRadius: _lerpDouble(controlRadius, other.controlRadius, t),
      selectionRadius: _lerpDouble(selectionRadius, other.selectionRadius, t),
      panelRadius: _lerpDouble(panelRadius, other.panelRadius, t),
      pillRadius: _lerpDouble(pillRadius, other.pillRadius, t),
      motionFast: _lerpDuration(motionFast, other.motionFast, t),
      motionStandard: _lerpDuration(motionStandard, other.motionStandard, t),
      feedbackHold: _lerpDuration(feedbackHold, other.feedbackHold, t),
    );
  }

  static double _lerpDouble(double a, double b, double t) => a + (b - a) * t;

  static Duration _lerpDuration(Duration a, Duration b, double t) => Duration(
    microseconds: _lerpDouble(
      a.inMicroseconds.toDouble(),
      b.inMicroseconds.toDouble(),
      t,
    ).round(),
  );
}

abstract final class AppSpacing {
  static const x1 = 4.0;
  static const x2 = 8.0;
  static const x3 = 12.0;
  static const x4 = 16.0;
  static const x6 = 24.0;
  static const x8 = 32.0;
}

abstract final class AppBreakpoints {
  static const supportedMinimumWidth = 1280.0;
  static const supportedMinimumHeight = 720.0;
  static const scanStacked = 960.0;
  static const activityStacked = 1040.0;
}

extension AppThemeTokens on BuildContext {
  AppSemanticColors get appColors =>
      Theme.of(this).extension<AppSemanticColors>() ?? AppSemanticColors.light;

  AppDesignTokens get appTokens =>
      Theme.of(this).extension<AppDesignTokens>() ?? AppDesignTokens.standard;

  AppComponentColors get appComponents =>
      Theme.of(this).extension<AppComponentColors>() ??
      AppComponentColors.light;
}
