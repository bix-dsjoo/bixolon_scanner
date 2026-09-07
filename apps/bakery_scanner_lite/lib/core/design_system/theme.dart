import 'package:flutter/material.dart';

import 'tokens.dart';

abstract final class AppColors {
  static const workspace = AppPalette.workspace;
  static const surface = AppPalette.surface;
  static const elevated = AppPalette.elevated;
  static const preview = AppPalette.preview;
  static const ink = AppPalette.ink;
  static const muted = AppPalette.muted;
  static const subtle = AppPalette.subtle;
  static const divider = AppPalette.outline;
  static const primary = AppPalette.brand;
  static const focus = AppPalette.focusRing;
  static const primarySoft = AppPalette.brandSoft;
  static const success = AppPalette.success;
  static const successSoft = AppPalette.successSoft;
  static const attention = AppPalette.attention;
  static const attentionSoft = AppPalette.attentionSoft;
  static const error = AppPalette.error;
  static const errorSoft = AppPalette.errorSoft;
}

ThemeData buildAppTheme() {
  const tokens = AppDesignTokens.standard;
  const scheme = ColorScheme.light(
    primary: AppColors.primary,
    onPrimary: AppColors.ink,
    secondary: AppColors.attention,
    onSecondary: Colors.white,
    error: AppColors.error,
    onError: Colors.white,
    surface: AppColors.surface,
    onSurface: AppColors.ink,
    outline: AppColors.divider,
  );
  final base = ThemeData(
    useMaterial3: true,
    visualDensity: VisualDensity.standard,
    materialTapTargetSize: MaterialTapTargetSize.padded,
    colorScheme: scheme,
    scaffoldBackgroundColor: AppColors.workspace,
    fontFamily: AppTypography.family,
    fontFamilyFallback: AppTypography.fallbackFamilies,
    extensions: const [
      AppSemanticColors.light,
      AppComponentColors.light,
      AppDesignTokens.standard,
    ],
  );
  final baseTextTheme = base.textTheme.apply(
    fontFamily: AppTypography.family,
    fontFamilyFallback: AppTypography.fallbackFamilies,
  );
  return base.copyWith(
    textTheme: baseTextTheme.copyWith(
      headlineMedium: baseTextTheme.headlineMedium?.copyWith(
        fontSize: AppTypography.titleSize,
        height: AppTypography.height(
          AppTypography.titleLineHeight,
          AppTypography.titleSize,
        ),
        fontWeight: AppTypography.bold,
        color: AppColors.ink,
        letterSpacing: AppTypography.titleTracking,
      ),
      headlineSmall: baseTextTheme.headlineSmall?.copyWith(
        fontSize: AppTypography.sectionSize,
        height: AppTypography.height(
          AppTypography.sectionLineHeight,
          AppTypography.sectionSize,
        ),
        fontWeight: AppTypography.bold,
        color: AppColors.ink,
        letterSpacing: AppTypography.sectionTracking,
      ),
      titleLarge: baseTextTheme.titleLarge?.copyWith(
        fontSize: AppTypography.sectionSize,
        height: AppTypography.height(
          AppTypography.sectionLineHeight,
          AppTypography.sectionSize,
        ),
        fontWeight: AppTypography.bold,
        color: AppColors.ink,
      ),
      titleMedium: baseTextTheme.titleMedium?.copyWith(
        fontSize: AppTypography.bodySize,
        height: AppTypography.height(
          AppTypography.bodyLineHeight,
          AppTypography.bodySize,
        ),
        fontWeight: AppTypography.bold,
        color: AppColors.ink,
      ),
      bodyLarge: baseTextTheme.bodyLarge?.copyWith(
        fontSize: AppTypography.bodySize,
        height: AppTypography.height(
          AppTypography.bodyLineHeight,
          AppTypography.bodySize,
        ),
        fontWeight: AppTypography.regular,
        color: AppColors.ink,
      ),
      bodyMedium: baseTextTheme.bodyMedium?.copyWith(
        fontSize: AppTypography.supportingSize,
        height: AppTypography.height(
          AppTypography.supportingLineHeight,
          AppTypography.supportingSize,
        ),
        fontWeight: AppTypography.regular,
        color: AppColors.ink,
      ),
      bodySmall: baseTextTheme.bodySmall?.copyWith(
        fontSize: AppTypography.captionSize,
        height: AppTypography.height(
          AppTypography.captionLineHeight,
          AppTypography.captionSize,
        ),
        fontWeight: AppTypography.regular,
        color: AppColors.muted,
      ),
      labelLarge: baseTextTheme.labelLarge?.copyWith(
        fontSize: AppTypography.actionSize,
        height: AppTypography.height(
          AppTypography.actionLineHeight,
          AppTypography.actionSize,
        ),
        fontWeight: AppTypography.bold,
        color: AppColors.ink,
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style:
          FilledButton.styleFrom(
            minimumSize: Size(112, tokens.actionHeight),
            padding: const EdgeInsets.symmetric(
              horizontal: AppSpacing.x4,
              vertical: AppSpacing.x3,
            ),
            foregroundColor: AppColors.ink,
            disabledForegroundColor: AppComponentColors.light.disabledContent,
            disabledBackgroundColor: AppComponentColors.light.disabledSurface,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(tokens.controlRadius),
            ),
          ).copyWith(
            side: _buttonSide(
              focused: BorderSide(
                color: AppColors.ink,
                width: tokens.focusRingWidth,
              ),
              resting: BorderSide.none,
            ),
            overlayColor: _buttonOverlay(),
          ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style:
          OutlinedButton.styleFrom(
            minimumSize: Size(104, tokens.controlHeight),
            padding: const EdgeInsets.symmetric(
              horizontal: AppSpacing.x4,
              vertical: AppSpacing.x2,
            ),
            foregroundColor: AppColors.ink,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(tokens.controlRadius),
            ),
          ).copyWith(
            side: _buttonSide(
              focused: BorderSide(
                color: AppColors.focus,
                width: tokens.focusRingWidth,
              ),
              resting: const BorderSide(color: AppColors.divider),
            ),
            overlayColor: _buttonOverlay(),
          ),
    ),
    textButtonTheme: TextButtonThemeData(
      style:
          TextButton.styleFrom(
            foregroundColor: AppColors.ink,
            minimumSize: Size(tokens.controlHeight, tokens.controlHeight),
            padding: const EdgeInsets.symmetric(
              horizontal: AppSpacing.x3,
              vertical: AppSpacing.x2,
            ),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(tokens.controlRadius),
            ),
          ).copyWith(
            side: _buttonSide(
              focused: BorderSide(
                color: AppColors.focus,
                width: tokens.focusRingWidth,
              ),
              resting: BorderSide.none,
            ),
            overlayColor: _buttonOverlay(),
          ),
    ),
    iconButtonTheme: IconButtonThemeData(
      style: IconButton.styleFrom(
        minimumSize: Size(tokens.controlHeight, tokens.controlHeight),
        foregroundColor: AppColors.ink,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(tokens.controlRadius),
        ),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: AppColors.workspace,
      isDense: true,
      hintStyle: baseTextTheme.bodyMedium?.copyWith(color: AppColors.muted),
      constraints: BoxConstraints(minHeight: tokens.controlHeight),
      contentPadding: const EdgeInsets.symmetric(
        horizontal: AppSpacing.x3,
        vertical: AppSpacing.x2,
      ),
      prefixIconConstraints: const BoxConstraints(minWidth: 40, minHeight: 40),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(tokens.controlRadius),
        borderSide: const BorderSide(color: AppColors.divider),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(tokens.controlRadius),
        borderSide: const BorderSide(color: AppColors.divider),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(tokens.controlRadius),
        borderSide: BorderSide(
          color: AppColors.focus,
          width: tokens.focusRingWidth,
        ),
      ),
    ),
    chipTheme: base.chipTheme.copyWith(
      side: const BorderSide(color: AppColors.divider),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(tokens.controlRadius),
      ),
      backgroundColor: AppColors.surface,
      selectedColor: AppColors.primarySoft,
      labelStyle: baseTextTheme.bodySmall?.copyWith(
        color: AppColors.ink,
        fontSize: AppTypography.compactControlSize,
        height: AppTypography.height(
          AppTypography.compactControlLineHeight,
          AppTypography.compactControlSize,
        ),
        fontWeight: AppTypography.semibold,
      ),
      padding: const EdgeInsets.symmetric(horizontal: 8),
    ),
    dividerTheme: const DividerThemeData(
      color: AppColors.divider,
      thickness: 1,
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: AppColors.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(tokens.panelRadius),
      ),
    ),
    focusColor: AppColors.focus.withValues(alpha: .12),
    hoverColor: AppColors.ink.withValues(alpha: .04),
    splashColor: AppColors.primary.withValues(alpha: .12),
    scrollbarTheme: ScrollbarThemeData(
      thumbColor: WidgetStateProperty.resolveWith(
        (states) => states.contains(WidgetState.hovered)
            ? AppColors.muted
            : AppColors.subtle,
      ),
      thickness: const WidgetStatePropertyAll(8),
      radius: Radius.circular(tokens.controlRadius),
    ),
  );
}

WidgetStateProperty<BorderSide?> _buttonSide({
  required BorderSide focused,
  required BorderSide resting,
}) {
  return WidgetStateProperty.resolveWith((states) {
    if (states.contains(WidgetState.focused)) {
      return focused;
    }
    return resting;
  });
}

WidgetStateProperty<Color?> _buttonOverlay() {
  return WidgetStateProperty.resolveWith((states) {
    if (states.contains(WidgetState.pressed)) {
      return AppColors.primary.withValues(alpha: .12);
    }
    if (states.contains(WidgetState.hovered)) {
      return AppColors.ink.withValues(alpha: .04);
    }
    if (states.contains(WidgetState.focused)) {
      return Colors.transparent;
    }
    return null;
  });
}
