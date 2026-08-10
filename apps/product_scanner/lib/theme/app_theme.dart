import 'package:flutter/material.dart';

abstract final class AppColors {
  static const workspace = Color(0xFFF6F6F6);
  static const surface = Color(0xFFFFFFFF);
  static const elevated = Color(0xFFFAFAFA);
  static const preview = Color(0xFF101010);
  static const ink = Color(0xFF171717);
  static const muted = Color(0xFF737373);
  static const subtle = Color(0xFFA3A3A3);
  static const divider = Color(0xFFE5E5E5);
  static const primary = Color(0xFF1D4ED8);
  static const primarySoft = Color(0xFFF3F6FC);
  static const success = Color(0xFF16865A);
  static const successSoft = Color(0xFFF3F8F5);
  static const attention = Color(0xFFB45F06);
  static const attentionSoft = Color(0xFFFCF8F1);
  static const error = Color(0xFFB42318);
  static const errorSoft = Color(0xFFFCF4F3);
}

ThemeData buildAppTheme() {
  const scheme = ColorScheme.light(
    primary: AppColors.primary,
    onPrimary: Colors.white,
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
    colorScheme: scheme,
    scaffoldBackgroundColor: AppColors.workspace,
    fontFamily: 'Pretendard',
    fontFamilyFallback: const ['Segoe UI', 'Malgun Gothic', 'Arial'],
  );
  return base.copyWith(
    textTheme: base.textTheme.copyWith(
      headlineSmall: const TextStyle(
        fontSize: 20,
        height: 1.25,
        fontWeight: FontWeight.w700,
        color: AppColors.ink,
      ),
      titleLarge: const TextStyle(
        fontSize: 17,
        height: 1.35,
        fontWeight: FontWeight.w700,
        color: AppColors.ink,
      ),
      titleMedium: const TextStyle(
        fontSize: 15,
        height: 1.4,
        fontWeight: FontWeight.w600,
        color: AppColors.ink,
      ),
      bodyLarge: const TextStyle(
        fontSize: 16,
        height: 1.55,
        fontWeight: FontWeight.w400,
        color: AppColors.ink,
      ),
      bodyMedium: const TextStyle(
        fontSize: 14,
        height: 1.5,
        fontWeight: FontWeight.w400,
        color: AppColors.ink,
      ),
      bodySmall: const TextStyle(
        fontSize: 12,
        height: 1.45,
        fontWeight: FontWeight.w400,
        color: AppColors.muted,
      ),
      labelLarge: const TextStyle(
        fontSize: 14,
        height: 1.2,
        fontWeight: FontWeight.w600,
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size(104, 36),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size(104, 36),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
        side: const BorderSide(color: AppColors.divider),
        foregroundColor: AppColors.ink,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: AppColors.primary,
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: AppColors.workspace,
      isDense: true,
      hintStyle: const TextStyle(color: AppColors.muted),
      constraints: const BoxConstraints(minHeight: 36, maxHeight: 36),
      contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      prefixIconConstraints: const BoxConstraints(minWidth: 34, minHeight: 34),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(4),
        borderSide: const BorderSide(color: AppColors.divider),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(4),
        borderSide: const BorderSide(color: AppColors.divider),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(4),
        borderSide: const BorderSide(color: AppColors.primary, width: 2),
      ),
    ),
    dividerTheme: const DividerThemeData(
      color: AppColors.divider,
      thickness: 1,
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: AppColors.surface,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
    ),
  );
}
