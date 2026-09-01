import 'package:flutter/material.dart';

import 'core/design_system/theme.dart';
import 'features/activity/presentation/activity_screen.dart';
import 'features/capture_library/application/bread_capture_controller.dart';
import 'features/capture_library/presentation/bread_capture_screen.dart';
import 'features/scanner/application/scanner_controller.dart';
import 'features/scanner/presentation/scanner_screen.dart';
import 'shared/version_info.dart';

class ProductScannerApp extends StatelessWidget {
  const ProductScannerApp({
    super.key,
    required this.controller,
    this.autoInitialize = true,
    this.disposeController = true,
    this.breadCaptureController,
  });

  final ScannerController controller;
  final bool autoInitialize;
  final bool disposeController;
  final BreadCaptureController? breadCaptureController;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BIXOLON Bakery AI Scanner v${VersionInfo.current}',
      debugShowCheckedModeBanner: false,
      theme: buildAppTheme(),
      home: ScannerScreen(
        controller: controller,
        autoInitialize: autoInitialize,
        disposeController: disposeController,
        breadCaptureWorkspaceBuilder: breadCaptureController == null
            ? null
            : (context, {required active}) => BreadCaptureScreen(
                controller: breadCaptureController!,
                active: active,
              ),
        activityWorkspaceBuilder:
            (
              context, {
              required active,
              required canChooseImageShortcut,
              required onChooseImageShortcut,
              required onNavigateToScan,
            }) => ActivityScreen(
              loadLogs: controller.loadScanLogs,
              dataRevision: controller.activityDataRevision,
              latestSavedScanId: controller.latestSavedScanId,
              active: active,
              canChooseImageShortcut: canChooseImageShortcut,
              onChooseImageShortcut: onChooseImageShortcut,
              onNavigateToScan: onNavigateToScan,
              exportLogs: controller.exportReviewArchive,
            ),
      ),
    );
  }
}
