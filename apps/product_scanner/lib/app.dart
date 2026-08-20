import 'package:flutter/material.dart';

import 'core/design_system/theme.dart';
import 'features/activity/presentation/activity_screen.dart';
import 'features/scanner/application/scanner_controller.dart';
import 'features/scanner/presentation/scanner_screen.dart';

class ProductScannerApp extends StatelessWidget {
  const ProductScannerApp({
    super.key,
    required this.controller,
    this.autoInitialize = true,
    this.disposeController = true,
  });

  final ScannerController controller;
  final bool autoInitialize;
  final bool disposeController;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BIXOLON Scanner',
      debugShowCheckedModeBanner: false,
      theme: buildAppTheme(),
      home: ScannerScreen(
        controller: controller,
        autoInitialize: autoInitialize,
        disposeController: disposeController,
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
            ),
      ),
    );
  }
}
