import 'package:flutter/material.dart';

import 'controllers/scanner_controller.dart';
import 'scanner/scanner_screen.dart';
import 'theme/app_theme.dart';

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
      ),
    );
  }
}
