import 'dart:ui' show AppExitResponse;
import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'features/scanner/scan_controller.dart';
import 'lite_app.dart';
import 'shared/input.dart';
import 'shared/run_log.dart';
import 'shared/worker_client.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  Directory support;
  try {
    support = await getApplicationSupportDirectory();
  } catch (_) {
    support = Directory(
      '${Platform.environment['LOCALAPPDATA'] ?? Directory.systemTemp.path}/BIXOLON/BakeryAIScannerLite',
    );
  }
  final controller = ScanController(
    WindowsPhotoInputs(),
    BundledScanner(),
    FileRunLogs(
      Directory('${support.path}/logs'),
      Directory(
        '${Directory.systemTemp.path}/BIXOLON-BakeryAIScannerLite-emergency',
      ),
    ),
  );
  AppLifecycleListener(
    onExitRequested: () async {
      await controller.close();
      return AppExitResponse.exit;
    },
  );
  runApp(LiteApp(controller: controller));
  unawaited(controller.initialize());
}
