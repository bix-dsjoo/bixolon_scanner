import 'package:flutter/widgets.dart';

import 'app.dart';
import 'features/capture_library/application/bread_capture_controller.dart';
import 'features/capture_library/data/bread_capture_repository.dart';
import 'features/scanner/application/scanner_controller.dart';
import 'features/scanner/data/scanner_api.dart';
import 'shared/catalog/product_catalog.dart';
import 'shared/input/image_input.dart';
import 'shared/logging/scan_log_repository.dart';
import 'shared/version_info.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  const baseUrl = String.fromEnvironment(
    'SCANNER_API_BASE_URL',
    defaultValue: 'http://127.0.0.1:8000',
  );
  final catalog = await ProductCatalog.load();
  final cameraGateway = WindowsCameraGateway();
  final controller = ScannerController(
    WorkerScannerApi(
      baseUrl: baseUrl,
      waitForReady: true,
      expectedVersion: VersionInfo.current,
    ),
    cameraGateway,
    WindowsImageFileGateway(),
    FileScanLogRepository(),
    catalog,
  );
  const configuredCaptureRoot = String.fromEnvironment('BREAD_CAPTURE_ROOT');
  final captureRepository = await FileBreadCaptureRepository.create(
    configuredRoot: configuredCaptureRoot,
  );
  final breadCaptureController = BreadCaptureController(
    captureRepository,
    cameraGateway,
    WindowsBreadCaptureDirectoryPicker(),
  );
  runApp(
    ProductScannerApp(
      controller: controller,
      breadCaptureController: breadCaptureController,
    ),
  );
}
