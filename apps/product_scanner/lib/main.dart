import 'package:flutter/widgets.dart';

import 'app.dart';
import 'features/scanner/application/scanner_controller.dart';
import 'features/scanner/data/image_input.dart';
import 'features/scanner/data/scanner_api.dart';
import 'shared/catalog/product_catalog.dart';
import 'shared/logging/scan_log_repository.dart';
import 'shared/version_info.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  const baseUrl = String.fromEnvironment(
    'SCANNER_API_BASE_URL',
    defaultValue: 'http://127.0.0.1:8000',
  );
  final catalog = await ProductCatalog.load();
  final controller = ScannerController(
    WorkerScannerApi(
      baseUrl: baseUrl,
      waitForReady: true,
      expectedVersion: VersionInfo.current,
    ),
    WindowsCameraGateway(),
    WindowsImageFileGateway(),
    FileScanLogRepository(),
    catalog,
  );
  runApp(ProductScannerApp(controller: controller));
}
