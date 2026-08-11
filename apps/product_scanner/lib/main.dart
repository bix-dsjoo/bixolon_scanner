import 'package:flutter/widgets.dart';

import 'app.dart';
import 'catalog/product_catalog.dart';
import 'controllers/scanner_controller.dart';
import 'services/image_input.dart';
import 'services/scan_log_repository.dart';
import 'services/scanner_api.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  const baseUrl = String.fromEnvironment(
    'SCANNER_API_BASE_URL',
    defaultValue: 'http://127.0.0.1:8000',
  );
  final catalog = await ProductCatalog.load();
  final controller = ScannerController(
    WorkerScannerApi(baseUrl: baseUrl, waitForReady: true),
    WindowsCameraGateway(),
    WindowsImageFileGateway(),
    FileScanLogRepository(),
    catalog,
  );
  runApp(ProductScannerApp(controller: controller));
}
