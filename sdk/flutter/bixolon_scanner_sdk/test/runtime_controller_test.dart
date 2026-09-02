import 'dart:io';

import 'package:bixolon_scanner_sdk/bixolon_scanner_sdk.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('builds an OpenVINO launch environment without forcing GPU', () {
    final temporaryRoot = Directory.systemTemp.createTempSync(
      'bixolon-scanner-sdk-',
    );
    try {
      final worker = File('${temporaryRoot.path}\\worker\\bixolon-worker.exe')
        ..createSync(recursive: true);
      final metadata = File(
        '${temporaryRoot.path}\\bundle\\model-package\\metadata.json',
      )..createSync(recursive: true);
      final catalog = File(
        '${temporaryRoot.path}\\bundle\\store-catalog\\catalog.json',
      )..createSync(recursive: true);
      final layout = BixolonRuntimeLayout(
        workerExecutable: worker.path,
        modelPackageDirectory: metadata.parent.path,
        storeCatalogDirectory: catalog.parent.path,
      );

      final configuration = BixolonWorkerLaunchConfiguration.openVino(
        layout: layout,
        port: 18080,
        useIntelGpuForEmbedder: false,
        allowEmbedderCpuFallback: false,
        terminateWithApp: false,
      );

      expect(configuration.environment['BIXOLON_PROVIDER'], 'openvino');
      expect(configuration.environment['BIXOLON_EMBEDDER_PROVIDER'], 'same');
      expect(
        configuration.environment['BIXOLON_EMBEDDER_FALLBACK_PROVIDER'],
        'none',
      );
      expect(configuration.environment['BIXOLON_PORT'], '18080');
      expect(configuration.terminateWithApp, isFalse);
    } finally {
      temporaryRoot.deleteSync(recursive: true);
    }
  });
}
