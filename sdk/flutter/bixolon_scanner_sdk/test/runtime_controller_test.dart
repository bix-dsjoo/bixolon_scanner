import 'dart:io';

import 'package:bixolon_scanner_sdk/bixolon_scanner_sdk.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('builds a CPU-only launch environment', () {
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

      final configuration = BixolonWorkerLaunchConfiguration.cpu(
        layout: layout,
        port: 18080,
        terminateWithApp: false,
      );

      expect(configuration.environment['BIXOLON_PROVIDER'], 'cpu');
      expect(configuration.environment['BIXOLON_EMBEDDER_PROVIDER'], 'same');
      expect(
        configuration.environment['BIXOLON_EMBEDDER_FALLBACK_PROVIDER'],
        'none',
      );
      expect(configuration.environment['BIXOLON_PORT'], '18080');
      expect(configuration.terminateWithApp, isFalse);

      final n100 = BixolonWorkerLaunchConfiguration.n100(layout: layout);
      expect(n100.environment['BIXOLON_PROVIDER'], 'cpu');
      expect(n100.environment['BIXOLON_EMBEDDER_PROVIDER'], 'openvino_gpu');
      expect(n100.environment['BIXOLON_EMBEDDER_FALLBACK_PROVIDER'], 'same');
      expect(
        n100.environment['BIXOLON_PROVIDER_EXECUTION_CPU_FALLBACK'],
        'true',
      );
      expect(n100.environment['BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS'], '4');
      expect(n100.environment['BIXOLON_OPENVINO_GPU_PRECISION'], 'f16');
      expect(n100.environment['BIXOLON_REUSE_VERIFIER_EMBEDDINGS'], 'true');
      expect(n100.environment['BIXOLON_PARALLEL_VERIFICATION'], 'true');
      expect(n100.environment['BIXOLON_VERIFIER_PROVIDER'], 'cpu');
      expect(n100.environment['BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS'], '4');

      final profile = File(
        '${metadata.parent.parent.path}/worker-profile.json',
      );
      profile.writeAsStringSync('''{"schema_version":"1.0","detector_workers":1,
        "detector_intra_op_threads":8,"embedder_intra_op_threads":12}''');
      final tuned = BixolonWorkerLaunchConfiguration.cpu(layout: layout);
      expect(tuned.environment['BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS'], '8');
      expect(tuned.environment['BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS'], '12');
      final overridden = BixolonWorkerLaunchConfiguration.cpu(
        layout: layout,
        extraEnvironment: {'BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS': '4'},
      );
      expect(
        overridden.environment['BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS'],
        '4',
      );
      profile.writeAsStringSync(
        '{"schema_version":"1.0","detector_workers":0}',
      );
      expect(
        () => BixolonWorkerLaunchConfiguration.cpu(layout: layout),
        throwsFormatException,
      );
    } finally {
      temporaryRoot.deleteSync(recursive: true);
    }
  });
}
