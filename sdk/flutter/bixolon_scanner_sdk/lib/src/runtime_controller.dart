import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';

import 'client.dart';
import 'exceptions.dart';
import 'models.dart';

class BixolonRuntimeLayout {
  const BixolonRuntimeLayout({
    required this.workerExecutable,
    required this.modelPackageDirectory,
    required this.storeCatalogDirectory,
  });

  final String workerExecutable;
  final String modelPackageDirectory;
  final String storeCatalogDirectory;

  void validate() {
    for (final path in <String>[
      workerExecutable,
      '$modelPackageDirectory\\metadata.json',
      '$storeCatalogDirectory\\catalog.json',
    ]) {
      if (!File(path).existsSync()) {
        throw BixolonWorkerStartException(
          'Required runtime file is missing: $path',
          code: 'RUNTIME_FILE_MISSING',
        );
      }
    }
  }

  static BixolonRuntimeLayout discover({
    String? runtimeRoot,
    String? storeBundleRoot,
  }) {
    if (!Platform.isWindows) {
      throw const BixolonWorkerStartException(
        'Local Worker auto-start is supported only on Windows.',
        code: 'UNSUPPORTED_LOCAL_PLATFORM',
      );
    }
    final executableDirectory = File(Platform.resolvedExecutable).parent;
    final runtime = Directory(
      runtimeRoot ??
          Platform.environment['BIXOLON_RUNTIME_ROOT'] ??
          '${executableDirectory.path}\\bixolon_runtime',
    );
    final bundle = Directory(
      storeBundleRoot ??
          Platform.environment['BIXOLON_STORE_BUNDLE_DIR'] ??
          _activeBundlePath() ??
          '${runtime.path}\\store-bundle',
    );
    return BixolonRuntimeLayout(
      workerExecutable: '${runtime.path}\\worker\\bixolon-worker.exe',
      modelPackageDirectory: '${bundle.path}\\model-package',
      storeCatalogDirectory: '${bundle.path}\\store-catalog',
    );
  }

  static String? _activeBundlePath() {
    final programData = Platform.environment['PROGRAMDATA'];
    if (programData == null || programData.isEmpty) return null;
    final pointer = File('$programData\\BIXOLON\\Scanner\\active-bundle.json');
    if (!pointer.existsSync()) return null;
    try {
      final value = jsonDecode(pointer.readAsStringSync());
      if (value is Map<String, dynamic> && value['bundle_path'] is String) {
        return value['bundle_path'] as String;
      }
    } catch (_) {
      return null;
    }
    return null;
  }
}

class BixolonWorkerLaunchConfiguration {
  const BixolonWorkerLaunchConfiguration({
    required this.executable,
    required this.workingDirectory,
    required this.environment,
    this.terminateWithApp = true,
  });

  /// N100: CPU detector, Intel UHD embedder and explicit CPU recovery.
  factory BixolonWorkerLaunchConfiguration.n100({
    required BixolonRuntimeLayout layout,
    String host = '127.0.0.1',
    int port = 8000,
    bool terminateWithApp = true,
  }) => BixolonWorkerLaunchConfiguration.cpu(
    layout: layout,
    host: host,
    port: port,
    terminateWithApp: terminateWithApp,
    extraEnvironment: const {
      'BIXOLON_EMBEDDER_PROVIDER': 'openvino_gpu',
      'BIXOLON_EMBEDDER_FALLBACK_PROVIDER': 'same',
      'BIXOLON_PROVIDER_EXECUTION_CPU_FALLBACK': 'true',
      'BIXOLON_VERIFIER_PROVIDER': 'cpu',
      'BIXOLON_CPU_DETECTOR_WORKERS': '1',
      'BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS': '2',
      'BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS': '4',
    },
  );

  factory BixolonWorkerLaunchConfiguration.cpu({
    required BixolonRuntimeLayout layout,
    String host = '127.0.0.1',
    int port = 8000,
    bool terminateWithApp = true,
    Map<String, String> extraEnvironment = const {},
  }) {
    layout.validate();
    return BixolonWorkerLaunchConfiguration(
      executable: layout.workerExecutable,
      workingDirectory: File(layout.workerExecutable).parent.path,
      terminateWithApp: terminateWithApp,
      environment: <String, String>{
        'BIXOLON_PACKAGE_DIR': layout.modelPackageDirectory,
        'BIXOLON_CATALOG_DIR': layout.storeCatalogDirectory,
        'BIXOLON_PROVIDER': 'cpu',
        'BIXOLON_EMBEDDER_PROVIDER': 'same',
        'BIXOLON_EMBEDDER_FALLBACK_PROVIDER': 'none',
        'BIXOLON_HOST': host,
        'BIXOLON_PORT': '$port',
        'BIXOLON_REQUEST_TIMEOUT_SECONDS': '60',
        'BIXOLON_LOG_TO_STDERR': '0',
        ..._cpuThreadEnvironment(layout),
        ...extraEnvironment,
      },
    );
  }

  factory BixolonWorkerLaunchConfiguration.openVino({
    required BixolonRuntimeLayout layout,
    String host = '127.0.0.1',
    int port = 8000,
    bool useIntelGpuForEmbedder = true,
    bool allowEmbedderCpuFallback = true,
    bool terminateWithApp = true,
    Map<String, String> extraEnvironment = const {},
  }) {
    layout.validate();
    return BixolonWorkerLaunchConfiguration(
      executable: layout.workerExecutable,
      workingDirectory: File(layout.workerExecutable).parent.path,
      terminateWithApp: terminateWithApp,
      environment: <String, String>{
        'BIXOLON_PACKAGE_DIR': layout.modelPackageDirectory,
        'BIXOLON_CATALOG_DIR': layout.storeCatalogDirectory,
        'BIXOLON_PROVIDER': 'openvino',
        'BIXOLON_EMBEDDER_PROVIDER': useIntelGpuForEmbedder
            ? 'openvino_gpu'
            : 'same',
        'BIXOLON_EMBEDDER_FALLBACK_PROVIDER': allowEmbedderCpuFallback
            ? 'same'
            : 'none',
        'BIXOLON_HOST': host,
        'BIXOLON_PORT': '$port',
        'BIXOLON_REQUEST_TIMEOUT_SECONDS': '60',
        'BIXOLON_LOG_TO_STDERR': '0',
        ...extraEnvironment,
      },
    );
  }

  final String executable;
  final String workingDirectory;
  final Map<String, String> environment;
  final bool terminateWithApp;

  static Map<String, String> _cpuThreadEnvironment(
    BixolonRuntimeLayout layout,
  ) {
    final profile = File(
      '${Directory(layout.modelPackageDirectory).parent.path}/worker-profile.json',
    );
    if (!profile.existsSync()) return const {};
    final value = jsonDecode(profile.readAsStringSync());
    if (value is! Map<String, dynamic> || value['schema_version'] != '1.0') {
      throw const FormatException('Invalid Store Model CPU profile');
    }
    const fields = {
      'detector_workers': 'BIXOLON_CPU_DETECTOR_WORKERS',
      'detector_intra_op_threads': 'BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS',
      'embedder_intra_op_threads': 'BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS',
    };
    final result = <String, String>{};
    for (final entry in fields.entries) {
      final count = value[entry.key];
      if (count is! int ||
          count < 1 ||
          (entry.key == 'detector_workers' && count > 4)) {
        throw const FormatException('Invalid Store Model CPU thread count');
      }
      result[entry.value] = '$count';
    }
    return result;
  }
}

class BixolonWorkerController {
  const BixolonWorkerController();

  static const MethodChannel _channel = MethodChannel(
    'bixolon_scanner_sdk/worker',
  );

  Future<void> start(BixolonWorkerLaunchConfiguration configuration) async {
    if (!Platform.isWindows) {
      throw const BixolonWorkerStartException(
        'Local Worker process control is supported only on Windows.',
        code: 'UNSUPPORTED_LOCAL_PLATFORM',
      );
    }
    try {
      await _channel.invokeMethod<bool>('ensureWorkerStarted', {
        'executable': configuration.executable,
        'workingDirectory': configuration.workingDirectory,
        'environment': configuration.environment,
        'terminateWithApp': configuration.terminateWithApp,
      });
    } on PlatformException catch (error) {
      throw BixolonWorkerStartException(
        error.message ?? 'Could not start the local Worker.',
        cause: error,
      );
    }
  }

  Future<bool> get isRunning async {
    if (!Platform.isWindows) return false;
    return await _channel.invokeMethod<bool>('isWorkerRunning') ?? false;
  }

  Future<void> stop() async {
    if (!Platform.isWindows) return;
    await _channel.invokeMethod<void>('stopWorker');
  }
}

class BixolonScannerSession {
  BixolonScannerSession._({
    required this.client,
    required this.readiness,
    this.workerController,
  });

  final BixolonScannerClient client;
  final WorkerReadiness readiness;
  final BixolonWorkerController? workerController;

  static Future<BixolonScannerSession> connect({
    Uri? baseUri,
    String? expectedModelVersion,
    Duration startupTimeout = const Duration(seconds: 60),
  }) async {
    final client = BixolonScannerClient(
      baseUri: baseUri,
      expectedModelVersion: expectedModelVersion,
    );
    try {
      final readiness = await client.waitUntilReady(timeout: startupTimeout);
      return BixolonScannerSession._(client: client, readiness: readiness);
    } catch (_) {
      client.close();
      rethrow;
    }
  }

  static Future<BixolonScannerSession> startLocal({
    BixolonRuntimeLayout? layout,
    BixolonWorkerController workerController = const BixolonWorkerController(),
    BixolonWorkerLaunchConfiguration? launchConfiguration,
    Uri? baseUri,
    String? expectedModelVersion,
    Duration startupTimeout = const Duration(seconds: 60),
  }) async {
    final resolvedLayout = layout ?? BixolonRuntimeLayout.discover();
    final configuration =
        launchConfiguration ??
        BixolonWorkerLaunchConfiguration.cpu(layout: resolvedLayout);
    await workerController.start(configuration);

    final client = BixolonScannerClient(
      baseUri: baseUri,
      expectedModelVersion: expectedModelVersion,
    );
    try {
      final readiness = await client.waitUntilReady(timeout: startupTimeout);
      return BixolonScannerSession._(
        client: client,
        readiness: readiness,
        workerController: workerController,
      );
    } catch (_) {
      client.close();
      rethrow;
    }
  }

  void close() => client.close();

  Future<void> shutdown({bool stopLocalWorker = true}) async {
    client.close();
    if (stopLocalWorker) {
      await workerController?.stop();
    }
  }
}
