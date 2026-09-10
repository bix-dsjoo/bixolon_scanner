import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'result.dart';

abstract interface class ScannerService {
  Future<void> start();
  Future<ScanResult> scan(Uint8List bytes, String fileName, String attemptId);
  Future<void> close();
}

class WorkerClient implements ScannerService {
  WorkerClient({required this.baseUri, http.Client? client})
    : _client = client ?? http.Client();
  final Uri baseUri;
  final http.Client _client;
  @override
  Future<void> start() async {
    final response = await _client
        .get(baseUri.resolve('/health/ready'))
        .timeout(const Duration(seconds: 3));
    final data = jsonDecode(response.body) as Map<String, dynamic>;
    if (response.statusCode != 200 ||
        data['status'] != 'ready' ||
        data['worker_version'] != workerVersion ||
        versionFields.any(
          (key) => data[key] != null && data[key] != workerVersion,
        )) {
      throw const FormatException('Worker not ready or incompatible');
    }
  }

  @override
  Future<ScanResult> scan(
    Uint8List bytes,
    String fileName,
    String attemptId,
  ) async {
    try {
      final request = http.MultipartRequest('POST', baseUri.resolve('/v1/scan'))
        ..files.add(
          http.MultipartFile.fromBytes('image', bytes, filename: fileName),
        );
      final response = await (() async => http.Response.fromStream(
        await _client.send(request),
      ))().timeout(const Duration(seconds: 65));
      final result = ScanResult.fromWorker(
        jsonDecode(response.body) as Map<String, dynamic>,
      );
      if ((response.statusCode != 200) != (result.status == 'ERROR')) {
        return ScanResult.clientError(attemptId, 'CLIENT_RESPONSE_INVALID');
      }
      return result;
    } on TimeoutException {
      return ScanResult.clientError(attemptId, 'CLIENT_REQUEST_TIMEOUT');
    } on FormatException {
      return ScanResult.clientError(attemptId, 'CLIENT_RESPONSE_INVALID');
    } catch (_) {
      return ScanResult.clientError(attemptId, 'CLIENT_CONNECTION_FAILED');
    }
  }

  @override
  Future<void> close() async => _client.close();
}

// One owned Worker on a private loopback port; no endpoint/model settings in Lite.
class BundledScanner implements ScannerService {
  Process? _process;
  WorkerClient? _client;
  bool _closed = false;
  Future<void>? _starting;
  @override
  Future<void> start() => _starting ??= _start();
  Future<void> _start() async {
    final root = File(Platform.resolvedExecutable).parent.path;
    final worker = '$root/worker';
    final socket = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
    final port = socket.port;
    await socket.close();
    if (_closed) throw StateError('Closed');
    _process = await Process.start(
      '$worker/bixolon-worker.exe',
      [],
      workingDirectory: worker,
      mode: ProcessStartMode.normal,
      environment: {
        'BIXOLON_PACKAGE_DIR': '$worker/model-package',
        'BIXOLON_CATALOG_DIR': '$worker/store-catalog',
        'BIXOLON_HOST': '127.0.0.1',
        'BIXOLON_PORT': '$port',
        'BIXOLON_PROVIDER': 'cpu',
        'BIXOLON_EMBEDDER_PROVIDER': 'same',
        'BIXOLON_EMBEDDER_FALLBACK_PROVIDER': 'none',
        'BIXOLON_CPU_DETECTOR_WORKERS': '1',
        'BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS': '8',
        'BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS': '12',
        'BIXOLON_REQUEST_TIMEOUT_SECONDS': '60',
      },
      runInShell: false,
    );
    _process!.stdout.drain<void>();
    _process!.stderr.drain<void>();
    var exited = false;
    _process!.exitCode.then((_) => exited = true);
    _client = WorkerClient(baseUri: Uri.parse('http://127.0.0.1:$port'));
    final end = DateTime.now().add(const Duration(seconds: 120));
    while (!_closed && !exited && DateTime.now().isBefore(end)) {
      try {
        await _client!.start();
        return;
      } catch (_) {
        await Future<void>.delayed(const Duration(milliseconds: 400));
      }
    }
    _process?.kill();
    throw StateError('Worker startup failed');
  }

  @override
  Future<ScanResult> scan(
    Uint8List bytes,
    String fileName,
    String attemptId,
  ) async {
    try {
      await start();
    } catch (_) {
      return ScanResult.clientError(attemptId, 'CLIENT_WORKER_UNAVAILABLE');
    }
    return _client!.scan(bytes, fileName, attemptId);
  }

  @override
  Future<void> close() async {
    _closed = true;
    await _client?.close();
    _process?.kill();
  }
}
