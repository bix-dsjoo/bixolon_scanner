import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'exceptions.dart';
import 'models.dart';

class BixolonScannerClient {
  BixolonScannerClient({
    Uri? baseUri,
    http.Client? httpClient,
    this.scanTimeout = const Duration(seconds: 65),
    this.readinessTimeout = const Duration(seconds: 180),
    this.readinessPollInterval = const Duration(milliseconds: 250),
    this.expectedModelVersion,
  }) : baseUri = baseUri ?? Uri.parse('http://127.0.0.1:8000'),
       _httpClient = httpClient ?? http.Client();

  final Uri baseUri;
  final Duration scanTimeout;
  final Duration readinessTimeout;
  final Duration readinessPollInterval;
  final String? expectedModelVersion;
  final http.Client _httpClient;
  Future<void> _serialTail = Future<void>.value();

  Uri _endpoint(String path) => baseUri.resolve(path);

  Future<WorkerReadiness> waitUntilReady({Duration? timeout}) async {
    final effectiveTimeout = timeout ?? readinessTimeout;
    final deadline = DateTime.now().add(effectiveTimeout);
    Object? lastFailure;
    while (DateTime.now().isBefore(deadline)) {
      try {
        final response = await _httpClient
            .get(_endpoint('/health/ready'))
            .timeout(const Duration(seconds: 1));
        if (response.statusCode >= 200 && response.statusCode < 300) {
          final value = jsonDecode(response.body);
          if (value is! Map<String, dynamic>) {
            throw const FormatException(
              'Readiness response must be an object.',
            );
          }
          final readiness = WorkerReadiness.fromJson(value);
          _validateVersions(readiness.versions);
          return readiness;
        }
      } on BixolonContractException {
        rethrow;
      } catch (error) {
        lastFailure = error;
      }
      await Future<void>.delayed(readinessPollInterval);
    }
    throw BixolonTransportException(
      'Worker did not become ready within ${effectiveTimeout.inSeconds} seconds.',
      code: 'WORKER_READINESS_TIMEOUT',
      cause: lastFailure,
    );
  }

  Future<ScanResponse> scanBytes(
    Uint8List imageBytes, {
    required String filename,
  }) => _serial(() => _scanBytes(imageBytes, filename: filename));

  Future<ScanResponse> scanFile(String imagePath) async {
    final file = File(imagePath);
    try {
      return scanBytes(
        await file.readAsBytes(),
        filename: file.uri.pathSegments.last,
      );
    } on FileSystemException catch (error) {
      throw BixolonTransportException(
        'Could not read the scan image.',
        code: 'SCAN_IMAGE_READ_FAILED',
        cause: error,
      );
    }
  }

  Future<ScanResponse> _scanBytes(
    Uint8List imageBytes, {
    required String filename,
  }) async {
    try {
      final request = http.MultipartRequest('POST', _endpoint('/v1/scan'))
        ..files.add(
          http.MultipartFile.fromBytes('image', imageBytes, filename: filename),
        );
      final streamed = await _httpClient.send(request).timeout(scanTimeout);
      final body = await streamed.stream.bytesToString();
      final response = ScanResponse.fromBody(body);
      _validateVersions(response.versions);
      return response;
    } on TimeoutException catch (error) {
      throw BixolonTransportException(
        'Scan request timed out.',
        code: 'SCAN_TIMEOUT',
        cause: error,
      );
    } on FormatException catch (error) {
      throw BixolonContractException(
        'Worker returned an invalid scan response.',
        cause: error,
      );
    } on BixolonScannerException {
      rethrow;
    } catch (error) {
      throw BixolonTransportException(
        'Could not communicate with the Worker.',
        cause: error,
      );
    }
  }

  Future<T> _serial<T>(Future<T> Function() action) {
    final completer = Completer<T>();
    _serialTail = _serialTail.then((_) async {
      try {
        completer.complete(await action());
      } catch (error, stackTrace) {
        completer.completeError(error, stackTrace);
      }
    });
    return completer.future;
  }

  void _validateVersions(ComponentVersions versions) {
    if (!versions.hasOneProductVersion) {
      throw const BixolonContractException(
        'Worker Runtime and Catalog versions are inconsistent.',
        code: 'WORKER_VERSION_MISMATCH',
      );
    }
    if (expectedModelVersion != null &&
        versions.nonNullValues.any(
          (version) => version != expectedModelVersion,
        )) {
      throw BixolonContractException(
        'Expected model $expectedModelVersion but Worker reported a different version.',
        code: 'UNEXPECTED_MODEL_VERSION',
      );
    }
  }

  void close() => _httpClient.close();
}
