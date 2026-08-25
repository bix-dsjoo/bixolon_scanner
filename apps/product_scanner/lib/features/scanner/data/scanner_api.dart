import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../../../shared/models/scan_models.dart';

abstract interface class ScannerApi {
  Future<ScanResponse> scan({
    required Uint8List imageBytes,
    required String fileName,
  });
}

/// Optional startup contract for APIs that need a local Worker to finish model
/// loading before the first scan. Controllers can begin this work while the
/// camera initializes instead of charging it to the first capture.
abstract interface class ScannerApiPreflight {
  bool get isPrepared;

  Future<String?> prepare();
}

class ScannerApiTimings {
  const ScannerApiTimings({
    required this.totalMs,
    required this.readinessMs,
    required this.requestBuildMs,
    required this.httpRoundTripMs,
    required this.responseBodyReadMs,
    required this.responseParseMs,
    this.provider,
  });

  final double totalMs;
  final double readinessMs;
  final double requestBuildMs;
  final double httpRoundTripMs;
  final double responseBodyReadMs;
  final double responseParseMs;
  final String? provider;
}

abstract interface class ScannerApiTimingSource {
  ScannerApiTimings? takeLastTimings();
}

enum ScannerErrorRecovery { retryAnalysis, replaceInput }

class ScannerApiException implements Exception {
  const ScannerApiException(
    this.message, {
    this.reasonCodes = const [],
    this.recovery = ScannerErrorRecovery.retryAnalysis,
  });

  final String message;
  final List<String> reasonCodes;
  final ScannerErrorRecovery recovery;

  @override
  String toString() => message;
}

class WorkerScannerApi
    implements ScannerApi, ScannerApiPreflight, ScannerApiTimingSource {
  WorkerScannerApi({
    required this.baseUrl,
    http.Client? client,
    this.timeout = const Duration(seconds: 65),
    this.waitForReady = false,
    this.expectedVersion,
    this.readinessTimeout = const Duration(seconds: 180),
    this.readinessPollInterval = const Duration(milliseconds: 250),
  }) : _client = client ?? http.Client();

  final String baseUrl;
  final Duration timeout;
  final bool waitForReady;
  final String? expectedVersion;
  final Duration readinessTimeout;
  final Duration readinessPollInterval;
  final http.Client _client;
  ScannerApiTimings? _lastTimings;
  bool _prepared = false;
  String? _preparedProvider;
  Future<String?>? _preparation;

  @override
  bool get isPrepared => !waitForReady || _prepared;

  @override
  Future<String?> prepare() {
    if (!waitForReady) return Future<String?>.value(null);
    if (_prepared) return Future<String?>.value(_preparedProvider);
    final active = _preparation;
    if (active != null) return active;
    final preparation = _prepareAndCache();
    _preparation = preparation;
    return preparation;
  }

  Future<String?> _prepareAndCache() async {
    try {
      final provider = await _waitUntilReady();
      _preparedProvider = provider;
      _prepared = true;
      return provider;
    } finally {
      _preparation = null;
    }
  }

  void _invalidatePreparation() {
    _prepared = false;
    _preparedProvider = null;
  }

  @override
  ScannerApiTimings? takeLastTimings() {
    final value = _lastTimings;
    _lastTimings = null;
    return value;
  }

  @override
  Future<ScanResponse> scan({
    required Uint8List imageBytes,
    required String fileName,
  }) async {
    _lastTimings = null;
    final total = Stopwatch()..start();
    try {
      var readinessMs = 0.0;
      String? provider;
      if (waitForReady) {
        final readiness = Stopwatch()..start();
        provider = await prepare();
        readiness.stop();
        readinessMs = readiness.elapsedMicroseconds / 1000.0;
      }
      final requestBuild = Stopwatch()..start();
      final request =
          http.MultipartRequest('POST', Uri.parse('$baseUrl/v1/scan'))
            ..files.add(
              http.MultipartFile.fromBytes(
                'image',
                imageBytes,
                filename: fileName,
              ),
            );
      requestBuild.stop();
      final httpRoundTrip = Stopwatch()..start();
      final streamed = await _client.send(request).timeout(timeout);
      final responseBodyRead = Stopwatch()..start();
      final body = await streamed.stream.bytesToString();
      responseBodyRead.stop();
      httpRoundTrip.stop();
      final responseParse = Stopwatch()..start();
      final response = ScanResponse.fromBody(body);
      responseParse.stop();
      total.stop();
      _lastTimings = ScannerApiTimings(
        totalMs: total.elapsedMicroseconds / 1000.0,
        readinessMs: readinessMs,
        requestBuildMs: requestBuild.elapsedMicroseconds / 1000.0,
        httpRoundTripMs: httpRoundTrip.elapsedMicroseconds / 1000.0,
        responseBodyReadMs: responseBodyRead.elapsedMicroseconds / 1000.0,
        responseParseMs: responseParse.elapsedMicroseconds / 1000.0,
        provider: provider,
      );
      if (streamed.statusCode < 200 ||
          streamed.statusCode >= 300 ||
          response.status == ScanStatus.error) {
        final presentation = _presentationForReasons(response.reasonCodes);
        throw ScannerApiException(
          presentation.message,
          reasonCodes: response.reasonCodes,
          recovery: presentation.recovery,
        );
      }
      return response;
    } on ScannerApiException {
      rethrow;
    } on TimeoutException {
      _invalidatePreparation();
      throw const ScannerApiException('분석 시간이 너무 오래 걸리고 있어요. 다시 분석해 주세요.');
    } on FormatException {
      _invalidatePreparation();
      throw const ScannerApiException('분석 서버의 응답을 확인할 수 없어요.');
    } catch (_) {
      _invalidatePreparation();
      throw const ScannerApiException('분석 서버에 연결할 수 없어요.');
    }
  }

  Future<String?> _waitUntilReady() async {
    final deadline = DateTime.now().add(readinessTimeout);
    final readyUrl = Uri.parse('$baseUrl/health/ready');
    while (true) {
      try {
        final response = await _client
            .get(readyUrl)
            .timeout(const Duration(seconds: 1));
        if (response.statusCode >= 200 && response.statusCode < 300) {
          String? provider;
          if (expectedVersion != null) {
            final decoded = jsonDecode(response.body);
            if (decoded is! Map<String, dynamic>) {
              throw const ScannerApiException(
                'Worker readiness 응답 형식이 올바르지 않습니다.',
                reasonCodes: ['WORKER_READINESS_INVALID'],
              );
            }
            final versions = <Object?>[
              decoded['worker_version'],
              decoded['detector_version'],
              decoded['classifier_version'],
              decoded['embedder_version'],
              decoded['detector_policy_version'],
              decoded['classifier_policy_version'],
              decoded['catalog_version'],
            ];
            final reported = versions.where((value) => value != null).toList();
            if (reported.isEmpty ||
                reported.any((value) => value != expectedVersion)) {
              throw const ScannerApiException(
                '앱과 분석 구성의 버전이 맞지 않습니다. 같은 버전의 BIXOLON SCANNER를 사용해 주세요.',
                reasonCodes: ['VERSION_MISMATCH'],
              );
            }
            provider = decoded['provider'] as String?;
          }
          return provider;
        }
      } on ScannerApiException {
        rethrow;
      } catch (_) {
        // The local Worker may still be starting or warming its model sessions.
      }

      if (!DateTime.now().isBefore(deadline)) {
        throw const ScannerApiException(
          '분석 서버를 시작하지 못했어요. 앱을 종료한 뒤 다시 실행해 주세요.',
        );
      }
      await Future<void>.delayed(readinessPollInterval);
    }
  }

  static ({String message, ScannerErrorRecovery recovery})
  _presentationForReasons(List<String> reasons) {
    if (reasons.contains('IMAGE_TOO_LARGE')) {
      return (
        message: '이미지 용량이 너무 커요. 다른 이미지를 선택해 주세요.',
        recovery: ScannerErrorRecovery.replaceInput,
      );
    }
    if (reasons.contains('UNSUPPORTED_IMAGE_FORMAT') ||
        reasons.contains('CORRUPT_IMAGE')) {
      return (
        message: 'JPEG 또는 PNG 이미지를 선택해 주세요.',
        recovery: ScannerErrorRecovery.replaceInput,
      );
    }
    return (
      message: '분석하지 못했어요. 잠시 후 다시 분석해 주세요.',
      recovery: ScannerErrorRecovery.retryAnalysis,
    );
  }
}
