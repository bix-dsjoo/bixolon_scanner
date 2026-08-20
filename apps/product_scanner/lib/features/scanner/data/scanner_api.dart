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

class WorkerScannerApi implements ScannerApi {
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

  @override
  Future<ScanResponse> scan({
    required Uint8List imageBytes,
    required String fileName,
  }) async {
    try {
      if (waitForReady) {
        await _waitUntilReady();
      }
      final request =
          http.MultipartRequest('POST', Uri.parse('$baseUrl/v1/scan'))
            ..files.add(
              http.MultipartFile.fromBytes(
                'image',
                imageBytes,
                filename: fileName,
              ),
            );
      final streamed = await _client.send(request).timeout(timeout);
      final body = await streamed.stream.bytesToString();
      final response = ScanResponse.fromBody(body);
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
      throw const ScannerApiException('분석 시간이 너무 오래 걸리고 있어요. 다시 분석해 주세요.');
    } on FormatException {
      throw const ScannerApiException('분석 서버의 응답을 확인할 수 없어요.');
    } catch (_) {
      throw const ScannerApiException('분석 서버에 연결할 수 없어요.');
    }
  }

  Future<void> _waitUntilReady() async {
    final deadline = DateTime.now().add(readinessTimeout);
    final readyUrl = Uri.parse('$baseUrl/health/ready');
    while (true) {
      try {
        final response = await _client
            .get(readyUrl)
            .timeout(const Duration(seconds: 1));
        if (response.statusCode >= 200 && response.statusCode < 300) {
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
          }
          return;
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
