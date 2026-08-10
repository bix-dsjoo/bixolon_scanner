import 'dart:async';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../models/scan_models.dart';

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
    this.timeout = const Duration(seconds: 35),
  }) : _client = client ?? http.Client();

  final String baseUrl;
  final Duration timeout;
  final http.Client _client;

  @override
  Future<ScanResponse> scan({
    required Uint8List imageBytes,
    required String fileName,
  }) async {
    try {
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
