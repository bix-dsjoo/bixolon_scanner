import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:product_scanner/models/scan_models.dart';
import 'package:product_scanner/services/scanner_api.dart';

void main() {
  test('JPEG bytes를 image multipart field로 Worker에 전송한다', () async {
    final client = MockClient((request) async {
      expect(request.method, 'POST');
      expect(request.url.toString(), 'http://127.0.0.1:8000/v1/scan');
      expect(
        request.headers['content-type'],
        startsWith('multipart/form-data'),
      );
      expect(
        utf8.decode(request.bodyBytes, allowMalformed: true),
        contains('name="image"'),
      );
      return http.Response(_approvedBody, 200);
    });
    final api = WorkerScannerApi(
      baseUrl: 'http://127.0.0.1:8000',
      client: client,
    );

    final response = await api.scan(
      imageBytes: Uint8List.fromList([0xff, 0xd8, 0xff, 0xd9]),
      fileName: 'scan.jpg',
    );

    expect(response.status, ScanStatus.approved);
    expect(response.items.single.prediction?.classId, 'bread_06');
  });

  test('Worker 준비가 끝난 뒤 scan 요청을 전송한다', () async {
    var readinessChecks = 0;
    final methods = <String>[];
    final client = MockClient((request) async {
      methods.add('${request.method} ${request.url.path}');
      if (request.url.path == '/health/ready') {
        readinessChecks += 1;
        return http.Response(
          readinessChecks == 1
              ? '{"status":"not_ready"}'
              : '{"status":"ready"}',
          readinessChecks == 1 ? 503 : 200,
        );
      }
      return http.Response(_approvedBody, 200);
    });
    final api = WorkerScannerApi(
      baseUrl: 'http://127.0.0.1:8000',
      client: client,
      waitForReady: true,
      readinessPollInterval: Duration.zero,
    );

    final response = await api.scan(
      imageBytes: Uint8List.fromList([0xff, 0xd8, 0xff, 0xd9]),
      fileName: 'scan.jpg',
    );

    expect(response.status, ScanStatus.approved);
    expect(methods, [
      'GET /health/ready',
      'GET /health/ready',
      'POST /v1/scan',
    ]);
  });

  test('Worker ERROR 응답은 ScannerApiException으로 변환한다', () async {
    final client = MockClient(
      (_) async => http.Response('''{
          "request_id":"request_error_1234",
          "status":"ERROR",
          "reason_codes":["CORRUPT_IMAGE"],
          "items":[],
          "processing_time_ms":1.2,
          "model_versions":{"detector":null,"classifier":null}
        }''', 422),
    );
    final api = WorkerScannerApi(
      baseUrl: 'http://127.0.0.1:8000',
      client: client,
    );

    await expectLater(
      api.scan(imageBytes: Uint8List(2), fileName: 'broken.jpg'),
      throwsA(
        isA<ScannerApiException>()
            .having((error) => error.reasonCodes, 'reasonCodes', [
              'CORRUPT_IMAGE',
            ])
            .having(
              (error) => error.recovery,
              'recovery',
              ScannerErrorRecovery.replaceInput,
            )
            .having((error) => error.message, 'message', contains('JPEG')),
      ),
    );
  });

  test('HTTP 200의 Worker ERROR도 성공 결과가 아닌 재분석 오류로 처리한다', () async {
    final client = MockClient(
      (_) async => http.Response('''{
          "request_id":"request_error_200",
          "status":"ERROR",
          "reason_codes":["MODEL_RUNTIME_FAILURE"],
          "items":[],
          "processing_time_ms":2.0,
          "model_versions":{"detector":null,"classifier":null}
        }''', 200),
    );
    final api = WorkerScannerApi(
      baseUrl: 'http://127.0.0.1:8000',
      client: client,
    );

    await expectLater(
      api.scan(imageBytes: Uint8List(2), fileName: 'scan.jpg'),
      throwsA(
        isA<ScannerApiException>()
            .having(
              (error) => error.recovery,
              'recovery',
              ScannerErrorRecovery.retryAnalysis,
            )
            .having((error) => error.message, 'message', contains('다시 분석')),
      ),
    );
  });

  test('서버 실행 ERROR는 현재 이미지 재분석 복구를 유지한다', () async {
    final client = MockClient(
      (_) async => http.Response('''{
          "request_id":"request_error_5678",
          "status":"ERROR",
          "reason_codes":["MODEL_RUNTIME_FAILURE"],
          "items":[],
          "processing_time_ms":2.1,
          "model_versions":{"detector":null,"classifier":null}
        }''', 500),
    );
    final api = WorkerScannerApi(
      baseUrl: 'http://127.0.0.1:8000',
      client: client,
    );

    await expectLater(
      api.scan(imageBytes: Uint8List(2), fileName: 'scan.jpg'),
      throwsA(
        isA<ScannerApiException>()
            .having(
              (error) => error.recovery,
              'recovery',
              ScannerErrorRecovery.retryAnalysis,
            )
            .having((error) => error.message, 'message', contains('다시 분석')),
      ),
    );
  });
}

const _approvedBody = '''
{
  "request_id":"request_ok_123456",
  "status":"APPROVED",
  "reason_codes":[],
  "items":[{
    "item_id":"item_001",
    "bbox":{"x":1,"y":2,"width":30,"height":40},
    "status":"APPROVED",
    "reason_codes":[],
    "prediction":{"class_id":"bread_06","class_name":"Croissant"},
    "top3":[],
    "confidence":0.99
  }],
  "processing_time_ms":31.4,
  "model_versions":{"detector":"0.1.1","classifier":"0.1.1"}
}
''';
