import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:bixolon_scanner_sdk/bixolon_scanner_sdk.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

const _versions = <String, dynamic>{
  'worker_version': '0.1.15',
  'detector_version': '0.1.15',
  'classifier_version': '0.1.15',
  'embedder_version': '0.1.15',
  'detector_policy_version': '0.1.15',
  'classifier_policy_version': '0.1.15',
  'catalog_version': '0.1.15',
};

void main() {
  test(
    'connects to an already running Worker without process control',
    () async {
      final client = BixolonScannerClient(
        expectedModelVersion: '0.1.15',
        httpClient: MockClient((request) async {
          expect(request.url.path, '/health/ready');
          return http.Response(
            jsonEncode(<String, dynamic>{
              'status': 'ready',
              'provider': 'cpu',
              ..._versions,
            }),
            200,
          );
        }),
      );

      final readiness = await client.waitUntilReady();
      expect(readiness.provider, 'cpu');
      expect(readiness.versions.hasOneProductVersion, isTrue);
      client.close();
    },
  );

  test('parses an approved scan response', () async {
    final client = BixolonScannerClient(
      httpClient: MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/v1/scan');
        expect(
          request.headers['content-type'],
          startsWith('multipart/form-data'),
        );
        return http.Response(
          jsonEncode(<String, dynamic>{
            'request_id': 'request-1',
            'status': 'SEGMENTATION',
            'reason_codes': <String>[],
            'segmentations': <Map<String, dynamic>>[
              <String, dynamic>{
                'segmentation_id': 'segmentation_001',
                'bbox': <String, int>{
                  'x': 10,
                  'y': 20,
                  'width': 30,
                  'height': 40,
                },
                'status': 'APPROVED',
                'reason_codes': <String>[],
                'prediction': <String, String>{
                  'class_id': 'bread_06',
                  'class_name': 'Croissant',
                },
                'top3': <Object>[],
                'confidence': 0.987,
              },
            ],
            'processing_time_ms': 72.1,
            ..._versions,
          }),
          200,
        );
      }),
    );

    final response = await client.scanBytes(
      Uint8List.fromList(<int>[0xff, 0xd8, 0xff]),
      filename: 'capture.jpg',
    );
    expect(response.status, ScanStatus.segmentation);
    expect(response.segmentations.single.status, SegmentationStatus.approved);
    expect(response.segmentations.single.prediction?.classId, 'bread_06');
    client.close();
  });

  test('rejects inconsistent component versions', () async {
    final client = BixolonScannerClient(
      readinessPollInterval: Duration.zero,
      httpClient: MockClient(
        (_) async => http.Response(
          jsonEncode(<String, dynamic>{
            'status': 'ready',
            'provider': 'cpu',
            ..._versions,
            'catalog_version': '0.1.12',
          }),
          200,
        ),
      ),
    );

    await expectLater(
      client.waitUntilReady(),
      throwsA(
        isA<BixolonContractException>().having(
          (error) => error.code,
          'code',
          'WORKER_VERSION_MISMATCH',
        ),
      ),
    );
    client.close();
  });

  test('scanFile sends the original filename', () async {
    final temporary = File(
      '${Directory.systemTemp.path}${Platform.pathSeparator}pos-capture.png',
    );
    await temporary.writeAsBytes(<int>[0x89, 0x50, 0x4e, 0x47]);
    final client = BixolonScannerClient(
      httpClient: MockClient((request) async {
        expect(
          latin1.decode(request.bodyBytes),
          contains('filename="pos-capture.png"'),
        );
        return http.Response(
          jsonEncode(<String, dynamic>{
            'request_id': 'request-error-1',
            'status': 'ERROR',
            'reason_codes': <String>['INVALID_IMAGE'],
            'segmentations': <Object>[],
            'processing_time_ms': 1.0,
            ..._versions,
          }),
          422,
        );
      }),
    );
    try {
      final response = await client.scanFile(temporary.path);
      expect(response.status, ScanStatus.error);
    } finally {
      client.close();
      await temporary.delete();
    }
  });

  test('rejects an UNKNOWN response without sorted Top-3 evidence', () async {
    final client = BixolonScannerClient(
      httpClient: MockClient(
        (_) async => http.Response(
          jsonEncode(<String, dynamic>{
            'request_id': 'request-unknown-1',
            'status': 'SEGMENTATION',
            'reason_codes': <String>['SEGMENT_BELOW_APPROVAL_THRESHOLD'],
            'segmentations': <Map<String, dynamic>>[
              <String, dynamic>{
                'segmentation_id': 'segmentation_001',
                'bbox': <String, int>{
                  'x': 1,
                  'y': 1,
                  'width': 10,
                  'height': 10,
                },
                'status': 'UNKNOWN',
                'reason_codes': <String>['BELOW_APPROVAL_THRESHOLD'],
                'prediction': null,
                'top3': <Map<String, dynamic>>[
                  <String, dynamic>{
                    'class_id': 'bread_01',
                    'class_name': 'Walnut Donut',
                    'confidence': 0.4,
                  },
                  <String, dynamic>{
                    'class_id': 'bread_02',
                    'class_name': 'Croffle',
                    'confidence': 0.5,
                  },
                ],
                'confidence': 0.4,
              },
            ],
            'processing_time_ms': 2.0,
            ..._versions,
          }),
          200,
        ),
      ),
    );

    await expectLater(
      client.scanBytes(Uint8List(1), filename: 'scan.jpg'),
      throwsA(isA<BixolonContractException>()),
    );
    client.close();
  });
}
