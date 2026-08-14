import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/catalog/product_catalog.dart';
import 'package:product_scanner/models/scan_models.dart';

import 'support/test_catalog.dart';

void main() {
  test('Worker UNKNOWN 응답을 계약에 맞게 파싱한다', () {
    final response = ScanResponse.fromBody('''
      {
        "request_id": "request_12345678",
        "status": "UNKNOWN",
        "reason_codes": ["ITEM_BELOW_APPROVAL_THRESHOLD"],
        "items": [
          {
            "item_id": "item_001",
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 40},
            "status": "UNKNOWN",
            "reason_codes": ["BELOW_APPROVAL_THRESHOLD"],
            "prediction": null,
            "top3": [
              {"class_id": "bread_06", "class_name": "Croissant", "confidence": 0.7},
              {"class_id": "bread_13", "class_name": "Muffin", "confidence": 0.2}
            ],
            "confidence": 0.7
          }
        ],
        "processing_time_ms": 82.4,
        "model_versions": {"detector": "0.1.1", "classifier": "0.1.1"}
      }
    ''');

    expect(response.status, ScanStatus.unknown);
    expect(response.items.single.status, ItemStatus.unknown);
    expect(response.items.single.top3.first.classId, 'bread_06');
    expect(response.items.single.bbox.width, 30);
  });

  test('상품 카탈로그는 한국어 표시명과 원본 영문명·class id 검색을 함께 지원한다', () {
    expect(testCatalog.search('머핀').single.displayName, '머핀');
    expect(testCatalog.search('muffin').single.displayName, '머핀');
    expect(testCatalog.search('bread_11').single.displayName, '베이글');
    expect(testCatalog.search('muffin').single.className, 'Muffin');
  });

  test('기존 로그 상품명은 class id를 우선해 한국어 표시명으로 복원한다', () {
    expect(
      testCatalog.displayNameFor(
        classId: 'bread_13',
        className: 'Legacy name',
        fallback: 'Legacy name',
      ),
      '머핀',
    );
    expect(
      testCatalog.displayNameFor(className: 'Muffin', fallback: 'Muffin'),
      '머핀',
    );
    expect(
      testCatalog.displayNameFor(fallback: 'Retired product'),
      'Retired product',
    );
  });

  test('한국어 표시명이 없는 카탈로그 항목을 거부한다', () {
    expect(
      () => ProductCatalog.fromJsonBody('''
        {
          "schema_version":"1.0",
          "products":[{"class_id":"bread_13","class_name":"Muffin"}]
        }
      '''),
      throwsFormatException,
    );
  });

  test('번들 운영 카탈로그 20개는 한국어 표시명과 영문 검색을 함께 제공한다', () async {
    final catalog = await ProductCatalog.load();

    expect(catalog.products, hasLength(20));
    expect(catalog.search('머핀').single.displayName, '머핀');
    expect(catalog.search('Muffin').single.displayName, '머핀');
    expect(catalog.search('bread_13').single.displayName, '머핀');
    expect(
      catalog.products.every(
        (product) => product.displayName != product.className,
      ),
      isTrue,
    );
  });

  test('지원하지 않는 카탈로그 schema version을 거부한다', () {
    expect(
      () =>
          ProductCatalog.fromJsonBody('{"schema_version":"2.0","products":[]}'),
      throwsFormatException,
    );
  });
  mainOfficialContractTests();
}

void mainOfficialContractTests() {
  test('parses official segmentation response and direct versions', () {
    final response = ScanResponse.fromJson({
      'request_id': 'request-100',
      'status': 'SEGMENTATION',
      'reason_codes': ['SEGMENT_RECAPTURE_REQUIRED'],
      'segmentations': [
        {
          'segmentation_id': 'segmentation_001',
          'bbox': {'x': 1, 'y': 2, 'width': 30, 'height': 40},
          'status': 'SEGMENT_RECAPTURE',
          'reason_codes': ['CLASSIFIER_QUALITY_CLASS'],
          'prediction': null,
          'top3': <dynamic>[],
          'confidence': 0.99,
        },
      ],
      'processing_time_ms': 10.0,
      'worker_version': '1.0.0',
      'detector_version': '1.0.0',
      'classifier_version': '1.0.0',
    });

    expect(response.status, ScanStatus.unknown);
    expect(response.items.single.status, ItemStatus.segmentRecapture);
    expect(response.modelVersions.worker, '1.0.0');
  });

  test(
    'parses contained duplicate UNKNOWN with Top-3 and all model versions',
    () {
      final response = ScanResponse.fromJson({
        'request_id': 'request-duplicate',
        'status': 'SEGMENTATION',
        'reason_codes': ['SEGMENT_DUPLICATE_REVIEW_REQUIRED'],
        'segmentations': [
          {
            'segmentation_id': 'segmentation_004',
            'bbox': {'x': 10, 'y': 20, 'width': 300, 'height': 400},
            'status': 'UNKNOWN',
            'reason_codes': ['DETECTOR_CONTAINED_DUPLICATE'],
            'prediction': null,
            'top3': [
              {
                'class_id': 'bread_15',
                'class_name': 'Sandwich',
                'confidence': 1.0,
              },
              {'class_id': 'bread_04', 'class_name': 'Scon', 'confidence': 0.0},
              {
                'class_id': 'bread_14',
                'class_name': 'Red Bean Bread',
                'confidence': 0.0,
              },
            ],
            'confidence': 1.0,
          },
        ],
        'processing_time_ms': 66.6,
        'worker_version': '1.0.0',
        'detector_version': '1.0.0',
        'classifier_version': '1.0.0',
      });

      expect(response.status, ScanStatus.unknown);
      expect(response.reasonCodes, ['SEGMENT_DUPLICATE_REVIEW_REQUIRED']);
      expect(response.items.single.reasonCodes, [
        'DETECTOR_CONTAINED_DUPLICATE',
      ]);
      expect(response.items.single.top3, hasLength(3));
      expect(response.items.single.top3.first.classId, 'bread_15');
      expect(response.modelVersions.worker, '1.0.0');
      expect(response.modelVersions.detector, '1.0.0');
      expect(response.modelVersions.classifier, '1.0.0');
    },
  );
}
