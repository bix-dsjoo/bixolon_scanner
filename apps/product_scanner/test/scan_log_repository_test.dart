import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:product_scanner/models/scan_models.dart';
import 'package:product_scanner/services/scan_log_repository.dart';

void main() {
  test('원본 이미지와 최초/최종 판정을 분리한 JSON을 저장한다', () async {
    final support = await Directory.systemTemp.createTemp(
      'product-scanner-log-',
    );
    addTearDown(() async {
      if (await support.exists()) await support.delete(recursive: true);
    });
    final repository = FileScanLogRepository(
      applicationSupportDirectory: () async => support,
    );
    final detection = ReviewDetection.fromScanItem(_unknownItem)
      ..state = DetectionState.confirmed
      ..finalProduct = const Product(
        classId: 'bread_13',
        className: 'Muffin',
        displayName: 'Muffin',
      )
      ..confirmationMethod = ConfirmationMethod.top3Selected;

    await repository.save(
      ScanLogRecord(
        scanId: 'request_log_1234',
        analyzedAt: DateTime.utc(2026, 8, 10, 1),
        confirmedAt: DateTime.utc(2026, 8, 10, 1, 1),
        inputMode: InputMode.image,
        imageBytes: Uint8List.fromList([1, 2, 3]),
        imageFileName: 'scan.png',
        processingTimeMs: 72.1,
        modelVersions: const ModelVersions(
          detector: '0.1.1',
          classifier: '0.1.1',
        ),
        detections: [detection],
      ),
    );

    final root = Directory(p.join(support.path, 'ProductScanner', 'scan_logs'));
    final image = File(p.join(root.path, 'request_log_1234.png'));
    final jsonFile = File(p.join(root.path, 'request_log_1234.json'));
    expect(await image.readAsBytes(), [1, 2, 3]);
    final payload =
        jsonDecode(await jsonFile.readAsString()) as Map<String, dynamic>;
    final savedDetection =
        (payload['detections'] as List).single as Map<String, dynamic>;
    expect(payload['input_mode'], 'IMAGE');
    expect(savedDetection['initial_ai_status'], 'TOP3_CANDIDATES');
    expect(savedDetection['final_product']['class_id'], 'bread_13');
    expect(savedDetection['confirmation_method'], 'TOP3_SELECTED');

    final logs = await repository.list();
    expect(logs, hasLength(1));
    expect(logs.single.scanId, 'request_log_1234');
    expect(logs.single.items.single.productName, 'Muffin');
  });
}

const _unknownItem = ScanItem(
  itemId: 'item_001',
  bbox: BoundingBox(x: 1, y: 2, width: 3, height: 4),
  status: ItemStatus.unknown,
  reasonCodes: ['BELOW_APPROVAL_THRESHOLD'],
  prediction: null,
  top3: [
    Candidate(
      classId: 'bread_13',
      className: 'Muffin',
      displayName: 'Muffin',
      confidence: .7,
    ),
  ],
  confidence: .7,
);
