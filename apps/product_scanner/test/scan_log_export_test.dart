import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:product_scanner/models/scan_models.dart';
import 'package:product_scanner/services/scan_log_repository.dart';

void main() {
  test('필터/전체 검수 ZIP은 고정 구조·checksum·legacy 범위를 지킨다', () async {
    final support = await Directory.systemTemp.createTemp('review-export-');
    addTearDown(() async {
      if (await support.exists()) await support.delete(recursive: true);
    });
    final repository = FileScanLogRepository(
      applicationSupportDirectory: () async => support,
      captureSessionId: 'session-test',
    );
    final reviewedAt = DateTime.utc(2026, 8, 24, 3);
    final detection = ReviewDetection.fromScanItem(_approvedItem);
    await repository.save(
      ScanLogRecord(
        scanId: 'scan_export_1',
        analyzedAt: reviewedAt,
        confirmedAt: reviewedAt,
        inputMode: InputMode.image,
        imageBytes: Uint8List.fromList([0xff, 0xd8, 0xff, 0xd9]),
        imageFileName: 'input.jpg',
        processingTimeMs: 600,
        modelVersions: const ModelVersions(
          worker: '0.1.1',
          detector: '0.1.1',
          classifier: '0.1.1',
        ),
        detections: [detection],
        operatorReview: OperatorReview(
          verdict: OperatorReviewVerdict.accepted,
          reviewedAt: reviewedAt,
          inferredIssueCodes: const {},
          issueCodes: const {},
          objects: [OperatorReviewObject.fromDetection(detection)],
        ),
      ),
    );
    final legacy = await repository.legacyFeedbackDirectory();
    await legacy.create(recursive: true);
    await File(p.join(legacy.path, 'legacy.json')).writeAsString('{}');

    final logs = await repository.list();
    final filteredZip = p.join(support.path, 'filtered.zip');
    await repository.exportReviewArchive(
      targetPath: filteredZip,
      records: logs,
    );
    final filtered = ZipDecoder().decodeBytes(
      await File(filteredZip).readAsBytes(),
      verify: true,
    );
    final filteredNames = filtered.files.map((file) => file.name).toSet();
    expect(
      filteredNames,
      containsAll(<String>{
        'manifest.json',
        'records/scan_export_1.json',
        'images/scan_export_1.jpg',
        'checksums.sha256',
      }),
    );
    expect(
      filteredNames.any((name) => name.startsWith('legacy_feedback/')),
      isFalse,
    );
    final manifest =
        jsonDecode(
              utf8.decode(
                filtered.files
                        .singleWhere((file) => file.name == 'manifest.json')
                        .content
                    as List<int>,
              ),
            )
            as Map<String, dynamic>;
    expect(manifest['scope'], 'FILTERED');
    expect(manifest['record_count'], 1);
    expect(jsonEncode(manifest), isNot(contains(support.path)));
    final checksums = utf8.decode(
      filtered.files
              .singleWhere((file) => file.name == 'checksums.sha256')
              .content
          as List<int>,
    );
    expect(checksums, contains('manifest.json'));
    expect(checksums, contains('records/scan_export_1.json'));

    final allZip = p.join(support.path, 'all.zip');
    await repository.exportReviewArchive(targetPath: allZip);
    final all = ZipDecoder().decodeBytes(
      await File(allZip).readAsBytes(),
      verify: true,
    );
    expect(
      all.files.map((file) => file.name),
      contains('legacy_feedback/legacy.json'),
    );
  });
}

const _approvedItem = ScanItem(
  itemId: 'model_1',
  bbox: BoundingBox(x: 10, y: 20, width: 100, height: 80),
  status: ItemStatus.approved,
  reasonCodes: [],
  prediction: Product(
    classId: 'bread_01',
    className: 'Bagel',
    displayName: '베이글',
  ),
  top3: [],
  confidence: .95,
);
