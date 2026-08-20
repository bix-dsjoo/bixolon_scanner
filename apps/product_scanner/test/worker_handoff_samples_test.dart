import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/shared/models/scan_models.dart';

void main() {
  String sample(String name) => File(
    '../../docs/contracts/examples/0.0.2/$name.json',
  ).readAsStringSync();

  test('0.0.2 APPROVED sample parses', () {
    final response = ScanResponse.fromBody(sample('approved'));

    expect(response.status, ScanStatus.approved);
    expect(response.items.single.status, ItemStatus.approved);
    expect(response.modelVersions.worker, '0.0.2');
  });

  test('0.0.2 UNKNOWN sample parses with ordered Top-3', () {
    final response = ScanResponse.fromBody(sample('unknown'));

    expect(response.status, ScanStatus.unknown);
    expect(response.items.single.status, ItemStatus.unknown);
    expect(
      response.items.single.top3.map((candidate) => candidate.classId),
      ['bread_02', 'bread_03', 'bread_19'],
    );
  });

  test('0.0.2 SEGMENT_RECAPTURE sample parses', () {
    final response = ScanResponse.fromBody(sample('segment-recapture'));

    expect(response.status, ScanStatus.unknown);
    expect(response.items.single.status, ItemStatus.segmentRecapture);
  });

  test('0.0.2 IMAGE_RECAPTURE sample parses separately from ERROR', () {
    final recapture = ScanResponse.fromBody(sample('image-recapture'));
    final error = ScanResponse.fromBody(sample('error'));

    expect(recapture.status, ScanStatus.recapture);
    expect(recapture.items, isEmpty);
    expect(error.status, ScanStatus.error);
    expect(error.items, isEmpty);
  });
}
