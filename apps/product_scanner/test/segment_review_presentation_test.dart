import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/features/scanner/presentation/segment_review_presentation.dart';
import 'package:product_scanner/models/scan_models.dart';

void main() {
  test('contained duplicate is explained as a review, not recapture', () {
    final presentation = presentSegmentReview(_duplicateItem);

    expect(presentation.inspectorTitle, contains('중복'));
    expect(presentation.detail, contains('박스'));
    expect(presentation.shortLabel, '중복 박스 확인');
  });

  test('segment recapture gives a direct capture-or-confirm action', () {
    final presentation = presentSegmentReview(_segmentRecaptureItem);

    expect(presentation.inspectorTitle, contains('잘린 상품'));
    expect(presentation.detail, contains('다시 촬영'));
    expect(presentation.shortLabel, '다시 촬영 필요');
  });
}

const _duplicateItem = ScanItem(
  itemId: 'segmentation_001',
  bbox: BoundingBox(x: 1, y: 2, width: 30, height: 40),
  status: ItemStatus.unknown,
  reasonCodes: ['DETECTOR_CONTAINED_DUPLICATE'],
  prediction: null,
  top3: [
    Candidate(
      classId: 'bread_15',
      className: 'Sandwich',
      displayName: '샌드위치',
      confidence: 1,
    ),
  ],
  confidence: 1,
);

const _segmentRecaptureItem = ScanItem(
  itemId: 'segmentation_002',
  bbox: BoundingBox(x: 1, y: 2, width: 30, height: 40),
  status: ItemStatus.segmentRecapture,
  reasonCodes: ['DETECTOR_BORDER_CLIPPED'],
  prediction: null,
  top3: [],
  confidence: .5,
);
