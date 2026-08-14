import '../../../shared/models/scan_models.dart';

class SegmentReviewPresentation {
  const SegmentReviewPresentation({
    required this.inspectorTitle,
    required this.rowTitle,
    required this.detail,
    required this.shortLabel,
  });

  final String inspectorTitle;
  final String rowTitle;
  final String detail;
  final String shortLabel;
}

SegmentReviewPresentation presentSegmentReview(ScanItem item) {
  if (item.reasonCodes.contains('DETECTOR_CONTAINED_DUPLICATE')) {
    return const SegmentReviewPresentation(
      inspectorTitle: '중복으로 검출된 상품인지 확인해 주세요',
      rowTitle: '중복 검출 가능성이 있어요',
      detail: '같은 상품을 가리키는 박스가 겹쳐 있어요. 미리보기의 박스를 확인하고 맞는 상품을 선택해 주세요.',
      shortLabel: '중복 박스 확인',
    );
  }
  if (item.status == ItemStatus.segmentRecapture) {
    if (item.reasonCodes.contains('DETECTOR_BORDER_CLIPPED')) {
      return const SegmentReviewPresentation(
        inspectorTitle: '잘린 상품을 다시 확인해 주세요',
        rowTitle: '잘린 상품을 다시 확인해 주세요',
        detail: '상품 전체가 보이도록 다시 촬영하거나, 상품이 맞다면 검색해서 직접 확정해 주세요.',
        shortLabel: '다시 촬영 필요',
      );
    }
    return const SegmentReviewPresentation(
      inspectorTitle: '상품 상태를 다시 확인해 주세요',
      rowTitle: '다시 촬영이 필요한 상품이에요',
      detail: '상품이 선명하게 보이도록 다시 촬영하거나, 상품이 맞다면 검색해서 직접 확정해 주세요.',
      shortLabel: '다시 촬영 필요',
    );
  }
  return const SegmentReviewPresentation(
    inspectorTitle: '상품을 확인해 주세요',
    rowTitle: '상품 확인이 필요해요',
    detail: '선택하면 다음 확인 항목으로 이동해요.',
    shortLabel: '확인 필요',
  );
}
