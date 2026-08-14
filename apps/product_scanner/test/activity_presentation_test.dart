import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/activity/activity_presentation.dart';
import 'package:product_scanner/services/scan_log_repository.dart';

void main() {
  test('구버전 빈 상품과 확정 방식은 내부 sentinel 대신 한국어로 안내한다', () {
    const item = ScanLogItemSummary(
      itemId: 'item_001',
      productName: 'Unknown',
      confidence: .9,
      userModified: false,
      confirmationMethod: 'UNKNOWN',
    );

    expect(activityProductLabel(item), '상품 정보 없음');
    expect(activityItemMatchesQuery(item, '상품 정보'), isTrue);
    expect(activityItemMatchesQuery(item, 'unknown'), isFalse);
    expect(activityConfirmationMethodLabel('UNKNOWN'), '확정 방식 확인 불가');
    expect(activityConfirmationMethodLabel('FUTURE_METHOD'), '확정 방식 확인 불가');
  });

  test('Activity 요약은 한국어·영문·class id 검색과 일치한 상품을 먼저 표시한다', () {
    const items = [
      ScanLogItemSummary(
        itemId: 'item_001',
        productName: '머핀',
        confidence: .9,
        userModified: false,
        confirmationMethod: 'AUTO_APPROVED',
        classId: 'bread_13',
        className: 'Muffin',
      ),
      ScanLogItemSummary(
        itemId: 'item_002',
        productName: '베이글',
        confidence: .8,
        userModified: false,
        confirmationMethod: 'AUTO_APPROVED',
        classId: 'bread_11',
        className: 'Bagel',
      ),
    ];

    expect(summarizeActivityProducts(items, query: ''), '머핀 외 1개');
    expect(summarizeActivityProducts(items, query: '베이글'), '베이글 외 1개');
    expect(summarizeActivityProducts(items, query: 'bagel'), '베이글 외 1개');
    expect(summarizeActivityProducts(items, query: 'bread_11'), '베이글 외 1개');
  });

  test('중복 검토 reason은 한국어와 raw code 모두 검색할 수 있다', () {
    const item = ScanLogItemSummary(
      itemId: 'segmentation_004',
      productName: '소보로빵',
      confidence: 1,
      userModified: true,
      confirmationMethod: 'TOP3_SELECTED',
      reasonCodes: ['DETECTOR_CONTAINED_DUPLICATE'],
    );

    expect(activityItemReasonLabel(item), '중복 검출 검토');
    expect(activityItemMatchesQuery(item, '중복 검출'), isTrue);
    expect(
      activityItemMatchesQuery(item, 'detector_contained_duplicate'),
      isTrue,
    );
  });
}
