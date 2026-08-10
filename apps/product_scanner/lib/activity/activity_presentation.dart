import '../services/scan_log_repository.dart';
import '../theme/app_copy.dart';

String activityProductLabel(ScanLogItemSummary item) {
  final value = item.productName.trim();
  if (value.isEmpty || value.toUpperCase() == 'UNKNOWN') {
    return AppActivityCopy.productUnavailable;
  }
  return value;
}

bool activityItemMatchesQuery(ScanLogItemSummary item, String query) {
  final normalized = query.trim().toLowerCase();
  if (normalized.isEmpty) return true;
  return activityProductLabel(item).toLowerCase().contains(normalized) ||
      (item.className?.toLowerCase().contains(normalized) ?? false) ||
      (item.classId?.toLowerCase().contains(normalized) ?? false);
}

String summarizeActivityProducts(
  List<ScanLogItemSummary> items, {
  required String query,
}) {
  if (items.isEmpty) return '상품 없음';
  if (items.length == 1) return activityProductLabel(items.first);
  final normalized = query.trim().toLowerCase();
  final primary = normalized.isEmpty
      ? items.first
      : items.firstWhere(
          (item) => activityItemMatchesQuery(item, normalized),
          orElse: () => items.first,
        );
  return '${activityProductLabel(primary)} 외 ${items.length - 1}개';
}

String activityConfirmationMethodLabel(String value) => switch (value) {
  'AUTO_APPROVED' => '자동 승인',
  'TOP3_SELECTED' => 'Top-3 선택',
  'SEARCH_SELECTED' => '검색 선택',
  'USER_CORRECTED' => '사용자 수정',
  _ => AppActivityCopy.confirmationMethodUnavailable,
};
