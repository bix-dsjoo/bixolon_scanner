import '../../../core/design_system/copy.dart';
import '../../scanner/presentation/recapture_presentation.dart';
import '../data/scan_log_repository.dart';

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

bool activityLogMatchesQuery(ScanLogSummary log, String query) {
  final normalized = query.trim().toLowerCase();
  if (normalized.isEmpty) return true;
  if (log.scanId.toLowerCase().contains(normalized) ||
      log.reasonCodes.any(
        (reason) => reason.toLowerCase().contains(normalized),
      ) ||
      log.items.any((item) => activityItemMatchesQuery(item, normalized))) {
    return true;
  }
  if (!log.isRecapture) return false;
  final presentation = presentRecaptureReasons(
    reasonCodes: log.reasonCodes,
    inputMode: log.inputMode,
  );
  return presentation.title.toLowerCase().contains(normalized) ||
      presentation.detail.toLowerCase().contains(normalized);
}

String activityLogContentLabel(ScanLogSummary log, {required String query}) {
  if (!log.isRecapture) {
    return summarizeActivityProducts(log.items, query: query);
  }
  return presentRecaptureReasons(
    reasonCodes: log.reasonCodes,
    inputMode: log.inputMode,
  ).title;
}

String activityLogResultLabel(ScanLogSummary log) {
  if (log.isRecapture) return '재촬영';
  final modifiedCount = log.items.where((item) => item.userModified).length;
  return modifiedCount > 0 ? '$modifiedCount개 수정' : '자동 확정';
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
