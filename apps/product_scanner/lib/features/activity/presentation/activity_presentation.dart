import '../../../core/design_system/copy.dart';
import '../../../shared/logging/scan_log_repository.dart';
import '../../../shared/models/scan_models.dart';
import '../../../shared/presentation/recapture_presentation.dart';

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
      (item.classId?.toLowerCase().contains(normalized) ?? false) ||
      item.reasonCodes.any(
        (reason) => reason.toLowerCase().contains(normalized),
      ) ||
      activityItemReasonLabel(item).toLowerCase().contains(normalized);
}

bool activityLogMatchesQuery(ScanLogSummary log, String query) {
  final normalized = query.trim().toLowerCase();
  if (normalized.isEmpty) return true;
  if (log.scanId.toLowerCase().contains(normalized) ||
      log.reasonCodes.any(
        (reason) => reason.toLowerCase().contains(normalized),
      ) ||
      (log.operatorReview?.issueCodes.any(
            (code) =>
                operatorIssueCodeValue(
                  code,
                ).toLowerCase().contains(normalized) ||
                activityOperatorIssueLabel(
                  code,
                ).toLowerCase().contains(normalized),
          ) ??
          false) ||
      activityLogResultLabel(log).toLowerCase().contains(normalized) ||
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
  if (log.isLegacy) return '기존 기록';
  return switch (log.operatorReview!.verdict) {
    OperatorReviewVerdict.accepted => '그대로 저장',
    OperatorReviewVerdict.corrected =>
      '${log.items.where((item) => item.userModified).length}건 수정',
    OperatorReviewVerdict.recaptureAgreed => '재촬영 저장',
    OperatorReviewVerdict.recaptureUnnecessary => '재촬영 해제',
    OperatorReviewVerdict.recaptureRequired => '재촬영으로 변경',
  };
}

String activityOperatorIssueLabel(OperatorIssueCode code) => switch (code) {
  OperatorIssueCode.bboxIncorrect => '박스 잘못 검출',
  OperatorIssueCode.missedObject => '객체 미검출',
  OperatorIssueCode.falsePositiveObject => '없는 객체 검출',
  OperatorIssueCode.wrongTop1 => '상품 오예측',
  OperatorIssueCode.candidateMissing => '후보에 없음',
  OperatorIssueCode.unnecessarySegmentRecapture => '객체 재촬영 불필요',
  OperatorIssueCode.unnecessaryRecapture => '전체 재촬영 불필요',
  OperatorIssueCode.missedRecapture => '놓친 재촬영',
};

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

String activityItemReasonLabel(ScanLogItemSummary item) {
  if (item.reasonCodes.contains('DETECTOR_CONTAINED_DUPLICATE')) {
    return '중복 검출 검토';
  }
  if (item.reasonCodes.contains('DETECTOR_BORDER_CLIPPED')) {
    return '잘린 상품 검토';
  }
  if (item.reasonCodes.contains('CLASSIFIER_QUALITY_CLASS')) {
    return '상품 상태 검토';
  }
  if (item.reasonCodes.contains('BELOW_APPROVAL_THRESHOLD')) {
    return '낮은 신뢰도 검토';
  }
  return '';
}
