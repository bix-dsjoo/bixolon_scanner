import '../../../shared/models/scan_models.dart';
import '../../../shared/logging/scan_log_repository.dart';
import '../presentation/activity_presentation.dart';

enum ActivityInputFilter { all, camera, image }

enum ActivityDateFilter { all, today, sevenDays, thirtyDays }

enum ActivitySortOrder { newest, oldest }

List<ScanLogSummary> filterActivityLogs({
  required List<ScanLogSummary> logs,
  required String query,
  required ActivityInputFilter inputFilter,
  required ActivityDateFilter dateFilter,
  required ActivitySortOrder sortOrder,
  DateTime? now,
}) {
  final normalized = query.trim().toLowerCase();
  final localNow = (now ?? DateTime.now()).toLocal();
  final today = DateTime(localNow.year, localNow.month, localNow.day);
  final start = switch (dateFilter) {
    ActivityDateFilter.all => null,
    ActivityDateFilter.today => today,
    ActivityDateFilter.sevenDays => today.subtract(const Duration(days: 6)),
    ActivityDateFilter.thirtyDays => today.subtract(const Duration(days: 29)),
  };
  final endExclusive = dateFilter == ActivityDateFilter.all
      ? null
      : today.add(const Duration(days: 1));

  final filtered = logs
      .where((log) {
        final queryMatches =
            normalized.isEmpty || activityLogMatchesQuery(log, normalized);
        final inputMatches = switch (inputFilter) {
          ActivityInputFilter.all => true,
          ActivityInputFilter.camera => log.inputMode == InputMode.camera,
          ActivityInputFilter.image => log.inputMode == InputMode.image,
        };
        final recordedAt = log.recordedAt.toLocal();
        final dateMatches =
            start == null ||
            (!recordedAt.isBefore(start) && recordedAt.isBefore(endExclusive!));
        return queryMatches && inputMatches && dateMatches;
      })
      .toList(growable: false);

  filtered.sort(
    sortOrder == ActivitySortOrder.newest
        ? (a, b) => b.recordedAt.compareTo(a.recordedAt)
        : (a, b) => a.recordedAt.compareTo(b.recordedAt),
  );
  return filtered;
}
