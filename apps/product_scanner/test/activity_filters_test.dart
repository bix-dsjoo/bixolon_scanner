import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/activity/activity_filters.dart';
import 'package:product_scanner/models/scan_models.dart';
import 'package:product_scanner/services/scan_log_repository.dart';

void main() {
  final now = DateTime(2026, 8, 10, 12);

  test('검색 AND 입력원 AND 기간 순으로 필터를 결합한다', () {
    final logs = [
      _log(
        'camera-muffin',
        DateTime(2026, 8, 10, 9),
        InputMode.camera,
        'Muffin',
      ),
      _log(
        'image-muffin',
        DateTime(2026, 8, 10, 10),
        InputMode.image,
        'Muffin',
      ),
      _log('image-egg', DateTime(2026, 8, 10, 11), InputMode.image, 'Egg Tart'),
    ];

    final result = filterActivityLogs(
      logs: logs,
      query: 'mUfFiN',
      inputFilter: ActivityInputFilter.image,
      dateFilter: ActivityDateFilter.today,
      sortOrder: ActivitySortOrder.newest,
      now: now,
    );

    expect(result.map((log) => log.scanId), ['image-muffin']);
  });

  test('7일과 30일은 오늘을 포함하고 각 시작일 00시를 경계로 한다', () {
    final logs = [
      _log('today-start', DateTime(2026, 8, 10), InputMode.camera, 'A'),
      _log('seven-start', DateTime(2026, 8, 4), InputMode.camera, 'B'),
      _log('before-seven', DateTime(2026, 8, 3, 23, 59), InputMode.camera, 'C'),
      _log('thirty-start', DateTime(2026, 7, 12), InputMode.camera, 'D'),
      _log(
        'before-thirty',
        DateTime(2026, 7, 11, 23, 59),
        InputMode.camera,
        'E',
      ),
      _log('tomorrow', DateTime(2026, 8, 11), InputMode.camera, 'F'),
    ];

    final sevenDays = filterActivityLogs(
      logs: logs,
      query: '',
      inputFilter: ActivityInputFilter.all,
      dateFilter: ActivityDateFilter.sevenDays,
      sortOrder: ActivitySortOrder.oldest,
      now: now,
    );
    expect(sevenDays.map((log) => log.scanId), ['seven-start', 'today-start']);

    final thirtyDays = filterActivityLogs(
      logs: logs,
      query: '',
      inputFilter: ActivityInputFilter.all,
      dateFilter: ActivityDateFilter.thirtyDays,
      sortOrder: ActivitySortOrder.oldest,
      now: now,
    );
    expect(thirtyDays.map((log) => log.scanId), [
      'thirty-start',
      'before-seven',
      'seven-start',
      'today-start',
    ]);
  });

  test('필터 적용 후 확정 시각으로 최신순과 오래된순을 정렬한다', () {
    final logs = [
      _log('middle', DateTime(2026, 8, 10, 10), InputMode.camera, 'A'),
      _log('oldest', DateTime(2026, 8, 10, 9), InputMode.camera, 'B'),
      _log('newest', DateTime(2026, 8, 10, 11), InputMode.camera, 'C'),
    ];

    List<String> ids(ActivitySortOrder order) => filterActivityLogs(
      logs: logs,
      query: '',
      inputFilter: ActivityInputFilter.all,
      dateFilter: ActivityDateFilter.all,
      sortOrder: order,
      now: now,
    ).map((log) => log.scanId).toList();

    expect(ids(ActivitySortOrder.newest), ['newest', 'middle', 'oldest']);
    expect(ids(ActivitySortOrder.oldest), ['oldest', 'middle', 'newest']);
  });

  test('한국어 표시 로그도 원본 영문명과 class id로 검색할 수 있다', () {
    final log = _log(
      'localized',
      DateTime(2026, 8, 10, 10),
      InputMode.camera,
      '머핀',
      classId: 'bread_13',
      className: 'Muffin',
    );

    List<String> search(String query) => filterActivityLogs(
      logs: [log],
      query: query,
      inputFilter: ActivityInputFilter.all,
      dateFilter: ActivityDateFilter.all,
      sortOrder: ActivitySortOrder.newest,
      now: now,
    ).map((item) => item.scanId).toList();

    expect(search('머핀'), ['localized']);
    expect(search('muffin'), ['localized']);
    expect(search('bread_13'), ['localized']);
  });
}

ScanLogSummary _log(
  String id,
  DateTime confirmedAt,
  InputMode mode,
  String product, {
  String? classId,
  String? className,
}) {
  return ScanLogSummary(
    scanId: id,
    analyzedAt: confirmedAt.subtract(const Duration(seconds: 1)),
    confirmedAt: confirmedAt,
    inputMode: mode,
    processingTimeMs: 50,
    modelVersions: const ModelVersions(detector: '1.0.0', classifier: '1.0.0'),
    items: [
      ScanLogItemSummary(
        itemId: 'item_001',
        productName: product,
        confidence: .9,
        userModified: false,
        confirmationMethod: 'AUTO_APPROVED',
        classId: classId,
        className: className,
      ),
    ],
  );
}
