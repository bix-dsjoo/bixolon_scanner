import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:product_scanner/models/scan_models.dart';
import 'package:product_scanner/services/scan_log_repository.dart';

void main() {
  test('Activity 최종 상태는 상품 확정과 UNKNOWN, SEGMENT_RECAPTURE를 구분한다', () {
    const approved = ScanLogItemSummary(
      itemId: 'approved',
      productName: '머핀',
      confidence: .9,
      userModified: true,
      confirmationMethod: 'SEARCH_SELECTED',
      classId: 'bread_01',
      initialStatus: ItemStatus.unknown,
    );
    const unknown = ScanLogItemSummary(
      itemId: 'unknown',
      productName: 'Unknown',
      confidence: .5,
      userModified: false,
      confirmationMethod: 'UNKNOWN',
      initialStatus: ItemStatus.unknown,
    );
    const segmentRecapture = ScanLogItemSummary(
      itemId: 'segment-recapture',
      productName: 'Unknown',
      confidence: .2,
      userModified: false,
      confirmationMethod: 'UNKNOWN',
      initialStatus: ItemStatus.segmentRecapture,
    );

    expect(approved.resultStatus, ItemStatus.approved);
    expect(unknown.resultStatus, ItemStatus.unknown);
    expect(segmentRecapture.resultStatus, ItemStatus.segmentRecapture);
  });

  test('원본 이미지와 최초/최종 판정을 분리한 JSON을 저장한다', () async {
    final support = await Directory.systemTemp.createTemp(
      'product-scanner-log-',
    );
    addTearDown(() async {
      if (await support.exists()) await support.delete(recursive: true);
    });
    final repository = FileScanLogRepository(
      applicationSupportDirectory: () async => support,
    );
    final detection = ReviewDetection.fromScanItem(_unknownItem)
      ..state = DetectionState.confirmed
      ..finalProduct = const Product(
        classId: 'bread_13',
        className: 'Muffin',
        displayName: 'Muffin',
      )
      ..confirmationMethod = ConfirmationMethod.top3Selected;

    await repository.save(
      ScanLogRecord(
        scanId: 'request_log_1234',
        analyzedAt: DateTime.utc(2026, 8, 10, 1),
        confirmedAt: DateTime.utc(2026, 8, 10, 1, 1),
        inputMode: InputMode.image,
        imageBytes: Uint8List.fromList([1, 2, 3]),
        imageFileName: 'scan.png',
        processingTimeMs: 72.1,
        workerStatus: ScanStatus.unknown,
        reasonCodes: const ['ITEM_BELOW_APPROVAL_THRESHOLD'],
        modelVersions: const ModelVersions(
          detector: '0.1.1',
          classifier: '0.1.1',
        ),
        operatorReview: OperatorReview(
          verdict: OperatorReviewVerdict.corrected,
          reviewedAt: DateTime.utc(2026, 8, 10, 1, 1),
          inferredIssueCodes: const {OperatorIssueCode.candidateMissing},
          issueCodes: const {OperatorIssueCode.candidateMissing},
          objects: [OperatorReviewObject.fromDetection(detection)],
        ),
        performance: _performance,
        detections: [detection],
      ),
    );

    final root = Directory(p.join(support.path, 'ProductScanner', 'scan_logs'));
    final image = File(p.join(root.path, 'request_log_1234.png'));
    final jsonFile = File(p.join(root.path, 'request_log_1234.json'));
    expect(await image.readAsBytes(), [1, 2, 3]);
    final payload =
        jsonDecode(await jsonFile.readAsString()) as Map<String, dynamic>;
    final savedDetection =
        (payload['detections'] as List).single as Map<String, dynamic>;
    expect(payload['input_mode'], 'IMAGE');
    expect(payload['log_schema_version'], 5);
    expect(payload['worker_status'], 'UNKNOWN');
    expect(payload['reason_codes'], ['ITEM_BELOW_APPROVAL_THRESHOLD']);
    expect(payload['recorded_at'], payload['confirmed_at']);
    expect(savedDetection['initial_ai_status'], 'TOP3_CANDIDATES');
    expect(savedDetection['reason_codes'], ['BELOW_APPROVAL_THRESHOLD']);
    expect(savedDetection['final_product']['class_id'], 'bread_13');
    expect(savedDetection['confirmation_method'], 'TOP3_SELECTED');
    expect(payload['performance']['end_to_end_ms'], 1498.7);
    expect(payload['performance']['provider'], 'openvino');
    expect(payload['performance']['worker_timings_ms']['detector_ms'], 210.4);
    expect(payload['operator_review']['verdict'], 'CORRECTED');
    expect(payload['operator_review']['issue_codes'], [
      'OPERATOR_CANDIDATE_MISSING',
    ]);

    final logs = await repository.list();
    expect(logs, hasLength(1));
    expect(logs.single.scanId, 'request_log_1234');
    expect(logs.single.items.single.productName, 'Muffin');
    expect(logs.single.items.single.classId, 'bread_13');
    expect(logs.single.items.single.className, 'Muffin');
    expect(logs.single.workerStatus, ScanStatus.unknown);
    expect(logs.single.reasonCodes, ['ITEM_BELOW_APPROVAL_THRESHOLD']);
    expect(logs.single.items.single.reasonCodes, ['BELOW_APPROVAL_THRESHOLD']);
    expect(logs.single.originalImagePath, image.path);
    expect(logs.single.recordedAt, logs.single.confirmedAt);
    expect(logs.single.performance?.endToEndMs, 1498.7);
    expect(logs.single.performance?.apiTotalMs, closeTo(711.1, 0.001));
    expect(logs.single.performance?.worker?.classifierMs, 350.2);
    expect(logs.single.logSchemaVersion, 5);
    expect(
      logs.single.operatorReview?.verdict,
      OperatorReviewVerdict.corrected,
    );
  });

  test('RECAPTURE 이미지와 reason code를 확정 시각 없이 저장한다', () async {
    final support = await Directory.systemTemp.createTemp(
      'product-scanner-recapture-log-',
    );
    addTearDown(() async {
      if (await support.exists()) await support.delete(recursive: true);
    });
    final repository = FileScanLogRepository(
      applicationSupportDirectory: () async => support,
    );
    final recordedAt = DateTime.utc(2026, 8, 11, 2, 3);

    await repository.save(
      ScanLogRecord(
        scanId: 'request_recapture_1234',
        analyzedAt: recordedAt.subtract(const Duration(seconds: 1)),
        confirmedAt: null,
        recordedAt: recordedAt,
        inputMode: InputMode.camera,
        imageBytes: Uint8List.fromList([4, 5, 6]),
        imageFileName: 'capture.jpg',
        processingTimeMs: 41.2,
        modelVersions: const ModelVersions(detector: '0.1.1', classifier: null),
        detections: const [],
        workerStatus: ScanStatus.recapture,
        reasonCodes: const ['DETECTOR_UNCERTAIN_OBJECT'],
      ),
    );

    final root = Directory(p.join(support.path, 'ProductScanner', 'scan_logs'));
    final payload =
        jsonDecode(
              await File(
                p.join(root.path, 'request_recapture_1234.json'),
              ).readAsString(),
            )
            as Map<String, dynamic>;
    expect(payload['worker_status'], 'RECAPTURE');
    expect(payload['confirmed_at'], isNull);
    expect(payload['recorded_at'], recordedAt.toIso8601String());
    expect(payload['reason_codes'], ['DETECTOR_UNCERTAIN_OBJECT']);
    expect(payload['detections'], isEmpty);
    expect(payload['model_versions']['classifier'], isNull);

    final log = (await repository.list()).single;
    expect(log.isRecapture, isTrue);
    expect(log.confirmedAt, isNull);
    expect(log.recordedAt, recordedAt);
    expect(log.items, isEmpty);
    expect(log.reasonCodes, ['DETECTOR_UNCERTAIN_OBJECT']);
    expect(
      log.originalImagePath,
      p.join(root.path, 'request_recapture_1234.jpg'),
    );
  });

  test('박스 미검출 추가는 원본 모델 결과와 operator_review에 함께 저장한다', () async {
    final support = await Directory.systemTemp.createTemp(
      'product-scanner-missed-object-log-',
    );
    addTearDown(() async {
      if (await support.exists()) await support.delete(recursive: true);
    });
    final repository = FileScanLogRepository(
      applicationSupportDirectory: () async => support,
      captureSessionId: 'capture-session-03',
    );
    final detection = ReviewDetection.fromScanItem(_unknownItem)
      ..finalProduct = _unknownItem.top3.first
      ..confirmationMethod = ConfirmationMethod.top3Selected;
    final added =
        ReviewDetection.operatorAdded(
            objectId: 'operator_1',
            bbox: const BoundingBox(x: 10, y: 20, width: 30, height: 40),
          )
          ..finalProduct = const Product(
            classId: 'bread_01',
            className: 'Bagel',
            displayName: '베이글',
          )
          ..confirmationMethod = ConfirmationMethod.searchSelected;

    await repository.save(
      ScanLogRecord(
        scanId: 'request_missed_1234',
        analyzedAt: DateTime.utc(2026, 8, 11, 3),
        confirmedAt: null,
        inputMode: InputMode.camera,
        imageBytes: Uint8List.fromList([7, 8, 9]),
        imageFileName: 'capture.jpg',
        processingTimeMs: 55.4,
        modelVersions: const ModelVersions(
          detector: '0.1.1',
          classifier: '0.1.1',
        ),
        detections: [detection, added],
        workerStatus: ScanStatus.unknown,
        reasonCodes: const ['ITEM_BELOW_APPROVAL_THRESHOLD'],
        operatorReview: OperatorReview(
          verdict: OperatorReviewVerdict.corrected,
          reviewedAt: DateTime.utc(2026, 8, 11, 3, 1),
          inferredIssueCodes: const {OperatorIssueCode.missedObject},
          issueCodes: const {OperatorIssueCode.missedObject},
          objects: [
            OperatorReviewObject.fromDetection(detection),
            OperatorReviewObject.fromDetection(added),
          ],
        ),
      ),
    );

    final root = Directory(p.join(support.path, 'ProductScanner', 'scan_logs'));
    final payload =
        jsonDecode(
              await File(
                p.join(root.path, 'request_missed_1234.json'),
              ).readAsString(),
            )
            as Map<String, dynamic>;
    expect(payload['capture_session_id'], 'capture-session-03');
    expect(payload['worker_status'], 'UNKNOWN');
    expect(payload['reason_codes'], ['ITEM_BELOW_APPROVAL_THRESHOLD']);
    expect(payload['detection_count'], 1);
    expect(payload['operator_review']['verdict'], 'CORRECTED');
    expect(payload['operator_review']['inferred_issue_codes'], [
      'OPERATOR_MISSED_OBJECT',
    ]);
    expect(payload['operator_review']['objects'], hasLength(2));
    expect(
      payload['operator_review']['objects'][1]['source_detection_id'],
      isNull,
    );
    expect(payload['operator_review']['objects'][1]['disposition'], 'ADD');
    expect(
      await File(p.join(root.path, 'request_missed_1234.jpg')).readAsBytes(),
      [7, 8, 9],
    );
    final logs = await repository.list();
    expect(logs.single.items, hasLength(2));
    expect(logs.single.items.last.disposition, OperatorObjectDisposition.add);
  });

  test('v1 로그는 확정 시각과 안전한 이미지 이름으로 호환 로드한다', () async {
    final support = await Directory.systemTemp.createTemp(
      'product-scanner-legacy-log-',
    );
    addTearDown(() async {
      if (await support.exists()) await support.delete(recursive: true);
    });
    final repository = FileScanLogRepository(
      applicationSupportDirectory: () async => support,
    );
    final root = Directory(p.join(support.path, 'ProductScanner', 'scan_logs'));
    await root.create(recursive: true);
    final image = File(p.join(root.path, 'legacy.jpg'));
    await image.writeAsBytes([1]);
    await File(p.join(root.path, 'legacy.json')).writeAsString(
      jsonEncode({
        'scan_id': 'legacy',
        'analyzed_at': '2026-08-10T01:00:00.000Z',
        'confirmed_at': '2026-08-10T01:01:00.000Z',
        'input_mode': 'CAMERA',
        'original_image': 'legacy.jpg',
        'processing_time_ms': 50,
        'model_versions': {'detector': '0.1.0', 'classifier': '0.1.0'},
        'detections': <Object>[],
      }),
    );
    await File(p.join(root.path, 'unsafe.json')).writeAsString(
      jsonEncode({
        'scan_id': 'unsafe',
        'analyzed_at': '2026-08-10T02:00:00.000Z',
        'confirmed_at': '2026-08-10T02:01:00.000Z',
        'input_mode': 'IMAGE',
        'original_image': '../outside.jpg',
        'processing_time_ms': 51,
        'model_versions': {'detector': '0.1.0', 'classifier': '0.1.0'},
        'detections': <Object>[],
      }),
    );
    await File(p.join(root.path, 'broken.json')).writeAsString('{');

    final logs = await repository.list();

    expect(logs, hasLength(2));
    final legacy = logs.singleWhere((log) => log.scanId == 'legacy');
    expect(legacy.workerStatus, ScanStatus.approved);
    expect(legacy.recordedAt, legacy.confirmedAt);
    expect(legacy.originalImagePath, image.path);
    expect(legacy.isLegacy, isTrue);
    final unsafe = logs.singleWhere((log) => log.scanId == 'unsafe');
    expect(unsafe.originalImagePath, isNull);
  });
}

const _unknownItem = ScanItem(
  itemId: 'item_001',
  bbox: BoundingBox(x: 1, y: 2, width: 3, height: 4),
  status: ItemStatus.unknown,
  reasonCodes: ['BELOW_APPROVAL_THRESHOLD'],
  prediction: null,
  top3: [
    Candidate(
      classId: 'bread_13',
      className: 'Muffin',
      displayName: 'Muffin',
      confidence: .7,
    ),
  ],
  confidence: .7,
);

const _performance = ScanPerformanceMetrics(
  imageWidth: 4032,
  imageHeight: 3024,
  imageSizeBytes: 3145728,
  provider: 'openvino',
  cameraCaptureMs: 401.2,
  fileReadMs: 35.4,
  flutterImageDecodeMs: 280.6,
  readinessMs: 4.1,
  requestBuildMs: 0.7,
  httpRoundTripMs: 705.8,
  responseBodyReadMs: 1.2,
  responseParseMs: 0.5,
  resultMappingMs: 2.1,
  resultFirstFrameMs: 67.1,
  endToEndMs: 1498.7,
  worker: WorkerStageTimings(
    requestTotalMs: 606.2,
    uploadReadMs: 2.0,
    decodeMs: 31.0,
    queueWaitMs: 0.1,
    pipelineMs: 568.4,
    detectorMs: 210.4,
    classifierMs: 350.2,
    decisionMs: 7.8,
    serverOverheadMs: 4.7,
  ),
);
