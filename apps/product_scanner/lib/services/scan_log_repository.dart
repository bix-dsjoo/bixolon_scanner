import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../models/scan_models.dart';

class ScanOperatorFeedback {
  const ScanOperatorFeedback._({
    required this.type,
    required this.expectedStatus,
    required this.expectedReason,
    required this.annotationStatus,
    required this.minimumMissingObjectCount,
  });

  static const missedObject = ScanOperatorFeedback._(
    type: 'MISSED_OBJECT',
    expectedStatus: 'RECAPTURE',
    expectedReason: 'DETECTOR_MISSED_OBJECT',
    annotationStatus: 'PENDING_BBOX_CLASS_REVIEW',
    minimumMissingObjectCount: 1,
  );

  final String type;
  final String expectedStatus;
  final String expectedReason;
  final String annotationStatus;
  final int minimumMissingObjectCount;

  Map<String, dynamic> toJson() => {
    'type': type,
    'expected_status': expectedStatus,
    'expected_reason': expectedReason,
    'annotation_status': annotationStatus,
    'minimum_missing_object_count': minimumMissingObjectCount,
  };
}

class ScanLogRecord {
  const ScanLogRecord({
    required this.scanId,
    required this.analyzedAt,
    required this.confirmedAt,
    DateTime? recordedAt,
    required this.inputMode,
    required this.imageBytes,
    required this.imageFileName,
    required this.processingTimeMs,
    required this.modelVersions,
    required this.detections,
    this.workerStatus = ScanStatus.approved,
    this.reasonCodes = const [],
    this.operatorFeedback,
  }) : recordedAt = recordedAt ?? confirmedAt ?? analyzedAt;

  final String scanId;
  final DateTime analyzedAt;
  final DateTime? confirmedAt;
  final DateTime recordedAt;
  final InputMode inputMode;
  final Uint8List imageBytes;
  final String imageFileName;
  final double processingTimeMs;
  final ModelVersions modelVersions;
  final List<ReviewDetection> detections;
  final ScanStatus workerStatus;
  final List<String> reasonCodes;
  final ScanOperatorFeedback? operatorFeedback;
}

class ScanLogItemSummary {
  const ScanLogItemSummary({
    required this.itemId,
    required this.productName,
    required this.confidence,
    required this.userModified,
    required this.confirmationMethod,
    this.classId,
    this.className,
  });

  final String itemId;
  final String productName;
  final double confidence;
  final bool userModified;
  final String confirmationMethod;
  final String? classId;
  final String? className;

  ScanLogItemSummary withProductName(String value) => ScanLogItemSummary(
    itemId: itemId,
    productName: value,
    confidence: confidence,
    userModified: userModified,
    confirmationMethod: confirmationMethod,
    classId: classId,
    className: className,
  );
}

class ScanLogSummary {
  const ScanLogSummary({
    required this.scanId,
    required this.analyzedAt,
    required this.confirmedAt,
    DateTime? recordedAt,
    required this.inputMode,
    required this.processingTimeMs,
    required this.modelVersions,
    required this.items,
    this.workerStatus = ScanStatus.approved,
    this.reasonCodes = const [],
    this.originalImagePath,
  }) : recordedAt = recordedAt ?? confirmedAt ?? analyzedAt;

  final String scanId;
  final DateTime analyzedAt;
  final DateTime? confirmedAt;
  final DateTime recordedAt;
  final InputMode inputMode;
  final double processingTimeMs;
  final ModelVersions modelVersions;
  final List<ScanLogItemSummary> items;
  final ScanStatus workerStatus;
  final List<String> reasonCodes;
  final String? originalImagePath;

  bool get isRecapture => workerStatus == ScanStatus.recapture;
}

abstract interface class ScanLogRepository {
  Future<void> save(ScanLogRecord record);

  Future<List<ScanLogSummary>> list({int limit = 100});
}

class FileScanLogRepository implements ScanLogRepository {
  FileScanLogRepository({
    Future<Directory> Function()? applicationSupportDirectory,
    String? captureSessionId,
  }) : _applicationSupportDirectory =
           applicationSupportDirectory ?? getApplicationSupportDirectory,
       _captureSessionId = captureSessionId ?? _newCaptureSessionId();

  final Future<Directory> Function() _applicationSupportDirectory;
  final String _captureSessionId;

  Future<Directory> _logDirectory() async {
    final support = await _applicationSupportDirectory();
    return Directory(p.join(support.path, 'ProductScanner', 'scan_logs'));
  }

  Future<Directory> _missedObjectFeedbackDirectory() async {
    final support = await _applicationSupportDirectory();
    return Directory(
      p.join(support.path, 'ProductScanner', 'feedback_logs', 'missed_object'),
    );
  }

  @override
  Future<void> save(ScanLogRecord record) async {
    if (record.workerStatus == ScanStatus.error) {
      throw ArgumentError.value(
        record.workerStatus,
        'workerStatus',
        'ERROR responses are not activity records',
      );
    }
    final root = record.operatorFeedback == null
        ? await _logDirectory()
        : await _missedObjectFeedbackDirectory();
    await root.create(recursive: true);

    final extension = _safeExtension(record.imageFileName);
    final imageName = '${record.scanId}$extension';
    final imageFile = File(p.join(root.path, imageName));
    await imageFile.writeAsBytes(record.imageBytes, flush: true);

    final payload = <String, dynamic>{
      'log_schema_version': 2,
      'scan_id': record.scanId,
      'capture_session_id': _captureSessionId,
      'worker_status': _workerStatusValue(record.workerStatus),
      'reason_codes': record.reasonCodes,
      'analyzed_at': record.analyzedAt.toUtc().toIso8601String(),
      'recorded_at': record.recordedAt.toUtc().toIso8601String(),
      'confirmed_at': record.confirmedAt?.toUtc().toIso8601String(),
      'input_mode': record.inputMode == InputMode.camera ? 'CAMERA' : 'IMAGE',
      'original_image': imageName,
      'processing_time_ms': record.processingTimeMs,
      'detection_count': record.detections.length,
      'model_versions': record.modelVersions.toJson(),
      'detections': record.detections
          .map((detection) => detection.toLogJson())
          .toList(),
      if (record.operatorFeedback case final feedback?)
        'operator_feedback': feedback.toJson(),
    };

    final target = File(p.join(root.path, '${record.scanId}.json'));
    final temporary = File('${target.path}.tmp');
    await temporary.writeAsString(
      const JsonEncoder.withIndent('  ').convert(payload),
      flush: true,
    );
    if (await target.exists()) await target.delete();
    await temporary.rename(target.path);
  }

  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async {
    final root = await _logDirectory();
    if (!await root.exists()) return const [];
    final files = await root
        .list()
        .where(
          (entity) => entity is File && p.extension(entity.path) == '.json',
        )
        .cast<File>()
        .toList();
    final logs = <ScanLogSummary>[];
    for (final file in files) {
      try {
        final decoded = jsonDecode(await file.readAsString());
        if (decoded is! Map<String, dynamic>) continue;
        final detections = decoded['detections'];
        if (detections is! List) continue;
        final rawVersions = decoded['model_versions'];
        if (rawVersions is! Map<String, dynamic>) continue;
        final analyzedAt = DateTime.parse(decoded['analyzed_at'] as String);
        final confirmedAt = switch (decoded['confirmed_at']) {
          final String value => DateTime.parse(value),
          _ => null,
        };
        final recordedAt = switch (decoded['recorded_at']) {
          final String value => DateTime.parse(value),
          _ => confirmedAt ?? analyzedAt,
        };
        final workerStatus = _parseWorkerStatus(
          decoded['worker_status'],
          detections,
        );
        final reasonCodes = switch (decoded['reason_codes']) {
          final List values => values.whereType<String>().toList(
            growable: false,
          ),
          _ => const <String>[],
        };
        final originalImagePath = await _resolveStoredImagePath(
          root,
          decoded['original_image'],
        );
        logs.add(
          ScanLogSummary(
            scanId: decoded['scan_id'] as String,
            analyzedAt: analyzedAt,
            confirmedAt: confirmedAt,
            recordedAt: recordedAt,
            inputMode: decoded['input_mode'] == 'CAMERA'
                ? InputMode.camera
                : InputMode.image,
            processingTimeMs: (decoded['processing_time_ms'] as num).toDouble(),
            modelVersions: ModelVersions.fromJson(rawVersions),
            workerStatus: workerStatus,
            reasonCodes: reasonCodes,
            originalImagePath: originalImagePath,
            items: detections
                .map((value) {
                  final detection = value as Map<String, dynamic>;
                  final product =
                      detection['final_product'] as Map<String, dynamic>?;
                  return ScanLogItemSummary(
                    itemId: detection['detection_id'] as String,
                    productName: product?['class_name'] as String? ?? 'Unknown',
                    confidence: (detection['initial_confidence'] as num)
                        .toDouble(),
                    userModified: detection['user_modified'] as bool? ?? false,
                    confirmationMethod:
                        detection['confirmation_method'] as String? ??
                        'UNKNOWN',
                    classId: product?['class_id'] as String?,
                    className: product?['class_name'] as String?,
                  );
                })
                .toList(growable: false),
          ),
        );
      } on Object {
        // A partially written or manually modified file must not hide valid logs.
      }
    }
    logs.sort((a, b) => b.recordedAt.compareTo(a.recordedAt));
    return logs.take(limit).toList(growable: false);
  }

  Future<String?> _resolveStoredImagePath(
    Directory root,
    Object? storedName,
  ) async {
    if (storedName is! String ||
        storedName.isEmpty ||
        p.isAbsolute(storedName) ||
        p.basename(storedName) != storedName) {
      return null;
    }
    final file = File(p.join(root.path, storedName));
    return await file.exists() ? file.path : null;
  }

  String _safeExtension(String fileName) {
    final value = p.extension(fileName).toLowerCase();
    return value == '.png' ? '.png' : '.jpg';
  }
}

String _newCaptureSessionId() {
  final timestamp = DateTime.now().toUtc().toIso8601String().replaceAll(
    RegExp(r'[^0-9]'),
    '',
  );
  return 'app-$timestamp-$pid';
}

String _workerStatusValue(ScanStatus status) => switch (status) {
  ScanStatus.approved => 'APPROVED',
  ScanStatus.unknown => 'UNKNOWN',
  ScanStatus.recapture => 'RECAPTURE',
  ScanStatus.error => 'ERROR',
};

ScanStatus _parseWorkerStatus(Object? value, List<dynamic> detections) {
  return switch (value) {
    'APPROVED' => ScanStatus.approved,
    'UNKNOWN' => ScanStatus.unknown,
    'RECAPTURE' => ScanStatus.recapture,
    'ERROR' => throw const FormatException('ERROR activity record is invalid'),
    null =>
      detections.any(
            (value) =>
                value is Map<String, dynamic> &&
                value['initial_ai_status'] == 'TOP3_CANDIDATES',
          )
          ? ScanStatus.unknown
          : ScanStatus.approved,
    _ => throw const FormatException('Unsupported activity worker status'),
  };
}
