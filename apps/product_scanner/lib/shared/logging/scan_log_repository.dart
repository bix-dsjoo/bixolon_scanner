import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive_io.dart';
import 'package:crypto/crypto.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../models/scan_models.dart';
import '../version_info.dart';

class ScanPerformanceMetrics {
  const ScanPerformanceMetrics({
    required this.imageWidth,
    required this.imageHeight,
    required this.imageSizeBytes,
    required this.cameraCaptureMs,
    required this.fileReadMs,
    required this.flutterImageDecodeMs,
    required this.readinessMs,
    required this.requestBuildMs,
    required this.httpRoundTripMs,
    required this.responseBodyReadMs,
    required this.responseParseMs,
    required this.resultMappingMs,
    required this.resultFirstFrameMs,
    required this.endToEndMs,
    this.provider,
    this.worker,
  });

  factory ScanPerformanceMetrics.fromJson(Map<String, dynamic> json) {
    final image = json['image'] as Map<String, dynamic>;
    final app = json['app_timings_ms'] as Map<String, dynamic>;
    return ScanPerformanceMetrics(
      imageWidth: image['width'] as int,
      imageHeight: image['height'] as int,
      imageSizeBytes: image['size_bytes'] as int,
      provider: json['provider'] as String?,
      cameraCaptureMs: (app['camera_capture_ms'] as num).toDouble(),
      fileReadMs: (app['file_read_ms'] as num).toDouble(),
      flutterImageDecodeMs: (app['flutter_image_decode_ms'] as num).toDouble(),
      readinessMs: (app['readiness_ms'] as num).toDouble(),
      requestBuildMs: (app['request_build_ms'] as num).toDouble(),
      httpRoundTripMs: (app['http_round_trip_ms'] as num).toDouble(),
      responseBodyReadMs: (app['response_body_read_ms'] as num).toDouble(),
      responseParseMs: (app['response_parse_ms'] as num).toDouble(),
      resultMappingMs: (app['result_mapping_ms'] as num).toDouble(),
      resultFirstFrameMs: (app['result_first_frame_ms'] as num).toDouble(),
      endToEndMs: (json['end_to_end_ms'] as num).toDouble(),
      worker: switch (json['worker_timings_ms']) {
        final Map<String, dynamic> value => WorkerStageTimings.fromJson(value),
        _ => null,
      },
    );
  }

  final int imageWidth;
  final int imageHeight;
  final int imageSizeBytes;
  final String? provider;
  final double cameraCaptureMs;
  final double fileReadMs;
  final double flutterImageDecodeMs;
  final double readinessMs;
  final double requestBuildMs;
  final double httpRoundTripMs;
  final double responseBodyReadMs;
  final double responseParseMs;
  final double resultMappingMs;
  final double resultFirstFrameMs;
  final double endToEndMs;
  final WorkerStageTimings? worker;

  double get apiTotalMs =>
      readinessMs + requestBuildMs + httpRoundTripMs + responseParseMs;

  Map<String, dynamic> toJson() => {
    'schema_version': 1,
    'provider': provider,
    'image': {
      'width': imageWidth,
      'height': imageHeight,
      'size_bytes': imageSizeBytes,
    },
    'app_timings_ms': {
      'camera_capture_ms': cameraCaptureMs,
      'file_read_ms': fileReadMs,
      'flutter_image_decode_ms': flutterImageDecodeMs,
      'readiness_ms': readinessMs,
      'request_build_ms': requestBuildMs,
      'http_round_trip_ms': httpRoundTripMs,
      'response_body_read_ms': responseBodyReadMs,
      'response_parse_ms': responseParseMs,
      'result_mapping_ms': resultMappingMs,
      'result_first_frame_ms': resultFirstFrameMs,
      'api_total_ms': apiTotalMs,
    },
    'worker_timings_ms': worker?.toJson(),
    'end_to_end_ms': endToEndMs,
  };
}

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
    this.operatorReview,
    this.operatorFeedback,
    this.performance,
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
  final OperatorReview? operatorReview;
  @Deprecated('v5에서는 operatorReview를 사용합니다.')
  final ScanOperatorFeedback? operatorFeedback;
  final ScanPerformanceMetrics? performance;
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
    this.reasonCodes = const [],
    this.modelBbox,
    this.finalBbox,
    this.modelProduct,
    this.disposition = OperatorObjectDisposition.keep,
    this.issueCodes = const {},
    this.initialStatus,
  });

  final String itemId;
  final String productName;
  final double confidence;
  final bool userModified;
  final String confirmationMethod;
  final String? classId;
  final String? className;
  final List<String> reasonCodes;
  final BoundingBox? modelBbox;
  final BoundingBox? finalBbox;
  final Product? modelProduct;
  final OperatorObjectDisposition disposition;
  final Set<OperatorIssueCode> issueCodes;
  final ItemStatus? initialStatus;

  ItemStatus get resultStatus {
    final hasFinalProduct =
        classId?.isNotEmpty == true ||
        className?.isNotEmpty == true ||
        (productName.isNotEmpty && productName != 'Unknown');
    if (hasFinalProduct) return ItemStatus.approved;
    return initialStatus ?? ItemStatus.unknown;
  }

  ScanLogItemSummary withProductName(
    String value, {
    Product? localizedModelProduct,
  }) => ScanLogItemSummary(
    itemId: itemId,
    productName: value,
    confidence: confidence,
    userModified: userModified,
    confirmationMethod: confirmationMethod,
    classId: classId,
    className: className,
    reasonCodes: reasonCodes,
    modelBbox: modelBbox,
    finalBbox: finalBbox,
    modelProduct: localizedModelProduct ?? modelProduct,
    disposition: disposition,
    issueCodes: issueCodes,
    initialStatus: initialStatus,
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
    this.performance,
    this.logSchemaVersion = 1,
    this.operatorReview,
    this.recordFilePath,
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
  final ScanPerformanceMetrics? performance;
  final int logSchemaVersion;
  final OperatorReview? operatorReview;
  final String? recordFilePath;

  bool get isRecapture => workerStatus == ScanStatus.recapture;
  bool get isLegacy => logSchemaVersion < 5 || operatorReview == null;
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

  Future<Directory> legacyFeedbackDirectory() async {
    final support = await _applicationSupportDirectory();
    return Directory(
      p.join(support.path, 'ProductScanner', 'feedback_logs', 'missed_object'),
    );
  }

  Future<void> exportReviewArchive({
    required String targetPath,
    List<ScanLogSummary>? records,
  }) async {
    final selectedRecords = records ?? await list(limit: 0x7fffffff);
    final scope = records == null ? 'ALL' : 'FILTERED';
    final payloadFiles = <_ExportFile>[];
    for (final record in selectedRecords) {
      final safeId = _safeArchiveId(record.scanId);
      final recordPath = record.recordFilePath;
      if (recordPath != null && await File(recordPath).exists()) {
        payloadFiles.add(
          _ExportFile(
            source: File(recordPath),
            archivePath: 'records/$safeId.json',
            level: ZipFileEncoder.gzip,
          ),
        );
      }
      final imagePath = record.originalImagePath;
      if (imagePath != null && await File(imagePath).exists()) {
        final extension = p.extension(imagePath).toLowerCase() == '.png'
            ? '.png'
            : '.jpg';
        payloadFiles.add(
          _ExportFile(
            source: File(imagePath),
            archivePath: 'images/$safeId$extension',
            level: ZipFileEncoder.store,
          ),
        );
      }
    }
    if (scope == 'ALL') {
      final legacyRoot = await legacyFeedbackDirectory();
      if (await legacyRoot.exists()) {
        await for (final entity in legacyRoot.list(recursive: true)) {
          if (entity is! File) continue;
          final relative = p.relative(entity.path, from: legacyRoot.path);
          payloadFiles.add(
            _ExportFile(
              source: entity,
              archivePath: p.posix.join(
                'legacy_feedback',
                p.posix.fromUri(p.toUri(relative)),
              ),
              level: _isCompressedImage(entity.path)
                  ? ZipFileEncoder.store
                  : ZipFileEncoder.gzip,
            ),
          );
        }
      }
    }

    final files = <Map<String, dynamic>>[];
    final checksumLines = <String>[];
    for (final file in payloadFiles) {
      final checksum = await _fileSha256(file.source);
      final size = await file.source.length();
      files.add({
        'path': file.archivePath,
        'size_bytes': size,
        'sha256': checksum,
      });
      checksumLines.add('$checksum  ${file.archivePath}');
    }
    final manifestText = const JsonEncoder.withIndent('  ').convert({
      'export_schema_version': 1,
      'product_version': VersionInfo.current,
      'created_at': DateTime.now().toUtc().toIso8601String(),
      'scope': scope,
      'record_count': selectedRecords.length,
      'files': files,
    });
    final manifestHash = sha256.convert(utf8.encode(manifestText)).toString();
    checksumLines.insert(0, '$manifestHash  manifest.json');
    final checksumText = '${checksumLines.join('\n')}\n';

    final target = File(targetPath);
    await target.parent.create(recursive: true);
    final partial = File('$targetPath.partial');
    if (await partial.exists()) await partial.delete();
    final encoder = ZipFileEncoder();
    try {
      encoder.create(partial.path, level: ZipFileEncoder.gzip);
      encoder.addArchiveFile(ArchiveFile.string('manifest.json', manifestText));
      for (final file in payloadFiles) {
        await encoder.addFile(file.source, file.archivePath, file.level);
      }
      encoder.addArchiveFile(
        ArchiveFile.string('checksums.sha256', checksumText),
      );
      await encoder.close();
      if (await target.exists()) await target.delete();
      await partial.rename(target.path);
    } catch (_) {
      try {
        await encoder.close();
      } on Object {
        // The encoder may not have completed initialization.
      }
      if (await partial.exists()) await partial.delete();
      rethrow;
    }
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
    final root = await _logDirectory();
    await root.create(recursive: true);

    final extension = _safeExtension(record.imageFileName);
    final imageName = '${record.scanId}$extension';
    final imageFile = File(p.join(root.path, imageName));
    final temporaryImage = File('${imageFile.path}.tmp');
    await temporaryImage.writeAsBytes(record.imageBytes, flush: true);
    if (await imageFile.exists()) await imageFile.delete();
    await temporaryImage.rename(imageFile.path);

    final payload = <String, dynamic>{
      'log_schema_version': 5,
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
      'detection_count': record.detections
          .where((detection) => !detection.operatorAdded)
          .length,
      'model_versions': record.modelVersions.toJson(),
      if (record.performance case final performance?)
        'performance': performance.toJson(),
      'detections': record.detections
          .where((detection) => !detection.operatorAdded)
          .map((detection) => detection.toLogJson())
          .toList(),
      'operator_review': (record.operatorReview ?? _defaultReview(record))
          .toJson(),
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
        final performance = switch (decoded['performance']) {
          final Map<String, dynamic> value => ScanPerformanceMetrics.fromJson(
            value,
          ),
          _ => null,
        };
        final schemaVersion = switch (decoded['log_schema_version']) {
          final int value => value,
          final num value => value.toInt(),
          _ => 1,
        };
        final operatorReview = switch (decoded['operator_review']) {
          final Map<String, dynamic> value => OperatorReview.fromJson(value),
          _ => null,
        };
        final reviewObjects = {
          for (final object
              in operatorReview?.objects ?? const <OperatorReviewObject>[])
            object.sourceDetectionId ?? object.objectId: object,
        };
        final modelItems = detections.map((value) {
          final detection = value as Map<String, dynamic>;
          final detectionId = detection['detection_id'] as String;
          final product = detection['final_product'] as Map<String, dynamic>?;
          final modelProductJson =
              detection['initial_ai_prediction'] as Map<String, dynamic>?;
          final top3 = switch (detection['top3']) {
            final List values => values.whereType<Map<String, dynamic>>(),
            _ => const Iterable<Map<String, dynamic>>.empty(),
          };
          final reviewObject = reviewObjects[detectionId];
          final finalProduct =
              reviewObject?.finalProduct ??
              (product == null ? null : Product.fromJson(product));
          return ScanLogItemSummary(
            itemId: detectionId,
            productName: finalProduct?.className ?? 'Unknown',
            confidence: (detection['initial_confidence'] as num).toDouble(),
            userModified:
                reviewObject?.disposition != OperatorObjectDisposition.keep ||
                (detection['user_modified'] as bool? ?? false),
            confirmationMethod:
                detection['confirmation_method'] as String? ?? 'UNKNOWN',
            classId: finalProduct?.classId,
            className: finalProduct?.className,
            reasonCodes: switch (detection['reason_codes']) {
              final List values => values.whereType<String>().toList(
                growable: false,
              ),
              _ => const <String>[],
            },
            modelBbox: switch (detection['bbox']) {
              final Map<String, dynamic> value => BoundingBox.fromJson(value),
              _ => null,
            },
            finalBbox:
                reviewObject?.finalBbox ??
                switch (detection['bbox']) {
                  final Map<String, dynamic> value => BoundingBox.fromJson(
                    value,
                  ),
                  _ => null,
                },
            modelProduct: modelProductJson != null
                ? Product.fromJson(modelProductJson)
                : top3.isEmpty
                ? null
                : Product.fromJson(top3.first),
            disposition:
                reviewObject?.disposition ?? OperatorObjectDisposition.keep,
            issueCodes: reviewObject?.issueCodes ?? const <OperatorIssueCode>{},
            initialStatus: switch (detection['initial_ai_status']) {
              'AUTO_APPROVED' => ItemStatus.approved,
              'TOP3_CANDIDATES' => ItemStatus.unknown,
              'SEGMENT_RECAPTURE' => ItemStatus.segmentRecapture,
              _ => null,
            },
          );
        }).toList();
        final addedItems =
            (operatorReview?.objects ?? const <OperatorReviewObject>[])
                .where(
                  (object) =>
                      object.sourceDetectionId == null &&
                      object.disposition == OperatorObjectDisposition.add,
                )
                .map(
                  (object) => ScanLogItemSummary(
                    itemId: object.objectId,
                    productName: object.finalProduct?.className ?? 'Unknown',
                    confidence: 0,
                    userModified: true,
                    confirmationMethod: 'OPERATOR_ADDED',
                    classId: object.finalProduct?.classId,
                    className: object.finalProduct?.className,
                    finalBbox: object.finalBbox,
                    disposition: object.disposition,
                    issueCodes: object.issueCodes,
                  ),
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
            performance: performance,
            logSchemaVersion: schemaVersion,
            operatorReview: operatorReview,
            recordFilePath: file.path,
            items: [...modelItems, ...addedItems],
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

  OperatorReview _defaultReview(ScanLogRecord record) {
    final inferred = <OperatorIssueCode>{
      for (final detection in record.detections)
        ...detection.inferredIssueCodes,
    };
    final verdict = record.workerStatus == ScanStatus.recapture
        ? OperatorReviewVerdict.recaptureAgreed
        : inferred.isEmpty
        ? OperatorReviewVerdict.accepted
        : OperatorReviewVerdict.corrected;
    return OperatorReview(
      verdict: verdict,
      reviewedAt: record.recordedAt,
      inferredIssueCodes: inferred,
      issueCodes: inferred,
      objects: record.detections
          .map(OperatorReviewObject.fromDetection)
          .toList(growable: false),
    );
  }
}

String _newCaptureSessionId() {
  final timestamp = DateTime.now().toUtc().toIso8601String().replaceAll(
    RegExp(r'[^0-9]'),
    '',
  );
  return 'app-$timestamp-$pid';
}

String _safeArchiveId(String value) {
  final sanitized = value.replaceAll(RegExp(r'[^A-Za-z0-9._-]'), '_');
  return sanitized.isEmpty ? 'scan' : sanitized;
}

bool _isCompressedImage(String path) {
  final extension = p.extension(path).toLowerCase();
  return extension == '.jpg' || extension == '.jpeg' || extension == '.png';
}

Future<String> _fileSha256(File file) async =>
    (await sha256.bind(file.openRead()).first).toString();

class _ExportFile {
  const _ExportFile({
    required this.source,
    required this.archivePath,
    required this.level,
  });

  final File source;
  final String archivePath;
  final int level;
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
