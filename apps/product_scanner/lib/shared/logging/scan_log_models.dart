part of 'scan_log_repository.dart';

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
