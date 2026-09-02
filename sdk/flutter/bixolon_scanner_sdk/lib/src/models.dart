import 'dart:convert';

enum ScanStatus {
  segmentation,
  imageRecapture,
  error;

  static ScanStatus parse(String value) => switch (value) {
    'SEGMENTATION' => ScanStatus.segmentation,
    'IMAGE_RECAPTURE' => ScanStatus.imageRecapture,
    'ERROR' => ScanStatus.error,
    _ => throw FormatException('Unsupported scan status: $value'),
  };
}

enum SegmentationStatus {
  approved,
  unknown,
  segmentRecapture;

  static SegmentationStatus parse(String value) => switch (value) {
    'APPROVED' => SegmentationStatus.approved,
    'UNKNOWN' => SegmentationStatus.unknown,
    'SEGMENT_RECAPTURE' => SegmentationStatus.segmentRecapture,
    _ => throw FormatException('Unsupported segmentation status: $value'),
  };
}

class BoundingBox {
  const BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  factory BoundingBox.fromJson(Map<String, dynamic> json) => BoundingBox(
    x: json['x'] as int,
    y: json['y'] as int,
    width: json['width'] as int,
    height: json['height'] as int,
  );

  final int x;
  final int y;
  final int width;
  final int height;
}

class Prediction {
  const Prediction({required this.classId, required this.className});

  factory Prediction.fromJson(Map<String, dynamic> json) => Prediction(
    classId: json['class_id'] as String,
    className: json['class_name'] as String,
  );

  final String classId;
  final String className;
}

class Candidate extends Prediction {
  const Candidate({
    required super.classId,
    required super.className,
    required this.confidence,
  });

  factory Candidate.fromJson(Map<String, dynamic> json) => Candidate(
    classId: json['class_id'] as String,
    className: json['class_name'] as String,
    confidence: (json['confidence'] as num).toDouble(),
  );

  final double confidence;
}

class Segmentation {
  const Segmentation({
    required this.segmentationId,
    required this.boundingBox,
    required this.status,
    required this.reasonCodes,
    required this.prediction,
    required this.top3,
    required this.confidence,
  });

  factory Segmentation.fromJson(Map<String, dynamic> json) => Segmentation(
    segmentationId: json['segmentation_id'] as String,
    boundingBox: BoundingBox.fromJson(json['bbox'] as Map<String, dynamic>),
    status: SegmentationStatus.parse(json['status'] as String),
    reasonCodes: List<String>.unmodifiable(
      (json['reason_codes'] as List).cast<String>(),
    ),
    prediction: switch (json['prediction']) {
      final Map<String, dynamic> value => Prediction.fromJson(value),
      _ => null,
    },
    top3: List<Candidate>.unmodifiable(
      (json['top3'] as List).map(
        (value) => Candidate.fromJson(value as Map<String, dynamic>),
      ),
    ),
    confidence: (json['confidence'] as num).toDouble(),
  );

  final String segmentationId;
  final BoundingBox boundingBox;
  final SegmentationStatus status;
  final List<String> reasonCodes;
  final Prediction? prediction;
  final List<Candidate> top3;
  final double confidence;
}

class ComponentVersions {
  const ComponentVersions({
    required this.worker,
    required this.detector,
    required this.classifier,
    required this.embedder,
    required this.detectorPolicy,
    required this.classifierPolicy,
    required this.catalog,
  });

  factory ComponentVersions.fromPublicJson(Map<String, dynamic> json) =>
      ComponentVersions(
        worker: json['worker_version'] as String?,
        detector: json['detector_version'] as String?,
        classifier: json['classifier_version'] as String?,
        embedder: json['embedder_version'] as String?,
        detectorPolicy: json['detector_policy_version'] as String?,
        classifierPolicy: json['classifier_policy_version'] as String?,
        catalog: json['catalog_version'] as String?,
      );

  final String? worker;
  final String? detector;
  final String? classifier;
  final String? embedder;
  final String? detectorPolicy;
  final String? classifierPolicy;
  final String? catalog;

  Iterable<String> get nonNullValues => <String?>[
    worker,
    detector,
    classifier,
    embedder,
    detectorPolicy,
    classifierPolicy,
    catalog,
  ].whereType<String>();

  bool get hasOneProductVersion => nonNullValues.toSet().length <= 1;
}

class WorkerReadiness {
  const WorkerReadiness({required this.provider, required this.versions});

  factory WorkerReadiness.fromJson(Map<String, dynamic> json) {
    if (json['status'] != 'ready') {
      throw const FormatException('Worker is not ready.');
    }
    return WorkerReadiness(
      provider: json['provider'] as String?,
      versions: ComponentVersions.fromPublicJson(json),
    );
  }

  final String? provider;
  final ComponentVersions versions;
}

class ScanResponse {
  const ScanResponse({
    required this.requestId,
    required this.status,
    required this.reasonCodes,
    required this.segmentations,
    required this.processingTimeMs,
    required this.versions,
  });

  factory ScanResponse.fromJson(Map<String, dynamic> json) => ScanResponse(
    requestId: json['request_id'] as String,
    status: ScanStatus.parse(json['status'] as String),
    reasonCodes: List<String>.unmodifiable(
      (json['reason_codes'] as List).cast<String>(),
    ),
    segmentations: List<Segmentation>.unmodifiable(
      (json['segmentations'] as List).map(
        (value) => Segmentation.fromJson(value as Map<String, dynamic>),
      ),
    ),
    processingTimeMs: (json['processing_time_ms'] as num).toDouble(),
    versions: ComponentVersions.fromPublicJson(json),
  );

  factory ScanResponse.fromBody(String body) {
    final value = jsonDecode(body);
    if (value is! Map<String, dynamic>) {
      throw const FormatException('Scan response must be a JSON object.');
    }
    return ScanResponse.fromJson(value);
  }

  final String requestId;
  final ScanStatus status;
  final List<String> reasonCodes;
  final List<Segmentation> segmentations;
  final double processingTimeMs;
  final ComponentVersions versions;
}
