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

  factory BoundingBox.fromJson(Map<String, dynamic> json) {
    final value = BoundingBox(
      x: json['x'] as int,
      y: json['y'] as int,
      width: json['width'] as int,
      height: json['height'] as int,
    );
    if (value.x < 0 || value.y < 0 || value.width <= 0 || value.height <= 0) {
      throw const FormatException(
        'Bounding box values are outside the public contract.',
      );
    }
    return value;
  }

  final int x;
  final int y;
  final int width;
  final int height;
}

class Prediction {
  const Prediction({required this.classId, required this.className});

  factory Prediction.fromJson(Map<String, dynamic> json) {
    final value = Prediction(
      classId: json['class_id'] as String,
      className: json['class_name'] as String,
    );
    if (value.classId.isEmpty || value.className.isEmpty) {
      throw const FormatException('Prediction identifiers must not be empty.');
    }
    return value;
  }

  final String classId;
  final String className;
}

class Candidate extends Prediction {
  const Candidate({
    required super.classId,
    required super.className,
    required this.confidence,
  });

  factory Candidate.fromJson(Map<String, dynamic> json) {
    final value = Candidate(
      classId: json['class_id'] as String,
      className: json['class_name'] as String,
      confidence: (json['confidence'] as num).toDouble(),
    );
    if (value.classId.isEmpty ||
        value.className.isEmpty ||
        value.confidence < 0 ||
        value.confidence > 1) {
      throw const FormatException(
        'Candidate values are outside the public contract.',
      );
    }
    return value;
  }

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

  factory Segmentation.fromJson(Map<String, dynamic> json) {
    final value = Segmentation(
      segmentationId: json['segmentation_id'] as String,
      boundingBox: BoundingBox.fromJson(json['bbox'] as Map<String, dynamic>),
      status: SegmentationStatus.parse(json['status'] as String),
      reasonCodes: List<String>.unmodifiable(
        (json['reason_codes'] as List).cast<String>(),
      ),
      prediction: switch (json['prediction']) {
        final Map<String, dynamic> prediction => Prediction.fromJson(
          prediction,
        ),
        null => null,
        _ => throw const FormatException(
          'prediction must be an object or null.',
        ),
      },
      top3: List<Candidate>.unmodifiable(
        (json['top3'] as List).map(
          (candidate) => Candidate.fromJson(candidate as Map<String, dynamic>),
        ),
      ),
      confidence: (json['confidence'] as num).toDouble(),
    );
    value._validateContract();
    return value;
  }

  final String segmentationId;
  final BoundingBox boundingBox;
  final SegmentationStatus status;
  final List<String> reasonCodes;
  final Prediction? prediction;
  final List<Candidate> top3;
  final double confidence;

  void _validateContract() {
    if (!RegExp(r'^segmentation_\d{3,}$').hasMatch(segmentationId) ||
        confidence < 0 ||
        confidence > 1 ||
        top3.length > 3) {
      throw const FormatException(
        'Segmentation values are outside the public contract.',
      );
    }
    switch (status) {
      case SegmentationStatus.approved:
        if (prediction == null || top3.isNotEmpty || reasonCodes.isNotEmpty) {
          throw const FormatException(
            'APPROVED requires prediction and empty reasons and candidates.',
          );
        }
      case SegmentationStatus.unknown:
        const allowedReasons = <String>{
          'BELOW_APPROVAL_THRESHOLD',
          'CLASSIFIER_AMBIGUOUS_TOP2',
          'CLASSIFIER_CATALOG_CONFLICT',
          'DETECTOR_CONTAINED_DUPLICATE',
        };
        if (prediction != null ||
            top3.isEmpty ||
            reasonCodes.length != 1 ||
            !allowedReasons.contains(reasonCodes.single)) {
          throw const FormatException(
            'UNKNOWN requires Top-3 evidence and exactly one supported reason.',
          );
        }
        for (var index = 1; index < top3.length; index++) {
          if (top3[index - 1].confidence < top3[index].confidence) {
            throw const FormatException(
              'UNKNOWN Top-3 must be sorted by confidence.',
            );
          }
        }
      case SegmentationStatus.segmentRecapture:
        if (prediction != null ||
            top3.isNotEmpty ||
            reasonCodes.length != 1 ||
            reasonCodes.single != 'SEGMENT_RECAPTURE_REQUIRED') {
          throw const FormatException(
            'SEGMENT_RECAPTURE requires its common reason and no prediction.',
          );
        }
    }
  }
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

  bool get hasWorkerVersion => worker != null && worker!.isNotEmpty;
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

  factory ScanResponse.fromJson(Map<String, dynamic> json) {
    final value = ScanResponse(
      requestId: json['request_id'] as String,
      status: ScanStatus.parse(json['status'] as String),
      reasonCodes: List<String>.unmodifiable(
        (json['reason_codes'] as List).cast<String>(),
      ),
      segmentations: List<Segmentation>.unmodifiable(
        (json['segmentations'] as List).map(
          (segmentation) =>
              Segmentation.fromJson(segmentation as Map<String, dynamic>),
        ),
      ),
      processingTimeMs: (json['processing_time_ms'] as num).toDouble(),
      versions: ComponentVersions.fromPublicJson(json),
    );
    value._validateContract();
    return value;
  }

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

  void _validateContract() {
    if (requestId.length < 8 ||
        processingTimeMs < 0 ||
        !versions.hasWorkerVersion) {
      throw const FormatException(
        'Scan response values are outside the public contract.',
      );
    }
    switch (status) {
      case ScanStatus.imageRecapture:
        if (segmentations.isNotEmpty ||
            reasonCodes.length != 1 ||
            reasonCodes.single != 'IMAGE_RECAPTURE_REQUIRED' ||
            versions.detector == null) {
          throw const FormatException(
            'IMAGE_RECAPTURE response violates the public contract.',
          );
        }
      case ScanStatus.error:
        if (segmentations.isNotEmpty || reasonCodes.isEmpty) {
          throw const FormatException(
            'ERROR response violates the public contract.',
          );
        }
      case ScanStatus.segmentation:
        if (segmentations.isEmpty ||
            versions.detector == null ||
            versions.classifier == null) {
          throw const FormatException(
            'SEGMENTATION response violates the public contract.',
          );
        }
        final expectedReasons = <String>[];
        if (segmentations.any(
          (item) =>
              item.status == SegmentationStatus.unknown &&
              item.reasonCodes.single != 'DETECTOR_CONTAINED_DUPLICATE',
        )) {
          expectedReasons.add('SEGMENT_BELOW_APPROVAL_THRESHOLD');
        }
        if (segmentations.any(
          (item) => item.reasonCodes.contains('DETECTOR_CONTAINED_DUPLICATE'),
        )) {
          expectedReasons.add('SEGMENT_DUPLICATE_REVIEW_REQUIRED');
        }
        if (segmentations.any(
          (item) => item.status == SegmentationStatus.segmentRecapture,
        )) {
          expectedReasons.add('SEGMENT_RECAPTURE_REQUIRED');
        }
        if (!_sameStrings(reasonCodes, expectedReasons)) {
          throw const FormatException(
            'SEGMENTATION aggregate reasons do not match segment outcomes.',
          );
        }
    }
  }

  static bool _sameStrings(List<String> left, List<String> right) {
    if (left.length != right.length) return false;
    for (var index = 0; index < left.length; index++) {
      if (left[index] != right[index]) return false;
    }
    return true;
  }
}
