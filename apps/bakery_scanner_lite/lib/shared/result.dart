const workerVersion = '0.1.17';

const finalStates = {'SEGMENTATION', 'IMAGE_RECAPTURE', 'ERROR'};
const objectStates = {'APPROVED', 'UNKNOWN', 'SEGMENT_RECAPTURE'};
const versionFields = [
  'worker_version',
  'detector_version',
  'classifier_version',
  'embedder_version',
  'detector_policy_version',
  'classifier_policy_version',
  'catalog_version',
];

List<String> readReasons(dynamic value) {
  if (value is! List || value.any((v) => v is! String)) {
    throw const FormatException('Invalid reasons');
  }
  return List<String>.unmodifiable(value.cast<String>());
}

class ObjectResult {
  ObjectResult(
    this.status,
    Iterable<String> reasons, {
    this.box,
    Iterable<TopCandidate> top3 = const [],
  }) : reasons = List.unmodifiable(reasons),
       top3 = List.unmodifiable(top3);
  final String status;
  final List<String> reasons;
  // Immutable model geometry is retained for viewing historical scan images.
  final DetectionBox? box;
  final List<TopCandidate> top3;
  Map<String, Object?> toJson() => {
    'status': status,
    'reason_codes': reasons,
    if (box != null) 'bbox': box!.toJson(),
    'top3': top3.map((candidate) => candidate.toJson()).toList(),
  };
}

class TopCandidate {
  const TopCandidate(this.classId, this.className);
  final String classId, className;
  Map<String, Object?> toJson() => {
    'class_id': classId,
    'class_name': className,
  };
}

List<TopCandidate> readTop3(dynamic value) {
  if (value == null) {
    return const []; // Older Lite journals did not record candidates.
  }
  if (value is! List || value.length > 3) {
    throw const FormatException('Invalid Top-3');
  }
  return List.unmodifiable(
    value.map((dynamic candidate) {
      if (candidate is! Map ||
          candidate['class_id'] is! String ||
          (candidate['class_id'] as String).isEmpty ||
          (candidate['class_name'] != null &&
              candidate['class_name'] is! String)) {
        throw const FormatException('Invalid candidate');
      }
      return TopCandidate(
        candidate['class_id'] as String,
        (candidate['class_name'] ?? candidate['class_id']) as String,
      );
    }),
  );
}

class DetectionBox {
  const DetectionBox(this.x, this.y, this.width, this.height);
  final double x, y, width, height;
  Map<String, double> toJson() => {
    'x': x,
    'y': y,
    'width': width,
    'height': height,
  };
  factory DetectionBox.fromJson(dynamic json) {
    if (json is! Map) throw const FormatException('Missing detection box');
    final values = [
      'x',
      'y',
      'width',
      'height',
    ].map((key) => json[key]).toList();
    if (values.any((v) => v is! num || !v.isFinite || v < 0) ||
        values[2] == 0 ||
        values[3] == 0) {
      throw const FormatException('Invalid detection box');
    }
    return DetectionBox(
      values[0].toDouble(),
      values[1].toDouble(),
      values[2].toDouble(),
      values[3].toDouble(),
    );
  }
}

// Only this projection crosses into the Lite application and persistence layers.
class ScanResult {
  ScanResult({
    required this.requestId,
    required this.status,
    required Iterable<String> reasons,
    required Iterable<ObjectResult> objects,
    required this.processingMs,
    required Map<String, String?> versions,
    this.origin = 'worker',
  }) : reasons = List.unmodifiable(reasons),
       objects = List.unmodifiable(objects),
       versions = Map.unmodifiable(versions);
  final String requestId;
  final String status;
  final List<String> reasons;
  final List<ObjectResult> objects;
  final double? processingMs;
  final Map<String, String?> versions;
  final String origin;

  factory ScanResult.fromWorker(Map<String, dynamic> json) {
    final status = json['status'];
    final id = json['request_id'];
    final time = json['processing_time_ms'];
    final segments = json['segmentations'];
    if (!finalStates.contains(status) ||
        id is! String ||
        id.isEmpty ||
        time is! num ||
        !time.isFinite ||
        time < 0 ||
        segments is! List ||
        (status == 'SEGMENTATION' ? segments.isEmpty : segments.isNotEmpty)) {
      throw const FormatException('Invalid Worker result');
    }
    final versions = <String, String?>{};
    for (final key in versionFields) {
      final value = json[key];
      if (value != null && value != workerVersion) {
        throw const FormatException('Worker version mismatch');
      }
      versions[key] = value as String?;
    }
    if (versions['worker_version'] != workerVersion) {
      throw const FormatException('Worker version missing');
    }
    final objects = segments.map((dynamic item) {
      if (item is! Map || !objectStates.contains(item['status'])) {
        throw const FormatException('Invalid object status');
      }
      return ObjectResult(
        item['status'] as String,
        readReasons(item['reason_codes']),
        box: DetectionBox.fromJson(item['bbox']),
        top3: readTop3(item['top3']),
      );
    });
    return ScanResult(
      requestId: id,
      status: status,
      reasons: readReasons(json['reason_codes']),
      objects: objects,
      processingMs: time.toDouble(),
      versions: versions,
    );
  }

  factory ScanResult.clientError(String attemptId, String reason) => ScanResult(
    requestId: attemptId,
    status: 'ERROR',
    reasons: [reason],
    objects: const [],
    processingMs: null,
    versions: const {},
    origin: 'client',
  );

  Map<String, Object?> toJson() => {
    'request_id': requestId,
    'status': status,
    'reason_codes': reasons,
    'segmentations': objects.map((o) => o.toJson()).toList(),
    'processing_time_ms': processingMs,
    'versions': versions,
    'origin': origin,
  };
}
