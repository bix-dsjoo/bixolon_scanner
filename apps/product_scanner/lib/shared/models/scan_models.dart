import 'dart:convert';

enum ScanStatus { approved, unknown, recapture, error }

enum ItemStatus { approved, unknown, segmentRecapture }

enum InputMode { camera, image }

enum ProcessState { ready, capturing, analyzing, reviewing, submitting, error }

enum DetectionState { autoApproved, top3Candidates, confirmed }

enum ConfirmationMethod {
  autoApproved,
  top3Selected,
  searchSelected,
  userCorrected,
}

ScanStatus _scanStatus(String value) => switch (value) {
  'APPROVED' => ScanStatus.approved,
  'UNKNOWN' => ScanStatus.unknown,
  'RECAPTURE' => ScanStatus.recapture,
  'ERROR' => ScanStatus.error,
  _ => throw const FormatException('지원하지 않는 scan status입니다.'),
};

ItemStatus _itemStatus(String value) => switch (value) {
  'APPROVED' => ItemStatus.approved,
  'UNKNOWN' => ItemStatus.unknown,
  'SEGMENT_RECAPTURE' => ItemStatus.segmentRecapture,
  _ => throw const FormatException('지원하지 않는 item status입니다.'),
};

String confirmationMethodValue(ConfirmationMethod value) => switch (value) {
  ConfirmationMethod.autoApproved => 'AUTO_APPROVED',
  ConfirmationMethod.top3Selected => 'TOP3_SELECTED',
  ConfirmationMethod.searchSelected => 'SEARCH_SELECTED',
  ConfirmationMethod.userCorrected => 'USER_CORRECTED',
};

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

  Map<String, dynamic> toJson() => {
    'x': x,
    'y': y,
    'width': width,
    'height': height,
  };
}

class Product {
  const Product({
    required this.classId,
    required this.className,
    required this.displayName,
  });

  factory Product.fromJson(Map<String, dynamic> json) => Product(
    classId: json['class_id'] as String,
    className: json['class_name'] as String,
    displayName: json['class_name'] as String,
  );

  final String classId;
  final String className;
  final String displayName;

  Product withDisplayName(String value) =>
      Product(classId: classId, className: className, displayName: value);

  Map<String, dynamic> toJson() => {
    'class_id': classId,
    'class_name': className,
  };
}

class Candidate extends Product {
  const Candidate({
    required super.classId,
    required super.className,
    required super.displayName,
    required this.confidence,
  });

  factory Candidate.fromJson(Map<String, dynamic> json) => Candidate(
    classId: json['class_id'] as String,
    className: json['class_name'] as String,
    displayName: json['class_name'] as String,
    confidence: (json['confidence'] as num).toDouble(),
  );

  final double confidence;

  @override
  Candidate withDisplayName(String value) => Candidate(
    classId: classId,
    className: className,
    displayName: value,
    confidence: confidence,
  );

  @override
  Map<String, dynamic> toJson() => {
    ...super.toJson(),
    'confidence': confidence,
  };
}

class ScanItem {
  const ScanItem({
    required this.itemId,
    required this.bbox,
    required this.status,
    required this.reasonCodes,
    required this.prediction,
    required this.top3,
    required this.confidence,
  });

  factory ScanItem.fromJson(Map<String, dynamic> json) => ScanItem(
    itemId: (json['segmentation_id'] ?? json['item_id']) as String,
    bbox: BoundingBox.fromJson(json['bbox'] as Map<String, dynamic>),
    status: _itemStatus(json['status'] as String),
    reasonCodes: List<String>.from(json['reason_codes'] as List),
    prediction: json['prediction'] == null
        ? null
        : Product.fromJson(json['prediction'] as Map<String, dynamic>),
    top3: (json['top3'] as List)
        .map((value) => Candidate.fromJson(value as Map<String, dynamic>))
        .toList(growable: false),
    confidence: (json['confidence'] as num).toDouble(),
  );

  final String itemId;
  final BoundingBox bbox;
  final ItemStatus status;
  final List<String> reasonCodes;
  final Product? prediction;
  final List<Candidate> top3;
  final double confidence;
}

class ModelVersions {
  const ModelVersions({
    this.worker,
    this.detector,
    this.classifier,
    this.embedder,
    this.detectorPolicy,
    this.classifierPolicy,
    this.catalog,
  });

  factory ModelVersions.fromJson(Map<String, dynamic> json) => ModelVersions(
    worker: json['worker'] as String?,
    detector: json['detector'] as String?,
    classifier: json['classifier'] as String?,
    embedder: json['embedder'] as String?,
    detectorPolicy: json['detector_policy'] as String?,
    classifierPolicy: json['classifier_policy'] as String?,
    catalog: json['catalog'] as String?,
  );

  final String? worker;
  final String? detector;
  final String? classifier;
  final String? embedder;
  final String? detectorPolicy;
  final String? classifierPolicy;
  final String? catalog;

  Map<String, dynamic> toJson() => {
    'worker': worker,
    'detector': detector,
    'classifier': classifier,
    'embedder': embedder,
    'detector_policy': detectorPolicy,
    'classifier_policy': classifierPolicy,
    'catalog': catalog,
  };
}

class ScanResponse {
  const ScanResponse({
    required this.requestId,
    required this.status,
    required this.reasonCodes,
    required this.items,
    required this.processingTimeMs,
    required this.modelVersions,
  });

  factory ScanResponse.fromJson(Map<String, dynamic> json) {
    final rawItems = (json['segmentations'] ?? json['items']) as List;
    final items = rawItems
        .map((value) => ScanItem.fromJson(value as Map<String, dynamic>))
        .toList(growable: false);
    final rawStatus = json['status'] as String;
    final status = switch (rawStatus) {
      'SEGMENTATION'
          when items.any((item) => item.status != ItemStatus.approved) =>
        ScanStatus.unknown,
      'SEGMENTATION' => ScanStatus.approved,
      'IMAGE_RECAPTURE' => ScanStatus.recapture,
      _ => _scanStatus(rawStatus),
    };
    final versions = switch (json['model_versions']) {
      final Map<String, dynamic> value => ModelVersions.fromJson(value),
      _ => ModelVersions(
        worker: json['worker_version'] as String?,
        detector: json['detector_version'] as String?,
        classifier: json['classifier_version'] as String?,
        embedder: json['embedder_version'] as String?,
        detectorPolicy: json['detector_policy_version'] as String?,
        classifierPolicy: json['classifier_policy_version'] as String?,
        catalog: json['catalog_version'] as String?,
      ),
    };
    return ScanResponse(
      requestId: json['request_id'] as String,
      status: status,
      reasonCodes: List<String>.from(json['reason_codes'] as List),
      items: items,
      processingTimeMs: (json['processing_time_ms'] as num).toDouble(),
      modelVersions: versions,
    );
  }

  factory ScanResponse.fromBody(String body) {
    final decoded = jsonDecode(body);
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('Worker 응답이 JSON object가 아닙니다.');
    }
    return ScanResponse.fromJson(decoded);
  }

  final String requestId;
  final ScanStatus status;
  final List<String> reasonCodes;
  final List<ScanItem> items;
  final double processingTimeMs;
  final ModelVersions modelVersions;
}

class ReviewDetection {
  ReviewDetection({
    required this.source,
    required this.state,
    required this.finalProduct,
    required this.confirmationMethod,
  });

  factory ReviewDetection.fromScanItem(ScanItem item) => ReviewDetection(
    source: item,
    state: item.status == ItemStatus.approved
        ? DetectionState.autoApproved
        : DetectionState.top3Candidates,
    finalProduct: item.prediction,
    confirmationMethod: item.status == ItemStatus.approved
        ? ConfirmationMethod.autoApproved
        : null,
  );

  final ScanItem source;
  DetectionState state;
  Product? finalProduct;
  ConfirmationMethod? confirmationMethod;

  bool get isConfirmed => finalProduct != null;
  bool get wasUserChanged =>
      confirmationMethod != null &&
      confirmationMethod != ConfirmationMethod.autoApproved;

  Map<String, dynamic> toLogJson() => {
    'detection_id': source.itemId,
    'bbox': source.bbox.toJson(),
    'initial_ai_status': switch (source.status) {
      ItemStatus.approved => 'AUTO_APPROVED',
      ItemStatus.unknown => 'TOP3_CANDIDATES',
      ItemStatus.segmentRecapture => 'SEGMENT_RECAPTURE',
    },
    'initial_ai_prediction': source.prediction?.toJson(),
    'initial_confidence': source.confidence,
    'reason_codes': source.reasonCodes,
    'top3': source.top3.map((candidate) => candidate.toJson()).toList(),
    'final_product': finalProduct?.toJson(),
    'user_modified': wasUserChanged,
    'confirmation_method': confirmationMethod == null
        ? null
        : confirmationMethodValue(confirmationMethod!),
  };
}
