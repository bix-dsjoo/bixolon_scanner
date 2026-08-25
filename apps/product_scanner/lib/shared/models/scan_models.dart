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

enum OperatorReviewVerdict {
  accepted,
  corrected,
  recaptureAgreed,
  recaptureUnnecessary,
  recaptureRequired,
}

enum OperatorObjectDisposition { keep, correct, add, remove }

enum OperatorIssueCode {
  bboxIncorrect,
  missedObject,
  falsePositiveObject,
  wrongTop1,
  candidateMissing,
  unnecessarySegmentRecapture,
  unnecessaryRecapture,
  missedRecapture,
}

enum ReviewTool { select, editBox, addBox }

enum ReviewOverlayMode { model, review, compare }

String operatorReviewVerdictValue(OperatorReviewVerdict value) =>
    switch (value) {
      OperatorReviewVerdict.accepted => 'ACCEPTED',
      OperatorReviewVerdict.corrected => 'CORRECTED',
      OperatorReviewVerdict.recaptureAgreed => 'RECAPTURE_AGREED',
      OperatorReviewVerdict.recaptureUnnecessary => 'RECAPTURE_UNNECESSARY',
      OperatorReviewVerdict.recaptureRequired => 'RECAPTURE_REQUIRED',
    };

OperatorReviewVerdict operatorReviewVerdictFromValue(String value) =>
    switch (value) {
      'ACCEPTED' => OperatorReviewVerdict.accepted,
      'CORRECTED' => OperatorReviewVerdict.corrected,
      'RECAPTURE_AGREED' => OperatorReviewVerdict.recaptureAgreed,
      'RECAPTURE_UNNECESSARY' => OperatorReviewVerdict.recaptureUnnecessary,
      'RECAPTURE_REQUIRED' => OperatorReviewVerdict.recaptureRequired,
      _ => throw const FormatException('지원하지 않는 operator verdict입니다.'),
    };

String operatorObjectDispositionValue(OperatorObjectDisposition value) =>
    switch (value) {
      OperatorObjectDisposition.keep => 'KEEP',
      OperatorObjectDisposition.correct => 'CORRECT',
      OperatorObjectDisposition.add => 'ADD',
      OperatorObjectDisposition.remove => 'REMOVE',
    };

OperatorObjectDisposition operatorObjectDispositionFromValue(String value) =>
    switch (value) {
      'KEEP' => OperatorObjectDisposition.keep,
      'CORRECT' => OperatorObjectDisposition.correct,
      'ADD' => OperatorObjectDisposition.add,
      'REMOVE' => OperatorObjectDisposition.remove,
      _ => throw const FormatException('지원하지 않는 object disposition입니다.'),
    };

String operatorIssueCodeValue(OperatorIssueCode value) => switch (value) {
  OperatorIssueCode.bboxIncorrect => 'OPERATOR_BBOX_INCORRECT',
  OperatorIssueCode.missedObject => 'OPERATOR_MISSED_OBJECT',
  OperatorIssueCode.falsePositiveObject => 'OPERATOR_FALSE_POSITIVE_OBJECT',
  OperatorIssueCode.wrongTop1 => 'OPERATOR_WRONG_TOP1',
  OperatorIssueCode.candidateMissing => 'OPERATOR_CANDIDATE_MISSING',
  OperatorIssueCode.unnecessarySegmentRecapture =>
    'OPERATOR_UNNECESSARY_SEGMENT_RECAPTURE',
  OperatorIssueCode.unnecessaryRecapture => 'OPERATOR_UNNECESSARY_RECAPTURE',
  OperatorIssueCode.missedRecapture => 'OPERATOR_MISSED_RECAPTURE',
};

OperatorIssueCode operatorIssueCodeFromValue(String value) => switch (value) {
  'OPERATOR_BBOX_INCORRECT' => OperatorIssueCode.bboxIncorrect,
  'OPERATOR_MISSED_OBJECT' => OperatorIssueCode.missedObject,
  'OPERATOR_FALSE_POSITIVE_OBJECT' => OperatorIssueCode.falsePositiveObject,
  'OPERATOR_WRONG_TOP1' => OperatorIssueCode.wrongTop1,
  'OPERATOR_CANDIDATE_MISSING' => OperatorIssueCode.candidateMissing,
  'OPERATOR_UNNECESSARY_SEGMENT_RECAPTURE' =>
    OperatorIssueCode.unnecessarySegmentRecapture,
  'OPERATOR_UNNECESSARY_RECAPTURE' => OperatorIssueCode.unnecessaryRecapture,
  'OPERATOR_MISSED_RECAPTURE' => OperatorIssueCode.missedRecapture,
  _ => throw const FormatException('지원하지 않는 operator issue code입니다.'),
};

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

  bool get isValid => width > 0 && height > 0 && x >= 0 && y >= 0;

  @override
  bool operator ==(Object other) =>
      other is BoundingBox &&
      other.x == x &&
      other.y == y &&
      other.width == width &&
      other.height == height;

  @override
  int get hashCode => Object.hash(x, y, width, height);
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

class WorkerStageTimings {
  const WorkerStageTimings({
    required this.requestTotalMs,
    required this.uploadReadMs,
    required this.decodeMs,
    required this.queueWaitMs,
    required this.pipelineMs,
    required this.detectorMs,
    required this.classifierMs,
    required this.decisionMs,
    required this.serverOverheadMs,
  });

  factory WorkerStageTimings.fromJson(Map<String, dynamic> json) =>
      WorkerStageTimings(
        requestTotalMs: (json['request_total_ms'] as num).toDouble(),
        uploadReadMs: (json['upload_read_ms'] as num).toDouble(),
        decodeMs: (json['decode_ms'] as num).toDouble(),
        queueWaitMs: (json['queue_wait_ms'] as num).toDouble(),
        pipelineMs: (json['pipeline_ms'] as num).toDouble(),
        detectorMs: (json['detector_ms'] as num).toDouble(),
        classifierMs: (json['classifier_ms'] as num).toDouble(),
        decisionMs: (json['decision_ms'] as num).toDouble(),
        serverOverheadMs: (json['server_overhead_ms'] as num).toDouble(),
      );

  final double requestTotalMs;
  final double uploadReadMs;
  final double decodeMs;
  final double queueWaitMs;
  final double pipelineMs;
  final double detectorMs;
  final double classifierMs;
  final double decisionMs;
  final double serverOverheadMs;

  Map<String, dynamic> toJson() => {
    'request_total_ms': requestTotalMs,
    'upload_read_ms': uploadReadMs,
    'decode_ms': decodeMs,
    'queue_wait_ms': queueWaitMs,
    'pipeline_ms': pipelineMs,
    'detector_ms': detectorMs,
    'classifier_ms': classifierMs,
    'decision_ms': decisionMs,
    'server_overhead_ms': serverOverheadMs,
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
    this.stageTimings,
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
      stageTimings: switch (json['stage_timings_ms']) {
        final Map<String, dynamic> value => WorkerStageTimings.fromJson(value),
        _ => null,
      },
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
  final WorkerStageTimings? stageTimings;
}

class ReviewDetection {
  ReviewDetection({
    required this.source,
    required this.state,
    required this.finalProduct,
    required this.confirmationMethod,
    BoundingBox? finalBbox,
    this.operatorAdded = false,
    this.removed = false,
  }) : finalBbox = finalBbox ?? source.bbox;

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

  factory ReviewDetection.operatorAdded({
    required String objectId,
    required BoundingBox bbox,
  }) => ReviewDetection(
    source: ScanItem(
      itemId: objectId,
      bbox: bbox,
      status: ItemStatus.unknown,
      reasonCodes: const [],
      prediction: null,
      top3: const [],
      confidence: 0,
    ),
    state: DetectionState.top3Candidates,
    finalProduct: null,
    confirmationMethod: null,
    finalBbox: bbox,
    operatorAdded: true,
  );

  final ScanItem source;
  DetectionState state;
  Product? finalProduct;
  ConfirmationMethod? confirmationMethod;
  BoundingBox finalBbox;
  final bool operatorAdded;
  bool removed;

  bool get isConfirmed =>
      removed || (finalBbox.isValid && finalProduct != null);
  bool get bboxChanged => operatorAdded || finalBbox != source.bbox;
  bool get wasUserChanged =>
      operatorAdded ||
      removed ||
      bboxChanged ||
      (confirmationMethod != null &&
          confirmationMethod != ConfirmationMethod.autoApproved);

  OperatorObjectDisposition get disposition {
    if (removed) return OperatorObjectDisposition.remove;
    if (operatorAdded) return OperatorObjectDisposition.add;
    if (wasUserChanged) return OperatorObjectDisposition.correct;
    return OperatorObjectDisposition.keep;
  }

  Set<OperatorIssueCode> get inferredIssueCodes {
    final values = <OperatorIssueCode>{};
    if (operatorAdded) values.add(OperatorIssueCode.missedObject);
    if (removed && !operatorAdded) {
      values.add(OperatorIssueCode.falsePositiveObject);
    }
    if (!operatorAdded && finalBbox != source.bbox) {
      values.add(OperatorIssueCode.bboxIncorrect);
    }
    final modelTop1 =
        source.prediction?.classId ??
        (source.top3.isEmpty ? null : source.top3.first.classId);
    final finalClassId = finalProduct?.classId;
    if (!removed &&
        !operatorAdded &&
        modelTop1 != null &&
        finalClassId != null &&
        modelTop1 != finalClassId) {
      values.add(OperatorIssueCode.wrongTop1);
      if (source.top3.every((candidate) => candidate.classId != finalClassId)) {
        values.add(OperatorIssueCode.candidateMissing);
      }
    }
    if (!removed &&
        source.status == ItemStatus.segmentRecapture &&
        finalProduct != null) {
      values.add(OperatorIssueCode.unnecessarySegmentRecapture);
    }
    return values;
  }

  ReviewDetection copy() => ReviewDetection(
    source: source,
    state: state,
    finalProduct: finalProduct,
    confirmationMethod: confirmationMethod,
    finalBbox: finalBbox,
    operatorAdded: operatorAdded,
    removed: removed,
  );

  Map<String, dynamic> toOperatorObjectJson() => {
    'object_id': source.itemId,
    'source_detection_id': operatorAdded ? null : source.itemId,
    'disposition': operatorObjectDispositionValue(disposition),
    'final_bbox': removed ? null : finalBbox.toJson(),
    'final_product': removed ? null : finalProduct?.toJson(),
    'issue_codes': inferredIssueCodes
        .map(operatorIssueCodeValue)
        .toList(growable: false),
  };

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

class OperatorReviewObject {
  const OperatorReviewObject({
    required this.objectId,
    required this.sourceDetectionId,
    required this.disposition,
    required this.finalBbox,
    required this.finalProduct,
    required this.issueCodes,
  });

  factory OperatorReviewObject.fromDetection(ReviewDetection detection) =>
      OperatorReviewObject(
        objectId: detection.source.itemId,
        sourceDetectionId: detection.operatorAdded
            ? null
            : detection.source.itemId,
        disposition: detection.disposition,
        finalBbox: detection.removed ? null : detection.finalBbox,
        finalProduct: detection.removed ? null : detection.finalProduct,
        issueCodes: detection.inferredIssueCodes,
      );

  factory OperatorReviewObject.fromJson(Map<String, dynamic> json) =>
      OperatorReviewObject(
        objectId: json['object_id'] as String,
        sourceDetectionId: json['source_detection_id'] as String?,
        disposition: operatorObjectDispositionFromValue(
          json['disposition'] as String,
        ),
        finalBbox: switch (json['final_bbox']) {
          final Map<String, dynamic> value => BoundingBox.fromJson(value),
          _ => null,
        },
        finalProduct: switch (json['final_product']) {
          final Map<String, dynamic> value => Product.fromJson(value),
          _ => null,
        },
        issueCodes: switch (json['issue_codes']) {
          final List values =>
            values.whereType<String>().map(operatorIssueCodeFromValue).toSet(),
          _ => const <OperatorIssueCode>{},
        },
      );

  final String objectId;
  final String? sourceDetectionId;
  final OperatorObjectDisposition disposition;
  final BoundingBox? finalBbox;
  final Product? finalProduct;
  final Set<OperatorIssueCode> issueCodes;

  Map<String, dynamic> toJson() => {
    'object_id': objectId,
    'source_detection_id': sourceDetectionId,
    'disposition': operatorObjectDispositionValue(disposition),
    'final_bbox': finalBbox?.toJson(),
    'final_product': finalProduct?.toJson(),
    'issue_codes': issueCodes
        .map(operatorIssueCodeValue)
        .toList(growable: false),
  };
}

class OperatorReview {
  const OperatorReview({
    required this.verdict,
    required this.reviewedAt,
    required this.inferredIssueCodes,
    required this.issueCodes,
    required this.objects,
    this.note,
  });

  factory OperatorReview.fromJson(Map<String, dynamic> json) => OperatorReview(
    verdict: operatorReviewVerdictFromValue(json['verdict'] as String),
    reviewedAt: DateTime.parse(json['reviewed_at'] as String),
    inferredIssueCodes: _operatorIssueCodesFromJson(
      json['inferred_issue_codes'],
    ),
    issueCodes: _operatorIssueCodesFromJson(json['issue_codes']),
    note: json['note'] as String?,
    objects: switch (json['objects']) {
      final List values =>
        values
            .whereType<Map<String, dynamic>>()
            .map(OperatorReviewObject.fromJson)
            .toList(growable: false),
      _ => const <OperatorReviewObject>[],
    },
  );

  final OperatorReviewVerdict verdict;
  final DateTime reviewedAt;
  final Set<OperatorIssueCode> inferredIssueCodes;
  final Set<OperatorIssueCode> issueCodes;
  final String? note;
  final List<OperatorReviewObject> objects;

  Map<String, dynamic> toJson() => {
    'verdict': operatorReviewVerdictValue(verdict),
    'reviewed_at': reviewedAt.toUtc().toIso8601String(),
    'inferred_issue_codes': inferredIssueCodes
        .map(operatorIssueCodeValue)
        .toList(growable: false),
    'issue_codes': issueCodes
        .map(operatorIssueCodeValue)
        .toList(growable: false),
    if (note case final value? when value.trim().isNotEmpty)
      'note': value.trim(),
    'objects': objects.map((object) => object.toJson()).toList(growable: false),
  };
}

Set<OperatorIssueCode> _operatorIssueCodesFromJson(Object? value) =>
    switch (value) {
      final List values =>
        values.whereType<String>().map(operatorIssueCodeFromValue).toSet(),
      _ => const <OperatorIssueCode>{},
    };
