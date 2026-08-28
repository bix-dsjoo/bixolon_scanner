part of 'scan_models.dart';

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

String confirmationMethodValue(ConfirmationMethod value) => switch (value) {
  ConfirmationMethod.autoApproved => 'AUTO_APPROVED',
  ConfirmationMethod.top3Selected => 'TOP3_SELECTED',
  ConfirmationMethod.searchSelected => 'SEARCH_SELECTED',
  ConfirmationMethod.userCorrected => 'USER_CORRECTED',
};

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
