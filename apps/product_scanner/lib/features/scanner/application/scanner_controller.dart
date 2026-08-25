import 'dart:async';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';

import '../../../core/design_system/tokens.dart';
import '../../../shared/catalog/product_catalog.dart';
import '../../../shared/logging/scan_log_repository.dart';
import '../../../shared/models/scan_models.dart';
import '../../../shared/presentation/recapture_presentation.dart';
import '../data/image_input.dart';
import '../data/scanner_api.dart';

enum CameraIssueType { unavailable, captureFailed }

@Deprecated('단일 검수 저장 상태는 processState를 사용합니다.')
enum RecaptureLogSaveState { idle, saving, saved, error }

@Deprecated('단일 검수 저장 상태는 processState를 사용합니다.')
enum MissedDetectionLogSaveState { idle, saving, saved, error }

class ScannerController extends ChangeNotifier {
  ScannerController(
    this._scannerApi,
    this._cameraGateway,
    this._imageFileGateway,
    this._scanLogRepository,
    this._catalog, {
    this.completionFeedbackDuration = AppMotion.feedbackHold,
  }) {
    final api = _scannerApi;
    final preflight = api is ScannerApiPreflight
        ? api as ScannerApiPreflight
        : null;
    workerReady = preflight == null || preflight.isPrepared;
    workerInitializing = !workerReady;
  }

  final ScannerApi _scannerApi;
  final CameraGateway _cameraGateway;
  final ImageFileGateway _imageFileGateway;
  final ScanLogRepository _scanLogRepository;
  final ProductCatalog _catalog;
  final Duration completionFeedbackDuration;
  Timer? _completionFeedbackTimer;
  Stopwatch? _resultFirstFrameStopwatch;
  double _performanceBeforeFirstFrameMs = 0.0;
  double _cameraCaptureMs = 0.0;
  double _fileReadMs = 0.0;
  double _flutterImageDecodeMs = 0.0;
  Future<void>? _imageDecodeFuture;
  var _imageGeneration = 0;
  bool _decodeOverlappedWithAnalysis = false;
  Future<void>? _workerPreparation;

  InputMode inputMode = InputMode.camera;
  ProcessState processState = ProcessState.ready;
  ScanResponse? response;
  List<ReviewDetection> detections = [];
  Uint8List? imageBytes;
  String? imageFileName;
  Size? imageSize;
  DateTime? analyzedAt;
  String? selectedItemId;
  String? searchItemId;
  String searchQuery = '';
  String? errorMessage;
  ScannerErrorRecovery errorRecovery = ScannerErrorRecovery.retryAnalysis;
  ReviewTool reviewTool = ReviewTool.select;
  ReviewOverlayMode overlayMode = ReviewOverlayMode.compare;
  bool operatorRequiresRecapture = false;
  Set<OperatorIssueCode> reviewIssueCodes = <OperatorIssueCode>{};
  String reviewNote = '';
  @Deprecated('processState를 사용합니다.')
  RecaptureLogSaveState recaptureLogSaveState = RecaptureLogSaveState.idle;
  @Deprecated('errorMessage를 사용합니다.')
  String? recaptureLogError;
  @Deprecated('processState를 사용합니다.')
  MissedDetectionLogSaveState missedDetectionLogSaveState =
      MissedDetectionLogSaveState.idle;
  @Deprecated('errorMessage를 사용합니다.')
  String? missedDetectionLogError;
  String? completionMessage;
  ScanPerformanceMetrics? performanceMetrics;
  int _activityDataRevision = 0;
  String? _latestSavedScanId;
  final List<_ReviewSnapshot> _undoStack = <_ReviewSnapshot>[];
  final List<_ReviewSnapshot> _redoStack = <_ReviewSnapshot>[];
  final Set<String> _savedRecaptureScanIds = <String>{};
  final Set<String> _savedMissedDetectionScanIds = <String>{};
  _ReviewSnapshot? _activeBoxEditStart;
  var _nextOperatorObjectNumber = 1;
  bool cameraInitializing = true;
  bool workerInitializing = false;
  bool workerReady = true;
  String? workerMessage;
  String? cameraMessage;
  CameraIssueType? cameraIssueType;
  bool _imageSelectionInProgress = false;
  bool _disposed = false;

  CameraController? get cameraController => _cameraGateway.controller;
  bool get isCameraReady => _cameraGateway.isReady;
  bool get hasActiveCameraIssue =>
      inputMode == InputMode.camera && cameraIssueType != null;
  bool get isCameraCheckActive =>
      inputMode == InputMode.camera && cameraInitializing;
  String get cameraIssueTitle => switch (cameraIssueType) {
    CameraIssueType.unavailable => '카메라를 사용할 수 없어요',
    CameraIssueType.captureFailed => '촬영하지 못했어요',
    null => '카메라를 확인해 주세요',
  };
  bool get isBusy =>
      processState == ProcessState.capturing ||
      processState == ProcessState.analyzing ||
      processState == ProcessState.submitting ||
      recaptureLogSaveState == RecaptureLogSaveState.saving ||
      missedDetectionLogSaveState == MissedDetectionLogSaveState.saving;
  int get activityDataRevision => _activityDataRevision;
  String? get latestSavedScanId => _latestSavedScanId;
  bool get isChoosingImage => _imageSelectionInProgress;
  bool get canChooseImage => !isBusy && !_imageSelectionInProgress;
  bool get canAnalyze =>
      !isBusy && workerReady && imageBytes != null && imageFileName != null;
  bool get isRecapture => response?.status == ScanStatus.recapture;
  bool get hasResults => response != null;
  List<ReviewDetection> get activeDetections => detections
      .where((detection) => !detection.removed)
      .toList(growable: false);
  int get confirmedCount =>
      activeDetections.where((detection) => detection.isConfirmed).length;
  int get incompleteCount =>
      activeDetections.where((detection) => !detection.isConfirmed).length;
  bool get allConfirmed => activeDetections.isNotEmpty && incompleteCount == 0;
  bool get canSaveReview =>
      !isBusy &&
      processState == ProcessState.reviewing &&
      response != null &&
      imageBytes != null &&
      imageFileName != null &&
      (operatorRequiresRecapture || allConfirmed);
  @Deprecated('박스를 추가한 뒤 단일 검수 저장을 사용합니다.')
  bool get canSaveMissedDetectionLog {
    final status = response?.status;
    return !isBusy &&
        imageBytes != null &&
        imageFileName != null &&
        detections.isNotEmpty &&
        (status == ScanStatus.approved || status == ScanStatus.unknown) &&
        missedDetectionLogSaveState != MissedDetectionLogSaveState.saved;
  }

  bool get hasUnsavedReview =>
      processState == ProcessState.reviewing && response != null;
  bool get hasUserChanges =>
      detections.any((detection) => detection.wasUserChanged) ||
      operatorRequiresRecapture != isRecapture;
  bool get canUndoReviewEdit => _undoStack.isNotEmpty;
  bool get canRedoReviewEdit => _redoStack.isNotEmpty;
  Set<OperatorIssueCode> get inferredIssueCodes {
    final values = <OperatorIssueCode>{
      for (final detection in detections) ...detection.inferredIssueCodes,
    };
    if (isRecapture && !operatorRequiresRecapture) {
      values.add(OperatorIssueCode.unnecessaryRecapture);
    } else if (!isRecapture && operatorRequiresRecapture) {
      values.add(OperatorIssueCode.missedRecapture);
    }
    return values;
  }

  ReviewDetection? get selectedDetection {
    final selected = selectedItemId;
    if (selected == null) return null;
    for (final detection in detections) {
      if (detection.source.itemId == selected) return detection;
    }
    return null;
  }

  int get selectedIndex {
    final selected = selectedItemId;
    if (selected == null) return -1;
    return detections.indexWhere(
      (detection) => detection.source.itemId == selected,
    );
  }

  List<Product> get searchResults => _catalog.search(searchQuery);

  Future<List<ScanLogSummary>> loadScanLogs() async {
    final logs = await _scanLogRepository.list();
    return logs
        .map(
          (log) => ScanLogSummary(
            scanId: log.scanId,
            analyzedAt: log.analyzedAt,
            confirmedAt: log.confirmedAt,
            recordedAt: log.recordedAt,
            inputMode: log.inputMode,
            processingTimeMs: log.processingTimeMs,
            modelVersions: log.modelVersions,
            workerStatus: log.workerStatus,
            reasonCodes: log.reasonCodes,
            originalImagePath: log.originalImagePath,
            performance: log.performance,
            logSchemaVersion: log.logSchemaVersion,
            operatorReview: log.operatorReview,
            recordFilePath: log.recordFilePath,
            items: log.items
                .map(
                  (item) => item.withProductName(
                    _catalog.displayNameFor(
                      classId: item.classId,
                      className: item.className,
                      fallback: item.productName,
                    ),
                    localizedModelProduct: item.modelProduct == null
                        ? null
                        : _catalog.localize(item.modelProduct!),
                  ),
                )
                .toList(growable: false),
          ),
        )
        .toList(growable: false);
  }

  Future<void> exportReviewArchive({
    required String targetPath,
    List<ScanLogSummary>? records,
  }) async {
    final repository = _scanLogRepository;
    if (repository is! FileScanLogRepository) {
      throw UnsupportedError('현재 저장소는 ZIP 내보내기를 지원하지 않습니다.');
    }
    await repository.exportReviewArchive(
      targetPath: targetPath,
      records: records,
    );
  }

  Candidate localizeCandidate(Candidate candidate) =>
      _catalog.localizeCandidate(candidate);

  Future<void> initialize() async {
    cameraInitializing = true;
    cameraMessage = null;
    cameraIssueType = null;
    _notify();
    final workerPreparation = prepareWorker();
    try {
      await _cameraGateway.initialize();
      cameraMessage = null;
      cameraIssueType = null;
    } catch (_) {
      cameraMessage = '카메라를 사용할 수 없어요. 연결 상태를 확인해 주세요.';
      cameraIssueType = CameraIssueType.unavailable;
    } finally {
      cameraInitializing = false;
      _notify();
    }
    await workerPreparation;
  }

  Future<void> prepareWorker() {
    final api = _scannerApi;
    if (api is! ScannerApiPreflight) {
      workerInitializing = false;
      workerReady = true;
      workerMessage = null;
      return Future<void>.value();
    }
    final preflight = api as ScannerApiPreflight;
    if (preflight.isPrepared) {
      workerInitializing = false;
      workerReady = true;
      workerMessage = null;
      return Future<void>.value();
    }
    final active = _workerPreparation;
    if (active != null) return active;
    final preparation = _prepareWorker(preflight);
    _workerPreparation = preparation;
    return preparation;
  }

  Future<void> _prepareWorker(ScannerApiPreflight api) async {
    workerInitializing = true;
    workerReady = false;
    workerMessage = null;
    _notify();
    try {
      await api.prepare();
      workerReady = true;
    } catch (_) {
      workerReady = false;
      workerMessage = '분석 엔진을 준비하지 못했어요. 연결 상태를 확인해 주세요.';
    } finally {
      workerInitializing = false;
      _workerPreparation = null;
      _notify();
    }
  }

  Future<void> reconnectCamera() async {
    if (cameraInitializing || isBusy) return;
    cameraInitializing = true;
    cameraMessage = null;
    cameraIssueType = null;
    _notify();
    try {
      await _cameraGateway.initialize();
      cameraIssueType = null;
    } catch (_) {
      cameraMessage = '카메라를 사용할 수 없어요. 연결 상태를 확인해 주세요.';
      cameraIssueType = CameraIssueType.unavailable;
    } finally {
      cameraInitializing = false;
      _notify();
    }
  }

  /// Restores the live camera after returning from another input or after the
  /// desktop app resumes. A forced restore replaces a controller that still
  /// reports initialized but may have lost its native device session.
  Future<void> restoreCamera({bool forceReconnect = false}) async {
    if (inputMode != InputMode.camera || cameraInitializing || isBusy) return;
    if (!forceReconnect && isCameraReady && cameraIssueType == null) return;
    await reconnectCamera();
  }

  Future<void> returnToCamera() async {
    if (isBusy) return;
    _resetSession();
    _notify();
    await restoreCamera();
  }

  Future<bool> chooseImage() async {
    if (!canChooseImage) return false;
    _imageSelectionInProgress = true;
    _notify();
    try {
      final selected = await _imageFileGateway.pick();
      if (selected == null) return false;
      await _setInputImage(selected, mode: InputMode.image);
      _clearResult();
      processState = ProcessState.ready;
      _notify();
      return true;
    } catch (_) {
      inputMode = InputMode.image;
      processState = ProcessState.error;
      errorMessage = '이미지를 열지 못했어요. 다른 이미지를 선택해 주세요.';
      errorRecovery = ScannerErrorRecovery.replaceInput;
      return true;
    } finally {
      _imageSelectionInProgress = false;
      _notify();
    }
  }

  Future<void> captureAndAnalyze() async {
    if (isBusy || !workerReady) return;
    processState = ProcessState.capturing;
    cameraMessage = null;
    cameraIssueType = null;
    _notify();
    try {
      final captured = await _cameraGateway.capture();
      await _setInputImage(
        captured,
        mode: InputMode.camera,
        waitForDecode: false,
      );
      processState = ProcessState.ready;
      await analyze();
    } catch (_) {
      processState = ProcessState.ready;
      cameraMessage = '카메라 응답을 받지 못했어요. 다시 연결해 주세요.';
      cameraIssueType = CameraIssueType.captureFailed;
      _notify();
    }
  }

  Future<void> analyze() async {
    final bytes = imageBytes;
    final fileName = imageFileName;
    if (bytes == null || fileName == null || isBusy || !workerReady) return;
    processState = ProcessState.analyzing;
    errorMessage = null;
    errorRecovery = ScannerErrorRecovery.retryAnalysis;
    response = null;
    detections = [];
    selectedItemId = null;
    searchItemId = null;
    _resetReviewState();
    performanceMetrics = null;
    _resultFirstFrameStopwatch = null;
    _notify();
    try {
      final apiStopwatch = Stopwatch()..start();
      final resultFuture = _scannerApi.scan(
        imageBytes: bytes,
        fileName: fileName,
      );
      final completed = await Future.wait<Object?>([
        resultFuture,
        _imageDecodeFuture ?? Future<void>.value(),
      ]);
      final result = completed.first! as ScanResponse;
      apiStopwatch.stop();
      final apiTimings = switch (_scannerApi) {
        final ScannerApiTimingSource source => source.takeLastTimings(),
        _ => null,
      };
      final mapping = Stopwatch()..start();
      response = result;
      analyzedAt = DateTime.now();
      detections = result.items
          .map((item) {
            final review = ReviewDetection.fromScanItem(item);
            final prediction = review.finalProduct;
            if (prediction != null) {
              review.finalProduct = _catalog.localize(prediction);
            }
            return review;
          })
          .toList(growable: false);
      operatorRequiresRecapture = result.status == ScanStatus.recapture;
      reviewIssueCodes = inferredIssueCodes;
      processState = ProcessState.reviewing;
      selectedItemId =
          _firstUnconfirmedId() ??
          (detections.isEmpty ? null : detections.first.source.itemId);
      mapping.stop();
      final imageDimensions = imageSize;
      final effectiveApiTotalMs =
          apiTimings?.totalMs ?? apiStopwatch.elapsedMicroseconds / 1000.0;
      final decodeAndApiMs = _decodeOverlappedWithAnalysis
          ? _flutterImageDecodeMs > effectiveApiTotalMs
                ? _flutterImageDecodeMs
                : effectiveApiTotalMs
          : _flutterImageDecodeMs + effectiveApiTotalMs;
      _performanceBeforeFirstFrameMs =
          _cameraCaptureMs +
          _fileReadMs +
          decodeAndApiMs +
          mapping.elapsedMicroseconds / 1000.0;
      performanceMetrics = ScanPerformanceMetrics(
        imageWidth: imageDimensions?.width.round() ?? 0,
        imageHeight: imageDimensions?.height.round() ?? 0,
        imageSizeBytes: bytes.length,
        provider: apiTimings?.provider,
        cameraCaptureMs: _cameraCaptureMs,
        fileReadMs: _fileReadMs,
        flutterImageDecodeMs: _flutterImageDecodeMs,
        readinessMs: apiTimings?.readinessMs ?? 0.0,
        requestBuildMs: apiTimings?.requestBuildMs ?? 0.0,
        httpRoundTripMs: apiTimings?.httpRoundTripMs ?? effectiveApiTotalMs,
        responseBodyReadMs: apiTimings?.responseBodyReadMs ?? 0.0,
        responseParseMs: apiTimings?.responseParseMs ?? 0.0,
        resultMappingMs: mapping.elapsedMicroseconds / 1000.0,
        resultFirstFrameMs: 0.0,
        endToEndMs: _performanceBeforeFirstFrameMs,
        worker: result.stageTimings,
      );
      _resultFirstFrameStopwatch = Stopwatch()..start();
    } on ScannerApiException catch (error) {
      processState = ProcessState.error;
      errorMessage = error.message;
      errorRecovery = error.recovery;
      final api = _scannerApi;
      final preflight = api is ScannerApiPreflight
          ? api as ScannerApiPreflight
          : null;
      if (preflight != null && !preflight.isPrepared) {
        workerReady = false;
        workerMessage = '분석 엔진 연결을 다시 확인해 주세요.';
      }
    } catch (_) {
      processState = ProcessState.error;
      errorMessage = '분석하지 못했어요. 잠시 후 다시 분석해 주세요.';
      errorRecovery = ScannerErrorRecovery.retryAnalysis;
    }
    _notify();
  }

  void recordResultFirstFrame(String requestId) {
    final current = performanceMetrics;
    final stopwatch = _resultFirstFrameStopwatch;
    if (response?.requestId != requestId ||
        current == null ||
        stopwatch == null) {
      return;
    }
    _resultFirstFrameStopwatch = null;
    stopwatch.stop();
    final renderMs = stopwatch.elapsedMicroseconds / 1000.0;
    performanceMetrics = ScanPerformanceMetrics(
      imageWidth: current.imageWidth,
      imageHeight: current.imageHeight,
      imageSizeBytes: current.imageSizeBytes,
      provider: current.provider,
      cameraCaptureMs: current.cameraCaptureMs,
      fileReadMs: current.fileReadMs,
      flutterImageDecodeMs: current.flutterImageDecodeMs,
      readinessMs: current.readinessMs,
      requestBuildMs: current.requestBuildMs,
      httpRoundTripMs: current.httpRoundTripMs,
      responseBodyReadMs: current.responseBodyReadMs,
      responseParseMs: current.responseParseMs,
      resultMappingMs: current.resultMappingMs,
      resultFirstFrameMs: renderMs,
      endToEndMs: _performanceBeforeFirstFrameMs + renderMs,
      worker: current.worker,
    );
  }

  void selectDetection(String itemId) {
    if (isBusy) return;
    selectedItemId = itemId;
    searchItemId = null;
    searchQuery = '';
    _notify();
  }

  void selectPreviousDetection() {
    _selectDetectionByOffset(-1);
  }

  void selectNextDetection() {
    _selectDetectionByOffset(1);
  }

  void _selectDetectionByOffset(int offset) {
    final available = activeDetections;
    if (isBusy || available.isEmpty) return;
    final current = available.indexWhere(
      (detection) => detection.source.itemId == selectedItemId,
    );
    final next = current < 0
        ? (offset > 0 ? 0 : available.length - 1)
        : (current + offset) % available.length;
    selectedItemId = available[next].source.itemId;
    searchItemId = null;
    searchQuery = '';
    _notify();
  }

  void showSearch(String itemId) {
    if (isBusy) return;
    selectedItemId = itemId;
    searchItemId = itemId;
    searchQuery = '';
    _notify();
  }

  void hideSearch() {
    searchItemId = null;
    searchQuery = '';
    _notify();
  }

  void updateSearch(String value) {
    searchQuery = value;
    _notify();
  }

  void confirmCandidate(String itemId, Candidate candidate) {
    _confirmProduct(
      itemId,
      _catalog.localizeCandidate(candidate),
      fromSearch: false,
    );
  }

  void confirmSearchProduct(String itemId, Product product) {
    _confirmProduct(itemId, product, fromSearch: true);
  }

  void _confirmProduct(
    String itemId,
    Product product, {
    required bool fromSearch,
  }) {
    final index = detections.indexWhere(
      (detection) => detection.source.itemId == itemId,
    );
    if (index < 0 || isBusy) return;
    final detection = detections[index];
    final productChanged =
        detection.finalProduct?.classId != product.classId ||
        !detection.isConfirmed;
    if (productChanged) _pushUndo();
    detection.finalProduct = product;
    detection.state = DetectionState.confirmed;
    if (productChanged) {
      detection.confirmationMethod =
          detection.source.status == ItemStatus.approved
          ? ConfirmationMethod.userCorrected
          : fromSearch
          ? ConfirmationMethod.searchSelected
          : ConfirmationMethod.top3Selected;
    }
    searchItemId = null;
    searchQuery = '';
    selectedItemId = _nextUnconfirmedId(index);
    _notify();
  }

  void setReviewTool(ReviewTool value) {
    if (isBusy || reviewTool == value) return;
    reviewTool = value;
    _notify();
  }

  void setOverlayMode(ReviewOverlayMode value) {
    if (overlayMode == value) return;
    overlayMode = value;
    _notify();
  }

  void setOperatorRequiresRecapture(bool value) {
    if (isBusy || operatorRequiresRecapture == value) return;
    _pushUndo();
    operatorRequiresRecapture = value;
    _syncIssueCodesToInference();
    _notify();
  }

  void updateReviewNote(String value) {
    reviewNote = value;
    _notify();
  }

  void toggleReviewIssueCode(OperatorIssueCode code) {
    if (isBusy) return;
    final next = Set<OperatorIssueCode>.from(reviewIssueCodes);
    if (!next.add(code)) next.remove(code);
    reviewIssueCodes = next;
    _notify();
  }

  void addDetection(BoundingBox bbox) {
    if (isBusy || !bbox.isValid) return;
    _pushUndo();
    final objectId = 'operator_${_nextOperatorObjectNumber++}';
    detections = [
      ...detections,
      ReviewDetection.operatorAdded(objectId: objectId, bbox: bbox),
    ];
    selectedItemId = objectId;
    reviewTool = ReviewTool.select;
    _syncIssueCodesToInference();
    showSearch(objectId);
  }

  void removeSelectedDetection() {
    final detection = selectedDetection;
    if (detection == null || isBusy) return;
    final removedAddedBox = detection.operatorAdded;
    _pushUndo();
    if (detection.operatorAdded) {
      detections = detections
          .where((item) => item.source.itemId != detection.source.itemId)
          .toList(growable: false);
    } else {
      detection.removed = true;
    }
    selectedItemId = activeDetections.isEmpty
        ? null
        : activeDetections.first.source.itemId;
    reviewTool = ReviewTool.select;
    _syncIssueCodesToInference();
    _showCompletionMessage(
      removedAddedBox
          ? '추가한 박스를 삭제했어요 · Ctrl+Z로 되돌릴 수 있어요'
          : '검출을 삭제했어요 · Ctrl+Z로 되돌릴 수 있어요',
    );
  }

  void beginBoxEdit() {
    if (selectedDetection == null || isBusy || _activeBoxEditStart != null) {
      return;
    }
    _activeBoxEditStart = _snapshot();
  }

  void updateSelectedBbox(BoundingBox bbox, {bool recordUndo = false}) {
    final detection = selectedDetection;
    final size = imageSize;
    if (detection == null || size == null || isBusy) return;
    final clamped = _clampBbox(bbox, size);
    if (clamped == detection.finalBbox) return;
    if (recordUndo) _pushUndo();
    detection.finalBbox = clamped;
    _syncIssueCodesToInference();
    _notify();
  }

  void commitBoxEdit() {
    final start = _activeBoxEditStart;
    _activeBoxEditStart = null;
    if (start == null) return;
    final current = selectedDetection;
    final before = start.detections.where(
      (detection) => detection.source.itemId == selectedItemId,
    );
    if (current == null ||
        before.isEmpty ||
        before.first.finalBbox == current.finalBbox) {
      return;
    }
    _undoStack.add(start);
    _redoStack.clear();
    _syncIssueCodesToInference();
    _notify();
  }

  void cancelBoxEdit() {
    final start = _activeBoxEditStart;
    if (start == null) return;
    _activeBoxEditStart = null;
    _restore(start);
  }

  void undoReviewEdit() {
    if (_undoStack.isEmpty || isBusy) return;
    _redoStack.add(_snapshot());
    _restore(_undoStack.removeLast());
  }

  void redoReviewEdit() {
    if (_redoStack.isEmpty || isBusy) return;
    _undoStack.add(_snapshot());
    _restore(_redoStack.removeLast());
  }

  void _pushUndo() {
    _undoStack.add(_snapshot());
    _redoStack.clear();
  }

  _ReviewSnapshot _snapshot() => _ReviewSnapshot(
    detections: detections.map((detection) => detection.copy()).toList(),
    selectedItemId: selectedItemId,
    operatorRequiresRecapture: operatorRequiresRecapture,
    reviewIssueCodes: Set<OperatorIssueCode>.from(reviewIssueCodes),
  );

  void _restore(_ReviewSnapshot snapshot) {
    detections = snapshot.detections
        .map((detection) => detection.copy())
        .toList(growable: false);
    selectedItemId = snapshot.selectedItemId;
    operatorRequiresRecapture = snapshot.operatorRequiresRecapture;
    reviewIssueCodes = Set<OperatorIssueCode>.from(snapshot.reviewIssueCodes);
    reviewTool = ReviewTool.select;
    searchItemId = null;
    searchQuery = '';
    _notify();
  }

  void _syncIssueCodesToInference() {
    reviewIssueCodes = inferredIssueCodes;
  }

  Future<void> submit() => saveReview();

  Future<void> saveReview() async {
    final activeResponse = response;
    final bytes = imageBytes;
    final fileName = imageFileName;
    if (!canSaveReview ||
        activeResponse == null ||
        bytes == null ||
        fileName == null) {
      return;
    }
    processState = ProcessState.submitting;
    errorMessage = null;
    _notify();
    try {
      final reviewedAt = DateTime.now();
      final inferred = inferredIssueCodes;
      final verdict = _reviewVerdict(inferred);
      await _scanLogRepository.save(
        ScanLogRecord(
          scanId: activeResponse.requestId,
          analyzedAt: analyzedAt ?? reviewedAt,
          confirmedAt: reviewedAt,
          recordedAt: reviewedAt,
          inputMode: inputMode,
          imageBytes: bytes,
          imageFileName: fileName,
          processingTimeMs: activeResponse.processingTimeMs,
          modelVersions: activeResponse.modelVersions,
          detections: detections,
          workerStatus: activeResponse.status,
          reasonCodes: activeResponse.reasonCodes,
          operatorReview: OperatorReview(
            verdict: verdict,
            reviewedAt: reviewedAt,
            inferredIssueCodes: inferred,
            issueCodes: inferred,
            objects: detections
                .map(OperatorReviewObject.fromDetection)
                .toList(growable: false),
          ),
          performance: performanceMetrics,
        ),
      );
      final completedInputMode = inputMode;
      _resetSession(nextInputMode: completedInputMode);
      _latestSavedScanId = activeResponse.requestId;
      _activityDataRevision += 1;
      _showCompletionMessage('결과를 저장했어요');
    } catch (_) {
      processState = ProcessState.reviewing;
      errorMessage = '저장하지 못했어요. 수정한 결과는 유지됐어요.';
      _notify();
    }
  }

  OperatorReviewVerdict _reviewVerdict(Set<OperatorIssueCode> inferred) {
    if (isRecapture) {
      return operatorRequiresRecapture
          ? OperatorReviewVerdict.recaptureAgreed
          : OperatorReviewVerdict.recaptureUnnecessary;
    }
    if (operatorRequiresRecapture) {
      return OperatorReviewVerdict.recaptureRequired;
    }
    return !hasUserChanges && inferred.isEmpty
        ? OperatorReviewVerdict.accepted
        : OperatorReviewVerdict.corrected;
  }

  @Deprecated('setOperatorRequiresRecapture와 saveReview를 사용합니다.')
  Future<void> saveRecaptureLog() async {
    final activeResponse = response;
    final bytes = imageBytes;
    final fileName = imageFileName;
    if (activeResponse == null ||
        activeResponse.status != ScanStatus.recapture ||
        bytes == null ||
        fileName == null ||
        recaptureLogSaveState == RecaptureLogSaveState.saving ||
        recaptureLogSaveState == RecaptureLogSaveState.saved) {
      return;
    }
    if (_savedRecaptureScanIds.contains(activeResponse.requestId)) {
      recaptureLogSaveState = RecaptureLogSaveState.saved;
      recaptureLogError = null;
      _notify();
      return;
    }
    recaptureLogSaveState = RecaptureLogSaveState.saving;
    recaptureLogError = null;
    _notify();
    try {
      final recordedAt = DateTime.now();
      await _scanLogRepository.save(
        ScanLogRecord(
          scanId: activeResponse.requestId,
          analyzedAt: analyzedAt ?? recordedAt,
          confirmedAt: null,
          recordedAt: recordedAt,
          inputMode: inputMode,
          imageBytes: bytes,
          imageFileName: fileName,
          processingTimeMs: activeResponse.processingTimeMs,
          modelVersions: activeResponse.modelVersions,
          detections: const [],
          workerStatus: ScanStatus.recapture,
          reasonCodes: activeResponse.reasonCodes,
          performance: performanceMetrics,
        ),
      );
      _savedRecaptureScanIds.add(activeResponse.requestId);
      recaptureLogSaveState = RecaptureLogSaveState.saved;
      _latestSavedScanId = activeResponse.requestId;
      _activityDataRevision += 1;
      _showCompletionMessage('재촬영 기록을 저장했어요');
    } catch (_) {
      recaptureLogSaveState = RecaptureLogSaveState.error;
      recaptureLogError = '재촬영 기록을 저장하지 못했어요. 다시 저장해 주세요.';
      _notify();
    }
  }

  @Deprecated('박스 추가 후 saveReview를 사용합니다.')
  Future<void> saveMissedDetectionLog() async {
    final activeResponse = response;
    final bytes = imageBytes;
    final fileName = imageFileName;
    final status = activeResponse?.status;
    if (activeResponse == null ||
        (status != ScanStatus.approved && status != ScanStatus.unknown) ||
        bytes == null ||
        fileName == null ||
        detections.isEmpty ||
        missedDetectionLogSaveState == MissedDetectionLogSaveState.saving ||
        missedDetectionLogSaveState == MissedDetectionLogSaveState.saved) {
      return;
    }
    if (_savedMissedDetectionScanIds.contains(activeResponse.requestId)) {
      missedDetectionLogSaveState = MissedDetectionLogSaveState.saved;
      missedDetectionLogError = null;
      _notify();
      return;
    }
    missedDetectionLogSaveState = MissedDetectionLogSaveState.saving;
    missedDetectionLogError = null;
    _notify();
    try {
      final recordedAt = DateTime.now();
      await _scanLogRepository.save(
        ScanLogRecord(
          scanId: activeResponse.requestId,
          analyzedAt: analyzedAt ?? recordedAt,
          confirmedAt: null,
          recordedAt: recordedAt,
          inputMode: inputMode,
          imageBytes: bytes,
          imageFileName: fileName,
          processingTimeMs: activeResponse.processingTimeMs,
          modelVersions: activeResponse.modelVersions,
          detections: detections,
          workerStatus: activeResponse.status,
          reasonCodes: activeResponse.reasonCodes,
          operatorFeedback: ScanOperatorFeedback.missedObject,
          performance: performanceMetrics,
        ),
      );
      _savedMissedDetectionScanIds.add(activeResponse.requestId);
      missedDetectionLogSaveState = MissedDetectionLogSaveState.saved;
      _showCompletionMessage('박스 미검출 기록을 저장했어요');
    } catch (_) {
      missedDetectionLogSaveState = MissedDetectionLogSaveState.error;
      missedDetectionLogError = '박스 미검출 기록을 저장하지 못했어요. 다시 저장해 주세요.';
      _notify();
    }
  }

  void resetSession() {
    if (isBusy) return;
    _resetSession();
    _notify();
  }

  String get recaptureTitle => _recapturePresentation.title;

  String get recaptureDetail => _recapturePresentation.detail;

  RecapturePresentation get _recapturePresentation => presentRecaptureReasons(
    reasonCodes: response?.reasonCodes ?? const <String>[],
    inputMode: inputMode,
  );

  Future<void> _setInputImage(
    InputImage image, {
    required InputMode mode,
    bool waitForDecode = true,
  }) async {
    imageBytes = image.bytes;
    imageFileName = image.fileName;
    imageSize = null;
    inputMode = mode;
    _cameraCaptureMs = image.cameraCaptureMs;
    _fileReadMs = image.fileReadMs;
    _flutterImageDecodeMs = 0.0;
    _decodeOverlappedWithAnalysis = !waitForDecode;
    final generation = ++_imageGeneration;
    final decodeFuture = _decodeInputImage(image.bytes, generation);
    _imageDecodeFuture = decodeFuture;
    if (waitForDecode) await decodeFuture;
  }

  Future<void> _decodeInputImage(Uint8List bytes, int generation) async {
    final imageDecode = Stopwatch()..start();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    try {
      imageDecode.stop();
      if (generation != _imageGeneration) return;
      imageSize = Size(
        frame.image.width.toDouble(),
        frame.image.height.toDouble(),
      );
      _flutterImageDecodeMs = imageDecode.elapsedMicroseconds / 1000.0;
    } finally {
      frame.image.dispose();
      codec.dispose();
    }
  }

  String? _firstUnconfirmedId() {
    for (final detection in detections) {
      if (!detection.removed && !detection.isConfirmed) {
        return detection.source.itemId;
      }
    }
    return null;
  }

  String? _nextUnconfirmedId(int currentIndex) {
    for (var offset = 1; offset <= detections.length; offset++) {
      final candidate = detections[(currentIndex + offset) % detections.length];
      if (!candidate.removed && !candidate.isConfirmed) {
        return candidate.source.itemId;
      }
    }
    return null;
  }

  void _clearResult() {
    response = null;
    detections = [];
    selectedItemId = null;
    searchItemId = null;
    searchQuery = '';
    analyzedAt = null;
    errorMessage = null;
    errorRecovery = ScannerErrorRecovery.retryAnalysis;
    _resetReviewState();
    performanceMetrics = null;
    _resultFirstFrameStopwatch = null;
  }

  void _resetSession({InputMode nextInputMode = InputMode.camera}) {
    inputMode = nextInputMode;
    processState = ProcessState.ready;
    imageBytes = null;
    _imageGeneration += 1;
    _imageDecodeFuture = null;
    _decodeOverlappedWithAnalysis = false;
    _cameraCaptureMs = 0.0;
    _fileReadMs = 0.0;
    _flutterImageDecodeMs = 0.0;
    _performanceBeforeFirstFrameMs = 0.0;
    imageFileName = null;
    imageSize = null;
    _clearResult();
  }

  void _resetReviewState() {
    reviewTool = ReviewTool.select;
    overlayMode = ReviewOverlayMode.compare;
    operatorRequiresRecapture = false;
    reviewIssueCodes = <OperatorIssueCode>{};
    reviewNote = '';
    _undoStack.clear();
    _redoStack.clear();
    _activeBoxEditStart = null;
    _nextOperatorObjectNumber = 1;
    recaptureLogSaveState = RecaptureLogSaveState.idle;
    recaptureLogError = null;
    missedDetectionLogSaveState = MissedDetectionLogSaveState.idle;
    missedDetectionLogError = null;
  }

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  void _showCompletionMessage(String message) {
    _completionFeedbackTimer?.cancel();
    completionMessage = message;
    _notify();
    _completionFeedbackTimer = Timer(completionFeedbackDuration, () {
      if (_disposed) return;
      completionMessage = null;
      notifyListeners();
    });
  }

  @override
  void dispose() {
    _disposed = true;
    _completionFeedbackTimer?.cancel();
    unawaited(_cameraGateway.dispose());
    super.dispose();
  }
}

BoundingBox _clampBbox(BoundingBox bbox, Size imageSize) {
  final imageWidth = imageSize.width.round().clamp(1, 1 << 30).toInt();
  final imageHeight = imageSize.height.round().clamp(1, 1 << 30).toInt();
  final width = bbox.width.clamp(1, imageWidth).toInt();
  final height = bbox.height.clamp(1, imageHeight).toInt();
  final x = bbox.x.clamp(0, imageWidth - width).toInt();
  final y = bbox.y.clamp(0, imageHeight - height).toInt();
  return BoundingBox(x: x, y: y, width: width, height: height);
}

class _ReviewSnapshot {
  const _ReviewSnapshot({
    required this.detections,
    required this.selectedItemId,
    required this.operatorRequiresRecapture,
    required this.reviewIssueCodes,
  });

  final List<ReviewDetection> detections;
  final String? selectedItemId;
  final bool operatorRequiresRecapture;
  final Set<OperatorIssueCode> reviewIssueCodes;
}
