import 'dart:async';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';

import '../catalog/product_catalog.dart';
import '../models/scan_models.dart';
import '../presentation/recapture_presentation.dart';
import '../services/image_input.dart';
import '../services/scan_log_repository.dart';
import '../services/scanner_api.dart';
import '../theme/app_tokens.dart';

enum CameraIssueType { unavailable, captureFailed }

enum RecaptureLogSaveState { idle, saving, saved, error }

enum MissedDetectionLogSaveState { idle, saving, saved, error }

class ScannerController extends ChangeNotifier {
  ScannerController(
    this._scannerApi,
    this._cameraGateway,
    this._imageFileGateway,
    this._scanLogRepository,
    this._catalog, {
    this.completionFeedbackDuration = AppMotion.feedbackHold,
  });

  final ScannerApi _scannerApi;
  final CameraGateway _cameraGateway;
  final ImageFileGateway _imageFileGateway;
  final ScanLogRepository _scanLogRepository;
  final ProductCatalog _catalog;
  final Duration completionFeedbackDuration;
  Timer? _completionFeedbackTimer;

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
  RecaptureLogSaveState recaptureLogSaveState = RecaptureLogSaveState.idle;
  String? recaptureLogError;
  MissedDetectionLogSaveState missedDetectionLogSaveState =
      MissedDetectionLogSaveState.idle;
  String? missedDetectionLogError;
  String? completionMessage;
  int _activityDataRevision = 0;
  String? _latestSavedScanId;
  final Set<String> _savedRecaptureScanIds = <String>{};
  final Set<String> _savedMissedDetectionScanIds = <String>{};
  bool cameraInitializing = true;
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
  bool get isRecapture => response?.status == ScanStatus.recapture;
  bool get canSaveMissedDetectionLog {
    final status = response?.status;
    return !isBusy &&
        imageBytes != null &&
        imageFileName != null &&
        detections.isNotEmpty &&
        (status == ScanStatus.approved || status == ScanStatus.unknown) &&
        missedDetectionLogSaveState != MissedDetectionLogSaveState.saved;
  }

  bool get hasResults => detections.isNotEmpty;
  int get confirmedCount =>
      detections.where((detection) => detection.isConfirmed).length;
  bool get allConfirmed =>
      detections.isNotEmpty && confirmedCount == detections.length;
  bool get hasUserChanges =>
      detections.any((detection) => detection.wasUserChanged);
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
            items: log.items
                .map(
                  (item) => item.withProductName(
                    _catalog.displayNameFor(
                      classId: item.classId,
                      className: item.className,
                      fallback: item.productName,
                    ),
                  ),
                )
                .toList(growable: false),
          ),
        )
        .toList(growable: false);
  }

  Candidate localizeCandidate(Candidate candidate) =>
      _catalog.localizeCandidate(candidate);

  Future<void> initialize() async {
    cameraInitializing = true;
    cameraMessage = null;
    cameraIssueType = null;
    _notify();
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
    if (isBusy) return;
    processState = ProcessState.capturing;
    cameraMessage = null;
    cameraIssueType = null;
    _notify();
    try {
      final captured = await _cameraGateway.capture();
      await _setInputImage(captured, mode: InputMode.camera);
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
    if (bytes == null || fileName == null || isBusy) return;
    processState = ProcessState.analyzing;
    errorMessage = null;
    errorRecovery = ScannerErrorRecovery.retryAnalysis;
    response = null;
    detections = [];
    selectedItemId = null;
    searchItemId = null;
    recaptureLogSaveState = RecaptureLogSaveState.idle;
    recaptureLogError = null;
    missedDetectionLogSaveState = MissedDetectionLogSaveState.idle;
    missedDetectionLogError = null;
    _notify();
    try {
      final result = await _scannerApi.scan(
        imageBytes: bytes,
        fileName: fileName,
      );
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
      processState = ProcessState.reviewing;
      if (result.status == ScanStatus.unknown) {
        selectedItemId = _firstUnconfirmedId();
      }
    } on ScannerApiException catch (error) {
      processState = ProcessState.error;
      errorMessage = error.message;
      errorRecovery = error.recovery;
    } catch (_) {
      processState = ProcessState.error;
      errorMessage = '분석하지 못했어요. 잠시 후 다시 분석해 주세요.';
      errorRecovery = ScannerErrorRecovery.retryAnalysis;
    }
    _notify();
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
    if (isBusy || detections.isEmpty) return;
    final current = selectedIndex;
    final next = current < 0
        ? (offset > 0 ? 0 : detections.length - 1)
        : (current + offset) % detections.length;
    selectedItemId = detections[next].source.itemId;
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

  Future<void> submit() async {
    final activeResponse = response;
    final bytes = imageBytes;
    final fileName = imageFileName;
    if (!allConfirmed ||
        activeResponse == null ||
        bytes == null ||
        fileName == null) {
      return;
    }
    processState = ProcessState.submitting;
    errorMessage = null;
    _notify();
    try {
      final confirmedAt = DateTime.now();
      await _scanLogRepository.save(
        ScanLogRecord(
          scanId: activeResponse.requestId,
          analyzedAt: analyzedAt ?? DateTime.now(),
          confirmedAt: confirmedAt,
          recordedAt: confirmedAt,
          inputMode: inputMode,
          imageBytes: bytes,
          imageFileName: fileName,
          processingTimeMs: activeResponse.processingTimeMs,
          modelVersions: activeResponse.modelVersions,
          detections: detections,
          workerStatus: activeResponse.status,
          reasonCodes: activeResponse.reasonCodes,
        ),
      );
      final count = detections.length;
      final completedInputMode = inputMode;
      _resetSession(nextInputMode: completedInputMode);
      _latestSavedScanId = activeResponse.requestId;
      _activityDataRevision += 1;
      _showCompletionMessage('$count개 상품을 확정했어요');
    } catch (_) {
      processState = ProcessState.reviewing;
      errorMessage = '저장하지 못했어요. 확인 결과는 유지됐어요.';
      _notify();
    }
  }

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
  }) async {
    final codec = await ui.instantiateImageCodec(image.bytes);
    final frame = await codec.getNextFrame();
    imageSize = Size(
      frame.image.width.toDouble(),
      frame.image.height.toDouble(),
    );
    frame.image.dispose();
    codec.dispose();
    imageBytes = image.bytes;
    imageFileName = image.fileName;
    inputMode = mode;
  }

  String? _firstUnconfirmedId() {
    for (final detection in detections) {
      if (!detection.isConfirmed) return detection.source.itemId;
    }
    return null;
  }

  String? _nextUnconfirmedId(int currentIndex) {
    for (var offset = 1; offset <= detections.length; offset++) {
      final candidate = detections[(currentIndex + offset) % detections.length];
      if (!candidate.isConfirmed) return candidate.source.itemId;
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
    recaptureLogSaveState = RecaptureLogSaveState.idle;
    recaptureLogError = null;
    missedDetectionLogSaveState = MissedDetectionLogSaveState.idle;
    missedDetectionLogError = null;
  }

  void _resetSession({InputMode nextInputMode = InputMode.camera}) {
    inputMode = nextInputMode;
    processState = ProcessState.ready;
    imageBytes = null;
    imageFileName = null;
    imageSize = null;
    _clearResult();
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
