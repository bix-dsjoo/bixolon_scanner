import 'dart:async';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';

import '../catalog/product_catalog.dart';
import '../models/scan_models.dart';
import '../services/image_input.dart';
import '../services/scan_log_repository.dart';
import '../services/scanner_api.dart';

class ScannerController extends ChangeNotifier {
  ScannerController(
    this._scannerApi,
    this._cameraGateway,
    this._imageFileGateway,
    this._scanLogRepository,
    this._catalog,
  );

  final ScannerApi _scannerApi;
  final CameraGateway _cameraGateway;
  final ImageFileGateway _imageFileGateway;
  final ScanLogRepository _scanLogRepository;
  final ProductCatalog _catalog;

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
  String? completionMessage;
  bool cameraInitializing = true;
  String? cameraMessage;
  bool _disposed = false;

  CameraController? get cameraController => _cameraGateway.controller;
  bool get isCameraReady => _cameraGateway.isReady;
  bool get isBusy =>
      processState == ProcessState.analyzing ||
      processState == ProcessState.submitting;
  bool get isRecapture => response?.status == ScanStatus.recapture;
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

  List<Product> get searchResults => _catalog.search(searchQuery);

  Candidate localizeCandidate(Candidate candidate) =>
      _catalog.localizeCandidate(candidate);

  Future<void> initialize() async {
    cameraInitializing = true;
    _notify();
    try {
      await _cameraGateway.initialize();
      cameraMessage = null;
    } catch (_) {
      cameraMessage = '카메라를 사용할 수 없어요. 연결 상태를 확인해 주세요.';
    } finally {
      cameraInitializing = false;
      _notify();
    }
  }

  Future<void> reconnectCamera() async {
    cameraInitializing = true;
    cameraMessage = null;
    _notify();
    try {
      await _cameraGateway.initialize();
    } catch (_) {
      cameraMessage = '카메라를 사용할 수 없어요. 연결 상태를 확인해 주세요.';
    } finally {
      cameraInitializing = false;
      _notify();
    }
  }

  Future<void> chooseImage() async {
    if (isBusy) return;
    try {
      final selected = await _imageFileGateway.pick();
      if (selected == null) return;
      await _setInputImage(selected, mode: InputMode.image);
      _clearResult();
      processState = ProcessState.ready;
      _notify();
    } catch (_) {
      processState = ProcessState.error;
      errorMessage = '이미지를 열지 못했어요. 다른 이미지를 선택해 주세요.';
      _notify();
    }
  }

  Future<void> captureAndAnalyze() async {
    if (isBusy) return;
    try {
      final captured = await _cameraGateway.capture();
      await _setInputImage(captured, mode: InputMode.camera);
      await analyze();
    } catch (_) {
      cameraMessage = '카메라로 촬영하지 못했어요. 다시 연결해 주세요.';
      _notify();
    }
  }

  Future<void> analyze() async {
    final bytes = imageBytes;
    final fileName = imageFileName;
    if (bytes == null || fileName == null || isBusy) return;
    processState = ProcessState.analyzing;
    errorMessage = null;
    response = null;
    detections = [];
    selectedItemId = null;
    searchItemId = null;
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
    } catch (_) {
      processState = ProcessState.error;
      errorMessage = '분석하지 못했어요. 잠시 후 다시 시도해 주세요.';
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
    detection.finalProduct = product;
    detection.state = DetectionState.confirmed;
    detection.confirmationMethod =
        detection.source.status == ItemStatus.approved
        ? ConfirmationMethod.userCorrected
        : fromSearch
        ? ConfirmationMethod.searchSelected
        : ConfirmationMethod.top3Selected;
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
      await _scanLogRepository.save(
        ScanLogRecord(
          scanId: activeResponse.requestId,
          analyzedAt: analyzedAt ?? DateTime.now(),
          confirmedAt: DateTime.now(),
          inputMode: inputMode,
          imageBytes: bytes,
          imageFileName: fileName,
          processingTimeMs: activeResponse.processingTimeMs,
          modelVersions: activeResponse.modelVersions,
          detections: detections,
        ),
      );
      final count = detections.length;
      _resetSession();
      completionMessage = '$count개 상품을 확정했어요';
      _notify();
      unawaited(
        Future<void>.delayed(const Duration(seconds: 3), () {
          if (_disposed) return;
          completionMessage = null;
          notifyListeners();
        }),
      );
    } catch (_) {
      processState = ProcessState.reviewing;
      errorMessage = '결과를 저장하지 못했어요. 확인한 상품은 그대로 유지돼요.';
      _notify();
    }
  }

  void resetSession() {
    if (isBusy) return;
    _resetSession();
    _notify();
  }

  String get recaptureTitle {
    final reasons = response?.reasonCodes ?? const <String>[];
    if (reasons.contains('DETECTOR_NO_OBJECT')) return '상품을 찾지 못했어요';
    return inputMode == InputMode.camera ? '다시 촬영해 주세요' : '다른 이미지를 선택해 주세요';
  }

  String get recaptureDetail {
    final reasons = response?.reasonCodes ?? const <String>[];
    if (reasons.contains('DETECTOR_NO_OBJECT')) {
      return '상품이 화면 안에 잘 보이도록 다시 촬영해 주세요.';
    }
    if (reasons.contains('DETECTOR_BLUR')) {
      return '이미지가 흔들렸어요. 카메라를 고정하고 다시 촬영해 주세요.';
    }
    if (reasons.contains('DETECTOR_UNDEREXPOSED')) {
      return '이미지가 너무 어두워요. 밝은 곳에서 다시 촬영해 주세요.';
    }
    if (reasons.contains('DETECTOR_OVEREXPOSED')) {
      return '이미지가 너무 밝아요. 빛 반사를 줄이고 다시 촬영해 주세요.';
    }
    if (reasons.contains('DETECTOR_BORDER_CLIPPED')) {
      return '일부 상품이 이미지 밖으로 잘려 있어요.';
    }
    if (reasons.contains('DETECTOR_CAPACITY_EXCEEDED') ||
        reasons.contains('DETECTOR_COUNT_MISMATCH')) {
      return '상품이 겹치지 않고 모두 보이도록 다시 촬영해 주세요.';
    }
    return '상품을 정확하게 확인하기 어려워요. 구도를 조정해 다시 시도해 주세요.';
  }

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
  }

  void _resetSession() {
    inputMode = InputMode.camera;
    processState = ProcessState.ready;
    imageBytes = null;
    imageFileName = null;
    imageSize = null;
    _clearResult();
  }

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    unawaited(_cameraGateway.dispose());
    super.dispose();
  }
}
