import '../../../shared/logging/scan_log_repository.dart';
import '../../../shared/models/scan_models.dart';
import '../data/scanner_api.dart';

class ScanPerformanceTracker {
  Stopwatch? _resultFirstFrameStopwatch;
  double _performanceBeforeFirstFrameMs = 0.0;
  double _cameraCaptureMs = 0.0;
  double _fileReadMs = 0.0;
  double _flutterImageDecodeMs = 0.0;
  Future<void>? _imageDecodeFuture;
  var _imageGeneration = 0;
  bool _decodeOverlappedWithAnalysis = false;

  Future<void> get imageDecodeFuture =>
      _imageDecodeFuture ?? Future<void>.value();

  int beginImage({
    required double cameraCaptureMs,
    required double fileReadMs,
    required bool waitForDecode,
  }) {
    _cameraCaptureMs = cameraCaptureMs;
    _fileReadMs = fileReadMs;
    _flutterImageDecodeMs = 0.0;
    _decodeOverlappedWithAnalysis = !waitForDecode;
    return ++_imageGeneration;
  }

  void trackDecode(Future<void> decodeFuture) {
    _imageDecodeFuture = decodeFuture;
  }

  bool isCurrentImage(int generation) => generation == _imageGeneration;

  void completeDecode(double elapsedMs) {
    _flutterImageDecodeMs = elapsedMs;
  }

  void resetAnalysis() {
    _resultFirstFrameStopwatch = null;
  }

  ScanPerformanceMetrics completeAnalysis({
    required int imageWidth,
    required int imageHeight,
    required int imageSizeBytes,
    required ScannerApiTimings? apiTimings,
    required double fallbackApiTotalMs,
    required double resultMappingMs,
    required WorkerStageTimings? workerTimings,
  }) {
    final effectiveApiTotalMs = apiTimings?.totalMs ?? fallbackApiTotalMs;
    final decodeAndApiMs = _decodeOverlappedWithAnalysis
        ? _flutterImageDecodeMs > effectiveApiTotalMs
              ? _flutterImageDecodeMs
              : effectiveApiTotalMs
        : _flutterImageDecodeMs + effectiveApiTotalMs;
    _performanceBeforeFirstFrameMs =
        _cameraCaptureMs + _fileReadMs + decodeAndApiMs + resultMappingMs;
    _resultFirstFrameStopwatch = Stopwatch()..start();
    return ScanPerformanceMetrics(
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      imageSizeBytes: imageSizeBytes,
      provider: apiTimings?.provider,
      cameraCaptureMs: _cameraCaptureMs,
      fileReadMs: _fileReadMs,
      flutterImageDecodeMs: _flutterImageDecodeMs,
      readinessMs: apiTimings?.readinessMs ?? 0.0,
      requestBuildMs: apiTimings?.requestBuildMs ?? 0.0,
      httpRoundTripMs: apiTimings?.httpRoundTripMs ?? effectiveApiTotalMs,
      responseBodyReadMs: apiTimings?.responseBodyReadMs ?? 0.0,
      responseParseMs: apiTimings?.responseParseMs ?? 0.0,
      resultMappingMs: resultMappingMs,
      resultFirstFrameMs: 0.0,
      endToEndMs: _performanceBeforeFirstFrameMs,
      worker: workerTimings,
    );
  }

  ScanPerformanceMetrics? recordResultFirstFrame(
    ScanPerformanceMetrics current,
  ) {
    final stopwatch = _resultFirstFrameStopwatch;
    if (stopwatch == null) return null;
    _resultFirstFrameStopwatch = null;
    stopwatch.stop();
    final renderMs = stopwatch.elapsedMicroseconds / 1000.0;
    return ScanPerformanceMetrics(
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

  void resetSession() {
    _imageGeneration += 1;
    _imageDecodeFuture = null;
    _decodeOverlappedWithAnalysis = false;
    _cameraCaptureMs = 0.0;
    _fileReadMs = 0.0;
    _flutterImageDecodeMs = 0.0;
    _performanceBeforeFirstFrameMs = 0.0;
    _resultFirstFrameStopwatch = null;
  }
}
