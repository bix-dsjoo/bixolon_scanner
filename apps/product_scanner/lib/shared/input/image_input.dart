import 'package:camera/camera.dart';
import 'package:file_selector/file_selector.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as imaging;

const cameraSquareOutputSize = 1080;

@visibleForTesting
Uint8List centerCropCameraJpeg(Uint8List sourceBytes) {
  if (sourceBytes.isEmpty) {
    throw const FormatException('카메라 이미지가 비어 있습니다.');
  }
  final decoded = imaging.decodeImage(sourceBytes);
  if (decoded == null) {
    throw const FormatException('카메라 JPEG를 디코딩하지 못했습니다.');
  }

  final oriented = imaging.bakeOrientation(decoded);
  // Crop native pixels around the center of both axes; never resize.
  // A 1920x1080 frame uses x=420, y=0 and retains the full height.
  final shortestSide = oriented.width < oriented.height
      ? oriented.width
      : oriented.height;
  final cropSize = shortestSide < cameraSquareOutputSize
      ? shortestSide
      : cameraSquareOutputSize;
  final cropped = imaging.copyCrop(
    oriented,
    x: (oriented.width - cropSize) ~/ 2,
    y: (oriented.height - cropSize) ~/ 2,
    width: cropSize,
    height: cropSize,
  );
  return Uint8List.fromList(imaging.encodeJpg(cropped, quality: 100));
}

class InputImage {
  const InputImage({
    required this.bytes,
    required this.fileName,
    this.cameraCaptureMs = 0.0,
    this.fileReadMs = 0.0,
  });

  final Uint8List bytes;
  final String fileName;
  final double cameraCaptureMs;
  final double fileReadMs;
}

abstract interface class CameraGateway {
  CameraController? get controller;
  bool get isReady;

  Future<void> initialize();
  Future<InputImage> capture();
  Future<void> dispose();
}

class WindowsCameraGateway implements CameraGateway {
  CameraController? _controller;

  @override
  CameraController? get controller => _controller;

  @override
  bool get isReady => _controller?.value.isInitialized ?? false;

  @override
  Future<void> initialize() async {
    final previous = _controller;
    _controller = null;
    await previous?.dispose();
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      throw CameraException('NO_CAMERA', '연결된 카메라가 없습니다.');
    }
    final controller = CameraController(
      cameras.first,
      ResolutionPreset.veryHigh,
      enableAudio: false,
      fps: 30,
    );
    try {
      await controller.initialize();
      _controller = controller;
    } catch (_) {
      await controller.dispose();
      rethrow;
    }
  }

  @override
  Future<InputImage> capture() async {
    final active = _controller;
    if (active == null || !active.value.isInitialized) {
      throw CameraException('CAMERA_NOT_READY', '카메라가 준비되지 않았습니다.');
    }
    final capture = Stopwatch()..start();
    final image = await active.takePicture();
    capture.stop();
    final fileRead = Stopwatch()..start();
    final sourceBytes = await image.readAsBytes();
    final bytes = await compute(centerCropCameraJpeg, sourceBytes);
    fileRead.stop();
    return InputImage(
      bytes: bytes,
      fileName: image.name,
      cameraCaptureMs: capture.elapsedMicroseconds / 1000.0,
      fileReadMs: fileRead.elapsedMicroseconds / 1000.0,
    );
  }

  @override
  Future<void> dispose() async {
    await _controller?.dispose();
    _controller = null;
  }
}

abstract interface class ImageFileGateway {
  Future<InputImage?> pick();
}

class WindowsImageFileGateway implements ImageFileGateway {
  static const XTypeGroup _images = XTypeGroup(
    label: 'JPEG 또는 PNG 이미지',
    extensions: ['jpg', 'jpeg', 'png'],
  );

  @override
  Future<InputImage?> pick() async {
    final file = await openFile(acceptedTypeGroups: const [_images]);
    if (file == null) return null;
    final fileRead = Stopwatch()..start();
    final bytes = await file.readAsBytes();
    fileRead.stop();
    return InputImage(
      bytes: bytes,
      fileName: file.name,
      fileReadMs: fileRead.elapsedMicroseconds / 1000.0,
    );
  }
}
