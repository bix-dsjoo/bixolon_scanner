import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:file_selector/file_selector.dart';

class InputImage {
  const InputImage({required this.bytes, required this.fileName});

  final Uint8List bytes;
  final String fileName;
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
      ResolutionPreset.max,
      enableAudio: false,
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
    final image = await active.takePicture();
    return InputImage(bytes: await image.readAsBytes(), fileName: image.name);
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
    return InputImage(bytes: await file.readAsBytes(), fileName: file.name);
  }
}
