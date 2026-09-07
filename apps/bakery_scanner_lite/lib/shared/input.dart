import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:file_selector/file_selector.dart';
import 'square_capture.dart';

class InputPhoto {
  const InputPhoto(this.bytes, this.name);
  final Uint8List bytes;
  final String name;
}

class InputReadFailure implements Exception {
  const InputReadFailure(this.name, this.reason);
  final String name;
  final String reason;
}

abstract interface class PhotoInputs {
  CameraController? get camera;
  Future<List<CameraDescription>> cameras();
  Future<void> select(CameraDescription description);
  Future<InputPhoto> capture();
  Future<InputPhoto?> pick();
  Future<String?> exportPath();
  Future<void> dispose();
}

class WindowsPhotoInputs implements PhotoInputs {
  @override
  CameraController? camera;
  @override
  Future<List<CameraDescription>> cameras() => availableCameras();
  @override
  Future<void> select(CameraDescription description) async {
    final previous = camera;
    camera = null;
    await previous?.dispose();
    final next = CameraController(
      description,
      ResolutionPreset.max,
      enableAudio: false,
    );
    try {
      await next.initialize();
      camera = next;
    } catch (_) {
      await next.dispose();
      rethrow;
    }
  }

  @override
  Future<InputPhoto> capture() async {
    final active = camera;
    if (active == null || !active.value.isInitialized) {
      throw StateError('Camera not ready');
    }
    final file = await active.takePicture();
    try {
      return InputPhoto(
        await squareCapture(await file.readAsBytes()),
        'capture.png',
      );
    } finally {
      // The camera plugin creates a temporary capture; no image archive is retained.
      final temporary = File(file.path);
      if (await temporary.exists()) await temporary.delete();
    }
  }

  @override
  Future<InputPhoto?> pick() async {
    final file = await openFile(
      acceptedTypeGroups: const [
        XTypeGroup(label: 'JPEG / PNG', extensions: ['jpg', 'jpeg', 'png']),
      ],
    );
    if (file == null) return null;
    try {
      if (await file.length() > 20 * 1024 * 1024) {
        throw InputReadFailure(file.name, 'CLIENT_IMAGE_TOO_LARGE');
      }
      return InputPhoto(await file.readAsBytes(), file.name);
    } on InputReadFailure {
      rethrow;
    } catch (_) {
      throw InputReadFailure(file.name, 'CLIENT_FILE_READ_FAILED');
    }
  }

  @override
  Future<String?> exportPath() async => (await getSaveLocation(
    suggestedName:
        'bixolon-lite-logs-${DateTime.now().millisecondsSinceEpoch}.zip',
    acceptedTypeGroups: const [
      XTypeGroup(label: '로그와 이미지 ZIP', extensions: ['zip']),
    ],
  ))?.path;
  @override
  Future<void> dispose() async {
    await camera?.dispose();
    camera = null;
  }
}
