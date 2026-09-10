import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:bakery_scanner_lite/shared/input.dart';
import 'package:bakery_scanner_lite/shared/camera_profile.dart';
import 'package:bakery_scanner_lite/shared/result.dart';
import 'package:bakery_scanner_lite/shared/run_log.dart';
import 'package:bakery_scanner_lite/shared/log_images.dart';
import 'package:bakery_scanner_lite/shared/worker_client.dart';

final pixel = base64Decode(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aO9sAAAAASUVORK5CYII=',
);
Map<String, dynamic> response({
  String status = 'SEGMENTATION',
  String object = 'UNKNOWN',
}) => {
  'request_id': 'request-test-001',
  'status': status,
  'reason_codes': status == 'IMAGE_RECAPTURE'
      ? ['IMAGE_RECAPTURE_REQUIRED']
      : [],
  'processing_time_ms': 42.5,
  'worker_version': '0.1.18',
  'detector_version': '0.1.18',
  'classifier_version': status == 'IMAGE_RECAPTURE' ? null : '0.1.18',
  'segmentations': status == 'SEGMENTATION'
      ? [
          {
            'status': object,
            'reason_codes': object == 'UNKNOWN'
                ? ['BELOW_APPROVAL_THRESHOLD']
                : [],
            'bbox': {'x': 1, 'y': 2, 'width': 3, 'height': 4},
            'confidence': .91,
            'prediction': {'class_id': 'secret-label'},
            'top3': [
              {
                'class_id': 'bread_01',
                'class_name': 'Almond Scone',
                'confidence': .93,
              },
              {
                'class_id': 'bread_02',
                'class_name': 'Cream Bun',
                'confidence': .06,
              },
              {
                'class_id': 'bread_03',
                'class_name': 'Baguette',
                'confidence': .01,
              },
            ],
          },
        ]
      : [],
};

class FakeInputs implements PhotoInputs {
  @override
  CameraController? get camera => null;
  final descriptions = const [
    CameraDescription(
      name: configuredCameraName,
      lensDirection: CameraLensDirection.external,
      sensorOrientation: 0,
    ),
    CameraDescription(
      name: 'USB camera',
      lensDirection: CameraLensDirection.external,
      sensorOrientation: 0,
    ),
  ];
  String? chosen;
  bool cancel = false;
  bool fail = false;
  String? export;
  Uint8List bytes = pixel;
  @override
  Future<List<CameraDescription>> cameras() async => descriptions;
  @override
  Future<void> select(CameraDescription description) async {
    chosen = description.name;
  }

  @override
  Future<InputPhoto> capture() async {
    if (fail) throw StateError('capture');
    return InputPhoto(bytes, 'capture.jpg');
  }

  @override
  Future<InputPhoto?> pick() async {
    if (fail) {
      throw const InputReadFailure('bad.jpg', 'CLIENT_FILE_READ_FAILED');
    }
    return cancel ? null : InputPhoto(bytes, 'sample.png');
  }

  @override
  Future<String?> exportPath() async => export;
  @override
  Future<void> dispose() async {}
}

class FakeScanner implements ScannerService {
  int scans = 0;
  Completer<ScanResult>? pending;
  ScanResult value = ScanResult.fromWorker(response());
  @override
  Future<void> start() async {}
  @override
  Future<ScanResult> scan(
    Uint8List bytes,
    String fileName,
    String attemptId,
  ) async {
    scans++;
    return pending == null ? value : pending!.future;
  }

  @override
  Future<void> close() async {}
}

class MemoryLogs implements RunLogs {
  final List<RunRecord> rows = [];
  final Map<String, Uint8List> images = {};
  bool fail = false;
  @override
  bool get storageWarning => fail;
  @override
  Future<RunRecord> append(RunRecord row, {Uint8List? image}) async {
    if (image != null) {
      images[row.attemptId] = image;
      row = row.withImage(StoredImage(row.attemptId, 'saved'));
    }
    final stored = row.stored(
      fail ? 'failed' : 'saved',
      fail ? 'LOG_WRITE_FAILED' : null,
    );
    rows.add(stored);
    return stored;
  }

  @override
  Future<List<RunRecord>> recent() async => {
    for (final row in rows) row.attemptId: row,
  }.values.toList().reversed.toList();
  @override
  Future<void> exportTo(String path) async {}
  @override
  Future<Uint8List?> imageFor(RunRecord record) async =>
      images[record.attemptId];
}
