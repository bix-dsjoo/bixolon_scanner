import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../models/scan_models.dart';

class ScanLogRecord {
  const ScanLogRecord({
    required this.scanId,
    required this.analyzedAt,
    required this.confirmedAt,
    required this.inputMode,
    required this.imageBytes,
    required this.imageFileName,
    required this.processingTimeMs,
    required this.modelVersions,
    required this.detections,
  });

  final String scanId;
  final DateTime analyzedAt;
  final DateTime confirmedAt;
  final InputMode inputMode;
  final Uint8List imageBytes;
  final String imageFileName;
  final double processingTimeMs;
  final ModelVersions modelVersions;
  final List<ReviewDetection> detections;
}

abstract interface class ScanLogRepository {
  Future<void> save(ScanLogRecord record);
}

class FileScanLogRepository implements ScanLogRepository {
  FileScanLogRepository({
    Future<Directory> Function()? applicationSupportDirectory,
  }) : _applicationSupportDirectory =
           applicationSupportDirectory ?? getApplicationSupportDirectory;

  final Future<Directory> Function() _applicationSupportDirectory;

  @override
  Future<void> save(ScanLogRecord record) async {
    final support = await _applicationSupportDirectory();
    final root = Directory(p.join(support.path, 'ProductScanner', 'scan_logs'));
    await root.create(recursive: true);

    final extension = _safeExtension(record.imageFileName);
    final imageName = '${record.scanId}$extension';
    final imageFile = File(p.join(root.path, imageName));
    await imageFile.writeAsBytes(record.imageBytes, flush: true);

    final payload = <String, dynamic>{
      'scan_id': record.scanId,
      'analyzed_at': record.analyzedAt.toUtc().toIso8601String(),
      'confirmed_at': record.confirmedAt.toUtc().toIso8601String(),
      'input_mode': record.inputMode == InputMode.camera ? 'CAMERA' : 'IMAGE',
      'original_image': imageName,
      'processing_time_ms': record.processingTimeMs,
      'detection_count': record.detections.length,
      'model_versions': record.modelVersions.toJson(),
      'detections': record.detections
          .map((detection) => detection.toLogJson())
          .toList(),
    };

    final target = File(p.join(root.path, '${record.scanId}.json'));
    final temporary = File('${target.path}.tmp');
    await temporary.writeAsString(
      const JsonEncoder.withIndent('  ').convert(payload),
      flush: true,
    );
    if (await target.exists()) await target.delete();
    await temporary.rename(target.path);
  }

  String _safeExtension(String fileName) {
    final value = p.extension(fileName).toLowerCase();
    return value == '.png' ? '.png' : '.jpg';
  }
}
