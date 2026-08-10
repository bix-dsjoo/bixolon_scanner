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

class ScanLogItemSummary {
  const ScanLogItemSummary({
    required this.itemId,
    required this.productName,
    required this.confidence,
    required this.userModified,
    required this.confirmationMethod,
    this.classId,
    this.className,
  });

  final String itemId;
  final String productName;
  final double confidence;
  final bool userModified;
  final String confirmationMethod;
  final String? classId;
  final String? className;

  ScanLogItemSummary withProductName(String value) => ScanLogItemSummary(
    itemId: itemId,
    productName: value,
    confidence: confidence,
    userModified: userModified,
    confirmationMethod: confirmationMethod,
    classId: classId,
    className: className,
  );
}

class ScanLogSummary {
  const ScanLogSummary({
    required this.scanId,
    required this.analyzedAt,
    required this.confirmedAt,
    required this.inputMode,
    required this.processingTimeMs,
    required this.modelVersions,
    required this.items,
  });

  final String scanId;
  final DateTime analyzedAt;
  final DateTime confirmedAt;
  final InputMode inputMode;
  final double processingTimeMs;
  final ModelVersions modelVersions;
  final List<ScanLogItemSummary> items;
}

abstract interface class ScanLogRepository {
  Future<void> save(ScanLogRecord record);

  Future<List<ScanLogSummary>> list({int limit = 100});
}

class FileScanLogRepository implements ScanLogRepository {
  FileScanLogRepository({
    Future<Directory> Function()? applicationSupportDirectory,
  }) : _applicationSupportDirectory =
           applicationSupportDirectory ?? getApplicationSupportDirectory;

  final Future<Directory> Function() _applicationSupportDirectory;

  Future<Directory> _logDirectory() async {
    final support = await _applicationSupportDirectory();
    return Directory(p.join(support.path, 'ProductScanner', 'scan_logs'));
  }

  @override
  Future<void> save(ScanLogRecord record) async {
    final root = await _logDirectory();
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

  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async {
    final root = await _logDirectory();
    if (!await root.exists()) return const [];
    final files = await root
        .list()
        .where(
          (entity) => entity is File && p.extension(entity.path) == '.json',
        )
        .cast<File>()
        .toList();
    final logs = <ScanLogSummary>[];
    for (final file in files) {
      try {
        final decoded = jsonDecode(await file.readAsString());
        if (decoded is! Map<String, dynamic>) continue;
        final detections = decoded['detections'];
        if (detections is! List) continue;
        final rawVersions = decoded['model_versions'];
        logs.add(
          ScanLogSummary(
            scanId: decoded['scan_id'] as String,
            analyzedAt: DateTime.parse(decoded['analyzed_at'] as String),
            confirmedAt: DateTime.parse(decoded['confirmed_at'] as String),
            inputMode: decoded['input_mode'] == 'CAMERA'
                ? InputMode.camera
                : InputMode.image,
            processingTimeMs: (decoded['processing_time_ms'] as num).toDouble(),
            modelVersions: ModelVersions.fromJson(
              rawVersions as Map<String, dynamic>,
            ),
            items: detections
                .map((value) {
                  final detection = value as Map<String, dynamic>;
                  final product =
                      detection['final_product'] as Map<String, dynamic>?;
                  return ScanLogItemSummary(
                    itemId: detection['detection_id'] as String,
                    productName: product?['class_name'] as String? ?? 'Unknown',
                    confidence: (detection['initial_confidence'] as num)
                        .toDouble(),
                    userModified: detection['user_modified'] as bool? ?? false,
                    confirmationMethod:
                        detection['confirmation_method'] as String? ??
                        'UNKNOWN',
                    classId: product?['class_id'] as String?,
                    className: product?['class_name'] as String?,
                  );
                })
                .toList(growable: false),
          ),
        );
      } on Object {
        // A partially written or manually modified file must not hide valid logs.
      }
    }
    logs.sort((a, b) => b.confirmedAt.compareTo(a.confirmedAt));
    return logs.take(limit).toList(growable: false);
  }

  String _safeExtension(String fileName) {
    final value = p.extension(fileName).toLowerCase();
    return value == '.png' ? '.png' : '.jpg';
  }
}
