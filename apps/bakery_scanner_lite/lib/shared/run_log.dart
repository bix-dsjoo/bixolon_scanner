import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'result.dart';
import 'log_images.dart';
import 'package:archive/archive_io.dart';

class RunRecord {
  RunRecord({
    required this.attemptId,
    required this.occurredAt,
    required this.inputMethod,
    required this.inputName,
    this.result,
    this.storageState = 'pending',
    this.storageReason,
    this.imageReference,
    this.imageStorageStatus = 'not_saved',
  });
  final String attemptId;
  final DateTime occurredAt;
  final String inputMethod;
  final String inputName;
  final ScanResult? result;
  final String storageState;
  final String? storageReason;
  final String? imageReference;
  final String imageStorageStatus;
  RunRecord withImage(StoredImage image) => RunRecord(
    attemptId: attemptId,
    occurredAt: occurredAt,
    inputMethod: inputMethod,
    inputName: inputName,
    result: result,
    storageState: storageState,
    storageReason: storageReason,
    imageReference: image.reference,
    imageStorageStatus: image.status,
  );
  RunRecord stored(String state, [String? reason]) => RunRecord(
    attemptId: attemptId,
    occurredAt: occurredAt,
    inputMethod: inputMethod,
    inputName: inputName,
    result: result,
    storageState: state,
    storageReason: reason,
    imageReference: imageReference,
    imageStorageStatus: imageStorageStatus,
  );
  Map<String, Object?> toJson() => {
    'schema_version': 2,
    'attempt_id': attemptId,
    'occurred_at': occurredAt.toUtc().toIso8601String(),
    'input_method': inputMethod,
    'input_name': inputName,
    'phase': result == null ? 'input_received' : 'inference_completed',
    'result': result?.toJson(),
    'log_storage_status': storageState,
    'log_storage_reason': storageReason,
    'image_reference': imageReference,
    'image_storage_status': imageStorageStatus,
  };
  factory RunRecord.fromJson(Map<String, dynamic> json) {
    final raw = json['result'] as Map<String, dynamic>?;
    ScanResult? result;
    if (raw != null) {
      if (!finalStates.contains(raw['status'])) throw const FormatException();
      final objects = (raw['segmentations'] as List).map((dynamic item) {
        if (!objectStates.contains(item['status'])) {
          throw const FormatException();
        }
        return ObjectResult(
          item['status'] as String,
          readReasons(item['reason_codes']),
          top3: readTop3(item['top3']),
          box: item['bbox'] == null
              ? null
              : DetectionBox.fromJson(item['bbox']),
        );
      });
      result = ScanResult(
        requestId: raw['request_id'] as String,
        status: raw['status'] as String,
        reasons: readReasons(raw['reason_codes']),
        objects: objects,
        processingMs: (raw['processing_time_ms'] as num?)?.toDouble(),
        versions: {
          for (final key in versionFields)
            if ((raw['versions'] as Map).containsKey(key))
              key: raw['versions'][key] as String?,
        },
        origin: raw['origin'] as String,
      );
    }
    return RunRecord(
      attemptId: json['attempt_id'] as String,
      occurredAt: DateTime.parse(json['occurred_at'] as String),
      inputMethod: json['input_method'] as String,
      inputName: json['input_name'] as String,
      result: result,
      storageState: json['log_storage_status'] as String,
      storageReason: json['log_storage_reason'] as String?,
      imageReference: json['image_reference'] as String?,
      imageStorageStatus:
          json['image_storage_status'] as String? ?? 'not_saved',
    );
  }
}

abstract interface class RunLogs {
  Future<RunRecord> append(RunRecord record, {Uint8List? image});
  Future<Uint8List?> imageFor(RunRecord record);
  Future<List<RunRecord>> recent();
  Future<void> exportTo(String path);
  bool get storageWarning;
}

class FileRunLogs implements RunLogs {
  FileRunLogs(this.directory, this.emergencyDirectory)
    : images = LogImages(directory, emergencyDirectory);
  final LogImages images;
  final Directory directory;
  final Directory emergencyDirectory;
  final Map<String, RunRecord> _unsaved = {};
  bool _warning = false;
  @override
  bool get storageWarning => _warning;

  Future<void> _write(Directory target, String filename, RunRecord row) async {
    await target.create(recursive: true);
    await File('${target.path}/$filename').writeAsString(
      '${jsonEncode(row.toJson())}\n',
      mode: FileMode.append,
      flush: true,
    );
  }

  @override
  Future<RunRecord> append(RunRecord record, {Uint8List? image}) async {
    if (image != null) {
      record = record.withImage(
        await images.store(record.attemptId, record.occurredAt, image),
      );
    }
    final imageFailed = [
      'failed',
      'emergency',
    ].contains(record.imageStorageStatus);
    if (imageFailed) _warning = true;
    final day = record.occurredAt.toUtc().toIso8601String().substring(0, 10);
    try {
      final saved = record.stored(
        imageFailed ? 'failed' : 'saved',
        imageFailed ? 'LOG_IMAGE_WRITE_FAILED' : null,
      );
      await _write(directory, '$day.jsonl', saved);
      _unsaved.remove(record.attemptId);
      return saved;
    } catch (_) {
      _warning = true;
      final failed = record.stored('failed', 'LOG_WRITE_FAILED');
      _unsaved[record.attemptId] = failed;
      // Independent emergency journal retains the failure itself and the sanitized result.
      try {
        await _write(emergencyDirectory, 'log-failures.jsonl', failed);
        _unsaved.remove(record.attemptId);
      } catch (_) {
        try {
          stderr.writeln(
            jsonEncode({
              'event': 'LOG_EMERGENCY_WRITE_FAILED',
              'attempt_id': record.attemptId,
              'occurred_at': record.occurredAt.toUtc().toIso8601String(),
            }),
          );
        } catch (_) {
          // GUI processes may have no stderr. The failed record remains in memory.
        }
      }
      return failed;
    }
  }

  Stream<RunRecord> _read(Directory target) async* {
    if (!await target.exists()) return;
    final files = await target
        .list()
        .where((f) => f is File && f.path.endsWith('.jsonl'))
        .toList();
    files.sort((a, b) => a.path.compareTo(b.path));
    for (final file in files.cast<File>()) {
      await for (final line
          in file
              .openRead()
              .transform(utf8.decoder)
              .transform(const LineSplitter())) {
        if (line.isEmpty) continue;
        try {
          yield RunRecord.fromJson(jsonDecode(line) as Map<String, dynamic>);
        } catch (_) {
          _warning = true;
        }
      }
    }
  }

  @override
  Future<List<RunRecord>> recent() async {
    try {
      await images.purge();
    } catch (_) {
      _warning = true;
    }
    final records = <String, RunRecord>{};
    for (final target in [emergencyDirectory, directory]) {
      try {
        await for (final row in _read(target)) {
          if (row.storageState == 'failed') _warning = true;
          final old = records[row.attemptId];
          if (old?.result == null || row.result != null) {
            records[row.attemptId] = row;
          }
          if (records.length > 500) records.remove(records.keys.first);
        }
      } catch (_) {
        _warning = true;
      }
    }
    records.addAll(_unsaved);
    final sorted = records.values.toList()
      ..sort((a, b) => b.occurredAt.compareTo(a.occurredAt));
    return sorted.take(200).toList(growable: false);
  }

  @override
  Future<Uint8List?> imageFor(RunRecord record) =>
      images.read(record.imageReference, record.occurredAt);

  @override
  Future<void> exportTo(String path) async {
    // Export the complete event journal, not just the recent-list window.
    final destination = File(path);
    String normalize(String value) =>
        Platform.isWindows ? value.replaceAll('\\', '/').toLowerCase() : value;
    final parent = normalize(await destination.parent.resolveSymbolicLinks());
    for (final source in [directory, emergencyDirectory]) {
      if (!await source.exists()) continue;
      final base = normalize(await source.resolveSymbolicLinks());
      if (parent == base || parent.startsWith('$base/')) {
        throw const FileSystemException(
          'Export must not overwrite the source journal',
        );
      }
    }
    final temporary = File(
      '$path.partial-${DateTime.now().microsecondsSinceEpoch}',
    );
    final zipTemporary = File(
      '$path.zip-partial-${DateTime.now().microsecondsSinceEpoch}',
    );
    final attachments = <String, RunRecord>{};
    final sink = temporary.openWrite();
    void writeRow(RunRecord row) {
      sink.writeln(jsonEncode(row.toJson()));
      if (row.imageReference != null) attachments[row.imageReference!] = row;
    }

    try {
      try {
        for (final target in [directory, emergencyDirectory]) {
          await for (final row in _read(target)) {
            writeRow(row);
          }
        }
        for (final row in _unsaved.values) {
          writeRow(row);
        }
        await sink.flush();
      } finally {
        await sink.close();
      }
      if (path.toLowerCase().endsWith('.zip')) {
        final zip = ZipFileEncoder()
          ..create(zipTemporary.path, level: ZipFileEncoder.store);
        try {
          await zip.addFile(temporary, 'logs.jsonl', ZipFileEncoder.store);
          final index = <String, String>{};
          for (final entry in attachments.entries) {
            final image = await imageFor(entry.value);
            if (image == null) {
              index[entry.key] = 'unavailable_or_expired';
              continue;
            }
            zip.addArchiveFile(
              ArchiveFile('images/${entry.key}', image.length, image),
            );
            index[entry.key] = 'included';
          }
          final encodedIndex = utf8.encode(jsonEncode(index));
          zip.addArchiveFile(
            ArchiveFile('image-index.json', encodedIndex.length, encodedIndex),
          );
        } finally {
          await zip.close();
        }
        await zipTemporary.rename(destination.path);
        await temporary.delete();
      } else {
        await temporary.rename(destination.path);
      }
    } catch (_) {
      if (await temporary.exists()) await temporary.delete();
      if (await zipTemporary.exists()) await zipTemporary.delete();
      rethrow;
    }
  }
}
