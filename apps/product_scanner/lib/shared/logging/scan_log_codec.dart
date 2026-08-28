part of 'scan_log_repository.dart';

String _newCaptureSessionId() {
  final timestamp = DateTime.now().toUtc().toIso8601String().replaceAll(
    RegExp(r'[^0-9]'),
    '',
  );
  return 'app-$timestamp-$pid';
}

String _safeArchiveId(String value) {
  final sanitized = value.replaceAll(RegExp(r'[^A-Za-z0-9._-]'), '_');
  return sanitized.isEmpty ? 'scan' : sanitized;
}

bool _isCompressedImage(String path) {
  final extension = p.extension(path).toLowerCase();
  return extension == '.jpg' || extension == '.jpeg' || extension == '.png';
}

Future<String> _fileSha256(File file) async =>
    (await sha256.bind(file.openRead()).first).toString();

class _ExportFile {
  const _ExportFile({
    required this.source,
    required this.archivePath,
    required this.level,
  });

  final File source;
  final String archivePath;
  final int level;
}

String _workerStatusValue(ScanStatus status) => switch (status) {
  ScanStatus.approved => 'APPROVED',
  ScanStatus.unknown => 'UNKNOWN',
  ScanStatus.recapture => 'RECAPTURE',
  ScanStatus.error => 'ERROR',
};

ScanStatus _parseWorkerStatus(Object? value, List<dynamic> detections) {
  return switch (value) {
    'APPROVED' => ScanStatus.approved,
    'UNKNOWN' => ScanStatus.unknown,
    'RECAPTURE' => ScanStatus.recapture,
    'ERROR' => throw const FormatException('ERROR activity record is invalid'),
    null =>
      detections.any(
            (value) =>
                value is Map<String, dynamic> &&
                value['initial_ai_status'] == 'TOP3_CANDIDATES',
          )
          ? ScanStatus.unknown
          : ScanStatus.approved,
    _ => throw const FormatException('Unsupported activity worker status'),
  };
}
