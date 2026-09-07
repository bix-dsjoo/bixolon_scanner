import 'dart:io';
import 'dart:typed_data';

const imageRetentionDays = 30;

class StoredImage {
  const StoredImage(this.reference, this.status);
  final String? reference;
  final String status;
}

class LogImages {
  LogImages(this.primary, this.emergency);
  final Directory primary, emergency;
  final Map<String, Uint8List> _memory = {};
  static final _safe = RegExp(r'^\d+-[a-zA-Z0-9_-]+\.(png|jpg|bin)$');
  DateTime? _lastPurge;

  Future<StoredImage> store(String id, DateTime time, Uint8List bytes) async {
    if (!RegExp(r'^[a-zA-Z0-9_-]+$').hasMatch(id)) {
      return const StoredImage(null, 'failed');
    }
    final extension = bytes.length >= 3 && bytes[0] == 255 && bytes[1] == 216
        ? 'jpg'
        : bytes.length >= 8 && bytes[0] == 137 && bytes[1] == 80
        ? 'png'
        : 'bin';
    final reference = '${time.millisecondsSinceEpoch}-$id.$extension';
    for (final root in [primary, emergency]) {
      try {
        final directory = Directory('${root.path}/images');
        await directory.create(recursive: true);
        final temporary = File('${directory.path}/$reference.partial');
        try {
          await temporary.writeAsBytes(bytes, flush: true);
          await temporary.rename('${directory.path}/$reference');
        } finally {
          if (await temporary.exists()) await temporary.delete();
        }
        return StoredImage(reference, root == primary ? 'saved' : 'emergency');
      } catch (_) {
        // Preserve inference and try the separate emergency location.
      }
    }
    _memory[reference] = bytes;
    while (_memory.length > 3) {
      _memory.remove(_memory.keys.first);
    }
    return StoredImage(reference, 'failed');
  }

  Future<Uint8List?> read(String? reference, DateTime occurredAt) async {
    if (reference == null ||
        !_safe.hasMatch(reference) ||
        DateTime.now().difference(occurredAt) >=
            const Duration(days: imageRetentionDays)) {
      return null;
    }
    if (_memory.containsKey(reference)) return _memory[reference];
    for (final root in [primary, emergency]) {
      final file = File('${root.path}/images/$reference');
      try {
        if (await FileSystemEntity.type(file.path, followLinks: false) !=
            FileSystemEntityType.file) {
          continue;
        }
        return await file.readAsBytes();
      } catch (_) {
        /* The other location may contain the retained copy. */
      }
    }
    return null;
  }

  Future<void> purge({DateTime? now}) async {
    final current = now ?? DateTime.now();
    if (_lastPurge != null &&
        current.difference(_lastPurge!) < const Duration(hours: 1)) {
      return;
    }
    _lastPurge = current;
    final cutoff = current
        .subtract(const Duration(days: imageRetentionDays))
        .millisecondsSinceEpoch;
    for (final root in [primary, emergency]) {
      final directory = Directory('${root.path}/images');
      if (!await directory.exists()) continue;
      await for (final entity in directory.list(followLinks: false)) {
        final name = entity.uri.pathSegments.last;
        if (entity is! File || !_safe.hasMatch(name)) continue;
        final time = int.tryParse(name.split('-').first);
        if (time != null && time < cutoff) await entity.delete();
      }
    }
    _memory.removeWhere(
      (key, _) => (int.tryParse(key.split('-').first) ?? cutoff) < cutoff,
    );
  }
}
