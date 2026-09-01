import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:file_selector/file_selector.dart';
import 'package:path/path.dart' as paths;
import 'package:path_provider/path_provider.dart';

import '../domain/bread_capture_models.dart';

abstract interface class BreadCaptureRepository {
  String get rootPath;

  Future<BreadCaptureLibrarySnapshot> load();
  Future<BreadCaptureProduct> addProduct(String name);
  Future<void> deleteProduct(BreadCaptureProduct product);
  Future<void> saveCapture(
    BreadCaptureProduct product,
    BreadCaptureSlot slot,
    Uint8List bytes,
  );
  Future<void> deleteCapture(
    BreadCaptureProduct product,
    BreadCaptureSlot slot,
  );
  Future<BreadCaptureLibrarySnapshot> changeRoot(String rootPath);
  Future<void> openRoot();
}

abstract interface class BreadCaptureDirectoryPicker {
  Future<String?> pick();
}

class WindowsBreadCaptureDirectoryPicker
    implements BreadCaptureDirectoryPicker {
  @override
  Future<String?> pick() => getDirectoryPath(confirmButtonText: '이 폴더 사용');
}

class FileBreadCaptureRepository implements BreadCaptureRepository {
  FileBreadCaptureRepository.forRoot(String rootPath)
    : _rootPath = paths.normalize(paths.absolute(rootPath)),
      _settingsFile = null;

  FileBreadCaptureRepository._(this._rootPath, {required this._settingsFile});

  static const _catalogFileName = 'capture_catalog.json';
  static final _directoryPattern = RegExp(r'^bread_(\d{2})(?:_(.+))?$');

  String _rootPath;
  final File? _settingsFile;

  static Future<FileBreadCaptureRepository> create({
    String? configuredRoot,
  }) async {
    final support = await getApplicationSupportDirectory();
    final settingsFile = File(
      paths.join(support.path, 'bread-capture-settings.json'),
    );
    String? savedRoot;
    if (await settingsFile.exists()) {
      try {
        final json = jsonDecode(await settingsFile.readAsString());
        if (json case {'capture_root': final String value}) savedRoot = value;
      } on Object {
        savedRoot = null;
      }
    }
    final documents = await getApplicationDocumentsDirectory();
    final root = configuredRoot?.trim().isNotEmpty == true
        ? configuredRoot!.trim()
        : savedRoot?.trim().isNotEmpty == true
        ? savedRoot!.trim()
        : paths.join(
            documents.path,
            'BIXOLON Bakery AI Scanner',
            'single_objects',
          );
    return FileBreadCaptureRepository._(
      paths.normalize(paths.absolute(root)),
      settingsFile: settingsFile,
    );
  }

  @override
  String get rootPath => _rootPath;

  Directory get _root => Directory(_rootPath);
  File get _catalogFile => File(paths.join(_rootPath, _catalogFileName));

  @override
  Future<BreadCaptureLibrarySnapshot> load() async {
    await _root.create(recursive: true);
    final catalog = await _readCatalog();
    final products = <BreadCaptureProduct>[];
    await for (final entity in _root.list(followLinks: false)) {
      if (entity is! Directory) continue;
      final directoryName = paths.basename(entity.path);
      final match = _directoryPattern.firstMatch(directoryName);
      if (match == null) continue;
      final categoryId = int.parse(match.group(1)!);
      if (categoryId < 1 || categoryId > 20) continue;
      final row = catalog[categoryId];
      final fallbackName = (match.group(2) ?? '빵 $categoryId').replaceAll(
        '_',
        ' ',
      );
      final captures = <String, String>{};
      for (final slot in BreadCaptureSlot.all) {
        final file = File(paths.join(entity.path, slot.fileName(categoryId)));
        if (await file.exists()) captures[slot.key] = file.path;
      }
      products.add(
        BreadCaptureProduct(
          categoryId: categoryId,
          name: row?.name ?? fallbackName,
          directoryName: directoryName,
          createdAt: row?.createdAt ?? (await entity.stat()).modified.toUtc(),
          captures: captures,
        ),
      );
    }
    products.sort((left, right) => left.categoryId.compareTo(right.categoryId));
    await _writeCatalog(products);
    return BreadCaptureLibrarySnapshot(
      rootPath: _rootPath,
      products: List.unmodifiable(products),
    );
  }

  @override
  Future<BreadCaptureProduct> addProduct(String name) async {
    final normalizedName = name.trim();
    if (normalizedName.isEmpty) throw ArgumentError('빵 이름을 입력해 주세요.');
    final snapshot = await load();
    final occupied = snapshot.products
        .map((product) => product.categoryId)
        .toSet();
    final categoryId = Iterable<int>.generate(
      20,
      (index) => index + 1,
    ).where((candidate) => !occupied.contains(candidate)).firstOrNull;
    if (categoryId == null) throw StateError('빵은 최대 20종까지 등록할 수 있어요.');
    final slug = _slug(normalizedName);
    final directoryName =
        'bread_${categoryId.toString().padLeft(2, '0')}_$slug';
    await Directory(
      paths.join(_rootPath, directoryName),
    ).create(recursive: false);
    final product = BreadCaptureProduct(
      categoryId: categoryId,
      name: normalizedName,
      directoryName: directoryName,
      createdAt: DateTime.now().toUtc(),
    );
    await _writeCatalog([...snapshot.products, product]);
    return product;
  }

  @override
  Future<void> deleteProduct(BreadCaptureProduct product) async {
    final snapshot = await load();
    final directory = _productDirectory(product);
    if (await directory.exists()) await directory.delete(recursive: true);
    await _writeCatalog(
      snapshot.products
          .where((candidate) => candidate.categoryId != product.categoryId)
          .toList(),
    );
  }

  @override
  Future<void> saveCapture(
    BreadCaptureProduct product,
    BreadCaptureSlot slot,
    Uint8List bytes,
  ) async {
    if (bytes.isEmpty) throw ArgumentError('촬영 이미지가 비어 있어요.');
    final directory = _productDirectory(product);
    await directory.create(recursive: false);
    final target = File(
      paths.join(directory.path, slot.fileName(product.categoryId)),
    );
    final temporary = File('${target.path}.tmp');
    await temporary.writeAsBytes(bytes, flush: true);
    if (await target.exists()) await target.delete();
    await temporary.rename(target.path);
  }

  @override
  Future<void> deleteCapture(
    BreadCaptureProduct product,
    BreadCaptureSlot slot,
  ) async {
    final file = File(
      paths.join(
        _productDirectory(product).path,
        slot.fileName(product.categoryId),
      ),
    );
    if (await file.exists()) await file.delete();
  }

  @override
  Future<BreadCaptureLibrarySnapshot> changeRoot(String rootPath) async {
    final normalized = paths.normalize(paths.absolute(rootPath.trim()));
    if (normalized.isEmpty) throw ArgumentError('저장 폴더를 선택해 주세요.');
    _rootPath = normalized;
    await _root.create(recursive: true);
    final settings = _settingsFile;
    if (settings != null) {
      await settings.parent.create(recursive: true);
      await _writeJsonAtomically(settings, {'capture_root': _rootPath});
    }
    return load();
  }

  @override
  Future<void> openRoot() async {
    await _root.create(recursive: true);
    await Process.start('explorer.exe', [_rootPath]);
  }

  Directory _productDirectory(BreadCaptureProduct product) {
    final directory = Directory(paths.join(_rootPath, product.directoryName));
    final normalizedRoot = paths.normalize(paths.absolute(_rootPath));
    final normalizedDirectory = paths.normalize(paths.absolute(directory.path));
    if (paths.dirname(normalizedDirectory) != normalizedRoot ||
        !_directoryPattern.hasMatch(paths.basename(normalizedDirectory))) {
      throw StateError('빵 촬영 폴더 경로가 올바르지 않아요.');
    }
    return Directory(normalizedDirectory);
  }

  Future<Map<int, _CatalogProduct>> _readCatalog() async {
    if (!await _catalogFile.exists()) return {};
    try {
      final body = jsonDecode(await _catalogFile.readAsString());
      final rows = body is Map<String, dynamic> ? body['products'] : null;
      if (rows is! List) return {};
      return {
        for (final row in rows)
          if (row is Map<String, dynamic> &&
              row['category_id'] is int &&
              row['name'] is String &&
              row['created_at'] is String)
            row['category_id'] as int: _CatalogProduct(
              name: row['name'] as String,
              createdAt: DateTime.parse(row['created_at'] as String).toUtc(),
            ),
      };
    } on Object {
      return {};
    }
  }

  Future<void> _writeCatalog(List<BreadCaptureProduct> products) async {
    products.sort((left, right) => left.categoryId.compareTo(right.categoryId));
    await _writeJsonAtomically(_catalogFile, {
      'schema_version': '1.0',
      'capture_contract': 'bread-pose-cutouts-20x2x5',
      'products': [
        for (final product in products)
          {
            'category_id': product.categoryId,
            'name': product.name,
            'directory_name': product.directoryName,
            'created_at': product.createdAt.toUtc().toIso8601String(),
          },
      ],
    });
  }

  Future<void> _writeJsonAtomically(File target, Object body) async {
    final temporary = File('${target.path}.tmp');
    await temporary.writeAsString(
      '${const JsonEncoder.withIndent('  ').convert(body)}\n',
      flush: true,
    );
    if (await target.exists()) await target.delete();
    await temporary.rename(target.path);
  }

  static String _slug(String name) {
    final slug = name
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9가-힣]+'), '_')
        .replaceAll(RegExp(r'^_+|_+$'), '');
    return slug.isEmpty ? 'item' : slug;
  }
}

class _CatalogProduct {
  const _CatalogProduct({required this.name, required this.createdAt});

  final String name;
  final DateTime createdAt;
}
