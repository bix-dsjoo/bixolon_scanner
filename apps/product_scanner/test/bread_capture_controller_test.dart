import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/features/capture_library/application/bread_capture_controller.dart';
import 'package:product_scanner/features/capture_library/data/bread_capture_repository.dart';
import 'package:product_scanner/features/capture_library/domain/bread_capture_models.dart';
import 'package:product_scanner/shared/input/image_input.dart';

void main() {
  test('촬영 확인 후 저장하고 다음 미완료 자세로 이동한다', () async {
    final repository = _MemoryBreadCaptureRepository();
    final camera = _FakeCameraGateway();
    final controller = BreadCaptureController(
      repository,
      camera,
      _FakeDirectoryPicker(),
    );

    await controller.activate();
    await controller.addProduct('호두 도넛');
    final firstSlot = BreadCaptureSlot.all.first;

    await controller.capture();

    expect(camera.initializeCalls, 1);
    expect(controller.pendingImageBytes, [4, 5, 6]);
    expect(controller.selectedSlot.key, firstSlot.key);

    await controller.acceptPendingCapture();

    expect(controller.pendingImageBytes, isNull);
    expect(controller.selectedProduct!.completedCaptureCount, 1);
    expect(controller.selectedSlot.key, BreadCaptureSlot.all[1].key);
    expect(repository.saved[firstSlot.key], [4, 5, 6]);

    await controller.deleteCapture(firstSlot);
    expect(controller.selectedProduct!.completedCaptureCount, 0);
    expect(controller.selectedSlot.key, firstSlot.key);
  });
}

class _FakeCameraGateway implements CameraGateway {
  bool ready = false;
  int initializeCalls = 0;

  @override
  CameraController? get controller => null;

  @override
  bool get isReady => ready;

  @override
  Future<void> initialize() async {
    initializeCalls += 1;
    ready = true;
  }

  @override
  Future<InputImage> capture() async =>
      InputImage(bytes: Uint8List.fromList([4, 5, 6]), fileName: 'capture.jpg');

  @override
  Future<void> dispose() async {}
}

class _FakeDirectoryPicker implements BreadCaptureDirectoryPicker {
  @override
  Future<String?> pick() async => null;
}

class _MemoryBreadCaptureRepository implements BreadCaptureRepository {
  final Map<String, Uint8List> saved = {};
  final List<BreadCaptureProduct> _products = [];

  @override
  String get rootPath => r'C:\captures';

  @override
  Future<BreadCaptureProduct> addProduct(String name) async {
    final product = BreadCaptureProduct(
      categoryId: 1,
      name: name,
      directoryName: 'bread_01_walnut_donut',
      createdAt: DateTime.utc(2026, 8, 31),
    );
    _products.add(product);
    return product;
  }

  @override
  Future<BreadCaptureLibrarySnapshot> changeRoot(String rootPath) => load();

  @override
  Future<void> deleteCapture(
    BreadCaptureProduct product,
    BreadCaptureSlot slot,
  ) async {
    saved.remove(slot.key);
  }

  @override
  Future<void> deleteProduct(BreadCaptureProduct product) async {
    _products.removeWhere((row) => row.categoryId == product.categoryId);
  }

  @override
  Future<BreadCaptureLibrarySnapshot> load() async {
    final products = [
      for (final product in _products)
        product.copyWith(
          captures: {for (final entry in saved.entries) entry.key: entry.key},
        ),
    ];
    return BreadCaptureLibrarySnapshot(rootPath: rootPath, products: products);
  }

  @override
  Future<void> openRoot() async {}

  @override
  Future<void> saveCapture(
    BreadCaptureProduct product,
    BreadCaptureSlot slot,
    Uint8List bytes,
  ) async {
    saved[slot.key] = Uint8List.fromList(bytes);
  }
}
