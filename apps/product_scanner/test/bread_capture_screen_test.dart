import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/core/design_system/theme.dart';
import 'package:product_scanner/features/capture_library/application/bread_capture_controller.dart';
import 'package:product_scanner/features/capture_library/data/bread_capture_repository.dart';
import 'package:product_scanner/features/capture_library/domain/bread_capture_models.dart';
import 'package:product_scanner/features/capture_library/presentation/bread_capture_screen.dart';
import 'package:product_scanner/shared/input/image_input.dart';

void main() {
  test('가로 카메라 종횡비를 뒤집지 않는다', () {
    const cameraAspectRatio = 16 / 9;

    expect(
      breadCaptureViewportAspectRatio(cameraAspectRatio),
      closeTo(cameraAspectRatio, 1e-9),
    );
  });

  testWidgets('Windows 미러 미리보기를 저장 사진 방향으로 보정한다', (tester) async {
    await tester.pumpWidget(
      Directionality(
        textDirection: TextDirection.ltr,
        child: breadCaptureUnmirroredPreview(
          const SizedBox(key: ValueKey('camera-texture')),
        ),
      ),
    );

    final correction = tester.widget<Transform>(find.byType(Transform));
    expect(correction.transform.storage[0], -1);
    expect(find.byKey(const ValueKey('camera-texture')), findsOneWidget);
  });

  testWidgets('빵 목록·카메라 가이드·10장 콘택트 시트를 한 화면에 표시한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final controller = BreadCaptureController(
      _ScreenRepository(),
      _UnavailableCameraGateway(),
      _NoopDirectoryPicker(),
    );
    await controller.initialize();

    await tester.pumpWidget(
      MaterialApp(
        theme: buildAppTheme(),
        home: Scaffold(
          body: BreadCaptureScreen(controller: controller, active: false),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('빵 촬영 관리'), findsOneWidget);
    expect(find.text('호두 도넛'), findsWidgets);
    expect(find.text('정상 면'), findsWidgets);
    expect(find.text('뒤집은 면'), findsWidgets);
    expect(
      find.byWidgetPredicate(
        (widget) =>
            widget.key is ValueKey<String> &&
            (widget.key! as ValueKey<String>).value.startsWith(
              'bread-capture-slot-',
            ),
      ),
      findsNWidgets(10),
    );
    expect(tester.takeException(), isNull);

    await expectLater(
      find.byType(BreadCaptureScreen),
      matchesGoldenFile('goldens/bread_capture_workspace_1280x720.png'),
    );

    tester.view.physicalSize = const Size(1280, 656);
    await tester.pump();
    expect(tester.takeException(), isNull);
  });
}

class _UnavailableCameraGateway implements CameraGateway {
  @override
  CameraController? get controller => null;

  @override
  bool get isReady => false;

  @override
  Future<void> initialize() async {}

  @override
  Future<InputImage> capture() => throw UnimplementedError();

  @override
  Future<void> dispose() async {}
}

class _NoopDirectoryPicker implements BreadCaptureDirectoryPicker {
  @override
  Future<String?> pick() async => null;
}

class _ScreenRepository implements BreadCaptureRepository {
  final BreadCaptureProduct product = BreadCaptureProduct(
    categoryId: 1,
    name: '호두 도넛',
    directoryName: 'bread_01_walnut_donut',
    createdAt: DateTime.utc(2026, 8, 31),
  );

  @override
  String get rootPath => r'C:\captures\single_objects';

  @override
  Future<BreadCaptureProduct> addProduct(String name) async => product;

  @override
  Future<BreadCaptureLibrarySnapshot> changeRoot(String rootPath) => load();

  @override
  Future<void> deleteCapture(
    BreadCaptureProduct product,
    BreadCaptureSlot slot,
  ) async {}

  @override
  Future<void> deleteProduct(BreadCaptureProduct product) async {}

  @override
  Future<BreadCaptureLibrarySnapshot> load() async =>
      BreadCaptureLibrarySnapshot(rootPath: rootPath, products: [product]);

  @override
  Future<void> openRoot() async {}

  @override
  Future<void> saveCapture(
    BreadCaptureProduct product,
    BreadCaptureSlot slot,
    Uint8List bytes,
  ) async {}
}
