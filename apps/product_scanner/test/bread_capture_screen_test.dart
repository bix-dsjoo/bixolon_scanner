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
import 'package:product_scanner/shared/input/square_camera_preview.dart';

void main() {
  testWidgets('카메라를 왜곡 없이 중앙 1080 정사각형으로 자르고 미러를 보정한다', (tester) async {
    final camera = _TestCameraController();
    await tester.pumpWidget(
      MaterialApp(
        home: Center(
          child: SizedBox(
            width: 400,
            height: 400,
            child: SquareCameraPreview(controller: camera),
          ),
        ),
      ),
    );

    final viewport = tester.widget<AspectRatio>(find.byType(AspectRatio).first);
    expect(viewport.aspectRatio, 1);
    final crop = tester.widget<Transform>(
      find.byKey(const ValueKey('square-camera-center-crop')),
    );
    expect(crop.transform.storage[0], closeTo(1, 1e-9));
    expect(crop.alignment, Alignment.center);
    final fitted = tester.widget<FittedBox>(find.byType(FittedBox));
    expect(fitted.fit, BoxFit.cover);
    expect(fitted.alignment, Alignment.center);
    final correction = tester.widget<Transform>(
      find.byKey(const ValueKey('camera-preview-mirror')),
    );
    expect(correction.transform.storage[0], -1);
    expect(find.byKey(const ValueKey('test-camera-preview')), findsOneWidget);
    await camera.dispose();
  });

  testWidgets('8MP 미리보기에서도 중앙의 원본 1080px 영역만 표시한다', (tester) async {
    final camera = _TestCameraController();
    camera.value = camera.value.copyWith(previewSize: const Size(3264, 2448));
    await tester.pumpWidget(
      MaterialApp(
        home: Center(
          child: SizedBox(
            width: 400,
            height: 400,
            child: SquareCameraPreview(controller: camera),
          ),
        ),
      ),
    );
    final crop = tester.widget<Transform>(
      find.byKey(const ValueKey('square-camera-center-crop')),
    );
    expect(crop.transform.storage[0], closeTo(2448 / 1080, 1e-9));
    expect(crop.alignment, Alignment.center);
    final fitted = tester.widget<FittedBox>(find.byType(FittedBox));
    expect(fitted.alignment, Alignment.center);
    await camera.dispose();
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

class _TestCameraController extends CameraController {
  _TestCameraController()
    : super(
        const CameraDescription(
          name: 'test-camera',
          lensDirection: CameraLensDirection.back,
          sensorOrientation: 0,
        ),
        ResolutionPreset.veryHigh,
        enableAudio: false,
        fps: 30,
      ) {
    value = value.copyWith(
      isInitialized: true,
      previewSize: const Size(1920, 1080),
    );
  }

  @override
  Widget buildPreview() => const ColoredBox(
    key: ValueKey('test-camera-preview'),
    color: Color(0xFF242424),
  );
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
