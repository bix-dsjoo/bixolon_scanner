import 'dart:async';
import 'dart:convert';
import 'dart:ui' show SemanticsAction, Tristate;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/app.dart';
import 'package:product_scanner/controllers/scanner_controller.dart';
import 'package:product_scanner/models/scan_models.dart';
import 'package:product_scanner/services/image_input.dart';
import 'package:product_scanner/services/scan_log_repository.dart';
import 'package:product_scanner/services/scanner_api.dart';
import 'package:product_scanner/theme/app_copy.dart';
import 'package:product_scanner/theme/app_tokens.dart';
import 'package:product_scanner/widgets/app_scan_guide.dart';

import 'support/test_catalog.dart';

void main() {
  testWidgets('카메라 확인 중·미연결·이미지 선택 완료의 준비 단계를 구분한다', (tester) async {
    final semantics = tester.ensureSemantics();
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final fileGateway = _EmptyFileGateway();
    final controller = ScannerController(
      _UnusedApi(),
      _EmptyCameraGateway(),
      fileGateway,
      _MemoryLogRepository(),
      testCatalog,
    );

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pump();

    expect(find.byType(AppScanGuide), findsNothing);
    expect(find.text('카메라 확인 중'), findsWidgets);
    expect(find.text('카메라를 준비하고 있어요'), findsWidgets);
    final initializingAction = find.widgetWithText(FilledButton, '연결 확인 중');
    expect(initializingAction, findsOneWidget);
    expect(tester.widget<FilledButton>(initializingAction).onPressed, isNull);
    final progress = find.bySemanticsLabel('카메라 연결 확인 중. 완료될 때까지 기다려 주세요');
    expect(progress, findsOneWidget);
    final progressData = tester.getSemantics(progress).getSemanticsData();
    expect(progressData.flagsCollection.isLiveRegion, isTrue);
    expect(progressData.flagsCollection.isButton, isTrue);
    expect(progressData.hasAction(SemanticsAction.tap), isFalse);
    expect(find.widgetWithText(OutlinedButton, '이미지 선택'), findsOneWidget);
    await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
    await tester.sendKeyDownEvent(LogicalKeyboardKey.keyO);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.keyO);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
    await tester.pump();
    expect(fileGateway.pickCalls, 1);
    expect(controller.cameraInitializing, isTrue);
    await tester.pump(const Duration(milliseconds: 600));
    await tester.pump();
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_camera_initializing_1280x720.png'),
    );

    controller
      ..cameraInitializing = false
      ..notifyListeners();
    await tester.pumpAndSettle();

    expect(find.byType(AppScanGuide), findsNothing);
    expect(find.text('입력 준비'), findsOneWidget);
    expect(find.text('카메라를 연결해 주세요'), findsWidgets);
    expect(find.widgetWithText(FilledButton, '다시 연결'), findsOneWidget);
    expect(find.widgetWithText(OutlinedButton, '이미지 선택'), findsOneWidget);
    expect(find.byType(FilledButton), findsOneWidget);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_ready_camera_off_1280x720.png'),
    );

    controller
      ..inputMode = InputMode.image
      ..imageBytes = _testInputImage.bytes
      ..imageFileName = _testInputImage.fileName
      ..imageSize = const Size(400, 400)
      ..processState = ProcessState.ready
      ..notifyListeners();
    await tester.pumpAndSettle();

    expect(find.byType(AppScanGuide), findsOneWidget);
    expect(find.text('분석 준비'), findsOneWidget);
    expect(find.text(AppPreviewCopy.selectedImage), findsOneWidget);
    expect(find.bySemanticsLabel('입력 미리보기, 선택한 이미지'), findsOneWidget);
    expect(find.text('이미지 분석 준비가 됐어요'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '분석하기'), findsOneWidget);
    expect(find.widgetWithText(OutlinedButton, '다른 이미지 선택'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '다시 연결'), findsNothing);
    expect(find.byType(FilledButton), findsOneWidget);
    expect(
      tester.getSize(find.widgetWithText(FilledButton, '분석하기')).height,
      greaterThanOrEqualTo(48),
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_ready_image_1280x720.png'),
    );

    controller.dispose();
    semantics.dispose();
  });

  testWidgets('카메라 연결 성공은 촬영 준비와 단일 Primary를 제공한다', (tester) async {
    final semantics = tester.ensureSemantics();
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    addTearDown(tester.platformDispatcher.clearTextScaleFactorTestValue);

    final controller = ScannerController(
      _UnusedApi(),
      _ReadyCameraGateway(),
      _EmptyFileGateway(),
      _MemoryLogRepository(),
      testCatalog,
    )..cameraInitializing = false;

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('카메라 연결됨'), findsOneWidget);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('카메라 연결됨'))
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    expect(find.text('촬영 준비'), findsOneWidget);
    expect(find.text(AppPreviewCopy.liveCamera), findsOneWidget);
    expect(find.bySemanticsLabel('입력 미리보기, 라이브 카메라'), findsOneWidget);
    expect(
      tester.getSize(find.byKey(const ValueKey('preview-source-label'))).height,
      greaterThanOrEqualTo(AppDesignTokens.standard.previewLabelMinHeight),
    );
    expect(find.text('상품을 촬영해 주세요'), findsOneWidget);
    expect(find.byKey(const ValueKey('test-camera-preview')), findsOneWidget);
    expect(find.widgetWithText(OutlinedButton, '이미지 선택'), findsOneWidget);
    final capture = find.widgetWithText(FilledButton, '촬영하기');
    expect(capture, findsOneWidget);
    expect(tester.widget<FilledButton>(capture).onPressed, isNotNull);
    expect(find.widgetWithText(FilledButton, '다시 연결'), findsNothing);
    expect(find.byType(FilledButton), findsOneWidget);
    expect(tester.getSize(capture).height, greaterThanOrEqualTo(48));
    expect(tester.takeException(), isNull);

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_ready_camera_1280x720.png'),
    );

    final imageSelect = find.widgetWithText(OutlinedButton, '이미지 선택');
    for (var index = 0; index < 12; index++) {
      await tester.sendKeyEvent(LogicalKeyboardKey.tab);
      await tester.pumpAndSettle();
      if (tester
              .getSemantics(imageSelect)
              .getSemanticsData()
              .flagsCollection
              .isFocused ==
          Tristate.isTrue) {
        break;
      }
    }
    expect(
      tester
          .getSemantics(imageSelect)
          .getSemanticsData()
          .flagsCollection
          .isFocused,
      Tristate.isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile(
        'goldens/scanner_ready_camera_secondary_focus_1280x720.png',
      ),
    );

    await tester.sendKeyEvent(LogicalKeyboardKey.tab);
    await tester.pumpAndSettle();
    expect(
      tester.getSemantics(capture).getSemanticsData().flagsCollection.isFocused,
      Tristate.isTrue,
    );
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile(
        'goldens/scanner_ready_camera_primary_focus_1280x720.png',
      ),
    );

    FocusManager.instance.primaryFocus?.unfocus();
    await tester.pumpAndSettle();

    tester.platformDispatcher.textScaleFactorTestValue = 1.5;
    await tester.pumpAndSettle();
    expect(find.widgetWithText(FilledButton, '촬영하기'), findsOneWidget);
    expect(find.byType(FilledButton), findsOneWidget);
    expect(tester.takeException(), isNull);
    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_ready_camera_text_150_1280x720.png'),
    );

    controller.dispose();
    semantics.dispose();
  });

  testWidgets('카메라 초기화 실패는 촬영 실패가 아닌 사용 불가 상태로 복구한다', (tester) async {
    final semantics = tester.ensureSemantics();
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final camera = _InitializationFailingCameraGateway();
    final controller = ScannerController(
      _UnusedApi(),
      camera,
      _EmptyFileGateway(),
      _MemoryLogRepository(),
      testCatalog,
    );

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await controller.initialize();
    await tester.pumpAndSettle();

    expect(controller.cameraIssueType, CameraIssueType.unavailable);
    expect(find.byType(AppScanGuide), findsNothing);
    expect(find.text('카메라 확인 필요'), findsWidgets);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('카메라 확인 필요'))
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isFalse,
    );
    expect(find.text('카메라를 사용할 수 없어요'), findsOneWidget);
    expect(find.text('카메라를 사용할 수 없어요. 연결 상태를 확인해 주세요.'), findsNWidgets(2));
    expect(find.text('촬영하지 못했어요'), findsNothing);
    expect(find.widgetWithText(OutlinedButton, '이미지 선택'), findsOneWidget);
    final reconnect = find.widgetWithText(FilledButton, '다시 연결');
    expect(reconnect, findsOneWidget);
    expect(find.byType(FilledButton), findsOneWidget);
    expect(tester.getSize(reconnect).height, greaterThanOrEqualTo(48));
    expect(tester.takeException(), isNull);
    await tester.pump();

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_camera_unavailable_1280x720.png'),
    );

    await tester.tap(reconnect);
    await tester.pumpAndSettle();
    expect(camera.initializeCalls, 2);
    expect(controller.cameraIssueType, CameraIssueType.unavailable);
    expect(find.text('카메라를 사용할 수 없어요'), findsOneWidget);
    controller.dispose();
    semantics.dispose();
  });

  testWidgets('촬영 실패는 카메라 확인 상태와 단일 재연결 행동으로 복구한다', (tester) async {
    final semantics = tester.ensureSemantics();
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final camera = _ReadyCameraGateway(captureFails: true);
    final controller = ScannerController(
      _UnusedApi(),
      camera,
      _EmptyFileGateway(),
      _MemoryLogRepository(),
      testCatalog,
    )..cameraInitializing = false;

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.widgetWithText(FilledButton, '촬영하기'));
    await tester.pumpAndSettle();

    expect(controller.processState, ProcessState.ready);
    expect(controller.isRecapture, isFalse);
    expect(find.text('카메라 확인 필요'), findsWidgets);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('카메라 확인 필요'))
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isFalse,
    );
    expect(find.text('촬영하지 못했어요'), findsOneWidget);
    expect(find.byType(AppScanGuide), findsNothing);
    expect(find.text('카메라 응답을 받지 못했어요. 다시 연결해 주세요.'), findsOneWidget);
    expect(find.text('분석 오류'), findsNothing);
    expect(find.text('재촬영 필요'), findsNothing);
    expect(find.widgetWithText(FilledButton, '촬영하기'), findsNothing);
    final reconnect = find.widgetWithText(FilledButton, '다시 연결');
    expect(reconnect, findsOneWidget);
    expect(find.byType(FilledButton), findsOneWidget);
    expect(tester.getSize(reconnect).height, greaterThanOrEqualTo(48));
    final cameraError = find.bySemanticsLabel(
      '촬영하지 못했어요. 카메라 응답을 받지 못했어요. 다시 연결해 주세요.',
    );
    expect(cameraError, findsOneWidget);
    final cameraErrorData = tester.getSemantics(cameraError).getSemanticsData();
    expect(cameraErrorData.flagsCollection.isLiveRegion, isTrue);
    expect(cameraErrorData.hasAction(SemanticsAction.tap), isFalse);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('다시 연결'))
          .getSemanticsData()
          .hasAction(SemanticsAction.tap),
      isTrue,
    );
    expect(tester.takeException(), isNull);

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_capture_error_1280x720.png'),
    );

    await tester.tap(reconnect);
    await tester.pumpAndSettle();
    expect(camera.initializeCalls, 1);
    expect(controller.cameraMessage, isNull);
    expect(find.text('카메라 연결됨'), findsOneWidget);
    expect(
      tester
          .getSemantics(find.bySemanticsLabel('카메라 연결됨'))
          .getSemanticsData()
          .flagsCollection
          .isLiveRegion,
      isTrue,
    );
    expect(find.widgetWithText(FilledButton, '촬영하기'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '다시 연결'), findsNothing);
    expect(find.byType(AppScanGuide), findsOneWidget);
    controller.dispose();
    semantics.dispose();
  });

  testWidgets('재촬영 중 카메라가 실패하면 이전 판정보다 재연결을 우선한다', (tester) async {
    tester.view.physicalSize = const Size(1280, 720);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    final camera = _ReadyCameraGateway(captureFails: true);
    final controller =
        ScannerController(
            _UnusedApi(),
            camera,
            _EmptyFileGateway(),
            _MemoryLogRepository(),
            testCatalog,
          )
          ..cameraInitializing = false
          ..inputMode = InputMode.camera
          ..processState = ProcessState.reviewing
          ..response = _recaptureResponse
          ..imageBytes = _testInputImage.bytes
          ..imageFileName = _testInputImage.fileName
          ..imageSize = const Size(400, 400);

    await tester.pumpWidget(
      ProductScannerApp(
        controller: controller,
        autoInitialize: false,
        disposeController: false,
      ),
    );
    await tester.runAsync(
      () => precacheImage(
        MemoryImage(_testInputImage.bytes),
        tester.element(find.byType(Scaffold).first),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('재촬영 필요'), findsWidgets);
    await tester.tap(find.widgetWithText(FilledButton, '다시 촬영'));
    await tester.pumpAndSettle();

    expect(controller.processState, ProcessState.reviewing);
    expect(controller.response, same(_recaptureResponse));
    expect(controller.hasActiveCameraIssue, isTrue);
    expect(find.byType(AppScanGuide), findsNothing);
    expect(find.text('카메라 확인 필요'), findsWidgets);
    expect(find.text('촬영하지 못했어요'), findsOneWidget);
    expect(find.text('재촬영 필요'), findsNothing);
    expect(find.text('상품을 찾지 못했어요'), findsNothing);
    expect(find.widgetWithText(FilledButton, '다시 촬영'), findsNothing);
    final reconnect = find.widgetWithText(FilledButton, '다시 연결');
    expect(reconnect, findsOneWidget);
    expect(find.byType(FilledButton), findsOneWidget);
    expect(tester.getSize(reconnect).height, greaterThanOrEqualTo(48));
    expect(tester.takeException(), isNull);

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_recapture_capture_error_1280x720.png'),
    );

    camera.deferNextInitialize();
    await tester.tap(reconnect);
    await tester.pump();
    expect(controller.isCameraCheckActive, isTrue);
    expect(find.byType(AppScanGuide), findsNothing);
    expect(find.text('카메라 확인 중'), findsWidgets);
    expect(find.text('카메라를 준비하고 있어요'), findsOneWidget);
    expect(find.text('재촬영 필요'), findsNothing);
    expect(find.text('상품을 찾지 못했어요'), findsNothing);
    expect(find.widgetWithText(FilledButton, '다시 촬영'), findsNothing);
    final checking = find.widgetWithText(FilledButton, '연결 확인 중');
    expect(checking, findsOneWidget);
    expect(tester.widget<FilledButton>(checking).onPressed, isNull);
    expect(find.byType(FilledButton), findsOneWidget);
    await tester.pump(const Duration(milliseconds: 600));
    await tester.pump();

    await expectLater(
      find.byType(Scaffold).first,
      matchesGoldenFile('goldens/scanner_recapture_reconnecting_1280x720.png'),
    );

    await controller.reconnectCamera();
    expect(camera.initializeCalls, 1);
    camera.completeDeferredInitialize();
    await tester.pumpAndSettle();
    expect(camera.initializeCalls, 1);
    expect(controller.hasActiveCameraIssue, isFalse);
    expect(controller.isCameraCheckActive, isFalse);
    expect(find.byType(AppScanGuide), findsOneWidget);
    expect(find.text('재촬영 필요'), findsWidgets);
    expect(find.widgetWithText(FilledButton, '다시 촬영'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, '다시 연결'), findsNothing);
    controller.dispose();
  });
}

final InputImage _testInputImage = InputImage(
  bytes: base64Decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  ),
  fileName: 'test.png',
);

const ScanResponse _recaptureResponse = ScanResponse(
  requestId: 'request_recapture',
  status: ScanStatus.recapture,
  reasonCodes: ['DETECTOR_NO_OBJECT'],
  items: [],
  processingTimeMs: 42,
  modelVersions: ModelVersions(detector: '0.1.1'),
);

class _UnusedApi implements ScannerApi {
  @override
  Future<ScanResponse> scan({
    required Uint8List imageBytes,
    required String fileName,
  }) => throw UnimplementedError();
}

class _EmptyCameraGateway implements CameraGateway {
  @override
  CameraController? get controller => null;

  @override
  bool get isReady => false;

  @override
  Future<InputImage> capture() => throw UnimplementedError();

  @override
  Future<void> dispose() async {}

  @override
  Future<void> initialize() async {}
}

class _ReadyCameraGateway implements CameraGateway {
  _ReadyCameraGateway({this.captureFails = false})
    : _controller = _TestCameraController();

  final _TestCameraController _controller;
  final bool captureFails;
  int initializeCalls = 0;
  Completer<void>? _initializeCompleter;

  void deferNextInitialize() {
    _initializeCompleter = Completer<void>();
  }

  void completeDeferredInitialize() {
    _initializeCompleter?.complete();
    _initializeCompleter = null;
  }

  @override
  CameraController get controller => _controller;

  @override
  bool get isReady => true;

  @override
  Future<InputImage> capture() async {
    if (captureFails) throw StateError('capture failed');
    return _testInputImage;
  }

  @override
  Future<void> dispose() async {}

  @override
  Future<void> initialize() async {
    initializeCalls += 1;
    final completer = _initializeCompleter;
    if (completer != null) await completer.future;
  }
}

class _InitializationFailingCameraGateway implements CameraGateway {
  int initializeCalls = 0;

  @override
  CameraController? get controller => null;

  @override
  bool get isReady => false;

  @override
  Future<InputImage> capture() => throw UnimplementedError();

  @override
  Future<void> dispose() async {}

  @override
  Future<void> initialize() {
    initializeCalls += 1;
    throw StateError('camera unavailable');
  }
}

class _TestCameraController extends CameraController {
  _TestCameraController()
    : super(
        const CameraDescription(
          name: 'test-camera',
          lensDirection: CameraLensDirection.back,
          sensorOrientation: 0,
        ),
        ResolutionPreset.low,
        enableAudio: false,
      ) {
    value = value.copyWith(
      isInitialized: true,
      previewSize: const Size(1280, 720),
    );
  }

  @override
  Widget buildPreview() => const ColoredBox(
    key: ValueKey('test-camera-preview'),
    color: Color(0xFF242424),
  );
}

class _EmptyFileGateway implements ImageFileGateway {
  int pickCalls = 0;

  @override
  Future<InputImage?> pick() async {
    pickCalls += 1;
    return null;
  }
}

class _MemoryLogRepository implements ScanLogRepository {
  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async => const [];

  @override
  Future<void> save(ScanLogRecord record) async {}
}
