import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' show SemanticsAction, Tristate;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:product_scanner/app.dart';
import 'package:product_scanner/controllers/scanner_controller.dart';
import 'package:product_scanner/models/scan_models.dart';
import 'package:product_scanner/services/image_input.dart';
import 'package:product_scanner/services/scan_log_repository.dart';
import 'package:product_scanner/services/scanner_api.dart';

import 'support/test_catalog.dart';

void main() {
  const viewports = <Size>[Size(1280, 720), Size(1440, 900)];

  for (final viewport in viewports) {
    group('${viewport.width.toInt()}x${viewport.height.toInt()} 운영 상태', () {
      for (final scenario in _scanScenarios) {
        testWidgets('${scenario.name}: 오버플로·Primary·접근 가능한 조작 영역을 유지한다', (
          tester,
        ) async {
          await _setViewport(tester, viewport);
          final semantics = tester.ensureSemantics();
          try {
            final controller = scenario.createController();
            addTearDown(controller.dispose);

            await tester.pumpWidget(
              ProductScannerApp(
                controller: controller,
                autoInitialize: false,
                disposeController: false,
              ),
            );
            if (scenario.continuousProgress) {
              await tester.pump(const Duration(milliseconds: 300));
            } else {
              await tester.pumpAndSettle();
            }

            expect(tester.getSize(find.byType(Scaffold).first), viewport);
            expect(find.text(scenario.expectedText), findsAtLeastNWidgets(1));
            _expectOperationalSurface(tester);
            _expectAccessibleTapTargets(tester);
          } finally {
            semantics.dispose();
          }
        });
      }

      for (final scenario in _activityScenarios) {
        testWidgets('${scenario.name}: 오버플로·Primary·접근 가능한 조작 영역을 유지한다', (
          tester,
        ) async {
          await _setViewport(tester, viewport);
          final semantics = tester.ensureSemantics();
          try {
            final repository = _MatrixLogRepository(
              logs: scenario.logs,
              failList: scenario.failList,
            );
            final controller = _baseController(repository);
            addTearDown(controller.dispose);

            await tester.pumpWidget(
              ProductScannerApp(
                controller: controller,
                autoInitialize: false,
                disposeController: false,
              ),
            );
            await tester.tap(find.text('활동'));
            await tester.pumpAndSettle();

            expect(tester.getSize(find.byType(Scaffold).first), viewport);
            expect(find.text(scenario.expectedText), findsAtLeastNWidgets(1));
            _expectOperationalSurface(tester);
            _expectAccessibleTapTargets(tester);
          } finally {
            semantics.dispose();
          }
        });
      }
    });
  }
}

Future<void> _setViewport(WidgetTester tester, Size viewport) async {
  tester.view.physicalSize = viewport;
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.resetPhysicalSize);
  addTearDown(tester.view.resetDevicePixelRatio);
}

void _expectOperationalSurface(WidgetTester tester) {
  expect(tester.takeException(), isNull);
  final primaryActions = find.byType(FilledButton);
  expect(primaryActions.evaluate().length, lessThanOrEqualTo(1));
  for (final element in primaryActions.evaluate()) {
    expect(
      tester.getSize(find.byWidget(element.widget)).height,
      greaterThanOrEqualTo(48),
    );
  }
}

void _expectAccessibleTapTargets(WidgetTester tester) {
  final sizeViolations = <String>[];
  final unnamedTargets = <String>[];
  final clippedTargets = <String>[];
  final nonFocusableTargets = <String>[];
  final workspaceRect = tester.getRect(find.byType(Scaffold).first);
  for (final node in find.semantics.byAction(SemanticsAction.tap).evaluate()) {
    final data = node.getSemanticsData();
    final rect = node.rect;
    final accessibleName = '${data.label} ${data.hint}'.trim();
    if (accessibleName.isEmpty) {
      unnamedTargets.add(
        '<레이블 없음>: ${rect.width.toStringAsFixed(1)}×${rect.height.toStringAsFixed(1)}',
      );
    }
    if (rect.width < 44 || rect.height < 44) {
      final label = data.label.isEmpty ? '<레이블 없음>' : data.label;
      sizeViolations.add(
        '$label: ${rect.width.toStringAsFixed(1)}×${rect.height.toStringAsFixed(1)}',
      );
    }
    if (!workspaceRect.contains(rect.topLeft) ||
        !workspaceRect.contains(rect.bottomRight - const Offset(.001, .001))) {
      final label = data.label.isEmpty ? '<레이블 없음>' : data.label;
      clippedTargets.add('$label: $rect');
    }
    if (data.flagsCollection.isFocused == Tristate.none) {
      final label = data.label.isEmpty ? '<레이블 없음>' : data.label;
      nonFocusableTargets.add(label);
    }
  }
  expect(
    sizeViolations,
    isEmpty,
    reason: '활성 조작 영역은 모두 최소 44×44px이어야 합니다: $sizeViolations',
  );
  expect(
    unnamedTargets,
    isEmpty,
    reason: '활성 조작 영역에는 접근 가능한 이름이 있어야 합니다: $unnamedTargets',
  );
  expect(
    clippedTargets,
    isEmpty,
    reason: '활성 조작 영역은 작업대 안에 완전히 보여야 합니다: $clippedTargets',
  );
  expect(
    nonFocusableTargets,
    isEmpty,
    reason: '활성 조작 영역은 키보드 포커스를 받을 수 있어야 합니다: $nonFocusableTargets',
  );
}

ScannerController _baseController(ScanLogRepository repository) =>
    ScannerController(
        _UnusedApi(),
        _UnavailableCameraGateway(),
        _EmptyFileGateway(),
        repository,
        testCatalog,
      )
      ..cameraInitializing = false
      ..inputMode = InputMode.image;

ScannerController _scanController({
  required ProcessState processState,
  ScanResponse? response,
  bool hasImage = true,
  String? completionMessage,
}) {
  final controller = _baseController(_MatrixLogRepository());
  if (hasImage) {
    controller
      ..imageBytes = _imageBytes
      ..imageFileName = 'matrix.png'
      ..imageSize = const Size(400, 400);
  }
  controller
    ..processState = processState
    ..response = response
    ..completionMessage = completionMessage;
  if (response != null && response.items.isNotEmpty) {
    controller
      ..detections = testReviewDetections(response)
      ..selectedItemId = response.items.first.itemId;
  }
  return controller;
}

ScannerController _submittingController() {
  final controller = _scanController(
    processState: ProcessState.reviewing,
    response: _unknownResponse,
  );
  controller.confirmCandidate(
    _unknownResponse.items[1].itemId,
    _unknownResponse.items[1].top3.first,
  );
  controller.processState = ProcessState.submitting;
  return controller;
}

ScannerController _errorController() =>
    _scanController(processState: ProcessState.error, response: _errorResponse)
      ..errorMessage = '분석 서버에 연결하지 못했어요.'
      ..errorRecovery = ScannerErrorRecovery.retryAnalysis;

class _ScanScenario {
  const _ScanScenario(
    this.name,
    this.expectedText,
    this.createController, {
    this.continuousProgress = false,
  });

  final String name;
  final String expectedText;
  final ScannerController Function() createController;
  final bool continuousProgress;
}

final _scanScenarios = <_ScanScenario>[
  _ScanScenario(
    '준비',
    '분석 준비',
    () => _scanController(processState: ProcessState.ready),
  ),
  _ScanScenario(
    '촬영 중',
    '촬영 중',
    () =>
        _scanController(processState: ProcessState.capturing, hasImage: false)
          ..inputMode = InputMode.camera,
    continuousProgress: true,
  ),
  _ScanScenario(
    '분석 중',
    '분석 중',
    () => _scanController(processState: ProcessState.analyzing),
    continuousProgress: true,
  ),
  _ScanScenario(
    'APPROVED',
    '검수 완료',
    () => _scanController(
      processState: ProcessState.reviewing,
      response: _approvedResponse,
    ),
  ),
  _ScanScenario(
    'UNKNOWN',
    '상품 확인이 필요해요',
    () => _scanController(
      processState: ProcessState.reviewing,
      response: _unknownResponse,
    ),
  ),
  _ScanScenario(
    'RECAPTURE',
    '이미지가 흔들렸어요',
    () => _scanController(
      processState: ProcessState.reviewing,
      response: _recaptureResponse,
    ),
  ),
  _ScanScenario('ERROR', '분석 서버에 연결하지 못했어요.', _errorController),
  _ScanScenario(
    '저장 중',
    '저장 중',
    _submittingController,
    continuousProgress: true,
  ),
  _ScanScenario(
    '완료',
    '1개 상품을 확정했어요',
    () => _scanController(
      processState: ProcessState.ready,
      hasImage: false,
      completionMessage: '1개 상품을 확정했어요',
    ),
  ),
];

class _ActivityScenario {
  const _ActivityScenario(
    this.name,
    this.expectedText, {
    this.logs = const <ScanLogSummary>[],
    this.failList = false,
  });

  final String name;
  final String expectedText;
  final List<ScanLogSummary> logs;
  final bool failList;
}

final _activityScenarios = <_ActivityScenario>[
  _ActivityScenario('Activity 목록·상세', '확정 상품', logs: <ScanLogSummary>[_log]),
  _ActivityScenario('Activity 빈 상태', '저장된 활동이 없어요'),
  _ActivityScenario('Activity 오류', '활동 기록을 불러오지 못했어요', failList: true),
];

final _approvedResponse = ScanResponse(
  requestId: 'matrix-approved',
  status: ScanStatus.approved,
  reasonCodes: const <String>[],
  items: <ScanItem>[_approvedItem],
  processingTimeMs: 70,
  modelVersions: const ModelVersions(detector: '0.1.1', classifier: '0.1.1'),
);

final _unknownResponse = ScanResponse(
  requestId: 'matrix-unknown',
  status: ScanStatus.unknown,
  reasonCodes: const <String>[],
  items: <ScanItem>[
    _approvedItem,
    ScanItem(
      itemId: 'item_002',
      bbox: const BoundingBox(x: 170, y: 150, width: 100, height: 120),
      status: ItemStatus.unknown,
      reasonCodes: const <String>['CLASSIFIER_LOW_CONFIDENCE'],
      prediction: null,
      top3: const <Candidate>[
        Candidate(
          classId: 'bread_13',
          className: 'Muffin',
          displayName: '머핀',
          confidence: .75,
        ),
        Candidate(
          classId: 'bread_04',
          className: 'Scon',
          displayName: '스콘',
          confidence: .16,
        ),
        Candidate(
          classId: 'bread_11',
          className: 'Bagel',
          displayName: '베이글',
          confidence: .09,
        ),
      ],
      confidence: .75,
    ),
  ],
  processingTimeMs: 74,
  modelVersions: const ModelVersions(detector: '0.1.1', classifier: '0.1.1'),
);

final _recaptureResponse = ScanResponse(
  requestId: 'matrix-recapture',
  status: ScanStatus.recapture,
  reasonCodes: const <String>['DETECTOR_BLUR'],
  items: const <ScanItem>[],
  processingTimeMs: 41,
  modelVersions: const ModelVersions(detector: '0.1.1'),
);

final _errorResponse = ScanResponse(
  requestId: 'matrix-error',
  status: ScanStatus.error,
  reasonCodes: const <String>['WORKER_UNAVAILABLE'],
  items: const <ScanItem>[],
  processingTimeMs: 0,
  modelVersions: const ModelVersions(),
);

const _approvedItem = ScanItem(
  itemId: 'item_001',
  bbox: BoundingBox(x: 60, y: 50, width: 120, height: 130),
  status: ItemStatus.approved,
  reasonCodes: <String>[],
  prediction: Product(
    classId: 'bread_06',
    className: 'Croissant',
    displayName: '크루아상',
  ),
  top3: <Candidate>[],
  confidence: .99,
);

final _log = ScanLogSummary(
  scanId: 'matrix-log',
  analyzedAt: DateTime(2026, 8, 11, 9),
  confirmedAt: DateTime(2026, 8, 11, 9, 1),
  inputMode: InputMode.image,
  processingTimeMs: 70,
  modelVersions: const ModelVersions(detector: '0.1.1', classifier: '0.1.1'),
  items: const <ScanLogItemSummary>[
    ScanLogItemSummary(
      itemId: 'item_001',
      productName: '크루아상',
      confidence: .99,
      userModified: false,
      confirmationMethod: 'AUTO_APPROVED',
      classId: 'bread_06',
      className: 'Croissant',
    ),
  ],
);

final Uint8List _imageBytes = base64Decode(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
);

class _UnusedApi implements ScannerApi {
  @override
  Future<ScanResponse> scan({
    required Uint8List imageBytes,
    required String fileName,
  }) {
    throw UnimplementedError();
  }
}

class _UnavailableCameraGateway implements CameraGateway {
  @override
  CameraController? get controller => null;

  @override
  bool get isReady => false;

  @override
  Future<void> initialize() async {}

  @override
  Future<InputImage> capture() {
    throw UnimplementedError();
  }

  @override
  Future<void> dispose() async {}
}

class _EmptyFileGateway implements ImageFileGateway {
  @override
  Future<InputImage?> pick() async => null;
}

class _MatrixLogRepository implements ScanLogRepository {
  _MatrixLogRepository({
    this.logs = const <ScanLogSummary>[],
    this.failList = false,
  });

  final List<ScanLogSummary> logs;
  final bool failList;

  @override
  Future<List<ScanLogSummary>> list({int limit = 100}) async {
    if (failList) throw StateError('matrix load failure');
    return logs.take(limit).toList(growable: false);
  }

  @override
  Future<void> save(ScanLogRecord record) async {}
}
