import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:bakery_scanner_lite/features/scanner/scan_controller.dart';
import 'package:bakery_scanner_lite/lite_app.dart';
import 'package:bakery_scanner_lite/shared/result.dart';
import 'package:bakery_scanner_lite/shared/run_log.dart';
import 'package:bakery_scanner_lite/shared/worker_client.dart';
import 'fakes.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  testWidgets('Small Windows window with larger text remains usable', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(800, 640);
    tester.view.devicePixelRatio = 1;
    tester.platformDispatcher.textScaleFactorTestValue = 1.3;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    addTearDown(tester.platformDispatcher.clearTextScaleFactorTestValue);
    final c = ScanController(FakeInputs(), FakeScanner(), MemoryLogs());
    await c.initialize();
    await c.run(camera: false);
    await tester.pumpWidget(LiteApp(controller: c));
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('pick')), findsOneWidget);
    expect(tester.takeException(), isNull);
    await tester.tap(find.text('실행 로그'));
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox());
    await c.close();
  });
  test(
    'Both journal writes can fail without losing the displayed result or export',
    () async {
      final temp = await Directory.systemTemp.createTemp('lite-both-fail-');
      addTearDown(() => temp.delete(recursive: true));
      final blocked = File('${temp.path}/blocked');
      await blocked.writeAsString('not a directory');
      final inputs = FakeInputs()..export = '${temp.path}/export.jsonl';
      final logs = FileRunLogs(
        Directory(blocked.path),
        Directory(blocked.path),
      );
      final c = ScanController(inputs, FakeScanner(), logs);
      await c.initialize();
      await c.run(camera: false);
      expect(c.result!.status, 'SEGMENTATION');
      expect(c.storageWarning, true);
      expect(c.lastRecord!.storageState, 'failed');
      await c.exportLogs();
      expect(c.exportMessage, '로그를 내보냈습니다.');
      final content = await File(inputs.export!).readAsString();
      expect(content, contains('LOG_WRITE_FAILED'));
      expect(content, contains('UNKNOWN'));
      await c.close();
    },
  );
  test('Worker projection contains only immutable Lite fields', () {
    final result = ScanResult.fromWorker(response());
    final encoded = jsonEncode(result.toJson());
    for (final forbidden in [
      'confidence',
      'prediction',
      'secret-candidate',
      'secret-label',
    ]) {
      expect(encoded, isNot(contains(forbidden)));
    }
    expect(result.objects.first.top3.map((c) => c.className), [
      'Almond Scone',
      'Cream Bun',
      'Baguette',
    ]);
    expect(() => result.objects.first.top3.clear(), throwsUnsupportedError);
    expect(() => result.objects.clear(), throwsUnsupportedError);
    expect(() => result.versions.clear(), throwsUnsupportedError);
    expect(() => result.objects.first.reasons.clear(), throwsUnsupportedError);
  });
  test(
    'Reject invalid status, version, empty segmentation and nonfinite time',
    () {
      for (final change in [
        {'status': 'OK'},
        {'worker_version': '0.1.15'},
        {'segmentations': []},
        {'processing_time_ms': double.nan},
      ]) {
        expect(
          () => ScanResult.fromWorker({...response(), ...change}),
          throwsFormatException,
        );
      }
    },
  );
  test('HTTP failures remain ERROR and never recapture', () async {
    final workerError = response(status: 'ERROR');
    final client = WorkerClient(
      baseUri: Uri.parse('http://localhost'),
      client: MockClient(
        (_) async => http.Response(jsonEncode(workerError), 500),
      ),
    );
    expect((await client.scan(pixel, 'test.png', 'attempt')).status, 'ERROR');
    await client.close();
    final bad = WorkerClient(
      baseUri: Uri.parse('http://localhost'),
      client: MockClient((_) async => http.Response('broken', 503)),
    );
    final result = await bad.scan(pixel, 'test.png', 'attempt');
    expect(result.status, 'ERROR');
    expect(result.reasons, ['CLIENT_RESPONSE_INVALID']);
    await bad.close();
  });
  test(
    'Automatic input/result logs; no duplicate scan or camera switch while running',
    () async {
      final inputs = FakeInputs(), scanner = FakeScanner(), logs = MemoryLogs();
      final c = ScanController(inputs, scanner, logs);
      await c.initialize();
      await c.selectCamera(inputs.descriptions.last);
      expect(inputs.chosen, 'USB camera');
      scanner.pending = Completer();
      final future = c.run(camera: true);
      await Future<void>.delayed(Duration.zero);
      await c.run(camera: false);
      await c.selectCamera(inputs.descriptions.first);
      expect(scanner.scans, 1);
      expect(inputs.chosen, 'USB camera');
      expect(logs.rows.single.result, isNull);
      scanner.pending!.complete(scanner.value);
      await future;
      expect(logs.rows.length, 2);
      expect(logs.rows.last.result!.requestId, 'request-test-001');
      expect(logs.rows.last.inputName, 'USB camera');
      expect(c.busy, false);
      await c.close();
    },
  );
  test(
    'Cancel preserves previous result; input failure retains filename; log failure does not block result',
    () async {
      final inputs = FakeInputs(), scanner = FakeScanner(), logs = MemoryLogs();
      final c = ScanController(inputs, scanner, logs);
      await c.initialize();
      logs.fail = true;
      await c.run(camera: false);
      expect(c.result!.status, 'SEGMENTATION');
      expect(c.storageWarning, true);
      final count = logs.rows.length;
      inputs.cancel = true;
      await c.run(camera: false);
      expect(logs.rows.length, count);
      expect(scanner.scans, 1);
      inputs.fail = true;
      await c.run(camera: false);
      expect(c.result!.status, 'ERROR');
      expect(logs.rows.last.inputName, 'bad.jpg');
      await c.close();
    },
  );
  test(
    'Disk journal survives restart, exports full sanitized events and no image bytes',
    () async {
      final temp = await Directory.systemTemp.createTemp('lite-log-test');
      addTearDown(() => temp.delete(recursive: true));
      final store = FileRunLogs(
        Directory('${temp.path}/logs'),
        Directory('${temp.path}/emergency'),
      );
      final row = RunRecord(
        attemptId: 'attempt',
        occurredAt: DateTime.now(),
        inputMethod: 'file',
        inputName: 'file.jpg',
        result: ScanResult.fromWorker(response()),
      );
      expect((await store.append(row)).storageState, 'saved');
      final restored = FileRunLogs(store.directory, store.emergencyDirectory);
      expect((await restored.recent()).single.result!.status, 'SEGMENTATION');
      final destination = '${temp.path}/export.jsonl';
      await restored.exportTo(destination);
      final text = await File(destination).readAsString();
      expect(text, contains('file.jpg'));
      for (final forbidden in [
        'confidence',
        'secret-candidate',
        base64Encode(pixel),
      ]) {
        expect(text, isNot(contains(forbidden)));
      }
      final original = (await store.directory.list().toList()).single.path;
      await expectLater(
        restored.exportTo(original),
        throwsA(isA<FileSystemException>()),
      );
    },
  );
  test(
    'Primary disk write failure retains a failed emergency record and exportable result',
    () async {
      final temp = await Directory.systemTemp.createTemp('lite-failure-test');
      addTearDown(() => temp.delete(recursive: true));
      await File('${temp.path}/blocked').writeAsString('occupied');
      final store = FileRunLogs(
        Directory('${temp.path}/blocked'),
        Directory('${temp.path}/emergency'),
      );
      final row = RunRecord(
        attemptId: 'failure',
        occurredAt: DateTime.now(),
        inputMethod: 'camera',
        inputName: 'USB camera',
        result: ScanResult.fromWorker(response(status: 'IMAGE_RECAPTURE')),
      );
      final saved = await store.append(row);
      expect(saved.storageState, 'failed');
      final restored = FileRunLogs(store.directory, store.emergencyDirectory);
      expect(
        (await restored.recent()).single.result!.status,
        'IMAGE_RECAPTURE',
      );
      expect(restored.storageWarning, true);
      expect(
        await File('${temp.path}/emergency/log-failures.jsonl').readAsString(),
        contains('LOG_WRITE_FAILED'),
      );
    },
  );

  testWidgets('Readonly scan and log surfaces; visual inspection artifact', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1280, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final loader = FontLoader('Pretendard')
      ..addFont(rootBundle.load('assets/fonts/PretendardVariable.ttf'));
    await tester.runAsync(loader.load);
    final icons = FontLoader('MaterialIcons')
      ..addFont(rootBundle.load('fonts/MaterialIcons-Regular.otf'));
    await tester.runAsync(icons.load);
    final inputs = FakeInputs(), scanner = FakeScanner(), logs = MemoryLogs();
    final c = ScanController(inputs, scanner, logs);
    await c.initialize();
    final boundary = GlobalKey();
    await tester.pumpWidget(
      RepaintBoundary(
        key: boundary,
        child: LiteApp(controller: c),
      ),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('pick')));
    await tester.pumpAndSettle();
    expect(find.text('UNKNOWN'), findsOneWidget);
    expect(find.text('1. Almond Scone'), findsOneWidget);
    expect(find.text('2. Cream Bun'), findsOneWidget);
    expect(find.text('3. Baguette'), findsOneWidget);
    expect(find.byType(TextField), findsNothing);
    expect(find.byType(EditableText), findsNothing);
    expect(find.text('로그 저장 완료'), findsOneWidget);
    expect(tester.takeException(), isNull);
    if (Platform.environment['LITE_VISUAL_OUTPUT'] case final String path) {
      final render =
          boundary.currentContext!.findRenderObject()! as RenderRepaintBoundary;
      await tester.runAsync(() async {
        final image = await render.toImage();
        final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
        await File(path).parent.create(recursive: true);
        await File(path).writeAsBytes(bytes!.buffer.asUint8List());
        image.dispose();
      });
    }
    await tester.tap(find.text('실행 로그'));
    await tester.pumpAndSettle();
    expect(find.text('전체 로그 내보내기'), findsOneWidget);
    expect(find.textContaining('sample.png'), findsOneWidget);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox());
    await c.close();
  });
  for (final status in ['IMAGE_RECAPTURE', 'ERROR']) {
    testWidgets('$status is visually distinct and read-only', (tester) async {
      final c = ScanController(
        FakeInputs(),
        FakeScanner()..value = ScanResult.fromWorker(response(status: status)),
        MemoryLogs(),
      );
      await c.initialize();
      await c.run(camera: false);
      tester.view.physicalSize = const Size(1280, 900);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      await tester.pumpWidget(LiteApp(controller: c));
      await tester.pumpAndSettle();
      expect(find.text(status), findsOneWidget);
      expect(
        find.textContaining(
          status == 'ERROR' ? '시스템 또는 입력 오류' : '이미지 전체를 다시 촬영',
        ),
        findsOneWidget,
      );
      expect(find.byType(TextField), findsNothing);
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox());
      await c.close();
    });
  }
}
