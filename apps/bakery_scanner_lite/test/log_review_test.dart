import 'dart:convert';
import 'dart:io';
import 'package:archive/archive.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:bakery_scanner_lite/core/design_system/scan_status.dart';
import 'package:bakery_scanner_lite/core/design_system/tokens.dart';
import 'package:bakery_scanner_lite/features/scanner/scan_controller.dart';
import 'package:bakery_scanner_lite/features/scanner/result_image.dart';
import 'package:bakery_scanner_lite/shared/log_images.dart';
import 'package:bakery_scanner_lite/shared/result.dart';
import 'package:bakery_scanner_lite/shared/run_log.dart';
import 'package:bakery_scanner_lite/lite_app.dart';
import 'fakes.dart';

void main() {
  test('Both recapture states are red while UNKNOWN remains amber', () {
    expect(scanStatusColor('IMAGE_RECAPTURE'), AppPalette.error);
    expect(scanStatusColor('SEGMENT_RECAPTURE'), AppPalette.error);
    expect(scanStatusColor('UNKNOWN'), AppPalette.attention);
    expect(scanStatusColor('APPROVED'), AppPalette.success);
  });
  test(
    'Restart restores the exact input, boxes, ranked candidates and ZIP export',
    () async {
      final temp = await Directory.systemTemp.createTemp('lite-image-log-');
      addTearDown(() => temp.delete(recursive: true));
      final logs = FileRunLogs(
        Directory('${temp.path}/logs'),
        Directory('${temp.path}/emergency'),
      );
      final c = ScanController(FakeInputs(), FakeScanner(), logs);
      await c.initialize();
      await c.run(camera: false);
      await c.close();
      final restored = FileRunLogs(logs.directory, logs.emergencyDirectory);
      final row = (await restored.recent()).single;
      expect(await restored.imageFor(row), pixel);
      expect(row.imageStorageStatus, 'saved');
      expect(row.result!.objects.single.box!.width, 3);
      expect(row.result!.objects.single.top3.map((c) => c.className), [
        'Almond Scone',
        'Cream Bun',
        'Baguette',
      ]);
      final path = '${temp.path}/export.zip';
      await restored.exportTo(path);
      final zip = ZipDecoder().decodeBytes(await File(path).readAsBytes());
      expect(zip.findFile('images/${row.imageReference}')!.content, pixel);
      final jsonl = utf8.decode(zip.findFile('logs.jsonl')!.content);
      expect(const LineSplitter().convert(jsonl).length, 2);
      expect(jsonl, contains('"top3"'));
      expect(jsonl, contains('"bbox"'));
      expect(jsonl, isNot(contains('confidence')));
      expect(jsonl, isNot(contains(base64Encode(pixel))));
    },
  );
  test(
    'Image write failure does not lose inference and uses emergency storage',
    () async {
      final temp = await Directory.systemTemp.createTemp('lite-image-fail-');
      addTearDown(() => temp.delete(recursive: true));
      final directory = await Directory('${temp.path}/logs').create();
      await File('${directory.path}/images').writeAsString('blocked');
      final logs = FileRunLogs(directory, Directory('${temp.path}/emergency'));
      final c = ScanController(FakeInputs(), FakeScanner(), logs);
      await c.initialize();
      await c.run(camera: false);
      expect(c.result!.status, 'SEGMENTATION');
      expect(c.lastRecord!.imageStorageStatus, 'emergency');
      expect(c.lastRecord!.storageReason, 'LOG_IMAGE_WRITE_FAILED');
      expect(c.storageWarning, true);
      expect(await logs.imageFor(c.lastRecord!), pixel);
      await c.close();
    },
  );
  test(
    'Retention removes only expired image files; metadata and fresh images remain',
    () async {
      final temp = await Directory.systemTemp.createTemp('lite-retention-');
      addTearDown(() => temp.delete(recursive: true));
      final store = LogImages(
        Directory('${temp.path}/logs'),
        Directory('${temp.path}/emergency'),
      );
      final now = DateTime.now();
      final old = await store.store(
        'old',
        now.subtract(const Duration(days: 31)),
        pixel,
      );
      final fresh = await store.store('fresh', now, pixel);
      final metadata = File('${store.primary.path}/record.jsonl');
      await metadata.writeAsString('metadata');
      await store.purge(now: now);
      expect(
        await File('${store.primary.path}/images/${old.reference}').exists(),
        false,
      );
      expect(await store.read(fresh.reference, now), pixel);
      expect(await metadata.readAsString(), 'metadata');
      expect(await store.read('../../record.jsonl', now), isNull);
    },
  );
  test('Old journals without image, box or Top-3 are still readable', () {
    final data = RunRecord(
      attemptId: 'old',
      occurredAt: DateTime.now(),
      inputMethod: 'file',
      inputName: 'old.png',
      result: ScanResult.fromWorker(response()),
    ).toJson();
    data['schema_version'] = 1;
    data.remove('image_reference');
    data.remove('image_storage_status');
    final object =
        ((data['result'] as Map)['segmentations'] as List).single as Map;
    object.remove('bbox');
    object.remove('top3');
    final row = RunRecord.fromJson(data);
    expect(row.imageReference, isNull);
    expect(row.result!.objects.single.box, isNull);
    expect(row.result!.objects.single.top3, isEmpty);
  });
  testWidgets('Expanded log shows its image, boxes and read-only Top-3', (
    tester,
  ) async {
    final logs = MemoryLogs();
    final c = ScanController(FakeInputs(), FakeScanner(), logs);
    await c.initialize();
    await c.run(camera: false);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: LogRecordView(record: c.lastRecord!, logs: logs),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.byType(ResultImage), findsOneWidget);
    expect(
      tester.widget<ResultImage>(find.byType(ResultImage)).objects.single.box,
      isNotNull,
    );
    expect(find.text('1. Almond Scone'), findsOneWidget);
    expect(find.byType(EditableText), findsNothing);
    expect(find.byType(TextField), findsNothing);
    expect(tester.takeException(), isNull);
    await tester.pumpWidget(const SizedBox());
    await c.close();
  });
}
