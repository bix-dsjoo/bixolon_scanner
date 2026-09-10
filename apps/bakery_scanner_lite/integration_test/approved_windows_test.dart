import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:bakery_scanner_lite/features/scanner/scan_controller.dart';
import 'package:bakery_scanner_lite/lite_app.dart';
import 'package:bakery_scanner_lite/shared/run_log.dart';
import 'package:bakery_scanner_lite/shared/worker_client.dart';
import '../test/fakes.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  testWidgets(
    'Packaged Worker approved names remain visible after log restart',
    (tester) async {
      final output = Directory(Platform.environment['LITE_APPROVED_OUTPUT']!);
      await output.create(recursive: true);
      final inputs = FakeInputs()
        ..bytes = await File(
          Platform.environment['LITE_APPROVED_IMAGE']!,
        ).readAsBytes();
      final logs = FileRunLogs(
        Directory('${output.path}/logs'),
        Directory('${output.path}/emergency'),
      );
      final c = ScanController(inputs, BundledScanner(), logs);
      final boundary = GlobalKey();
      try {
        await c.initialize();
        expect(c.workerReady, true);
        await c.run(camera: false);
        expect(c.result!.status, 'SEGMENTATION');
        final approved = c.result!.objects
            .where((o) => o.status == 'APPROVED')
            .toList();
        expect(approved, isNotEmpty);
        expect(approved.every((o) => o.prediction != null), true);
        await tester.pumpWidget(
          RepaintBoundary(
            key: boundary,
            child: LiteApp(controller: c),
          ),
        );
        await tester.pumpAndSettle();
        for (final object in approved) {
          expect(
            find.text('승인 상품: ${object.prediction!.className}'),
            findsWidgets,
          );
        }
        final render =
            boundary.currentContext!.findRenderObject()!
                as RenderRepaintBoundary;
        final picture = await render.toImage();
        final bytes = await picture.toByteData(format: ui.ImageByteFormat.png);
        await File(
          '${output.path}/approved-result.png',
        ).writeAsBytes(bytes!.buffer.asUint8List());
        picture.dispose();
        final restarted = FileRunLogs(logs.directory, logs.emergencyDirectory);
        final restored = (await restarted.recent()).first;
        expect(
          jsonEncode(restored.result!.toJson()),
          jsonEncode(c.result!.toJson()),
        );
        await tester.tap(find.text('실행 로그'));
        await tester.pumpAndSettle();
        await tester.tap(find.textContaining('sample.png').first);
        await tester.pumpAndSettle();
        for (final object in approved) {
          expect(
            find.text('승인 상품: ${object.prediction!.className}'),
            findsWidgets,
          );
        }
        final exported = '${output.path}/export.jsonl';
        await restarted.exportTo(exported);
        final content = await File(exported).readAsString();
        expect(content, isNot(contains('confidence')));
        for (final object in approved) {
          expect(content, contains(object.prediction!.className));
        }
        expect(tester.takeException(), isNull);
        await File('${output.path}/report.json').writeAsString(
          jsonEncode({
            'passed': true,
            'approved_names': approved
                .map((o) => o.prediction!.className)
                .toList(),
            'result': c.result!.toJson(),
          }),
        );
      } finally {
        await tester.pumpWidget(const SizedBox());
        await c.close();
      }
    },
  );
}
