import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:archive/archive.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:bakery_scanner_lite/features/scanner/scan_controller.dart';
import 'package:bakery_scanner_lite/lite_app.dart';
import 'package:bakery_scanner_lite/shared/input.dart';
import 'package:bakery_scanner_lite/shared/camera_profile.dart';
import 'package:bakery_scanner_lite/features/scanner/square_preview.dart';
import 'package:bakery_scanner_lite/shared/run_log.dart';
import 'package:bakery_scanner_lite/shared/worker_client.dart';
import '../test/fakes.dart';

// The native file chooser is replaced only at the input boundary. The Windows
// app, owned packaged Worker, inference, result projection and journals are real.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  testWidgets(
    'Windows 0.2.0 inference, read-only results and durable logs',
    (tester) async {
      final root = Platform.environment['LITE_E2E_ROOT']!;
      final output = Directory('$root/artifacts/lite/0.2.0/windows-e2e');
      await output.create(recursive: true);
      final report = <String, Object?>{};
      Future<void> checkpoint(String stage) async {
        report['stage'] = stage;
        await File('${output.path}/report.json').writeAsString(
          '${const JsonEncoder.withIndent('  ').convert(report)}\n',
          flush: true,
        );
      }

      final source = Directory(
        '$root/datasets/bread_dataset/operational_collections/2026-08-18/images',
      );
      Future<File> image(String prefix) async => (await source.list().toList())
          .whereType<File>()
          .firstWhere((file) => file.uri.pathSegments.last.startsWith(prefix));
      final inputs = FakeInputs();
      inputs.bytes = await (await image('001_')).readAsBytes();
      final runs = await Directory(
        '${output.path}/runs',
      ).create(recursive: true);
      final runFolder = await runs.createTemp('run-');
      final logs = FileRunLogs(
        Directory('${runFolder.path}/logs'),
        Directory('${runFolder.path}/emergency'),
      );
      final scanner = BundledScanner();
      final controller = ScanController(inputs, scanner, logs);
      final boundary = GlobalKey();
      try {
        await controller.initialize();
        expect(controller.workerReady, true);
        await tester.pumpWidget(
          RepaintBoundary(
            key: boundary,
            child: LiteApp(controller: controller),
          ),
        );
        await tester.pumpAndSettle();
        await tester.tap(find.byKey(const Key('pick')));
        for (var i = 0; controller.busy && i < 150; i++) {
          await tester.pump(const Duration(milliseconds: 500));
        }
        expect(controller.busy, false);
        expect(controller.result!.status, 'SEGMENTATION');
        expect(controller.result!.objects.length, 3);
        expect(controller.result!.objects.every((o) => o.box != null), true);
        expect(
          controller.result!.objects.every((o) => o.status == 'APPROVED'),
          true,
        );
        expect(controller.lastRecord!.storageState, 'saved');
        expect(find.byType(EditableText), findsNothing);
        expect(find.byType(TextField), findsNothing);
        report['valid_image'] = controller.result!.toJson();
        final validRecord = controller.lastRecord!;
        final validBytes = inputs.bytes;
        await tester.pumpAndSettle();
        final render =
            boundary.currentContext!.findRenderObject()!
                as RenderRepaintBoundary;
        final picture = await render.toImage();
        final pixels = await picture.toByteData(format: ui.ImageByteFormat.png);
        await File(
          '${output.path}/real-result.png',
        ).writeAsBytes(pixels!.buffer.asUint8List());
        picture.dispose();

        inputs.bytes = await (await image('092_')).readAsBytes();
        await controller.run(camera: false);
        expect(controller.result!.status, 'IMAGE_RECAPTURE');
        report['empty_image'] = controller.result!.toJson();
        inputs.bytes = utf8.encode('not an image');
        await controller.run(camera: false);
        expect(controller.result!.status, 'ERROR');
        report['invalid_image'] = controller.result!.toJson();
        expect(
          (await FileRunLogs(
            logs.directory,
            logs.emergencyDirectory,
          ).recent()).length,
          3,
        );
        final restored = FileRunLogs(logs.directory, logs.emergencyDirectory);
        expect(await restored.imageFor(validRecord), validBytes);
        inputs.export = '${output.path}/export.zip';
        await controller.exportLogs();
        expect(controller.exportMessage, '로그를 내보냈습니다.');
        final archive = ZipDecoder().decodeBytes(
          await File(inputs.export!).readAsBytes(),
        );
        final exported = utf8.decode(archive.findFile('logs.jsonl')!.content);
        expect(
          archive.findFile('images/${validRecord.imageReference}'),
          isNotNull,
        );
        expect(const LineSplitter().convert(exported).length, 6);
        for (final key in ['confidence', 'prediction', 'image_base64']) {
          expect(exported, isNot(contains('"$key"')));
        }
        await tester.tap(find.text('실행 로그'));
        await tester.pumpAndSettle();
        expect(find.text('전체 로그 내보내기'), findsOneWidget);
        expect(tester.takeException(), isNull);
        report['persistent_runs'] = 3;
        report['export_events'] = 6;
        await tester.pumpWidget(
          RepaintBoundary(
            key: boundary,
            child: MaterialApp(
              home: Scaffold(
                body: SingleChildScrollView(
                  child: LogRecordView(record: validRecord, logs: restored),
                ),
              ),
            ),
          ),
        );
        await tester.pumpAndSettle();
        await tester.runAsync(
          () => Future<void>.delayed(const Duration(seconds: 1)),
        );
        await tester.pumpAndSettle();
        expect(find.byType(RawImage), findsOneWidget);
        expect(find.byType(EditableText), findsNothing);
        final logRender =
            boundary.currentContext!.findRenderObject()!
                as RenderRepaintBoundary;
        final logPicture = await logRender.toImage();
        final logPixels = await logPicture.toByteData(
          format: ui.ImageByteFormat.png,
        );
        await File(
          '${output.path}/log-result.png',
        ).writeAsBytes(logPixels!.buffer.asUint8List());
        logPicture.dispose();
        report['archived_image_and_boxes'] = 'passed';
        report['zip_images'] = 'passed';

        // Exercise the actual camera plugin without archiving live camera images.
        final camera = WindowsPhotoInputs();
        try {
          await checkpoint('camera_enumeration');
          final devices = await camera.cameras();
          report['camera_count'] = devices.length;
          final configured = preferredCamera(devices);
          report['configured_camera_found'] = configured != null;
          if (configured != null) {
            await checkpoint('camera_initialization');
            await camera.select(configured);
            expect(camera.camera!.value.isInitialized, true);
            await checkpoint('camera_preview');
            await tester.pumpWidget(
              MaterialApp(
                home: Scaffold(
                  body: Center(
                    child: AspectRatio(
                      aspectRatio: 1,
                      child: SquarePreview(controller: camera.camera!),
                    ),
                  ),
                ),
              ),
            );
            await tester.pump(const Duration(seconds: 2));
            await checkpoint('camera_capture');
            final capture = await camera.capture();
            expect(capture.bytes.isNotEmpty, true);
            final captureCodec = await ui.instantiateImageCodec(capture.bytes);
            final capturedImage = (await captureCodec.getNextFrame()).image;
            expect(capturedImage.width, 2048);
            expect(capturedImage.height, 2048);
            report['camera_output_size'] = [
              capturedImage.width,
              capturedImage.height,
            ];
            capturedImage.dispose();
            captureCodec.dispose();
            final cameraResult = await scanner.scan(
              capture.bytes,
              capture.name,
              'lite-camera-smoke',
            );
            report['camera_capture_bytes'] = capture.bytes.length;
            report['camera_inference_status'] = cameraResult.status;
            report['camera_preview_capture'] = 'passed';
          } else {
            report['camera_preview_capture'] =
                'not_run_configured_camera_missing';
          }
        } catch (error) {
          if (error is! CameraException) rethrow;
          report['camera_preview_capture'] = 'unavailable';
          report['camera_error_type'] = error.runtimeType.toString();
          report['camera_error_code'] = error.code;
          report['camera_error_description'] = error.description;
        } finally {
          await tester.pumpWidget(const SizedBox());
          await tester.pump();
          await camera.dispose();
        }
        report['passes'] = true;
      } finally {
        await tester.pumpWidget(const SizedBox());
        await controller.close();
        await File('${output.path}/report.json').writeAsString(
          '${const JsonEncoder.withIndent('  ').convert(report)}\n',
        );
      }
    },
    timeout: const Timeout(Duration(minutes: 5)),
  );
}
