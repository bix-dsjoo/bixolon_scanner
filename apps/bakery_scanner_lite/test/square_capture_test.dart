import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:bakery_scanner_lite/shared/square_capture.dart';
import 'package:bakery_scanner_lite/shared/result.dart';
import 'package:bakery_scanner_lite/features/scanner/result_image.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test(
    'Capture and preview remove top 20 percent before the centered square',
    () {
      expect(
        captureRect(3840, 2160),
        const Rect.fromLTWH(1056, 432, 1728, 1728),
      );
      expect(captureRect(1920, 1080), const Rect.fromLTWH(528, 216, 864, 864));
      expect(captureRect(1080, 1920), const Rect.fromLTWH(0, 612, 1080, 1080));
    },
  );
  test(
    'Boxes track the actual image including portrait/landscape letterboxing',
    () {
      expect(
        displayBox(
          const DetectionBox(1024, 1024, 512, 512),
          const Size(2048, 2048),
          const Size(400, 400),
        ),
        const Rect.fromLTWH(200, 200, 100, 100),
      );
      expect(
        displayBox(
          const DetectionBox(0, 0, 2000, 1000),
          const Size(2000, 1000),
          const Size(400, 400),
        ),
        const Rect.fromLTWH(0, 100, 400, 200),
      );
      expect(
        () => DetectionBox.fromJson({'x': -1, 'y': 0, 'width': 1, 'height': 1}),
        throwsFormatException,
      );
    },
  );
  testWidgets('Top crop and horizontal flip affect actual 2048 square pixels', (
    tester,
  ) async {
    await tester.runAsync(() async {
      final recorder = ui.PictureRecorder();
      final canvas = Canvas(recorder);
      canvas.drawColor(Colors.red, BlendMode.src);
      canvas.drawRect(
        const Rect.fromLTWH(120, 40, 80, 160),
        Paint()..color = Colors.green,
      );
      canvas.drawRect(
        const Rect.fromLTWH(200, 40, 80, 160),
        Paint()..color = Colors.blue,
      );
      final picture = recorder.endRecording();
      final source = await picture.toImage(400, 200);
      final data = await source.toByteData(format: ui.ImageByteFormat.png);
      final result = await squareCapture(data!.buffer.asUint8List());
      final codec = await ui.instantiateImageCodec(result);
      final output = (await codec.getNextFrame()).image;
      expect(output.width, 2048);
      expect(output.height, 2048);
      final pixels = await output.toByteData();
      // Red top/margins are removed; source-right blue becomes output-left.
      for (final xy in [(10, 10), (512, 1024), (2030, 2030)]) {
        final index = (xy.$2 * 2048 + xy.$1) * 4;
        final channel = xy.$1 < 1024 ? 2 : 1;
        expect(
          pixels!.getUint8(index + channel),
          greaterThan(pixels.getUint8(index)),
        );
        expect(
          pixels.getUint8(index + channel),
          greaterThan(pixels.getUint8(index + (channel == 2 ? 1 : 2))),
        );
      }
      output.dispose();
      codec.dispose();
      source.dispose();
      picture.dispose();
    });
  });
}
