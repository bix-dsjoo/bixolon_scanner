import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

const capturePixels = 2048;

ui.Rect captureRect(double width, double height) {
  final top = height * .2;
  final remainingHeight = height - top;
  final side = math.min(width, remainingHeight);
  return ui.Rect.fromLTWH(
    (width - side) / 2,
    top + (remainingHeight - side) / 2,
    side,
    side,
  );
}

Future<Uint8List> squareCapture(Uint8List bytes) async {
  final codec = await ui.instantiateImageCodec(bytes);
  ui.Image? source;
  ui.Image? output;
  ui.Picture? picture;
  try {
    source = (await codec.getNextFrame()).image;
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder)
      ..translate(capturePixels.toDouble(), 0)
      ..scale(-1, 1);
    canvas.drawImageRect(
      source,
      captureRect(source.width.toDouble(), source.height.toDouble()),
      const ui.Rect.fromLTWH(0, 0, 2048, 2048),
      ui.Paint()..filterQuality = ui.FilterQuality.high,
    );
    picture = recorder.endRecording();
    output = await picture.toImage(capturePixels, capturePixels);
    final encoded = await output.toByteData(format: ui.ImageByteFormat.png);
    if (encoded == null) throw StateError('Capture encoding failed');
    return encoded.buffer.asUint8List();
  } finally {
    output?.dispose();
    picture?.dispose();
    source?.dispose();
    codec.dispose();
  }
}
