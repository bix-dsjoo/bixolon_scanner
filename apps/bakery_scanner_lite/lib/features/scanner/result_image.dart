import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import '../../core/design_system/tokens.dart';
import '../../core/design_system/scan_status.dart';
import '../../shared/result.dart';

Rect imageViewport(Size image, Size viewport) {
  final fitted = applyBoxFit(BoxFit.contain, image, viewport);
  return Alignment.center.inscribe(fitted.destination, Offset.zero & viewport);
}

Rect displayBox(DetectionBox box, Size image, Size viewport) {
  final area = imageViewport(image, viewport);
  return Rect.fromLTWH(
    area.left + box.x * area.width / image.width,
    area.top + box.y * area.height / image.height,
    box.width * area.width / image.width,
    box.height * area.height / image.height,
  );
}

class ResultImage extends StatefulWidget {
  const ResultImage({super.key, required this.bytes, required this.objects});
  final Uint8List bytes;
  final List<ObjectResult> objects;
  @override
  State<ResultImage> createState() => _ResultImageState();
}

class _ResultImageState extends State<ResultImage> {
  ui.Image? _image;
  int _generation = 0;
  bool _failed = false;
  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void didUpdateWidget(ResultImage old) {
    super.didUpdateWidget(old);
    if (!identical(old.bytes, widget.bytes)) _load();
  }

  Future<void> _load() async {
    final generation = ++_generation;
    _image?.dispose();
    _image = null;
    _failed = false;
    ui.Codec? codec;
    try {
      codec = await ui.instantiateImageCodec(widget.bytes);
      final image = (await codec.getNextFrame()).image;
      if (!mounted || generation != _generation) {
        image.dispose();
        return;
      }
      setState(() => _image = image);
    } catch (_) {
      if (mounted && generation == _generation) setState(() => _failed = true);
    } finally {
      codec?.dispose();
    }
  }

  @override
  void dispose() {
    _generation++;
    _image?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AspectRatio(
    aspectRatio: 1,
    child: ColoredBox(
      color: AppPalette.elevated,
      child: _image == null
          ? Center(
              child: Text(_failed ? '입력 이미지를 표시할 수 없습니다.' : '이미지를 불러오는 중입니다'),
            )
          : Semantics(
              label: '입력 이미지와 읽기 전용 검출 박스',
              child: CustomPaint(
                key: const Key('detection-boxes'),
                foregroundPainter: DetectionPainter(
                  Size(_image!.width.toDouble(), _image!.height.toDouble()),
                  widget.objects,
                ),
                child: RawImage(image: _image, fit: BoxFit.contain),
              ),
            ),
    ),
  );
}

class DetectionPainter extends CustomPainter {
  DetectionPainter(this.imageSize, this.objects);
  final Size imageSize;
  final List<ObjectResult> objects;
  @override
  void paint(Canvas canvas, Size size) {
    canvas.save();
    canvas.clipRect(imageViewport(imageSize, size));
    for (var i = 0; i < objects.length; i++) {
      final object = objects[i];
      final box = object.box;
      if (box == null) continue;
      final rect = displayBox(box, imageSize, size);
      final color = scanStatusColor(object.status);
      canvas.drawRect(
        rect,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
      final label = TextPainter(
        text: TextSpan(
          text: '${i + 1}',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.w700,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      canvas.drawRect(
        Rect.fromLTWH(rect.left, rect.top, label.width + 10, label.height + 4),
        Paint()..color = color,
      );
      label.paint(canvas, rect.topLeft + const Offset(5, 2));
      label.dispose();
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(DetectionPainter old) =>
      old.imageSize != imageSize || old.objects != objects;
}
