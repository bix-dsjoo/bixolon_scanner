import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as imaging;
import 'package:product_scanner/shared/input/image_input.dart';

void main() {
  test('1920x1080 JPEG는 중앙 1080x1080을 잘라 위아래를 모두 유지한다', () {
    final source = imaging.Image(width: 1920, height: 1080);
    imaging.fill(source, color: imaging.ColorRgb8(0, 255, 0));
    imaging.fillRect(
      source,
      x1: 0,
      y1: 0,
      x2: 419,
      y2: 1079,
      color: imaging.ColorRgb8(255, 0, 0),
    );
    imaging.fillRect(
      source,
      x1: 1500,
      y1: 0,
      x2: 1919,
      y2: 1079,
      color: imaging.ColorRgb8(0, 0, 255),
    );
    imaging.fillRect(
      source,
      x1: 0,
      y1: 0,
      x2: 1919,
      y2: 59,
      color: imaging.ColorRgb8(255, 255, 0),
    );
    imaging.fillRect(
      source,
      x1: 0,
      y1: 1020,
      x2: 1919,
      y2: 1079,
      color: imaging.ColorRgb8(255, 0, 255),
    );

    final encoded = Uint8List.fromList(imaging.encodeJpg(source, quality: 100));
    final result = imaging.decodeJpg(centerCropCameraJpeg(encoded));

    expect(result, isNotNull);
    expect(result!.width, cameraSquareOutputSize);
    expect(result.height, cameraSquareOutputSize);
    final center = result.getPixel(540, 540);
    expect(center.g, greaterThan(245));
    expect(center.r, lessThan(10));
    expect(center.b, lessThan(10));
    final leftEdge = result.getPixel(0, 540);
    final rightEdge = result.getPixel(1079, 540);
    expect(leftEdge.g, greaterThan(235));
    expect(rightEdge.g, greaterThan(235));
    final topEdge = result.getPixel(540, 0);
    expect(topEdge.g, greaterThan(235));
    expect(topEdge.r, greaterThan(235));
    expect(topEdge.b, lessThan(20));
    final bottomEdge = result.getPixel(540, 1079);
    expect(bottomEdge.g, lessThan(20));
    expect(bottomEdge.r, greaterThan(235));
    expect(bottomEdge.b, greaterThan(235));
  });

  for (final sample in [
    (width: 3264, height: 2448, x: 1092, y: 684, size: 1080),
    (width: 1080, height: 1920, x: 0, y: 420, size: 1080),
    (width: 640, height: 480, x: 80, y: 0, size: 480),
  ]) {
    test('${sample.width}x${sample.height} 원본의 중앙 픽셀을 크기 변경 없이 보존한다', () {
      final source = imaging.Image(width: sample.width, height: sample.height);
      imaging.fill(source, color: imaging.ColorRgb8(255, 0, 0));
      imaging.fillRect(
        source,
        x1: sample.x,
        y1: sample.y,
        x2: sample.x + sample.size - 1,
        y2: sample.y + sample.size - 1,
        color: imaging.ColorRgb8(0, 255, 0),
      );
      imaging.fillRect(
        source,
        x1: sample.x + 80,
        y1: sample.y + 40,
        x2: sample.x + 119,
        y2: sample.y + 79,
        color: imaging.ColorRgb8(0, 0, 255),
      );
      final result = imaging.decodeJpg(
        centerCropCameraJpeg(Uint8List.fromList(imaging.encodePng(source))),
      )!;
      expect(result.width, sample.size);
      expect(result.height, sample.size);
      for (final (x, y) in [
        (10, 10),
        (sample.size - 11, 10),
        (10, sample.size - 11),
        (sample.size - 11, sample.size - 11),
      ]) {
        final pixel = result.getPixel(x, y);
        expect(pixel.g, greaterThan(235));
        expect(pixel.r, lessThan(20));
      }
      final marker = result.getPixel(100, 60);
      expect(marker.b, greaterThan(235));
      expect(marker.g, lessThan(20));
    });
  }

  test('손상 이미지는 기본 사진으로 대체하지 않고 실패한다', () {
    expect(() => centerCropCameraJpeg(Uint8List(0)), throwsFormatException);
  });
}
