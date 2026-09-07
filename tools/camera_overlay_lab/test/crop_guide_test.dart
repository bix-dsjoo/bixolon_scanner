import 'package:camera_overlay_lab/crop_guide.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('1920x1080 미리보기에서 원본 픽셀 크기의 하단 정사각형을 계산한다', () {
    final rect = cropGuideRect(
      viewportSize: const Size(960, 540),
      sourceSize: const Size(1920, 1080),
      cropPixels: 1024,
      anchor: CropGuideAnchor.bottomCenter,
    );

    expect(rect, const Rect.fromLTWH(224, 28, 512, 512));
  });

  test('중앙 기준을 선택하면 정사각형을 수직 중앙에 둔다', () {
    final rect = cropGuideRect(
      viewportSize: const Size(960, 540),
      sourceSize: const Size(1920, 1080),
      cropPixels: 640,
      anchor: CropGuideAnchor.center,
    );

    expect(rect, const Rect.fromLTWH(320, 110, 320, 320));
  });

  test('1920x1080 미리보기에서 960 하단 가이드를 계산한다', () {
    final rect = cropGuideRect(
      viewportSize: const Size(960, 540),
      sourceSize: const Size(1920, 1080),
      cropPixels: 960,
      anchor: CropGuideAnchor.bottomCenter,
    );

    expect(rect, const Rect.fromLTWH(240, 60, 480, 480));
  });

  test('원본의 짧은 변보다 큰 가이드는 원본 범위로 제한한다', () {
    final rect = cropGuideRect(
      viewportSize: const Size(640, 480),
      sourceSize: const Size(640, 480),
      cropPixels: 1024,
      anchor: CropGuideAnchor.bottomCenter,
    );

    expect(rect, const Rect.fromLTWH(80, 0, 480, 480));
  });
}
