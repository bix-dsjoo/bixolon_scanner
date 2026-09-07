import 'dart:math' as math;

import 'package:flutter/widgets.dart';

enum CropGuideAnchor { bottomCenter, center }

Rect cropGuideRect({
  required Size viewportSize,
  required Size sourceSize,
  required int cropPixels,
  required CropGuideAnchor anchor,
}) {
  assert(cropPixels > 0);
  if (viewportSize.isEmpty || sourceSize.isEmpty) return Rect.zero;

  final scale = math.min(
    viewportSize.width / sourceSize.width,
    viewportSize.height / sourceSize.height,
  );
  final side = math.min(cropPixels.toDouble(), sourceSize.shortestSide) * scale;
  final left = (viewportSize.width - side) / 2;
  final top = switch (anchor) {
    CropGuideAnchor.bottomCenter => viewportSize.height - side,
    CropGuideAnchor.center => (viewportSize.height - side) / 2,
  };
  return Rect.fromLTWH(left, top, side, side);
}
