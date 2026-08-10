import 'package:flutter/material.dart';

import '../theme/app_tokens.dart';

/// BIXOLON의 대각선 모티프를 스캔 프리뷰에만 사용하는 촬영 가이드입니다.
///
/// 다른 선택·탐색·상태 컴포넌트에는 이 모티프를 재사용하지 않습니다.
class AppScanGuide extends StatelessWidget {
  const AppScanGuide({super.key});

  @override
  Widget build(BuildContext context) {
    return ExcludeSemantics(
      child: CustomPaint(
        painter: _AppScanGuidePainter(
          tokens: context.appTokens,
          outlineColor: context.appComponents.previewGuideOutline,
          accentColor: context.appComponents.previewGuideAccent,
        ),
      ),
    );
  }
}

class _AppScanGuidePainter extends CustomPainter {
  const _AppScanGuidePainter({
    required this.tokens,
    required this.outlineColor,
    required this.accentColor,
  });

  final AppDesignTokens tokens;
  final Color outlineColor;
  final Color accentColor;

  @override
  void paint(Canvas canvas, Size size) {
    final guide = Rect.fromCenter(
      center: size.center(Offset.zero),
      width: size.width * tokens.scanGuideFraction,
      height: size.height * tokens.scanGuideFraction,
    );
    final outline = Paint()
      ..color = outlineColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    final accent = Paint()
      ..color = accentColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = tokens.scanGuideStrokeWidth
      ..strokeCap = StrokeCap.round;
    canvas.drawRRect(
      RRect.fromRectAndRadius(guide, Radius.circular(tokens.panelRadius)),
      outline,
    );
    final segment = tokens.scanGuideCornerLength;
    canvas.drawLine(
      Offset(guide.left, guide.top + segment),
      Offset(guide.left + segment, guide.top),
      accent,
    );
    canvas.drawLine(
      Offset(guide.right - segment, guide.top),
      Offset(guide.right, guide.top + segment),
      accent,
    );
    canvas.drawLine(
      Offset(guide.left, guide.bottom - segment),
      Offset(guide.left + segment, guide.bottom),
      accent,
    );
    canvas.drawLine(
      Offset(guide.right - segment, guide.bottom),
      Offset(guide.right, guide.bottom - segment),
      accent,
    );
    final cross = tokens.scanGuideCenterMarkSize;
    canvas.drawLine(
      guide.center - Offset(cross, cross),
      guide.center + Offset(cross, cross),
      outline,
    );
    canvas.drawLine(
      guide.center + Offset(cross, -cross),
      guide.center + Offset(-cross, cross),
      outline,
    );
  }

  @override
  bool shouldRepaint(covariant _AppScanGuidePainter oldDelegate) {
    return oldDelegate.tokens.scanGuideFraction != tokens.scanGuideFraction ||
        oldDelegate.tokens.scanGuideCornerLength !=
            tokens.scanGuideCornerLength ||
        oldDelegate.tokens.scanGuideCenterMarkSize !=
            tokens.scanGuideCenterMarkSize ||
        oldDelegate.tokens.scanGuideStrokeWidth !=
            tokens.scanGuideStrokeWidth ||
        oldDelegate.tokens.panelRadius != tokens.panelRadius ||
        oldDelegate.outlineColor != outlineColor ||
        oldDelegate.accentColor != accentColor;
  }
}
