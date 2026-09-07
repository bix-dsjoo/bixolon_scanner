import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'image_input.dart';

/// Displays a centered 1080x1080 crop in preview-frame pixel coordinates.
///
/// `camera_windows` mirrors only its texture preview. The horizontal correction
/// keeps the orientation consistent with the unmirrored camera JPEG.
/// Preview and photo framing match when their source dimensions and FOV match.
class SquareCameraPreview extends StatelessWidget {
  const SquareCameraPreview({
    super.key,
    required this.controller,
    this.foreground,
  });

  final CameraController controller;
  final Widget? foreground;

  @override
  Widget build(BuildContext context) {
    final cameraAspectRatio = controller.value.aspectRatio;
    final safeAspectRatio = cameraAspectRatio.isFinite && cameraAspectRatio > 0
        ? cameraAspectRatio
        : 1.0;
    final sourceWidth = safeAspectRatio >= 1 ? safeAspectRatio : 1.0;
    final sourceHeight = safeAspectRatio >= 1 ? 1.0 : 1 / safeAspectRatio;
    final previewSize = controller.value.previewSize;
    final shortestSide =
        previewSize?.shortestSide ?? cameraSquareOutputSize.toDouble();
    final cropScale =
        shortestSide.isFinite && shortestSide > cameraSquareOutputSize
        ? shortestSide / cameraSquareOutputSize
        : 1.0;

    return AspectRatio(
      aspectRatio: 1,
      child: ClipRect(
        child: Stack(
          fit: StackFit.expand,
          children: [
            Transform.scale(
              key: const ValueKey('square-camera-center-crop'),
              scale: cropScale,
              alignment: Alignment.center,
              child: FittedBox(
                fit: BoxFit.cover,
                alignment: Alignment.center,
                child: SizedBox(
                  width: sourceWidth,
                  height: sourceHeight,
                  child: Transform.flip(
                    key: const ValueKey('camera-preview-mirror'),
                    flipX: true,
                    child: CameraPreview(controller),
                  ),
                ),
              ),
            ),
            ?foreground,
          ],
        ),
      ),
    );
  }
}
