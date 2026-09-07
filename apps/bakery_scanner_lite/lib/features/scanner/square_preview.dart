import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../../shared/square_capture.dart';

class SquarePreview extends StatelessWidget {
  const SquarePreview({super.key, required this.controller});
  final CameraController controller;
  @override
  Widget build(BuildContext context) {
    final size = controller.value.previewSize!;
    final crop = captureRect(size.width, size.height);
    return LayoutBuilder(
      builder: (context, constraints) {
        final scale = constraints.maxWidth / crop.width;
        // Pinned camera_windows 0.2.6+4 already mirrors its preview texture
        // (TextureHandler::mirror_preview_ = true). Captured PNGs are mirrored
        // by squareCapture; flipping this texture again would disagree with them.
        return ClipRect(
          child: Stack(
            children: [
              Positioned(
                left: -crop.left * scale,
                top: -crop.top * scale,
                width: size.width * scale,
                height: size.height * scale,
                child: CameraPreview(controller),
              ),
            ],
          ),
        );
      },
    );
  }
}
