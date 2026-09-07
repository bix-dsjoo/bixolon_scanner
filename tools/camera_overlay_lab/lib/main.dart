import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'crop_guide.dart';

const _ink = Color(0xFF102A33);
const _paper = Color(0xFFE8F0F1);
const _stage = Color(0xFF132A33);
const _muted = Color(0xFF61777E);
const _cyan = Color(0xFF42C4D3);
const _amber = Color(0xFFF2B84B);
const _violet = Color(0xFFA78BFA);
const _coral = Color(0xFFFF7268);

const _guideColors = <int, Color>{
  640: _cyan,
  768: _amber,
  960: _violet,
  1024: _coral,
};

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CameraOverlayLab());
}

class CameraOverlayLab extends StatelessWidget {
  const CameraOverlayLab({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Camera Crop Lab',
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: _paper,
        colorScheme: ColorScheme.fromSeed(
          seedColor: _cyan,
          brightness: Brightness.light,
          surface: _paper,
        ),
        fontFamily: 'Segoe UI',
      ),
      home: const CameraOverlayScreen(),
    );
  }
}

class CameraOverlayScreen extends StatefulWidget {
  const CameraOverlayScreen({super.key});

  @override
  State<CameraOverlayScreen> createState() => _CameraOverlayScreenState();
}

class _CameraOverlayScreenState extends State<CameraOverlayScreen> {
  final Set<int> _visibleGuides = {640, 768, 960, 1024};
  List<CameraDescription> _cameras = const [];
  CameraController? _controller;
  CropGuideAnchor _anchor = CropGuideAnchor.bottomCenter;
  int _cameraIndex = 0;
  bool _mirrorPreview = true;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    unawaited(_loadCameras());
  }

  Future<void> _loadCameras() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final cameras = await availableCameras();
      if (!mounted) return;
      _cameras = cameras;
      if (cameras.isEmpty) {
        setState(() {
          _loading = false;
          _error = '연결된 카메라가 없습니다.';
        });
        return;
      }
      _cameraIndex = _cameraIndex.clamp(0, cameras.length - 1);
      await _openCamera(_cameraIndex);
    } on CameraException catch (error) {
      _showCameraError('${error.code}: ${error.description ?? '카메라 오류'}');
    } catch (error) {
      _showCameraError(error.toString());
    }
  }

  Future<void> _openCamera(int index) async {
    final previous = _controller;
    _controller = null;
    await previous?.dispose();

    if (!mounted) return;
    setState(() {
      _loading = true;
      _error = null;
      _cameraIndex = index;
    });

    final controller = CameraController(
      _cameras[index],
      ResolutionPreset.veryHigh,
      enableAudio: false,
      fps: 30,
    );
    try {
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _controller = controller;
        _loading = false;
      });
    } on CameraException catch (error) {
      await controller.dispose();
      _showCameraError('${error.code}: ${error.description ?? '카메라 오류'}');
    }
  }

  void _showCameraError(String message) {
    if (!mounted) return;
    setState(() {
      _loading = false;
      _error = message;
    });
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final compact = constraints.maxWidth < 900;
            final preview = Expanded(
              child: _PreviewStage(
                controller: _controller,
                loading: _loading,
                error: _error,
                guideSizes: _visibleGuides,
                anchor: _anchor,
                mirrorPreview: _mirrorPreview,
                onRetry: _loadCameras,
              ),
            );
            final controls = _ControlPanel(
              cameras: _cameras,
              cameraIndex: _cameraIndex,
              controller: _controller,
              visibleGuides: _visibleGuides,
              anchor: _anchor,
              mirrorPreview: _mirrorPreview,
              onCameraChanged: _loading ? null : _openCamera,
              onGuideChanged: (size, visible) {
                setState(() {
                  if (visible) {
                    _visibleGuides.add(size);
                  } else {
                    _visibleGuides.remove(size);
                  }
                });
              },
              onAnchorChanged: (anchor) => setState(() => _anchor = anchor),
              onMirrorChanged: (value) =>
                  setState(() => _mirrorPreview = value),
              onRefresh: _loading ? null : _loadCameras,
            );
            if (compact) {
              return Column(
                children: [
                  preview,
                  SizedBox(height: 270, child: controls),
                ],
              );
            }
            return Row(
              children: [
                preview,
                SizedBox(width: 310, child: controls),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _PreviewStage extends StatelessWidget {
  const _PreviewStage({
    required this.controller,
    required this.loading,
    required this.error,
    required this.guideSizes,
    required this.anchor,
    required this.mirrorPreview,
    required this.onRetry,
  });

  final CameraController? controller;
  final bool loading;
  final String? error;
  final Set<int> guideSizes;
  final CropGuideAnchor anchor;
  final bool mirrorPreview;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final active = controller;
    return ColoredBox(
      color: _stage,
      child: Stack(
        fit: StackFit.expand,
        children: [
          if (active != null && active.value.isInitialized)
            _MeasuredCameraPreview(
              controller: active,
              guideSizes: guideSizes,
              anchor: anchor,
              mirrorPreview: mirrorPreview,
            )
          else
            _CameraEmptyState(loading: loading, error: error, onRetry: onRetry),
          const Positioned(left: 24, top: 20, child: _StageTitle()),
        ],
      ),
    );
  }
}

class _MeasuredCameraPreview extends StatelessWidget {
  const _MeasuredCameraPreview({
    required this.controller,
    required this.guideSizes,
    required this.anchor,
    required this.mirrorPreview,
  });

  final CameraController controller;
  final Set<int> guideSizes;
  final CropGuideAnchor anchor;
  final bool mirrorPreview;

  @override
  Widget build(BuildContext context) {
    final previewSize = controller.value.previewSize;
    if (previewSize == null || previewSize.isEmpty) {
      return const SizedBox.shrink();
    }
    return LayoutBuilder(
      builder: (context, constraints) {
        final viewport = Size(constraints.maxWidth, constraints.maxHeight);
        final fitted = applyBoxFit(BoxFit.contain, previewSize, viewport);
        final rect = Alignment.center.inscribe(
          fitted.destination,
          Offset.zero & viewport,
        );
        return Stack(
          children: [
            Positioned.fromRect(
              rect: rect,
              child: ClipRect(
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Transform.flip(
                      flipX: mirrorPreview,
                      child: CameraPreview(controller),
                    ),
                    IgnorePointer(
                      child: CustomPaint(
                        key: const ValueKey('pixel-crop-guides'),
                        painter: CropGuidePainter(
                          sourceSize: previewSize,
                          guideSizes: guideSizes,
                          anchor: anchor,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _StageTitle extends StatelessWidget {
  const _StageTitle();

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: _stage.withValues(alpha: .82),
        border: Border.all(color: Colors.white.withValues(alpha: .12)),
      ),
      child: const Padding(
        padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Text(
          'LIVE / CROP BENCH',
          style: TextStyle(
            color: Colors.white,
            fontFamily: 'Consolas',
            fontWeight: FontWeight.w700,
            fontSize: 12,
            letterSpacing: 1.2,
          ),
        ),
      ),
    );
  }
}

class _CameraEmptyState extends StatelessWidget {
  const _CameraEmptyState({
    required this.loading,
    required this.error,
    required this.onRetry,
  });

  final bool loading;
  final String? error;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 420),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (loading)
              const CircularProgressIndicator(color: _cyan)
            else
              const Icon(Icons.videocam_off_outlined, color: _cyan, size: 40),
            const SizedBox(height: 20),
            Text(
              loading ? '카메라를 여는 중입니다' : '카메라를 열 수 없습니다',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
            if (error != null) ...[
              const SizedBox(height: 8),
              Text(
                error!,
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white.withValues(alpha: .64)),
              ),
              const SizedBox(height: 20),
              OutlinedButton.icon(
                onPressed: onRetry,
                icon: const Icon(Icons.refresh),
                label: const Text('다시 연결'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: Colors.white,
                  side: const BorderSide(color: _cyan),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _ControlPanel extends StatelessWidget {
  const _ControlPanel({
    required this.cameras,
    required this.cameraIndex,
    required this.controller,
    required this.visibleGuides,
    required this.anchor,
    required this.mirrorPreview,
    required this.onCameraChanged,
    required this.onGuideChanged,
    required this.onAnchorChanged,
    required this.onMirrorChanged,
    required this.onRefresh,
  });

  final List<CameraDescription> cameras;
  final int cameraIndex;
  final CameraController? controller;
  final Set<int> visibleGuides;
  final CropGuideAnchor anchor;
  final bool mirrorPreview;
  final ValueChanged<int>? onCameraChanged;
  final void Function(int size, bool visible) onGuideChanged;
  final ValueChanged<CropGuideAnchor> onAnchorChanged;
  final ValueChanged<bool> onMirrorChanged;
  final VoidCallback? onRefresh;

  @override
  Widget build(BuildContext context) {
    final sourceSize = controller?.value.previewSize;
    return ColoredBox(
      color: _paper,
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(24, 26, 24, 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                const Expanded(
                  child: Text(
                    '촬영 영역 비교',
                    style: TextStyle(
                      color: _ink,
                      fontSize: 22,
                      height: 1.1,
                      fontWeight: FontWeight.w800,
                      letterSpacing: -.5,
                    ),
                  ),
                ),
                IconButton(
                  tooltip: '카메라 새로 고침',
                  onPressed: onRefresh,
                  icon: const Icon(Icons.refresh_rounded),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              sourceSize == null
                  ? '원본 해상도를 확인하는 중'
                  : '원본 ${sourceSize.width.round()} × ${sourceSize.height.round()} px',
              style: const TextStyle(
                color: _muted,
                fontFamily: 'Consolas',
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
            if (cameras.length > 1) ...[
              const SizedBox(height: 22),
              DropdownButtonFormField<int>(
                initialValue: cameraIndex,
                decoration: const InputDecoration(
                  labelText: '카메라',
                  border: OutlineInputBorder(),
                ),
                items: [
                  for (var index = 0; index < cameras.length; index++)
                    DropdownMenuItem(
                      value: index,
                      child: Text(
                        cameras[index].name,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                ],
                onChanged: onCameraChanged == null
                    ? null
                    : (value) {
                        if (value != null) onCameraChanged!(value);
                      },
              ),
            ],
            const SizedBox(height: 28),
            const _PanelLabel('표시할 정사각형'),
            const SizedBox(height: 10),
            for (final size in _guideColors.keys)
              _GuideToggle(
                size: size,
                color: _guideColors[size]!,
                selected: visibleGuides.contains(size),
                sourceSize: sourceSize,
                onChanged: (value) => onGuideChanged(size, value),
              ),
            const SizedBox(height: 24),
            const _PanelLabel('기준 위치'),
            const SizedBox(height: 10),
            SegmentedButton<CropGuideAnchor>(
              segments: const [
                ButtonSegment(
                  value: CropGuideAnchor.bottomCenter,
                  icon: Icon(Icons.vertical_align_bottom),
                  label: Text('하단'),
                ),
                ButtonSegment(
                  value: CropGuideAnchor.center,
                  icon: Icon(Icons.center_focus_strong),
                  label: Text('중앙'),
                ),
              ],
              selected: {anchor},
              onSelectionChanged: (values) => onAnchorChanged(values.first),
            ),
            const SizedBox(height: 14),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('미러 미리보기 보정'),
              subtitle: const Text('저장 사진과 같은 좌우 방향으로 봅니다.'),
              value: mirrorPreview,
              onChanged: onMirrorChanged,
            ),
            const SizedBox(height: 18),
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: .62),
                border: Border.all(color: _ink.withValues(alpha: .12)),
              ),
              child: const Text(
                '박스는 화면 비율이 아니라 카메라 원본 픽셀을 기준으로 표시합니다. '
                '1024 박스 밖의 영역은 실제 1024 정사각형 크롭에서 제외됩니다.',
                style: TextStyle(color: _muted, height: 1.45, fontSize: 12.5),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PanelLabel extends StatelessWidget {
  const _PanelLabel(this.label);

  final String label;

  @override
  Widget build(BuildContext context) {
    return Text(
      label.toUpperCase(),
      style: const TextStyle(
        color: _ink,
        fontFamily: 'Consolas',
        fontSize: 11,
        fontWeight: FontWeight.w700,
        letterSpacing: 1,
      ),
    );
  }
}

class _GuideToggle extends StatelessWidget {
  const _GuideToggle({
    required this.size,
    required this.color,
    required this.selected,
    required this.sourceSize,
    required this.onChanged,
  });

  final int size;
  final Color color;
  final bool selected;
  final Size? sourceSize;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) {
    final shortest = sourceSize?.shortestSide.round();
    final supported = shortest == null || shortest >= size;
    return InkWell(
      onTap: supported ? () => onChanged(!selected) : null,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 7),
        child: Row(
          children: [
            Container(width: 5, height: 36, color: color),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '$size × $size',
                    style: TextStyle(
                      color: supported ? _ink : _muted,
                      fontFamily: 'Consolas',
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  Text(
                    supported ? '원본 픽셀 기준' : '현재 해상도보다 큼',
                    style: const TextStyle(color: _muted, fontSize: 12),
                  ),
                ],
              ),
            ),
            Checkbox(
              value: selected && supported,
              onChanged: supported
                  ? (value) => onChanged(value ?? false)
                  : null,
              activeColor: color,
            ),
          ],
        ),
      ),
    );
  }
}

class CropGuidePainter extends CustomPainter {
  const CropGuidePainter({
    required this.sourceSize,
    required this.guideSizes,
    required this.anchor,
  });

  final Size sourceSize;
  final Set<int> guideSizes;
  final CropGuideAnchor anchor;

  @override
  void paint(Canvas canvas, Size size) {
    final orderedSizes = guideSizes.toList()..sort((a, b) => b.compareTo(a));
    for (final guideSize in orderedSizes) {
      if (guideSize > sourceSize.shortestSide) continue;
      final rect = cropGuideRect(
        viewportSize: size,
        sourceSize: sourceSize,
        cropPixels: guideSize,
        anchor: anchor,
      );
      final color = _guideColors[guideSize] ?? Colors.white;
      canvas.drawRect(
        rect.deflate(1.5),
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3,
      );
      _drawCornerTicks(canvas, rect.deflate(1.5), color);
      _drawLabel(canvas, rect, guideSize, color);
    }
  }

  void _drawCornerTicks(Canvas canvas, Rect rect, Color color) {
    const length = 18.0;
    final paint = Paint()
      ..color = color
      ..strokeWidth = 5
      ..strokeCap = StrokeCap.square;
    final paths = [
      (rect.topLeft, const Offset(length, 0), const Offset(0, length)),
      (rect.topRight, const Offset(-length, 0), const Offset(0, length)),
      (rect.bottomLeft, const Offset(length, 0), const Offset(0, -length)),
      (rect.bottomRight, const Offset(-length, 0), const Offset(0, -length)),
    ];
    for (final (origin, horizontal, vertical) in paths) {
      canvas.drawLine(origin, origin + horizontal, paint);
      canvas.drawLine(origin, origin + vertical, paint);
    }
  }

  void _drawLabel(Canvas canvas, Rect rect, int pixels, Color color) {
    final textPainter = TextPainter(
      text: TextSpan(
        text: '$pixels × $pixels',
        style: const TextStyle(
          color: _ink,
          fontFamily: 'Consolas',
          fontSize: 12,
          fontWeight: FontWeight.w700,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final labelRect = Rect.fromLTWH(
      rect.left + 8,
      rect.top + 8,
      textPainter.width + 16,
      textPainter.height + 10,
    );
    canvas.drawRect(labelRect, Paint()..color = color);
    textPainter.paint(canvas, labelRect.topLeft + const Offset(8, 5));
  }

  @override
  bool shouldRepaint(covariant CropGuidePainter oldDelegate) {
    return oldDelegate.sourceSize != sourceSize ||
        oldDelegate.anchor != anchor ||
        oldDelegate.guideSizes.length != guideSizes.length ||
        !oldDelegate.guideSizes.containsAll(guideSizes);
  }
}
