import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../controllers/scanner_controller.dart';
import '../models/scan_models.dart';
import '../theme/app_theme.dart';

class ScannerScreen extends StatefulWidget {
  const ScannerScreen({
    super.key,
    required this.controller,
    required this.autoInitialize,
    required this.disposeController,
  });

  final ScannerController controller;
  final bool autoInitialize;
  final bool disposeController;

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  @override
  void initState() {
    super.initState();
    if (widget.autoInitialize) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        widget.controller.initialize();
      });
    }
  }

  @override
  void dispose() {
    if (widget.disposeController) widget.controller.dispose();
    super.dispose();
  }

  Future<void> _requestReset() async {
    if (await _canDiscardChanges()) widget.controller.resetSession();
  }

  Future<void> _chooseImage() async {
    if (await _canDiscardChanges()) await widget.controller.chooseImage();
  }

  Future<bool> _canDiscardChanges() async {
    final controller = widget.controller;
    if (!controller.hasUserChanges) return true;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('다시 촬영할까요?'),
        content: const Text('현재 확인한 내용이 사라져요.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('취소'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('다시 촬영'),
          ),
        ],
      ),
    );
    return confirmed == true;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: AnimatedBuilder(
        animation: widget.controller,
        builder: (context, _) {
          final controller = widget.controller;
          return Column(
            children: [
              _TopBar(controller: controller),
              Expanded(
                child: Stack(
                  children: [
                    LayoutBuilder(
                      builder: (context, constraints) {
                        if (constraints.maxWidth < 1000) {
                          return Column(
                            children: [
                              Expanded(
                                flex: 6,
                                child: _PreviewPane(
                                  controller: controller,
                                  onReset: _requestReset,
                                  onChooseImage: _chooseImage,
                                ),
                              ),
                              const Divider(height: 1),
                              Expanded(
                                flex: 5,
                                child: _ResultPanel(controller: controller),
                              ),
                            ],
                          );
                        }
                        final resultWidth = (constraints.maxWidth * .35).clamp(
                          420.0,
                          540.0,
                        );
                        return Row(
                          children: [
                            Expanded(
                              child: _PreviewPane(
                                controller: controller,
                                onReset: _requestReset,
                                onChooseImage: _chooseImage,
                              ),
                            ),
                            const VerticalDivider(width: 1),
                            SizedBox(
                              width: resultWidth,
                              child: _ResultPanel(controller: controller),
                            ),
                          ],
                        );
                      },
                    ),
                    if (controller.completionMessage != null)
                      Positioned(
                        top: 20,
                        left: 0,
                        right: 0,
                        child: Center(
                          child: _CompletionBanner(
                            message: controller.completionMessage!,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _TopBar extends StatelessWidget {
  const _TopBar({required this.controller});

  final ScannerController controller;

  @override
  Widget build(BuildContext context) {
    final cameraReady = controller.isCameraReady;
    final statusColor = cameraReady
        ? AppColors.success
        : controller.cameraInitializing
        ? AppColors.muted
        : AppColors.attention;
    final statusText = cameraReady
        ? '카메라 연결됨'
        : controller.cameraInitializing
        ? '카메라 확인 중'
        : '카메라 확인 필요';
    return Container(
      height: 72,
      padding: const EdgeInsets.symmetric(horizontal: 26),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(bottom: BorderSide(color: AppColors.divider)),
      ),
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: AppColors.primarySoft,
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(
              Icons.document_scanner_outlined,
              color: AppColors.primary,
              size: 22,
            ),
          ),
          const SizedBox(width: 13),
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Product Scanner',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              const SizedBox(height: 1),
              Text('상품 확인 작업대', style: Theme.of(context).textTheme.bodySmall),
            ],
          ),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: AppColors.workspace,
              borderRadius: BorderRadius.circular(10),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: statusColor,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Text(statusText, style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _PreviewPane extends StatelessWidget {
  const _PreviewPane({
    required this.controller,
    required this.onReset,
    required this.onChooseImage,
  });

  final ScannerController controller;
  final VoidCallback onReset;
  final VoidCallback onChooseImage;

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: AppColors.surface,
      child: Column(
        children: [
          Expanded(child: _PreviewSurface(controller: controller)),
          _InputActionBar(
            controller: controller,
            onReset: onReset,
            onChooseImage: onChooseImage,
          ),
        ],
      ),
    );
  }
}

class _PreviewSurface extends StatelessWidget {
  const _PreviewSurface({required this.controller});

  final ScannerController controller;

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: AppColors.preview,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final viewport = Size(constraints.maxWidth, constraints.maxHeight);
          final imageSize = controller.imageSize;
          Rect? imageRect;
          if (controller.imageBytes != null && imageSize != null) {
            final fitted = applyBoxFit(BoxFit.contain, imageSize, viewport);
            imageRect = Alignment.center.inscribe(
              fitted.destination,
              Offset.zero & viewport,
            );
          }

          return Stack(
            fit: StackFit.expand,
            clipBehavior: Clip.hardEdge,
            children: [
              if (controller.imageBytes != null)
                Image.memory(
                  controller.imageBytes!,
                  fit: BoxFit.contain,
                  gaplessPlayback: true,
                  filterQuality: FilterQuality.medium,
                )
              else
                _LiveCamera(controller: controller),
              if (imageRect != null && controller.hasResults)
                ...controller.detections.map(
                  (detection) => _DetectionBox(
                    detection: detection,
                    imageRect: imageRect!,
                    imageSize: imageSize!,
                    selected:
                        controller.selectedItemId == detection.source.itemId,
                    onTap: () =>
                        controller.selectDetection(detection.source.itemId),
                  ),
                ),
              if (controller.processState == ProcessState.analyzing)
                const _AnalyzingOverlay(),
            ],
          );
        },
      ),
    );
  }
}

class _LiveCamera extends StatelessWidget {
  const _LiveCamera({required this.controller});

  final ScannerController controller;

  @override
  Widget build(BuildContext context) {
    final camera = controller.cameraController;
    if (camera != null && camera.value.isInitialized) {
      return Center(
        child: AspectRatio(
          aspectRatio: camera.value.aspectRatio,
          child: CameraPreview(camera),
        ),
      );
    }
    return Stack(
      children: [
        const Positioned.fill(
          child: CustomPaint(painter: _CameraGuidePainter()),
        ),
        Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (controller.cameraInitializing)
                const SizedBox(
                  width: 26,
                  height: 26,
                  child: CircularProgressIndicator(
                    strokeWidth: 2.5,
                    color: Colors.white70,
                  ),
                )
              else
                const Icon(
                  Icons.videocam_off_outlined,
                  color: Colors.white70,
                  size: 34,
                ),
              const SizedBox(height: 14),
              Text(
                controller.cameraInitializing
                    ? '카메라를 준비하고 있어요'
                    : controller.cameraMessage ?? '카메라를 연결해 주세요',
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white70, fontSize: 15),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _AnalyzingOverlay extends StatelessWidget {
  const _AnalyzingOverlay();

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: const Color(0xA6111820),
      child: Center(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
          decoration: BoxDecoration(
            color: const Color(0xE61B2531),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.white12),
          ),
          child: const Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 22,
                height: 22,
                child: CircularProgressIndicator(
                  strokeWidth: 2.5,
                  color: Colors.white,
                ),
              ),
              SizedBox(width: 14),
              Text(
                '이미지를 분석하고 있어요',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _DetectionBox extends StatelessWidget {
  const _DetectionBox({
    required this.detection,
    required this.imageRect,
    required this.imageSize,
    required this.selected,
    required this.onTap,
  });

  final ReviewDetection detection;
  final Rect imageRect;
  final Size imageSize;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final bbox = detection.source.bbox;
    final scaleX = imageRect.width / imageSize.width;
    final scaleY = imageRect.height / imageSize.height;
    final rect = Rect.fromLTWH(
      imageRect.left + bbox.x * scaleX,
      imageRect.top + bbox.y * scaleY,
      bbox.width * scaleX,
      bbox.height * scaleY,
    );
    final needsReview = !detection.isConfirmed;
    final color = needsReview ? AppColors.attention : AppColors.success;
    final index = int.tryParse(detection.source.itemId.split('_').last) ?? 0;
    final label = needsReview
        ? '$index  확인 필요'
        : '$index  ${detection.finalProduct?.displayName ?? '확정'}';
    return Positioned.fromRect(
      rect: rect,
      child: Semantics(
        button: true,
        label: '$label 상품 영역',
        child: MouseRegion(
          cursor: SystemMouseCursors.click,
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTap: onTap,
            child: Stack(
              clipBehavior: Clip.none,
              children: [
                Positioned.fill(
                  child: Container(
                    decoration: BoxDecoration(
                      color: selected ? color.withValues(alpha: .10) : null,
                      border: Border.all(color: color, width: selected ? 3 : 2),
                    ),
                  ),
                ),
                if (selected)
                  Positioned.fill(
                    child: CustomPaint(painter: _CornerBracketPainter(color)),
                  ),
                Positioned(
                  left: -1,
                  top: -1,
                  child: Container(
                    constraints: BoxConstraints(maxWidth: rect.width + 80),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 9,
                      vertical: 5,
                    ),
                    color: color,
                    child: Text(
                      label,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _CornerBracketPainter extends CustomPainter {
  const _CornerBracketPainter(this.color);

  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final length = (size.shortestSide * .18).clamp(12.0, 24.0);
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 5
      ..strokeCap = StrokeCap.square;
    final path = Path()
      ..moveTo(0, length)
      ..lineTo(0, 0)
      ..lineTo(length, 0)
      ..moveTo(size.width - length, 0)
      ..lineTo(size.width, 0)
      ..lineTo(size.width, length)
      ..moveTo(size.width, size.height - length)
      ..lineTo(size.width, size.height)
      ..lineTo(size.width - length, size.height)
      ..moveTo(length, size.height)
      ..lineTo(0, size.height)
      ..lineTo(0, size.height - length);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(_CornerBracketPainter oldDelegate) =>
      oldDelegate.color != color;
}

class _CameraGuidePainter extends CustomPainter {
  const _CameraGuidePainter();

  @override
  void paint(Canvas canvas, Size size) {
    final guide = Rect.fromCenter(
      center: size.center(Offset.zero),
      width: size.width * .62,
      height: size.height * .62,
    );
    final paint = Paint()
      ..color = Colors.white.withValues(alpha: .12)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    canvas.drawRect(guide, paint);
    canvas.drawLine(
      Offset(guide.center.dx, guide.top),
      Offset(guide.center.dx, guide.bottom),
      paint,
    );
    canvas.drawLine(
      Offset(guide.left, guide.center.dy),
      Offset(guide.right, guide.center.dy),
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _InputActionBar extends StatelessWidget {
  const _InputActionBar({
    required this.controller,
    required this.onReset,
    required this.onChooseImage,
  });

  final ScannerController controller;
  final VoidCallback onReset;
  final VoidCallback onChooseImage;

  @override
  Widget build(BuildContext context) {
    final busy = controller.isBusy;
    final hasImage = controller.imageBytes != null;
    final reviewingSuccess =
        controller.processState == ProcessState.reviewing &&
        !controller.isRecapture;
    return Container(
      constraints: const BoxConstraints(minHeight: 104),
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(top: BorderSide(color: AppColors.divider)),
      ),
      child: Row(
        children: [
          if (reviewingSuccess)
            OutlinedButton.icon(
              onPressed: busy
                  ? null
                  : controller.inputMode == InputMode.image
                  ? onChooseImage
                  : onReset,
              icon: Icon(
                controller.inputMode == InputMode.image
                    ? Icons.image_outlined
                    : Icons.refresh_rounded,
                size: 19,
              ),
              label: Text(
                controller.inputMode == InputMode.image ? '다른 이미지 선택' : '다시 촬영',
              ),
            )
          else if (controller.isRecapture &&
              controller.inputMode == InputMode.image) ...[
            const Spacer(),
            FilledButton.icon(
              onPressed: busy ? null : onChooseImage,
              icon: const Icon(Icons.image_outlined, size: 19),
              label: const Text('다른 이미지 선택'),
            ),
          ] else ...[
            OutlinedButton.icon(
              onPressed: busy ? null : onChooseImage,
              icon: const Icon(Icons.image_outlined, size: 19),
              label: Text(
                controller.inputMode == InputMode.image && hasImage
                    ? '다른 이미지 선택'
                    : '이미지 선택',
              ),
            ),
            const Spacer(),
            if (controller.processState == ProcessState.error && hasImage)
              FilledButton.icon(
                onPressed: busy ? null : controller.analyze,
                icon: const Icon(Icons.refresh_rounded, size: 19),
                label: const Text('다시 시도'),
              )
            else if (controller.inputMode == InputMode.image && hasImage)
              FilledButton.icon(
                onPressed: busy ? null : controller.analyze,
                icon: const Icon(Icons.auto_awesome_outlined, size: 19),
                label: const Text('분석하기'),
              )
            else if (!controller.isCameraReady)
              FilledButton.icon(
                onPressed: busy || controller.cameraInitializing
                    ? null
                    : controller.reconnectCamera,
                icon: const Icon(Icons.videocam_outlined, size: 19),
                label: const Text('다시 연결'),
              )
            else
              FilledButton.icon(
                onPressed: busy ? null : controller.captureAndAnalyze,
                icon: const Icon(Icons.camera_alt_outlined, size: 19),
                label: Text(controller.isRecapture ? '다시 촬영하기' : '촬영하기'),
              ),
          ],
        ],
      ),
    );
  }
}

class _ResultPanel extends StatelessWidget {
  const _ResultPanel({required this.controller});

  final ScannerController controller;

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: AppColors.workspace,
      child: Column(
        children: [
          _ResultHeader(controller: controller),
          Expanded(child: _resultBody()),
          if (controller.processState == ProcessState.reviewing &&
              !controller.isRecapture &&
              controller.hasResults)
            _ReviewFooter(controller: controller),
        ],
      ),
    );
  }

  Widget _resultBody() {
    if (controller.processState == ProcessState.analyzing) {
      return const _PanelMessage(
        icon: Icons.hourglass_top_rounded,
        title: '분석하고 있어요',
        detail: '이미지에서 상품을 찾고 있습니다.',
      );
    }
    if (controller.processState == ProcessState.error) {
      return _PanelMessage(
        icon: Icons.cloud_off_outlined,
        title: '분석하지 못했어요',
        detail: controller.errorMessage ?? '잠시 후 다시 시도해 주세요.',
        tone: AppColors.error,
      );
    }
    if (controller.isRecapture) {
      return _PanelMessage(
        icon: Icons.center_focus_weak_rounded,
        title: controller.recaptureTitle,
        detail: controller.recaptureDetail,
        tone: AppColors.attention,
      );
    }
    if (!controller.hasResults) {
      return const _PanelMessage(
        icon: Icons.inventory_2_outlined,
        title: '아직 분석된 상품이 없어요',
        detail: '촬영하거나 이미지를 선택해 주세요.',
      );
    }
    return _DetectionList(controller: controller);
  }
}

class _ResultHeader extends StatelessWidget {
  const _ResultHeader({required this.controller});

  final ScannerController controller;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 78,
      padding: const EdgeInsets.symmetric(horizontal: 24),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(bottom: BorderSide(color: AppColors.divider)),
      ),
      child: Row(
        children: [
          Text('상품', style: Theme.of(context).textTheme.headlineSmall),
          const Spacer(),
          if (controller.hasResults)
            Text(
              '${controller.detections.length}개',
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(color: AppColors.muted),
            ),
        ],
      ),
    );
  }
}

class _PanelMessage extends StatelessWidget {
  const _PanelMessage({
    required this.icon,
    required this.title,
    required this.detail,
    this.tone = AppColors.muted,
  });

  final IconData icon;
  final String title;
  final String detail;
  final Color tone;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 56,
              height: 56,
              decoration: BoxDecoration(
                color: tone.withValues(alpha: .10),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(icon, color: tone, size: 28),
            ),
            const SizedBox(height: 20),
            Text(
              title,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 8),
            Text(
              detail,
              textAlign: TextAlign.center,
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(color: AppColors.muted),
            ),
          ],
        ),
      ),
    );
  }
}

class _DetectionList extends StatelessWidget {
  const _DetectionList({required this.controller});

  final ScannerController controller;

  @override
  Widget build(BuildContext context) {
    return ListView.separated(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      itemCount: controller.detections.length,
      separatorBuilder: (_, _) => const SizedBox(height: 4),
      itemBuilder: (context, index) {
        final detection = controller.detections[index];
        return _DetectionRow(
          controller: controller,
          detection: detection,
          index: index + 1,
          selected: controller.selectedItemId == detection.source.itemId,
        );
      },
    );
  }
}

class _DetectionRow extends StatelessWidget {
  const _DetectionRow({
    required this.controller,
    required this.detection,
    required this.index,
    required this.selected,
  });

  final ScannerController controller;
  final ReviewDetection detection;
  final int index;
  final bool selected;

  @override
  Widget build(BuildContext context) {
    final needsReview = !detection.isConfirmed;
    final tone = needsReview ? AppColors.attention : AppColors.success;
    final duration = MediaQuery.disableAnimationsOf(context)
        ? Duration.zero
        : const Duration(milliseconds: 180);
    return AnimatedContainer(
      duration: duration,
      decoration: BoxDecoration(
        color: selected
            ? needsReview
                  ? AppColors.attentionSoft
                  : AppColors.successSoft
            : Colors.transparent,
        borderRadius: BorderRadius.circular(11),
        border: selected
            ? Border.all(color: tone.withValues(alpha: .28))
            : null,
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: () => controller.selectDetection(detection.source.itemId),
          borderRadius: BorderRadius.circular(11),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 13),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: 32,
                      height: 32,
                      alignment: Alignment.center,
                      decoration: BoxDecoration(
                        color: tone.withValues(alpha: .12),
                        borderRadius: BorderRadius.circular(9),
                      ),
                      child: Text(
                        '$index',
                        style: TextStyle(
                          color: tone,
                          fontWeight: FontWeight.w800,
                          fontSize: 14,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            needsReview
                                ? '상품 확인이 필요해요'
                                : detection.finalProduct!.displayName,
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          const SizedBox(height: 3),
                          Row(
                            children: [
                              Icon(
                                needsReview
                                    ? Icons.error_outline_rounded
                                    : Icons.check_circle_outline_rounded,
                                size: 15,
                                color: tone,
                              ),
                              const SizedBox(width: 5),
                              Text(
                                needsReview ? '상품을 선택해 주세요' : '확정',
                                style: Theme.of(
                                  context,
                                ).textTheme.bodySmall?.copyWith(color: tone),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    Icon(
                      selected ? Icons.expand_less : Icons.expand_more,
                      color: AppColors.muted,
                      size: 20,
                    ),
                  ],
                ),
                if (selected) ...[
                  const SizedBox(height: 14),
                  const Divider(height: 1),
                  const SizedBox(height: 14),
                  if (controller.searchItemId == detection.source.itemId)
                    _SearchProducts(
                      controller: controller,
                      detection: detection,
                    )
                  else if (detection.source.top3.isNotEmpty)
                    _CandidatePicker(
                      controller: controller,
                      detection: detection,
                    )
                  else
                    Align(
                      alignment: Alignment.centerLeft,
                      child: TextButton.icon(
                        onPressed: () =>
                            controller.showSearch(detection.source.itemId),
                        icon: const Icon(Icons.edit_outlined, size: 18),
                        label: const Text('상품 변경'),
                      ),
                    ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _CandidatePicker extends StatelessWidget {
  const _CandidatePicker({required this.controller, required this.detection});

  final ScannerController controller;
  final ReviewDetection detection;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          detection.isConfirmed ? '상품을 변경할까요?' : '어떤 상품인가요?',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 10),
        ...detection.source.top3.map((rawCandidate) {
          final candidate = controller.localizeCandidate(rawCandidate);
          final chosen = detection.finalProduct?.classId == candidate.classId;
          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: _ProductChoice(
              product: candidate,
              selected: chosen,
              onTap: () => controller.confirmCandidate(
                detection.source.itemId,
                rawCandidate,
              ),
            ),
          );
        }),
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton.icon(
            onPressed: () => controller.showSearch(detection.source.itemId),
            icon: const Icon(Icons.search_rounded, size: 19),
            label: const Text('다른 상품 검색'),
          ),
        ),
      ],
    );
  }
}

class _ProductChoice extends StatelessWidget {
  const _ProductChoice({
    required this.product,
    required this.selected,
    required this.onTap,
  });

  final Product product;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: selected ? AppColors.successSoft : AppColors.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(10),
        side: BorderSide(
          color: selected ? AppColors.success : AppColors.divider,
        ),
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(10),
        child: Padding(
          padding: const EdgeInsets.all(10),
          child: Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: AppColors.workspace,
                  borderRadius: BorderRadius.circular(9),
                ),
                child: const Icon(
                  Icons.bakery_dining_outlined,
                  color: AppColors.attention,
                  size: 23,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      product.displayName,
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    Text(
                      product.className,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                ),
              ),
              if (selected)
                const Icon(
                  Icons.check_circle_rounded,
                  color: AppColors.success,
                  size: 21,
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SearchProducts extends StatelessWidget {
  const _SearchProducts({required this.controller, required this.detection});

  final ScannerController controller;
  final ReviewDetection detection;

  @override
  Widget build(BuildContext context) {
    final results = controller.searchResults;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton.icon(
            onPressed: controller.hideSearch,
            icon: const Icon(Icons.arrow_back_rounded, size: 18),
            label: const Text('후보로 돌아가기'),
          ),
        ),
        const SizedBox(height: 6),
        TextField(
          autofocus: true,
          onChanged: controller.updateSearch,
          decoration: const InputDecoration(
            hintText: '상품명을 검색해 주세요',
            prefixIcon: Icon(Icons.search_rounded),
          ),
        ),
        const SizedBox(height: 12),
        Text('검색 결과', style: Theme.of(context).textTheme.bodySmall),
        const SizedBox(height: 6),
        if (results.isEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 20),
            child: Text(
              '일치하는 상품이 없어요.',
              textAlign: TextAlign.center,
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(color: AppColors.muted),
            ),
          )
        else
          ...results
              .take(8)
              .map(
                (product) => ListTile(
                  dense: true,
                  contentPadding: const EdgeInsets.symmetric(horizontal: 4),
                  title: Text(product.displayName),
                  subtitle: Text(product.className),
                  trailing: const Icon(Icons.chevron_right_rounded),
                  onTap: () => controller.confirmSearchProduct(
                    detection.source.itemId,
                    product,
                  ),
                ),
              ),
      ],
    );
  }
}

class _ReviewFooter extends StatelessWidget {
  const _ReviewFooter({required this.controller});

  final ScannerController controller;

  @override
  Widget build(BuildContext context) {
    final remaining = controller.detections.length - controller.confirmedCount;
    return Container(
      padding: const EdgeInsets.fromLTRB(24, 18, 24, 22),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(top: BorderSide(color: AppColors.divider)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (controller.errorMessage != null) ...[
            Container(
              margin: const EdgeInsets.only(bottom: 14),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppColors.errorSoft,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                children: [
                  const Icon(
                    Icons.error_outline_rounded,
                    color: AppColors.error,
                    size: 19,
                  ),
                  const SizedBox(width: 9),
                  Expanded(
                    child: Text(
                      controller.errorMessage!,
                      style: Theme.of(
                        context,
                      ).textTheme.bodySmall?.copyWith(color: AppColors.error),
                    ),
                  ),
                ],
              ),
            ),
          ],
          Text(
            controller.allConfirmed
                ? '${controller.detections.length}개 상품을 모두 확인했어요'
                : '${controller.confirmedCount} / ${controller.detections.length} 상품 확인 완료',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 5),
          Text(
            controller.allConfirmed
                ? '최종 확정하면 결과가 안전하게 저장됩니다.'
                : '확인이 필요한 상품이 $remaining개 있어요.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          if (controller.allConfirmed) ...[
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: controller.processState == ProcessState.submitting
                  ? null
                  : controller.submit,
              icon: controller.processState == ProcessState.submitting
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Icon(Icons.check_rounded, size: 20),
              label: Text(
                controller.processState == ProcessState.submitting
                    ? '저장하고 있어요'
                    : '최종 확정',
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _CompletionBanner extends StatelessWidget {
  const _CompletionBanner({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: AppColors.ink,
      borderRadius: BorderRadius.circular(12),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 13),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(
              Icons.check_circle_rounded,
              color: Color(0xFF68D6A5),
              size: 21,
            ),
            const SizedBox(width: 9),
            Text(
              message,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
