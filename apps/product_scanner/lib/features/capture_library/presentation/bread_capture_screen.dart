import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../../../core/design_system/components.dart';
import '../../../core/design_system/theme.dart';
import '../../../core/design_system/tokens.dart';
import '../application/bread_capture_controller.dart';
import '../domain/bread_capture_models.dart';

@visibleForTesting
double breadCaptureViewportAspectRatio(double cameraAspectRatio) {
  if (!cameraAspectRatio.isFinite || cameraAspectRatio <= 0) return 1;
  return cameraAspectRatio;
}

@visibleForTesting
Widget breadCaptureUnmirroredPreview(Widget preview) {
  // camera_windows mirrors only the texture preview; takePicture() keeps the
  // sensor orientation. Correct the preview so review and saved coordinates
  // remain identical.
  return Transform.flip(flipX: true, child: preview);
}

class BreadCaptureScreen extends StatefulWidget {
  const BreadCaptureScreen({
    super.key,
    required this.controller,
    required this.active,
  });

  final BreadCaptureController controller;
  final bool active;

  @override
  State<BreadCaptureScreen> createState() => _BreadCaptureScreenState();
}

class _BreadCaptureScreenState extends State<BreadCaptureScreen> {
  @override
  void initState() {
    super.initState();
    if (widget.active) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        widget.controller.activate();
      });
    }
  }

  @override
  void didUpdateWidget(covariant BreadCaptureScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!oldWidget.active && widget.active) widget.controller.activate();
  }

  Future<void> _showAddProductDialog() async {
    final textController = TextEditingController();
    final name = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('빵 추가'),
        content: SizedBox(
          width: context.appTokens.dialogWidth,
          child: TextField(
            key: const ValueKey('bread-capture-name-field'),
            controller: textController,
            autofocus: true,
            maxLength: 40,
            decoration: const InputDecoration(
              labelText: '빵 이름',
              hintText: '예: 호두 도넛',
            ),
            onSubmitted: (value) {
              if (value.trim().isNotEmpty) Navigator.of(context).pop(value);
            },
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('취소'),
          ),
          FilledButton(
            key: const ValueKey('bread-capture-add-confirm'),
            onPressed: () {
              final value = textController.text.trim();
              if (value.isNotEmpty) Navigator.of(context).pop(value);
            },
            child: const Text('빵 추가'),
          ),
        ],
      ),
    );
    textController.dispose();
    if (name != null) await widget.controller.addProduct(name);
  }

  Future<void> _confirmDeleteProduct() async {
    final product = widget.controller.selectedProduct;
    if (product == null) return;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AppConfirmDialog(
        title: '${product.name}을(를) 삭제할까요?',
        description:
            '등록 정보와 촬영한 ${product.completedCaptureCount}장의 사진이 함께 삭제돼요.',
        confirmLabel: '빵 삭제',
        onCancel: () => Navigator.of(context).pop(false),
        onConfirm: () => Navigator.of(context).pop(true),
      ),
    );
    if (confirmed == true) await widget.controller.deleteSelectedProduct();
  }

  Future<void> _confirmDeleteCapture(BreadCaptureSlot slot) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AppConfirmDialog(
        title: '${slot.label} 사진을 삭제할까요?',
        description: '삭제한 사진은 복구할 수 없으며 다시 촬영해야 해요.',
        confirmLabel: '사진 삭제',
        onCancel: () => Navigator.of(context).pop(false),
        onConfirm: () => Navigator.of(context).pop(true),
      ),
    );
    if (confirmed == true) await widget.controller.deleteCapture(slot);
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.controller,
      builder: (context, _) {
        final controller = widget.controller;
        if (controller.loading && controller.library == null) {
          return const AppLoadingState(message: '빵 촬영 목록을 불러오는 중이에요');
        }
        return Column(
          children: [
            _CaptureWorkspaceHeader(
              controller: controller,
              onAddProduct: controller.canAddProduct
                  ? _showAddProductDialog
                  : null,
            ),
            if (controller.errorMessage case final message?)
              AppInlineNotice(
                message: message,
                icon: Icons.error_outline_rounded,
                tone: context.appColors.error,
                backgroundColor: context.appColors.errorSoft,
                action: TextButton(
                  onPressed: controller.clearMessage,
                  child: const Text('닫기'),
                ),
              )
            else if (controller.completionMessage case final message?)
              AppInlineNotice(
                message: message,
                icon: Icons.check_circle_outline_rounded,
                tone: context.appColors.success,
                backgroundColor: context.appColors.successSoft,
                action: TextButton(
                  onPressed: controller.clearMessage,
                  child: const Text('닫기'),
                ),
              ),
            Expanded(
              child: controller.products.isEmpty
                  ? AppEmptyState(
                      icon: Icons.bakery_dining_outlined,
                      title: '첫 번째 빵을 등록해 주세요',
                      detail: '빵마다 정상 면 5장과 뒤집은 면 5장을 순서대로 촬영해 관리할 수 있어요.',
                      action: FilledButton.icon(
                        onPressed: controller.canAddProduct
                            ? _showAddProductDialog
                            : null,
                        icon: const Icon(Icons.add_rounded),
                        label: const Text('빵 추가'),
                      ),
                    )
                  : SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: SizedBox(
                        width: MediaQuery.sizeOf(context).width < 1180
                            ? 1180
                            : MediaQuery.sizeOf(context).width,
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            SizedBox(
                              width: 276,
                              child: _ProductRail(
                                controller: controller,
                                onAddProduct: controller.canAddProduct
                                    ? _showAddProductDialog
                                    : null,
                              ),
                            ),
                            const VerticalDivider(width: 1),
                            Expanded(
                              child: _GuidedCamera(controller: controller),
                            ),
                            const VerticalDivider(width: 1),
                            SizedBox(
                              width: 390,
                              child: _ContactSheetPanel(
                                controller: controller,
                                onDeleteProduct: _confirmDeleteProduct,
                                onDeleteCapture: _confirmDeleteCapture,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
            ),
          ],
        );
      },
    );
  }
}

class _CaptureWorkspaceHeader extends StatelessWidget {
  const _CaptureWorkspaceHeader({
    required this.controller,
    required this.onAddProduct,
  });

  final BreadCaptureController controller;
  final VoidCallback? onAddProduct;

  @override
  Widget build(BuildContext context) {
    final library = controller.library;
    final completed = library?.completedProductCount ?? 0;
    final total = controller.products.length;
    return Container(
      key: const ValueKey('bread-capture-header'),
      padding: const EdgeInsets.symmetric(
        horizontal: AppSpacing.x6,
        vertical: AppSpacing.x3,
      ),
      decoration: BoxDecoration(
        color: context.appColors.surface,
        border: Border(bottom: BorderSide(color: context.appColors.outline)),
      ),
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: context.appColors.brandSoft,
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(
              Icons.photo_camera_back_outlined,
              color: context.appColors.brand,
            ),
          ),
          const SizedBox(width: AppSpacing.x3),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('빵 촬영 관리', style: Theme.of(context).textTheme.titleMedium),
              Text(
                total == 0
                    ? '빵을 추가해 촬영을 시작하세요'
                    : '완료 $completed/$total종 · 사진 ${library?.capturedImageCount ?? 0}장',
                style: Theme.of(
                  context,
                ).textTheme.bodySmall?.copyWith(color: context.appColors.muted),
              ),
            ],
          ),
          const Spacer(),
          Flexible(
            child: Tooltip(
              message: library?.rootPath ?? '',
              child: Text(
                library?.rootPath ?? '저장 폴더 준비 중',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: Theme.of(
                  context,
                ).textTheme.bodySmall?.copyWith(color: context.appColors.muted),
              ),
            ),
          ),
          const SizedBox(width: AppSpacing.x3),
          OutlinedButton.icon(
            onPressed: controller.changingRoot ? null : controller.chooseRoot,
            icon: const Icon(Icons.folder_outlined, size: 18),
            label: const Text('저장 위치'),
          ),
          const SizedBox(width: AppSpacing.x2),
          IconButton(
            tooltip: '저장 폴더 열기',
            onPressed: controller.library == null ? null : controller.openRoot,
            icon: const Icon(Icons.folder_open_outlined),
          ),
          const SizedBox(width: AppSpacing.x2),
          FilledButton.icon(
            key: const ValueKey('bread-capture-add-product'),
            onPressed: onAddProduct,
            icon: const Icon(Icons.add_rounded, size: 18),
            label: const Text('빵 추가'),
          ),
        ],
      ),
    );
  }
}

class _ProductRail extends StatelessWidget {
  const _ProductRail({required this.controller, required this.onAddProduct});

  final BreadCaptureController controller;
  final VoidCallback? onAddProduct;

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: context.appColors.elevated,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppSpacing.x4,
              AppSpacing.x4,
              AppSpacing.x4,
              AppSpacing.x2,
            ),
            child: Row(
              children: [
                Text('빵 목록', style: Theme.of(context).textTheme.titleSmall),
                const Spacer(),
                Text(
                  '${controller.products.length}/20',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: context.appColors.muted,
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: ListView.separated(
              padding: const EdgeInsets.fromLTRB(
                AppSpacing.x3,
                AppSpacing.x2,
                AppSpacing.x3,
                AppSpacing.x3,
              ),
              itemCount: controller.products.length,
              separatorBuilder: (_, _) => const SizedBox(height: AppSpacing.x2),
              itemBuilder: (context, index) {
                final product = controller.products[index];
                final selected =
                    product.categoryId == controller.selectedProductId;
                return _ProductTile(
                  product: product,
                  selected: selected,
                  onTap: () => controller.selectProduct(product.categoryId),
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(AppSpacing.x3),
            child: OutlinedButton.icon(
              onPressed: onAddProduct,
              icon: const Icon(Icons.add_rounded),
              label: const Text('새 빵 등록'),
            ),
          ),
        ],
      ),
    );
  }
}

class _ProductTile extends StatelessWidget {
  const _ProductTile({
    required this.product,
    required this.selected,
    required this.onTap,
  });

  final BreadCaptureProduct product;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final progress =
        product.completedCaptureCount / BreadCaptureSlot.all.length;
    return Material(
      color: selected ? context.appColors.brandSoft : context.appColors.surface,
      borderRadius: BorderRadius.circular(context.appTokens.controlRadius),
      child: InkWell(
        key: ValueKey('bread-capture-product-${product.categoryId}'),
        onTap: onTap,
        borderRadius: BorderRadius.circular(context.appTokens.controlRadius),
        child: Container(
          padding: const EdgeInsets.all(AppSpacing.x3),
          decoration: BoxDecoration(
            border: Border.all(
              color: selected
                  ? context.appColors.brand
                  : context.appColors.outline,
              width: selected ? 2 : 1,
            ),
            borderRadius: BorderRadius.circular(
              context.appTokens.controlRadius,
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text(
                      product.name,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        fontWeight: AppTypography.bold,
                      ),
                    ),
                  ),
                  if (product.isComplete)
                    Icon(
                      Icons.check_circle_rounded,
                      color: context.appColors.success,
                      size: 19,
                    ),
                ],
              ),
              const SizedBox(height: AppSpacing.x1),
              Text(
                '${product.code} · ${product.completedCaptureCount}/10장',
                style: Theme.of(
                  context,
                ).textTheme.bodySmall?.copyWith(color: context.appColors.muted),
              ),
              const SizedBox(height: AppSpacing.x2),
              LinearProgressIndicator(
                value: progress,
                minHeight: 4,
                borderRadius: BorderRadius.circular(99),
                backgroundColor: context.appColors.outline,
                color: product.isComplete
                    ? context.appColors.success
                    : context.appColors.brand,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _GuidedCamera extends StatelessWidget {
  const _GuidedCamera({required this.controller});

  final BreadCaptureController controller;

  @override
  Widget build(BuildContext context) {
    final product = controller.selectedProduct!;
    final slot = controller.selectedSlot;
    final pending = controller.pendingImageBytes;
    return ColoredBox(
      color: AppColors.preview,
      child: Column(
        children: [
          Expanded(
            child: Stack(
              fit: StackFit.expand,
              children: [
                if (pending != null)
                  Image.memory(
                    pending,
                    key: const ValueKey('bread-capture-review-image'),
                    fit: BoxFit.contain,
                    gaplessPlayback: true,
                  )
                else
                  _LiveCameraSurface(
                    controller: controller,
                    guidePose: controller.isCameraReady ? slot.pose : null,
                  ),
                Positioned(
                  left: AppSpacing.x4,
                  top: AppSpacing.x4,
                  child: _PreviewBadge(
                    icon: pending == null
                        ? Icons.videocam_rounded
                        : Icons.image_outlined,
                    label: pending == null ? '라이브' : '촬영 확인',
                  ),
                ),
                Positioned(
                  right: AppSpacing.x4,
                  top: AppSpacing.x4,
                  child: _PreviewBadge(
                    icon: _poseIcon(slot.pose),
                    label: slot.label,
                  ),
                ),
                Positioned(
                  left: AppSpacing.x4,
                  right: AppSpacing.x4,
                  bottom: AppSpacing.x4,
                  child: _GuidanceStrip(
                    product: product,
                    slot: slot,
                    reviewing: pending != null,
                  ),
                ),
              ],
            ),
          ),
          AppActionBar(
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    pending == null
                        ? '한 개의 빵만 프레임 안에 크게 담아 주세요.'
                        : '윤곽, 표면과 초점이 선명한지 확인해 주세요.',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: context.appColors.muted,
                    ),
                  ),
                ),
                if (pending != null) ...[
                  OutlinedButton.icon(
                    onPressed: controller.saving
                        ? null
                        : controller.discardPendingCapture,
                    icon: const Icon(Icons.refresh_rounded),
                    label: const Text('다시 찍기'),
                  ),
                  const SizedBox(width: AppSpacing.x2),
                  AppProgressActionButton(
                    label: '이 사진 사용',
                    icon: const Icon(Icons.check_rounded),
                    onPressed: controller.acceptPendingCapture,
                    progressing: controller.saving,
                    progressLabel: '저장 중',
                    progressAnnouncement: '촬영 사진을 저장하고 있어요',
                  ),
                ] else
                  AppProgressActionButton(
                    label: !controller.isCameraReady
                        ? '카메라 연결'
                        : product.hasCapture(slot)
                        ? '다시 촬영'
                        : '촬영',
                    icon: Icon(
                      controller.isCameraReady
                          ? Icons.camera_alt_rounded
                          : Icons.videocam_outlined,
                    ),
                    onPressed: controller.isCameraReady
                        ? controller.capture
                        : controller.reconnectCamera,
                    progressing:
                        controller.capturing || controller.cameraInitializing,
                    progressLabel: controller.capturing ? '촬영 중' : '연결 중',
                    progressAnnouncement: controller.capturing
                        ? '사진을 촬영하고 있어요'
                        : '카메라를 연결하고 있어요',
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _LiveCameraSurface extends StatelessWidget {
  const _LiveCameraSurface({required this.controller, required this.guidePose});

  final BreadCaptureController controller;
  final BreadCapturePose? guidePose;

  @override
  Widget build(BuildContext context) {
    final camera = controller.cameraController;
    if (controller.cameraInitializing) {
      return const AppLoadingState(message: '카메라를 연결하는 중이에요');
    }
    if (camera == null || !camera.value.isInitialized) {
      return Center(
        child: Semantics(
          container: true,
          label: '카메라 연결이 필요해요. 연결 상태를 확인한 뒤 다시 연결해 주세요.',
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: .10),
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.videocam_off_outlined,
                  color: AppPalette.onPreviewMuted,
                ),
              ),
              const SizedBox(height: AppSpacing.x4),
              Text(
                '카메라 연결이 필요해요',
                style: Theme.of(
                  context,
                ).textTheme.titleMedium?.copyWith(color: AppPalette.onPreview),
              ),
              const SizedBox(height: AppSpacing.x2),
              Text(
                '카메라 연결 상태를 확인한 뒤 다시 연결해 주세요.',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: AppPalette.onPreviewMuted,
                ),
              ),
              const SizedBox(height: AppSpacing.x4),
              FilledButton.icon(
                onPressed: controller.reconnectCamera,
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('다시 연결'),
              ),
            ],
          ),
        ),
      );
    }
    return ClipRect(
      child: Center(
        child: AspectRatio(
          key: const ValueKey('bread-capture-live-viewport'),
          aspectRatio: breadCaptureViewportAspectRatio(
            camera.value.aspectRatio,
          ),
          child: Stack(
            fit: StackFit.expand,
            children: [
              breadCaptureUnmirroredPreview(CameraPreview(camera)),
              if (guidePose case final pose?) _CaptureGuideFrame(pose: pose),
            ],
          ),
        ),
      ),
    );
  }
}

class _CaptureGuideFrame extends StatelessWidget {
  const _CaptureGuideFrame({required this.pose});

  final BreadCapturePose pose;

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: CustomPaint(painter: _CaptureGridPainter(pose)),
    );
  }
}

class _CaptureGridPainter extends CustomPainter {
  const _CaptureGridPainter(this.pose);

  final BreadCapturePose pose;

  Offset _target(Size size) {
    final ratio = switch (pose) {
      BreadCapturePose.topLeft => const Offset(.24, .26),
      BreadCapturePose.topRight => const Offset(.76, .26),
      BreadCapturePose.center => const Offset(.50, .51),
      BreadCapturePose.bottomLeft => const Offset(.24, .76),
      BreadCapturePose.bottomRight => const Offset(.76, .76),
    };
    return Offset(size.width * ratio.dx, size.height * ratio.dy);
  }

  @override
  void paint(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..color = Colors.white.withValues(alpha: .22)
      ..strokeWidth = 1;
    for (final fraction in const [1 / 3, 2 / 3]) {
      canvas.drawLine(
        Offset(size.width * fraction, 0),
        Offset(size.width * fraction, size.height),
        gridPaint,
      );
      canvas.drawLine(
        Offset(0, size.height * fraction),
        Offset(size.width, size.height * fraction),
        gridPaint,
      );
    }

    final target = _target(size);
    final targetPaint = Paint()
      ..color = Colors.white.withValues(alpha: .82)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    canvas.drawOval(
      Rect.fromCenter(
        center: target,
        width: size.width * .34,
        height: size.height * .34,
      ),
      targetPaint,
    );
    canvas.drawCircle(target, 5, targetPaint..strokeWidth = 1.5);
  }

  @override
  bool shouldRepaint(_CaptureGridPainter oldDelegate) =>
      oldDelegate.pose != pose;
}

class _PreviewBadge extends StatelessWidget {
  const _PreviewBadge({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: AppSpacing.x3,
        vertical: AppSpacing.x2,
      ),
      decoration: BoxDecoration(
        color: context.appComponents.previewLabelSurface,
        borderRadius: BorderRadius.circular(99),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: context.appComponents.onPreview),
          const SizedBox(width: AppSpacing.x2),
          Text(
            label,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: context.appComponents.onPreview,
              fontWeight: AppTypography.bold,
            ),
          ),
        ],
      ),
    );
  }
}

class _GuidanceStrip extends StatelessWidget {
  const _GuidanceStrip({
    required this.product,
    required this.slot,
    required this.reviewing,
  });

  final BreadCaptureProduct product;
  final BreadCaptureSlot slot;
  final bool reviewing;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(AppSpacing.x3),
      decoration: BoxDecoration(
        color: context.appComponents.previewScrim,
        borderRadius: BorderRadius.circular(context.appTokens.controlRadius),
      ),
      child: Row(
        children: [
          Icon(
            reviewing
                ? Icons.fact_check_outlined
                : Icons.center_focus_strong_rounded,
            color: context.appComponents.onPreview,
          ),
          const SizedBox(width: AppSpacing.x3),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${product.name} · ${slot.label}',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: context.appComponents.onPreview,
                    fontWeight: AppTypography.bold,
                  ),
                ),
                Text(
                  reviewing
                      ? '손이나 다른 물체가 없고 빵 전체가 선명하면 사진을 사용하세요.'
                      : '${slot.pose.guidance} 카메라와 빵의 거리는 이전 촬영과 같게 유지하세요.',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: context.appComponents.onPreviewMuted,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ContactSheetPanel extends StatelessWidget {
  const _ContactSheetPanel({
    required this.controller,
    required this.onDeleteProduct,
    required this.onDeleteCapture,
  });

  final BreadCaptureController controller;
  final VoidCallback onDeleteProduct;
  final ValueChanged<BreadCaptureSlot> onDeleteCapture;

  @override
  Widget build(BuildContext context) {
    final product = controller.selectedProduct!;
    return ColoredBox(
      color: context.appColors.surface,
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(
              AppSpacing.x4,
              AppSpacing.x4,
              AppSpacing.x2,
              AppSpacing.x2,
            ),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        product.name,
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      Text(
                        '${product.code} · ${product.completedCaptureCount}/10장 완료',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: context.appColors.muted,
                        ),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  tooltip: '빵 삭제',
                  onPressed: controller.saving ? null : onDeleteProduct,
                  icon: Icon(
                    Icons.delete_outline_rounded,
                    color: context.appColors.error,
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppSpacing.x4),
            child: LinearProgressIndicator(
              value: product.completedCaptureCount / 10,
              minHeight: 6,
              borderRadius: BorderRadius.circular(99),
              backgroundColor: context.appColors.outline,
              color: product.isComplete
                  ? context.appColors.success
                  : context.appColors.brand,
            ),
          ),
          Expanded(
            child: ListView(
              padding: const EdgeInsets.all(AppSpacing.x4),
              children: [
                for (final side in BreadCaptureSide.values) ...[
                  Row(
                    children: [
                      Icon(
                        side == BreadCaptureSide.normal
                            ? Icons.light_mode_outlined
                            : Icons.flip_camera_android_outlined,
                        size: 18,
                        color: context.appColors.muted,
                      ),
                      const SizedBox(width: AppSpacing.x2),
                      Text(
                        side.label,
                        style: Theme.of(context).textTheme.titleSmall,
                      ),
                      const Spacer(),
                      Text(
                        '${BreadCaptureSlot.all.where((slot) => slot.side == side && product.hasCapture(slot)).length}/5',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: context.appColors.muted,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: AppSpacing.x2),
                  GridView.count(
                    physics: const NeverScrollableScrollPhysics(),
                    shrinkWrap: true,
                    crossAxisCount: 5,
                    mainAxisSpacing: AppSpacing.x2,
                    crossAxisSpacing: AppSpacing.x2,
                    childAspectRatio: .82,
                    children: [
                      for (final slot in BreadCaptureSlot.all.where(
                        (slot) => slot.side == side,
                      ))
                        _CaptureSlotCard(
                          product: product,
                          slot: slot,
                          selected: controller.selectedSlot.key == slot.key,
                          onTap: () => controller.selectSlot(slot),
                          onDelete: product.hasCapture(slot)
                              ? () => onDeleteCapture(slot)
                              : null,
                        ),
                    ],
                  ),
                  if (side != BreadCaptureSide.values.last)
                    const Padding(
                      padding: EdgeInsets.symmetric(vertical: AppSpacing.x4),
                      child: Divider(height: 1),
                    ),
                ],
                const SizedBox(height: AppSpacing.x2),
                Container(
                  padding: const EdgeInsets.all(AppSpacing.x3),
                  decoration: BoxDecoration(
                    color: context.appColors.elevated,
                    border: Border.all(color: context.appColors.outline),
                    borderRadius: BorderRadius.circular(
                      context.appTokens.controlRadius,
                    ),
                  ),
                  child: Text(
                    '촬영 기준 · 빵 1개 · 프레임 안쪽 · 동일 거리 · 손/도구 없음 · 초점과 표면 무늬 선명',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: context.appColors.muted,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _CaptureSlotCard extends StatelessWidget {
  const _CaptureSlotCard({
    required this.product,
    required this.slot,
    required this.selected,
    required this.onTap,
    required this.onDelete,
  });

  final BreadCaptureProduct product;
  final BreadCaptureSlot slot;
  final bool selected;
  final VoidCallback onTap;
  final VoidCallback? onDelete;

  @override
  Widget build(BuildContext context) {
    final path = product.capturePath(slot);
    return Material(
      color: selected
          ? context.appColors.brandSoft
          : context.appColors.elevated,
      borderRadius: BorderRadius.circular(8),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        key: ValueKey('bread-capture-slot-${slot.key}'),
        onTap: onTap,
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(
              color: selected
                  ? context.appColors.brand
                  : context.appColors.outline,
              width: selected ? 2 : 1,
            ),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (path != null)
                Image.file(
                  File(path),
                  fit: BoxFit.cover,
                  errorBuilder: (_, _, _) => const SizedBox(),
                )
              else
                Center(
                  child: Icon(
                    path == null ? _poseIcon(slot.pose) : Icons.check_rounded,
                    size: 20,
                    color: selected
                        ? context.appColors.brand
                        : context.appColors.subtle,
                  ),
                ),
              Positioned(
                left: 0,
                right: 0,
                bottom: 0,
                child: Container(
                  padding: const EdgeInsets.symmetric(vertical: 3),
                  color: path == null
                      ? context.appColors.surface.withValues(alpha: .92)
                      : AppPalette.previewScrim,
                  child: Text(
                    slot.pose.shortLabel,
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.labelSmall?.copyWith(
                      color: path == null
                          ? context.appColors.ink
                          : AppPalette.onPreview,
                      fontWeight: AppTypography.bold,
                    ),
                  ),
                ),
              ),
              if (path != null)
                Positioned(
                  right: 2,
                  top: 2,
                  child: Tooltip(
                    message: '사진 삭제',
                    child: InkWell(
                      onTap: onDelete,
                      borderRadius: BorderRadius.circular(99),
                      child: Container(
                        width: 22,
                        height: 22,
                        decoration: const BoxDecoration(
                          color: AppPalette.previewScrim,
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(
                          Icons.close_rounded,
                          size: 15,
                          color: AppPalette.onPreview,
                        ),
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

IconData _poseIcon(BreadCapturePose pose) => switch (pose) {
  BreadCapturePose.topLeft => Icons.north_west_rounded,
  BreadCapturePose.topRight => Icons.north_east_rounded,
  BreadCapturePose.center => Icons.center_focus_strong_rounded,
  BreadCapturePose.bottomLeft => Icons.south_west_rounded,
  BreadCapturePose.bottomRight => Icons.south_east_rounded,
};
