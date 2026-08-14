part of 'scanner_screen.dart';

class _PreviewPane extends StatelessWidget {
  const _PreviewPane({
    required this.controller,
    required this.previewSurfaceKey,
    required this.onReset,
    required this.onChooseImage,
    required this.onReturnToCamera,
    required this.onPrepareRecapture,
    required this.nextImageActionFocusNode,
    required this.cameraPrimaryActionFocusNode,
  });

  final ScannerController controller;
  final GlobalKey<_PreviewSurfaceState> previewSurfaceKey;
  final VoidCallback onReset;
  final VoidCallback onChooseImage;
  final VoidCallback onReturnToCamera;
  final VoidCallback onPrepareRecapture;
  final FocusNode nextImageActionFocusNode;
  final FocusNode cameraPrimaryActionFocusNode;

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: AppColors.surface,
      child: Column(
        children: [
          Expanded(
            child: _PreviewSurface(
              key: previewSurfaceKey,
              controller: controller,
            ),
          ),
          _InputActionBar(
            controller: controller,
            onReset: onReset,
            onChooseImage: onChooseImage,
            onReturnToCamera: onReturnToCamera,
            onPrepareRecapture: onPrepareRecapture,
            nextImageActionFocusNode: nextImageActionFocusNode,
            cameraPrimaryActionFocusNode: cameraPrimaryActionFocusNode,
          ),
        ],
      ),
    );
  }
}

class _PreviewSurface extends StatefulWidget {
  const _PreviewSurface({super.key, required this.controller});

  final ScannerController controller;

  @override
  State<_PreviewSurface> createState() => _PreviewSurfaceState();
}

class _PreviewSurfaceState extends State<_PreviewSurface> {
  final Map<String, FocusNode> _detectionFocusNodes = <String, FocusNode>{};

  void followSelectedDetectionFromKeyboard() {
    if (!_detectionFocusNodes.values.any((node) => node.hasFocus)) return;
    final selectedItemId = widget.controller.selectedItemId;
    if (selectedItemId == null) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final target = _detectionFocusNodes[selectedItemId];
      if (target?.canRequestFocus ?? false) target!.requestFocus();
    });
  }

  FocusNode _focusNodeFor(String itemId) => _detectionFocusNodes.putIfAbsent(
    itemId,
    () => FocusNode(debugLabel: 'preview-detection-$itemId'),
  );

  void _scheduleStaleFocusNodeCleanup(Set<String> activeItemIds) {
    final staleItemIds = _detectionFocusNodes.keys
        .where((itemId) => !activeItemIds.contains(itemId))
        .toList(growable: false);
    if (staleItemIds.isEmpty) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final currentItemIds = widget.controller.detections
          .map((detection) => detection.source.itemId)
          .toSet();
      for (final itemId in staleItemIds) {
        if (currentItemIds.contains(itemId)) continue;
        _detectionFocusNodes.remove(itemId)?.dispose();
      }
    });
  }

  @override
  void dispose() {
    for (final node in _detectionFocusNodes.values) {
      node.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: AppColors.preview,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final viewport = Size(constraints.maxWidth, constraints.maxHeight);
          final controller = widget.controller;
          final imageSize = controller.imageSize;
          _scheduleStaleFocusNodeCleanup(
            controller.detections
                .map((detection) => detection.source.itemId)
                .toSet(),
          );
          Rect? imageRect;
          if (controller.imageBytes != null && imageSize != null) {
            final fitted = applyBoxFit(BoxFit.contain, imageSize, viewport);
            imageRect = Alignment.center.inscribe(
              fitted.destination,
              Offset.zero & viewport,
            );
          }
          final selectedItemId = controller.selectedItemId;
          final orderedDetections = <ReviewDetection>[
            for (final detection in controller.detections)
              if (detection.source.itemId != selectedItemId) detection,
            for (final detection in controller.detections)
              if (detection.source.itemId == selectedItemId) detection,
          ];

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
              else if (controller.inputMode == InputMode.image)
                const _ImageInputPlaceholder()
              else
                _LiveCamera(controller: controller),
              if (imageRect != null && controller.hasResults)
                ...orderedDetections.map(
                  (detection) => _DetectionBox(
                    key: ValueKey('detection-box-${detection.source.itemId}'),
                    detection: detection,
                    index:
                        controller.detections.indexWhere(
                          (item) =>
                              item.source.itemId == detection.source.itemId,
                        ) +
                        1,
                    imageRect: imageRect!,
                    imageSize: imageSize!,
                    selected:
                        controller.selectedItemId == detection.source.itemId,
                    enabled: !controller.isBusy,
                    focusNode: _focusNodeFor(detection.source.itemId),
                    onTap: () =>
                        controller.selectDetection(detection.source.itemId),
                  ),
                ),
              if (controller.processState == ProcessState.capturing ||
                  controller.processState == ProcessState.analyzing)
                const _AnalyzingOverlay(),
              if (controller.inputMode != InputMode.image ||
                  controller.imageBytes != null)
                Positioned(
                  top: AppSpacing.x4,
                  left: AppSpacing.x4,
                  child: _PreviewSourceLabel(controller: controller),
                ),
            ],
          );
        },
      ),
    );
  }
}

class _PreviewSourceLabel extends StatelessWidget {
  const _PreviewSourceLabel({required this.controller});

  final ScannerController controller;

  @override
  Widget build(BuildContext context) {
    final (icon, label) = switch ((
      controller.inputMode,
      controller.imageBytes,
    )) {
      (InputMode.image, _?) => (
        Icons.image_outlined,
        AppPreviewCopy.selectedImage,
      ),
      (InputMode.camera, _?) => (
        Icons.photo_outlined,
        AppPreviewCopy.capturedImage,
      ),
      (InputMode.camera, _) when controller.isCameraReady => (
        Icons.videocam_outlined,
        AppPreviewCopy.liveCamera,
      ),
      _ => (Icons.videocam_off_outlined, AppPreviewCopy.cameraPreview),
    };
    return Semantics(
      key: const ValueKey('preview-source-label'),
      container: true,
      label: AppPreviewCopy.semanticLabel(label),
      child: ExcludeSemantics(
        child: Container(
          constraints: BoxConstraints(
            minHeight: context.appTokens.previewLabelMinHeight,
          ),
          padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.x3,
            vertical: AppSpacing.x1,
          ),
          decoration: BoxDecoration(
            color: context.appComponents.previewLabelSurface,
            borderRadius: BorderRadius.circular(
              context.appTokens.controlRadius,
            ),
            border: Border.all(
              color: context.appComponents.onPreview.withValues(alpha: .14),
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: context.appTokens.previewSourceMarkerSize,
                height: context.appTokens.previewSourceMarkerSize,
                transform: Matrix4.rotationZ(.785398),
                decoration: const BoxDecoration(color: AppColors.primary),
              ),
              const SizedBox(width: AppSpacing.x2),
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
        ),
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
          child: Transform.flip(flipX: true, child: CameraPreview(camera)),
        ),
      );
    }
    return Stack(
      children: [
        Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (controller.cameraInitializing)
                AppProgressVisual(
                  size: context.appTokens.pageProgressSize,
                  strokeWidth: 2.5,
                  color: context.appComponents.onPreviewMuted,
                )
              else
                Icon(
                  Icons.videocam_off_outlined,
                  color: context.appComponents.onPreviewMuted,
                  size: context.appTokens.compactVisualSize,
                ),
              const SizedBox(height: AppSpacing.x4),
              Text(
                controller.cameraInitializing
                    ? '카메라를 준비하고 있어요'
                    : controller.cameraMessage ?? '카메라를 연결해 주세요',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.labelLarge?.copyWith(
                  color: context.appComponents.onPreviewMuted,
                  fontWeight: AppTypography.regular,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _ImageInputPlaceholder extends StatelessWidget {
  const _ImageInputPlaceholder();

  @override
  Widget build(BuildContext context) {
    return Semantics(
      container: true,
      label: '이미지 미리보기 영역, 선택된 이미지 없음',
      child: ExcludeSemantics(
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.image_outlined,
                color: context.appComponents.onPreviewMuted,
                size: context.appTokens.compactVisualSize,
              ),
              const SizedBox(height: AppSpacing.x4),
              Text(
                '이미지를 선택해 주세요',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.labelLarge?.copyWith(
                  color: context.appComponents.onPreviewMuted,
                  fontWeight: AppTypography.regular,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _AnalyzingOverlay extends StatelessWidget {
  const _AnalyzingOverlay();

  @override
  Widget build(BuildContext context) {
    return ExcludeSemantics(
      child: ColoredBox(
        color: context.appComponents.previewScrim,
        child: Center(
          child: AppProgressVisual(
            size: context.appTokens.previewProgressSize,
            strokeWidth: 2.4,
            color: context.appComponents.onPreview,
          ),
        ),
      ),
    );
  }
}

class _DetectionBox extends StatefulWidget {
  const _DetectionBox({
    super.key,
    required this.detection,
    required this.index,
    required this.imageRect,
    required this.imageSize,
    required this.selected,
    required this.enabled,
    required this.focusNode,
    required this.onTap,
  });

  final ReviewDetection detection;
  final int index;
  final Rect imageRect;
  final Size imageSize;
  final bool selected;
  final bool enabled;
  final FocusNode focusNode;
  final VoidCallback onTap;

  @override
  State<_DetectionBox> createState() => _DetectionBoxState();
}

class _DetectionBoxState extends State<_DetectionBox> {
  bool _focused = false;

  @override
  void initState() {
    super.initState();
    _syncTraversalState();
  }

  void _syncTraversalState() {
    widget.focusNode.skipTraversal = !widget.enabled || !widget.selected;
  }

  @override
  void didUpdateWidget(covariant _DetectionBox oldWidget) {
    super.didUpdateWidget(oldWidget);
    _syncTraversalState();
    if (!widget.enabled && _focused) {
      widget.focusNode.unfocus();
      _focused = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    final bbox = widget.detection.source.bbox;
    final scaleX = widget.imageRect.width / widget.imageSize.width;
    final scaleY = widget.imageRect.height / widget.imageSize.height;
    final visualRect = Rect.fromLTWH(
      widget.imageRect.left + bbox.x * scaleX,
      widget.imageRect.top + bbox.y * scaleY,
      bbox.width * scaleX,
      bbox.height * scaleY,
    );
    final hitRect = _minimumInteractionRect(
      visualRect,
      widget.imageRect,
      context.appTokens.controlHeight,
    );
    final needsReview = !widget.detection.isConfirmed;
    final reviewPresentation = presentSegmentReview(widget.detection.source);
    final statusColor = needsReview ? AppColors.attention : AppColors.success;
    final label = widget.selected
        ? '${widget.index}  현재 검수'
        : needsReview
        ? '${widget.index}  ?'
        : '${widget.index}';
    final statusLabel = needsReview
        ? reviewPresentation.shortLabel
        : '${widget.detection.finalProduct?.displayName ?? '상품'}, 확정';
    final visualLeft = visualRect.left - hitRect.left;
    final visualTop = visualRect.top - hitRect.top;
    final tokens = context.appTokens;
    return Positioned.fromRect(
      rect: hitRect,
      child: Semantics(
        container: true,
        excludeSemantics: true,
        button: true,
        enabled: widget.enabled,
        selected: widget.selected,
        focused: widget.enabled ? _focused : null,
        label: widget.selected
            ? '${widget.index}번 현재 검수, $statusLabel 상품 영역'
            : '${widget.index}번 $statusLabel 상품 영역',
        onTap: widget.enabled ? widget.onTap : null,
        child: MouseRegion(
          cursor: widget.enabled
              ? SystemMouseCursors.click
              : SystemMouseCursors.basic,
          child: Material(
            type: MaterialType.transparency,
            child: InkWell(
              excludeFromSemantics: true,
              focusNode: widget.focusNode,
              canRequestFocus: widget.enabled,
              onTap: widget.enabled ? widget.onTap : null,
              focusColor: Colors.transparent,
              hoverColor: Colors.transparent,
              splashColor: Colors.transparent,
              highlightColor: Colors.transparent,
              onFocusChange: (focused) {
                if (_focused != focused) setState(() => _focused = focused);
              },
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  Positioned.fill(
                    left: visualLeft,
                    top: visualTop,
                    right: hitRect.width - visualLeft - visualRect.width,
                    bottom: hitRect.height - visualTop - visualRect.height,
                    child: Container(
                      key: ValueKey(
                        'detection-visual-${widget.detection.source.itemId}',
                      ),
                      decoration: BoxDecoration(
                        color: widget.selected
                            ? statusColor.withValues(alpha: .06)
                            : null,
                        border: Border.all(
                          color: statusColor,
                          width: widget.selected ? 2 : 1.5,
                        ),
                      ),
                    ),
                  ),
                  Positioned(
                    left: visualLeft - 1,
                    top: visualTop - 1,
                    child: Container(
                      constraints: BoxConstraints(
                        maxWidth: visualRect.width + 80,
                      ),
                      padding: const EdgeInsets.symmetric(
                        horizontal: AppSpacing.x2,
                        vertical: AppSpacing.x1,
                      ),
                      decoration: BoxDecoration(color: statusColor),
                      child: Text(
                        label,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          color: context.appComponents.onPreview,
                          fontSize: AppTypography.captionSize,
                          fontWeight: widget.selected
                              ? AppTypography.bold
                              : AppTypography.semibold,
                        ),
                      ),
                    ),
                  ),
                  Positioned.fill(
                    child: IgnorePointer(
                      child: AnimatedContainer(
                        key: ValueKey(
                          'detection-focus-${widget.detection.source.itemId}',
                        ),
                        duration: MediaQuery.disableAnimationsOf(context)
                            ? Duration.zero
                            : tokens.motionFast,
                        curve: AppMotion.interactionCurve,
                        decoration: BoxDecoration(
                          border: _focused
                              ? Border.all(
                                  color: context.appComponents.focusRing,
                                  width: tokens.focusRingWidth,
                                )
                              : null,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

Rect _minimumInteractionRect(Rect visual, Rect bounds, double minimumSize) {
  final width = visual.width < minimumSize ? minimumSize : visual.width;
  final height = visual.height < minimumSize ? minimumSize : visual.height;
  final boundedWidth = width > bounds.width ? bounds.width : width;
  final boundedHeight = height > bounds.height ? bounds.height : height;
  final left = (visual.center.dx - boundedWidth / 2).clamp(
    bounds.left,
    bounds.right - boundedWidth,
  );
  final top = (visual.center.dy - boundedHeight / 2).clamp(
    bounds.top,
    bounds.bottom - boundedHeight,
  );
  return Rect.fromLTWH(left, top, boundedWidth, boundedHeight);
}
