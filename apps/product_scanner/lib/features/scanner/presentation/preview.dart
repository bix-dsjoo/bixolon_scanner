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
                    visible: !detection.removed,
                    editable: false,
                    focusNode: _focusNodeFor(detection.source.itemId),
                    onTap: () =>
                        controller.selectDetection(detection.source.itemId),
                    onEditStart: controller.beginBoxEdit,
                    onEditUpdate: controller.updateSelectedBbox,
                    onEditEnd: controller.commitBoxEdit,
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
      return Center(child: SquareCameraPreview(controller: camera));
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
    required this.visible,
    required this.editable,
    required this.focusNode,
    required this.onTap,
    required this.onEditStart,
    required this.onEditUpdate,
    required this.onEditEnd,
  });

  final ReviewDetection detection;
  final int index;
  final Rect imageRect;
  final Size imageSize;
  final bool selected;
  final bool enabled;
  final bool visible;
  final bool editable;
  final FocusNode focusNode;
  final VoidCallback onTap;
  final VoidCallback onEditStart;
  final ValueChanged<BoundingBox> onEditUpdate;
  final VoidCallback onEditEnd;

  @override
  State<_DetectionBox> createState() => _DetectionBoxState();
}

class _DetectionBoxState extends State<_DetectionBox> {
  bool _focused = false;
  _ResizeCorner? _resizeCorner;

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
    if (!widget.visible) return const SizedBox.shrink();
    final bbox = widget.detection.finalBbox;
    final scaleX = widget.imageRect.width / widget.imageSize.width;
    final scaleY = widget.imageRect.height / widget.imageSize.height;
    final visualRect = Rect.fromLTWH(
      widget.imageRect.left + bbox.x * scaleX,
      widget.imageRect.top + bbox.y * scaleY,
      bbox.width * scaleX,
      bbox.height * scaleY,
    );
    final minimumHitRect = _minimumInteractionRect(
      visualRect,
      widget.imageRect,
      context.appTokens.controlHeight,
    );
    final hitRect = widget.editable && widget.selected
        ? _handleInteractionRect(minimumHitRect, visualRect, widget.imageRect)
        : minimumHitRect;
    final needsReview = !widget.detection.isConfirmed;
    final reviewPresentation = presentSegmentReview(widget.detection.source);
    final statusColor = _detectionStatusColor(widget.detection);
    final label = widget.index.toString();
    final labelWidth = math.max(
      AppSpacing.x6,
      AppSpacing.x4 + label.length * AppSpacing.x2,
    );
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
        explicitChildNodes: true,
        button: true,
        enabled: widget.enabled,
        selected: widget.selected,
        focused: widget.enabled ? _focused : null,
        label: widget.selected
            ? '${widget.index}번 현재 선택, $statusLabel 상품 영역'
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
              child: GestureDetector(
                behavior: HitTestBehavior.translucent,
                onPanStart: widget.enabled && widget.editable
                    ? (details) {
                        widget.onTap();
                        widget.onEditStart();
                        _resizeCorner = _nearestResizeCorner(
                          details.localPosition,
                          Rect.fromLTWH(
                            visualLeft,
                            visualTop,
                            visualRect.width,
                            visualRect.height,
                          ),
                        );
                      }
                    : null,
                onPanUpdate: widget.enabled && widget.editable
                    ? (details) {
                        final dx = details.delta.dx / scaleX;
                        final dy = details.delta.dy / scaleY;
                        widget.onEditUpdate(
                          _updatedBbox(
                            widget.detection.finalBbox,
                            dx,
                            dy,
                            _resizeCorner,
                          ),
                        );
                      }
                    : null,
                onPanEnd: widget.enabled && widget.editable
                    ? (_) {
                        _resizeCorner = null;
                        widget.onEditEnd();
                      }
                    : null,
                onPanCancel: widget.enabled && widget.editable
                    ? () {
                        _resizeCorner = null;
                        widget.onEditEnd();
                      }
                    : null,
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
                        child: DecoratedBox(
                          key: ValueKey(
                            'detection-outline-${widget.detection.source.itemId}',
                          ),
                          decoration: BoxDecoration(
                            color: widget.selected
                                ? statusColor.withValues(
                                    alpha: AppOpacity.selectedStatusSurface,
                                  )
                                : null,
                            border: Border.all(
                              color: statusColor,
                              width: widget.selected ? 3 : 2,
                            ),
                          ),
                        ),
                      ),
                    ),
                    Positioned(
                      left: visualLeft,
                      top: visualTop,
                      child: ExcludeSemantics(
                        child: Container(
                          key: ValueKey(
                            'detection-label-${widget.detection.source.itemId}',
                          ),
                          width: labelWidth,
                          height: AppSpacing.x6,
                          alignment: Alignment.center,
                          decoration: BoxDecoration(color: statusColor),
                          child: Text(
                            label,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              color: context.appComponents.onPreview,
                              fontSize: AppTypography.captionSize,
                              height: 1,
                              fontWeight: widget.selected
                                  ? AppTypography.bold
                                  : AppTypography.semibold,
                            ),
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
                    if (widget.editable && widget.selected)
                      ..._ResizeCorner.values.map(
                        (corner) => _ResizeHandle(
                          corner: corner,
                          visualRect: Rect.fromLTWH(
                            visualLeft,
                            visualTop,
                            visualRect.width,
                            visualRect.height,
                          ),
                          bbox: widget.detection.finalBbox,
                          onResize: (bbox) {
                            widget.onEditStart();
                            widget.onEditUpdate(bbox);
                            widget.onEditEnd();
                          },
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

Color _detectionStatusColor(ReviewDetection detection) {
  if (detection.isConfirmed) return AppColors.success;
  if (detection.source.status == ItemStatus.segmentRecapture) {
    return AppColors.error;
  }
  return AppColors.attention;
}

enum _ResizeCorner { topLeft, topRight, bottomLeft, bottomRight }

class _ResizeHandle extends StatefulWidget {
  const _ResizeHandle({
    required this.corner,
    required this.visualRect,
    required this.bbox,
    required this.onResize,
  });

  final _ResizeCorner corner;
  final Rect visualRect;
  final BoundingBox bbox;
  final ValueChanged<BoundingBox> onResize;

  @override
  State<_ResizeHandle> createState() => _ResizeHandleState();
}

class _ResizeHandleState extends State<_ResizeHandle> {
  final FocusNode _focusNode = FocusNode();
  bool _focused = false;

  @override
  void dispose() {
    _focusNode.dispose();
    super.dispose();
  }

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent) return KeyEventResult.ignored;
    final key = event.logicalKey;
    if (!{
      LogicalKeyboardKey.arrowUp,
      LogicalKeyboardKey.arrowDown,
      LogicalKeyboardKey.arrowLeft,
      LogicalKeyboardKey.arrowRight,
    }.contains(key)) {
      return KeyEventResult.ignored;
    }
    final amount = HardwareKeyboard.instance.isShiftPressed ? 10.0 : 1.0;
    final dx = key == LogicalKeyboardKey.arrowLeft
        ? -amount
        : key == LogicalKeyboardKey.arrowRight
        ? amount
        : 0.0;
    final dy = key == LogicalKeyboardKey.arrowUp
        ? -amount
        : key == LogicalKeyboardKey.arrowDown
        ? amount
        : 0.0;
    widget.onResize(_updatedBbox(widget.bbox, dx, dy, widget.corner));
    return KeyEventResult.handled;
  }

  @override
  Widget build(BuildContext context) {
    final center = switch (widget.corner) {
      _ResizeCorner.topLeft => widget.visualRect.topLeft,
      _ResizeCorner.topRight => widget.visualRect.topRight,
      _ResizeCorner.bottomLeft => widget.visualRect.bottomLeft,
      _ResizeCorner.bottomRight => widget.visualRect.bottomRight,
    };
    return Positioned(
      left: center.dx - 22,
      top: center.dy - 22,
      width: 44,
      height: 44,
      child: Semantics(
        container: true,
        focusable: true,
        focused: _focused,
        button: true,
        label: '${_cornerLabel(widget.corner)} 크기 조절 핸들',
        hint: '드래그하거나 방향키로 박스 크기를 조절합니다.',
        onTap: _focusNode.requestFocus,
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTap: _focusNode.requestFocus,
          child: Focus(
            focusNode: _focusNode,
            onFocusChange: (focused) {
              if (_focused != focused) setState(() => _focused = focused);
            },
            onKeyEvent: _handleKeyEvent,
            child: Center(
              child: Container(
                width: 12,
                height: 12,
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  border: Border.all(
                    color: _focused
                        ? context.appComponents.focusRing
                        : AppColors.attention,
                    width: _focused ? context.appTokens.focusRingWidth : 2,
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

String _cornerLabel(_ResizeCorner corner) => switch (corner) {
  _ResizeCorner.topLeft => '왼쪽 위',
  _ResizeCorner.topRight => '오른쪽 위',
  _ResizeCorner.bottomLeft => '왼쪽 아래',
  _ResizeCorner.bottomRight => '오른쪽 아래',
};

_ResizeCorner? _nearestResizeCorner(Offset point, Rect rect) {
  final corners = <_ResizeCorner, Offset>{
    _ResizeCorner.topLeft: rect.topLeft,
    _ResizeCorner.topRight: rect.topRight,
    _ResizeCorner.bottomLeft: rect.bottomLeft,
    _ResizeCorner.bottomRight: rect.bottomRight,
  };
  _ResizeCorner? nearest;
  var distance = double.infinity;
  for (final entry in corners.entries) {
    final candidate = (entry.value - point).distance;
    if (candidate < distance) {
      nearest = entry.key;
      distance = candidate;
    }
  }
  return distance <= 31 ? nearest : null;
}

BoundingBox _updatedBbox(
  BoundingBox bbox,
  double dx,
  double dy,
  _ResizeCorner? corner,
) {
  final deltaX = dx.round();
  final deltaY = dy.round();
  if (corner == null) {
    return BoundingBox(
      x: bbox.x + deltaX,
      y: bbox.y + deltaY,
      width: bbox.width,
      height: bbox.height,
    );
  }
  const minimum = 1;
  return switch (corner) {
    _ResizeCorner.topLeft => BoundingBox(
      x: bbox.x + deltaX,
      y: bbox.y + deltaY,
      width: (bbox.width - deltaX).clamp(minimum, 1 << 30).toInt(),
      height: (bbox.height - deltaY).clamp(minimum, 1 << 30).toInt(),
    ),
    _ResizeCorner.topRight => BoundingBox(
      x: bbox.x,
      y: bbox.y + deltaY,
      width: (bbox.width + deltaX).clamp(minimum, 1 << 30).toInt(),
      height: (bbox.height - deltaY).clamp(minimum, 1 << 30).toInt(),
    ),
    _ResizeCorner.bottomLeft => BoundingBox(
      x: bbox.x + deltaX,
      y: bbox.y,
      width: (bbox.width - deltaX).clamp(minimum, 1 << 30).toInt(),
      height: (bbox.height + deltaY).clamp(minimum, 1 << 30).toInt(),
    ),
    _ResizeCorner.bottomRight => BoundingBox(
      x: bbox.x,
      y: bbox.y,
      width: (bbox.width + deltaX).clamp(minimum, 1 << 30).toInt(),
      height: (bbox.height + deltaY).clamp(minimum, 1 << 30).toInt(),
    ),
  };
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

Rect _handleInteractionRect(Rect hitRect, Rect visualRect, Rect imageRect) {
  const handleRadius = 22.0;
  return hitRect
      .expandToInclude(visualRect.inflate(handleRadius))
      .intersect(imageRect.inflate(handleRadius));
}
