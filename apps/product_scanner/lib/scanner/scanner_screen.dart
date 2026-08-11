import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../activity/activity_screen.dart';
import '../controllers/scanner_controller.dart';
import '../models/scan_models.dart';
import '../services/scanner_api.dart';
import '../theme/app_copy.dart';
import '../theme/app_theme.dart';
import '../theme/app_tokens.dart';
import '../widgets/app_components.dart';

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

class _ScannerScreenState extends State<ScannerScreen>
    with WidgetsBindingObserver {
  _WorkspaceSection _section = _WorkspaceSection.scan;
  bool _activityMounted = false;
  final GlobalKey<_PreviewSurfaceState> _previewSurfaceKey =
      GlobalKey<_PreviewSurfaceState>();
  final GlobalKey<_ResultPanelState> _resultPanelKey =
      GlobalKey<_ResultPanelState>();
  final FocusNode _nextImageActionFocusNode = FocusNode(
    debugLabel: 'next-image-action',
  );
  final FocusNode _cameraPrimaryActionFocusNode = FocusNode(
    debugLabel: 'camera-primary-action',
  );
  late ProcessState _previousProcessState;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _previousProcessState = widget.controller.processState;
    widget.controller.addListener(_handleControllerChanged);
    if (widget.autoInitialize) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        widget.controller.initialize();
      });
    }
  }

  @override
  void didUpdateWidget(covariant ScannerScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.controller == widget.controller) return;
    oldWidget.controller.removeListener(_handleControllerChanged);
    _previousProcessState = widget.controller.processState;
    widget.controller.addListener(_handleControllerChanged);
  }

  void _handleControllerChanged() {
    final currentState = widget.controller.processState;
    final completedImageSave =
        _previousProcessState == ProcessState.submitting &&
        currentState == ProcessState.ready &&
        widget.controller.inputMode == InputMode.image &&
        widget.controller.imageBytes == null &&
        widget.controller.completionMessage != null;
    _previousProcessState = currentState;
    if (!completedImageSave ||
        _section != _WorkspaceSection.scan ||
        FocusManager.instance.primaryFocus?.debugLabel !=
            'final-confirmation-action') {
      return;
    }
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || !_nextImageActionFocusNode.canRequestFocus) return;
      _nextImageActionFocusNode.requestFocus();
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    widget.controller.removeListener(_handleControllerChanged);
    _nextImageActionFocusNode.dispose();
    _cameraPrimaryActionFocusNode.dispose();
    if (widget.disposeController) widget.controller.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      unawaited(widget.controller.restoreCamera(forceReconnect: true));
    }
  }

  Future<void> _requestReset() async {
    if (await _canDiscardChanges(
      title: '${AppActionCopy.recapture}할까요?',
      confirmLabel: AppActionCopy.recapture,
    )) {
      widget.controller.resetSession();
    }
  }

  Future<void> _returnToCamera() async {
    if (widget.controller.isBusy) return;
    if (!await _canDiscardChanges(
      title: '카메라로 돌아갈까요?',
      confirmLabel: AppActionCopy.returnToCamera,
    )) {
      return;
    }
    await widget.controller.returnToCamera();
  }

  Future<void> _prepareRecapture() async {
    if (widget.controller.isBusy) return;
    await widget.controller.returnToCamera();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || !_cameraPrimaryActionFocusNode.canRequestFocus) return;
      _cameraPrimaryActionFocusNode.requestFocus();
    });
  }

  void _showSection(_WorkspaceSection section) {
    setState(() {
      _section = section;
      if (section == _WorkspaceSection.activity) {
        _activityMounted = true;
      }
    });
    if (section == _WorkspaceSection.scan) {
      unawaited(widget.controller.restoreCamera());
    }
  }

  Future<void> _chooseImage({bool showScanOnSelection = false}) async {
    if (!widget.controller.canChooseImage) return;
    if (!await _canDiscardChanges(
      title: '다른 이미지를 선택할까요?',
      confirmLabel: AppActionCopy.chooseAnotherImage,
    )) {
      return;
    }
    final selectionHandled = await widget.controller.chooseImage();
    if (selectionHandled && showScanOnSelection && mounted) {
      setState(() => _section = _WorkspaceSection.scan);
    }
  }

  Future<bool> _canDiscardChanges({
    required String title,
    required String confirmLabel,
  }) async {
    final controller = widget.controller;
    if (!controller.hasUserChanges) return true;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AppConfirmDialog(
        title: title,
        description: '지금까지 확인한 상품 선택이 사라져요.',
        confirmLabel: confirmLabel,
        onCancel: () => Navigator.of(context).pop(false),
        onConfirm: () => Navigator.of(context).pop(true),
      ),
    );
    return confirmed == true;
  }

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent) return KeyEventResult.ignored;
    final key = event.logicalKey;
    final controlPressed = HardwareKeyboard.instance.isControlPressed;
    if (controlPressed && key == LogicalKeyboardKey.keyO) {
      if (!widget.controller.canChooseImage) return KeyEventResult.ignored;
      _chooseImage(showScanOnSelection: true);
      return KeyEventResult.handled;
    }
    if (_section != _WorkspaceSection.scan || _isEditingText()) {
      return KeyEventResult.ignored;
    }
    final controller = widget.controller;
    if (controller.isBusy && key != LogicalKeyboardKey.escape) {
      return KeyEventResult.ignored;
    }
    if (key == LogicalKeyboardKey.arrowUp) {
      controller.selectPreviousDetection();
      _previewSurfaceKey.currentState?.followSelectedDetectionFromKeyboard();
      return KeyEventResult.handled;
    }
    if (key == LogicalKeyboardKey.arrowDown) {
      controller.selectNextDetection();
      _previewSurfaceKey.currentState?.followSelectedDetectionFromKeyboard();
      return KeyEventResult.handled;
    }
    if (key == LogicalKeyboardKey.slash && controller.selectedItemId != null) {
      controller.showSearch(controller.selectedItemId!);
      return KeyEventResult.handled;
    }
    if (key == LogicalKeyboardKey.escape && controller.searchItemId != null) {
      final resultPanel = _resultPanelKey.currentState;
      if (resultPanel != null) {
        resultPanel.closeSearchFromKeyboard();
      } else {
        controller.hideSearch();
      }
      return KeyEventResult.handled;
    }
    return KeyEventResult.ignored;
  }

  bool _isEditingText() {
    final context = FocusManager.instance.primaryFocus?.context;
    if (context == null) return false;
    return context.widget is EditableText ||
        context.findAncestorWidgetOfExactType<EditableText>() != null;
  }

  Widget _scanWorkspace(ScannerController controller) {
    return Stack(
      children: [
        LayoutBuilder(
          builder: (context, constraints) {
            if (constraints.maxWidth < AppBreakpoints.scanStacked) {
              return Column(
                children: [
                  Expanded(
                    flex: 5,
                    child: _PreviewPane(
                      controller: controller,
                      previewSurfaceKey: _previewSurfaceKey,
                      onReset: _requestReset,
                      onChooseImage: _chooseImage,
                      onReturnToCamera: _returnToCamera,
                      onPrepareRecapture: _prepareRecapture,
                      nextImageActionFocusNode: _nextImageActionFocusNode,
                      cameraPrimaryActionFocusNode:
                          _cameraPrimaryActionFocusNode,
                    ),
                  ),
                  const Divider(height: 1),
                  Expanded(
                    flex: 7,
                    child: _ResultPanel(
                      key: _resultPanelKey,
                      controller: controller,
                    ),
                  ),
                ],
              );
            }
            final tokens = context.appTokens;
            final resultWidth =
                (constraints.maxWidth * tokens.scanResultPanelFraction).clamp(
                  tokens.scanResultPanelMinWidth,
                  tokens.scanResultPanelMaxWidth,
                );
            return Row(
              children: [
                Expanded(
                  child: _PreviewPane(
                    controller: controller,
                    previewSurfaceKey: _previewSurfaceKey,
                    onReset: _requestReset,
                    onChooseImage: _chooseImage,
                    onReturnToCamera: _returnToCamera,
                    onPrepareRecapture: _prepareRecapture,
                    nextImageActionFocusNode: _nextImageActionFocusNode,
                    cameraPrimaryActionFocusNode: _cameraPrimaryActionFocusNode,
                  ),
                ),
                const VerticalDivider(width: 1),
                SizedBox(
                  key: const ValueKey('scan-result-panel'),
                  width: resultWidth,
                  child: _ResultPanel(
                    key: _resultPanelKey,
                    controller: controller,
                  ),
                ),
              ],
            );
          },
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Focus(
      autofocus: true,
      onKeyEvent: _handleKeyEvent,
      child: Scaffold(
        body: AnimatedBuilder(
          animation: widget.controller,
          builder: (context, _) {
            final controller = widget.controller;
            return Stack(
              children: [
                Column(
                  children: [
                    _TopBar(
                      controller: controller,
                      section: _section,
                      onSectionChanged: _showSection,
                    ),
                    Expanded(
                      child: IndexedStack(
                        index: _section.index,
                        children: [
                          _scanWorkspace(controller),
                          if (_activityMounted)
                            ActivityScreen(
                              loadLogs: controller.loadScanLogs,
                              dataRevision: controller.activityDataRevision,
                              latestSavedScanId: controller.latestSavedScanId,
                              active: _section == _WorkspaceSection.activity,
                              canChooseImageShortcut: controller.canChooseImage,
                              onChooseImageShortcut: () =>
                                  _chooseImage(showScanOnSelection: true),
                              onNavigateToScan: () =>
                                  _showSection(_WorkspaceSection.scan),
                            )
                          else
                            const SizedBox.shrink(),
                        ],
                      ),
                    ),
                  ],
                ),
                if (controller.completionMessage != null)
                  Positioned(
                    top: context.appTokens.headerHeight + AppSpacing.x4,
                    left: 0,
                    right: 0,
                    child: IgnorePointer(
                      key: const ValueKey('global-completion-feedback'),
                      child: Center(
                        child: _CompletionBanner(
                          message: controller.completionMessage!,
                        ),
                      ),
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }
}

enum _WorkspaceSection { scan, activity }

class _TopBar extends StatelessWidget {
  const _TopBar({
    required this.controller,
    required this.section,
    required this.onSectionChanged,
  });

  final ScannerController controller;
  final _WorkspaceSection section;
  final ValueChanged<_WorkspaceSection> onSectionChanged;

  @override
  Widget build(BuildContext context) {
    final cameraReady = controller.isCameraReady;
    final imageModeActive =
        section == _WorkspaceSection.scan &&
        controller.inputMode == InputMode.image;
    final cameraAttentionRelevant =
        section == _WorkspaceSection.scan && !imageModeActive;
    final (
      statusColor,
      statusIcon,
      statusText,
    ) = controller.hasActiveCameraIssue && cameraAttentionRelevant
        ? (AppColors.attention, Icons.videocam_off_rounded, '카메라 확인 필요')
        : controller.isCameraCheckActive && cameraAttentionRelevant
        ? (AppColors.muted, Icons.sync_rounded, '카메라 확인 중')
        : switch ((
            cameraReady,
            controller.cameraInitializing,
            cameraAttentionRelevant,
          )) {
            (true, _, _) => (
              AppColors.success,
              Icons.videocam_rounded,
              '카메라 연결됨',
            ),
            (false, true, _) => (
              AppColors.muted,
              Icons.sync_rounded,
              '카메라 확인 중',
            ),
            (false, false, false) when imageModeActive => (
              AppColors.muted,
              Icons.image_outlined,
              '이미지 입력 · 카메라 미연결',
            ),
            (false, false, false) => (
              AppColors.muted,
              Icons.videocam_off_outlined,
              '카메라 미연결',
            ),
            _ => (AppColors.attention, Icons.videocam_off_rounded, '카메라 확인 필요'),
          };
    return Container(
      key: const ValueKey('app-top-bar'),
      height: context.appTokens.headerHeight,
      color: AppColors.workspace,
      child: Stack(
        children: [
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            height: context.appTokens.navigationIndicatorThickness,
            child: const ColoredBox(
              key: ValueKey('app-top-bar-accent'),
              color: AppColors.primary,
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppSpacing.x6),
            child: Row(
              children: [
                const AppBrandMark(),
                const SizedBox(width: AppSpacing.x8),
                _NavigationItem(
                  key: const ValueKey('navigation-scan'),
                  label: '스캔',
                  selected: section == _WorkspaceSection.scan,
                  onTap: () => onSectionChanged(_WorkspaceSection.scan),
                ),
                _NavigationItem(
                  key: const ValueKey('navigation-activity'),
                  label: '활동',
                  selected: section == _WorkspaceSection.activity,
                  onTap: () => onSectionChanged(_WorkspaceSection.activity),
                ),
                const Spacer(),
                AppStatusBadge(
                  label: statusText,
                  icon: statusIcon,
                  color: statusColor,
                  liveRegion: statusText == '카메라 연결됨',
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _NavigationItem extends StatefulWidget {
  const _NavigationItem({
    super.key,
    required this.label,
    required this.selected,
    required this.onTap,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;

  @override
  State<_NavigationItem> createState() => _NavigationItemState();
}

class _NavigationItemState extends State<_NavigationItem> {
  bool _focused = false;

  @override
  Widget build(BuildContext context) {
    final tokens = context.appTokens;
    final tabRadius = BorderRadius.only(
      topLeft: Radius.circular(tokens.controlRadius),
      topRight: Radius.circular(tokens.controlRadius),
    );
    final outlineColor = _focused
        ? context.appComponents.focusRing
        : widget.selected
        ? AppColors.primary
        : Colors.transparent;
    return Semantics(
      container: true,
      excludeSemantics: true,
      button: true,
      selected: widget.selected,
      focused: _focused,
      label: widget.label,
      onTap: widget.onTap,
      child: SizedBox(
        width: tokens.navigationItemWidth,
        height: tokens.headerHeight,
        child: Padding(
          padding: EdgeInsets.only(top: tokens.navigationItemTopInset),
          child: Material(
            color: Colors.transparent,
            borderRadius: tabRadius,
            clipBehavior: Clip.antiAlias,
            child: InkWell(
              onTap: widget.onTap,
              excludeFromSemantics: true,
              borderRadius: tabRadius,
              onFocusChange: (focused) {
                if (_focused != focused) setState(() => _focused = focused);
              },
              child: Stack(
                fit: StackFit.expand,
                children: [
                  AnimatedContainer(
                    key: ValueKey('navigation-tab-${widget.label}'),
                    duration: MediaQuery.disableAnimationsOf(context)
                        ? Duration.zero
                        : tokens.motionFast,
                    curve: AppMotion.interactionCurve,
                    decoration: BoxDecoration(
                      color: widget.selected
                          ? AppColors.surface
                          : Colors.transparent,
                      border: Border.all(
                        color: outlineColor,
                        width: tokens.navigationIndicatorThickness,
                      ),
                      borderRadius: tabRadius,
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      widget.label,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        fontWeight: widget.selected
                            ? AppTypography.bold
                            : AppTypography.semibold,
                        color: widget.selected
                            ? AppColors.ink
                            : AppColors.muted,
                      ),
                    ),
                  ),
                  if (widget.selected && !_focused)
                    Positioned(
                      left: tokens.navigationIndicatorThickness,
                      right: tokens.navigationIndicatorThickness,
                      bottom: 0,
                      height: tokens.navigationIndicatorThickness,
                      child: const ColoredBox(color: AppColors.surface),
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
    final statusColor = needsReview ? AppColors.attention : AppColors.success;
    final label = widget.selected
        ? '${widget.index}  현재 검수'
        : needsReview
        ? '${widget.index}  ?'
        : '${widget.index}';
    final statusLabel = needsReview
        ? '확인 필요'
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

class _InputActionBar extends StatelessWidget {
  const _InputActionBar({
    required this.controller,
    required this.onReset,
    required this.onChooseImage,
    required this.onReturnToCamera,
    required this.onPrepareRecapture,
    required this.nextImageActionFocusNode,
    required this.cameraPrimaryActionFocusNode,
  });

  final ScannerController controller;
  final VoidCallback onReset;
  final VoidCallback onChooseImage;
  final VoidCallback onReturnToCamera;
  final VoidCallback onPrepareRecapture;
  final FocusNode nextImageActionFocusNode;
  final FocusNode cameraPrimaryActionFocusNode;

  @override
  Widget build(BuildContext context) {
    final busy = controller.isBusy;
    final hasImage = controller.imageBytes != null;
    final capturing = controller.processState == ProcessState.capturing;
    final analyzing = controller.processState == ProcessState.analyzing;
    final reviewingSuccess =
        controller.processState == ProcessState.reviewing &&
        !controller.isRecapture;
    if (controller.processState == ProcessState.submitting) {
      return AppActionBar(
        child: Row(
          children: [
            const Icon(
              Icons.lock_clock_outlined,
              size: 18,
              color: AppColors.muted,
            ),
            const SizedBox(width: AppSpacing.x2),
            Text(
              '저장이 끝나면 다음 이미지를 준비할 수 있어요',
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(color: AppColors.muted),
            ),
          ],
        ),
      );
    }
    if (capturing || analyzing) {
      return AppActionBar(
        child: Row(
          children: [
            const Spacer(),
            AppProgressActionButton(
              focusNode: capturing ? cameraPrimaryActionFocusNode : null,
              onPressed: null,
              progressing: true,
              progressLabel: capturing
                  ? AppActionCopy.capturing
                  : AppActionCopy.analyzing,
              progressAnnouncement: capturing
                  ? AppActionCopy.capturingAnnouncement
                  : AppActionCopy.analyzingAnnouncement,
              icon: Icon(
                controller.inputMode == InputMode.image
                    ? Icons.auto_awesome_outlined
                    : Icons.camera_alt_outlined,
                size: 18,
              ),
              label: capturing ? AppActionCopy.capture : AppActionCopy.analyze,
            ),
          ],
        ),
      );
    }
    return AppActionBar(
      child: Row(
        children: [
          if (controller.inputMode == InputMode.image && !hasImage) ...[
            _ReturnToCameraButton(
              onPressed: controller.canChooseImage ? onReturnToCamera : null,
            ),
            const Spacer(),
            Tooltip(
              message: '${AppActionCopy.chooseImage} (Ctrl+O)',
              child: FilledButton.icon(
                focusNode: nextImageActionFocusNode,
                onPressed: controller.canChooseImage ? onChooseImage : null,
                icon: const Icon(Icons.image_outlined, size: 18),
                label: const Text(AppActionCopy.chooseImage),
              ),
            ),
          ] else if (controller.hasActiveCameraIssue) ...[
            Tooltip(
              message: '${AppActionCopy.chooseImage} (Ctrl+O)',
              child: OutlinedButton.icon(
                onPressed: controller.canChooseImage ? onChooseImage : null,
                icon: const Icon(Icons.image_outlined, size: 18),
                label: const Text(AppActionCopy.chooseImage),
              ),
            ),
            const Spacer(),
            FilledButton.icon(
              onPressed: busy || controller.cameraInitializing
                  ? null
                  : controller.reconnectCamera,
              icon: const Icon(Icons.videocam_outlined, size: 18),
              label: const Text(AppActionCopy.reconnect),
            ),
          ] else if (controller.isCameraCheckActive) ...[
            Tooltip(
              message: '${AppActionCopy.chooseImage} (Ctrl+O)',
              child: OutlinedButton.icon(
                onPressed: controller.canChooseImage ? onChooseImage : null,
                icon: const Icon(Icons.image_outlined, size: 18),
                label: const Text(AppActionCopy.chooseImage),
              ),
            ),
            const Spacer(),
            AppProgressActionButton(
              onPressed: null,
              progressing: true,
              progressLabel: AppActionCopy.checkingConnection,
              progressAnnouncement:
                  AppActionCopy.checkingConnectionAnnouncement,
              icon: const Icon(Icons.videocam_outlined, size: 18),
              label: AppActionCopy.reconnect,
            ),
          ] else if (reviewingSuccess) ...[
            if (controller.inputMode == InputMode.image) ...[
              _ReturnToCameraButton(
                onPressed: controller.canChooseImage ? onReturnToCamera : null,
              ),
              const SizedBox(width: AppSpacing.x2),
            ],
            Tooltip(
              message: controller.inputMode == InputMode.image
                  ? '${AppActionCopy.chooseAnotherImage} (Ctrl+O)'
                  : '현재 검수를 버리고 ${AppActionCopy.recapture}',
              child: OutlinedButton.icon(
                onPressed: !controller.canChooseImage
                    ? null
                    : controller.inputMode == InputMode.image
                    ? onChooseImage
                    : onReset,
                icon: Icon(
                  controller.inputMode == InputMode.image
                      ? Icons.image_outlined
                      : Icons.refresh_rounded,
                  size: 18,
                ),
                label: Text(
                  controller.inputMode == InputMode.image
                      ? AppActionCopy.chooseAnotherImage
                      : AppActionCopy.recapture,
                ),
              ),
            ),
          ] else if (controller.isRecapture &&
              controller.inputMode == InputMode.camera) ...[
            Tooltip(
              message: '${AppActionCopy.chooseImage} (Ctrl+O)',
              child: OutlinedButton.icon(
                onPressed: controller.canChooseImage ? onChooseImage : null,
                icon: const Icon(Icons.image_outlined, size: 18),
                label: const Text(AppActionCopy.chooseImage),
              ),
            ),
            const Spacer(),
            _RecaptureLogButton(controller: controller),
            const SizedBox(width: AppSpacing.x2),
            FilledButton.icon(
              focusNode: cameraPrimaryActionFocusNode,
              onPressed: controller.canChooseImage ? onPrepareRecapture : null,
              icon: const Icon(Icons.videocam_outlined, size: 18),
              label: const Text(AppActionCopy.returnToCapture),
            ),
          ] else if (controller.isRecapture &&
              controller.inputMode == InputMode.image) ...[
            _ReturnToCameraButton(
              onPressed: controller.canChooseImage ? onReturnToCamera : null,
            ),
            const Spacer(),
            _RecaptureLogButton(controller: controller),
            const SizedBox(width: AppSpacing.x2),
            FilledButton.icon(
              onPressed: controller.canChooseImage ? onChooseImage : null,
              icon: const Icon(Icons.image_outlined, size: 18),
              label: const Text(AppActionCopy.chooseAnotherImage),
            ),
          ] else if (controller.processState == ProcessState.error &&
              controller.errorRecovery ==
                  ScannerErrorRecovery.replaceInput) ...[
            if (controller.inputMode == InputMode.image)
              _ReturnToCameraButton(
                onPressed: controller.canChooseImage ? onReturnToCamera : null,
              ),
            const Spacer(),
            FilledButton.icon(
              onPressed: !controller.canChooseImage
                  ? null
                  : controller.inputMode == InputMode.image
                  ? onChooseImage
                  : onPrepareRecapture,
              focusNode: controller.inputMode == InputMode.camera
                  ? cameraPrimaryActionFocusNode
                  : null,
              icon: Icon(
                controller.inputMode == InputMode.image
                    ? Icons.image_outlined
                    : Icons.videocam_outlined,
                size: 18,
              ),
              label: Text(
                controller.inputMode == InputMode.image
                    ? AppActionCopy.chooseAnotherImage
                    : AppActionCopy.returnToCapture,
              ),
            ),
          ] else ...[
            if (controller.inputMode == InputMode.image) ...[
              _ReturnToCameraButton(
                onPressed: controller.canChooseImage ? onReturnToCamera : null,
              ),
              const SizedBox(width: AppSpacing.x2),
            ],
            Tooltip(
              message: '${AppActionCopy.chooseImage} (Ctrl+O)',
              child: OutlinedButton.icon(
                onPressed: controller.canChooseImage ? onChooseImage : null,
                icon: const Icon(Icons.image_outlined, size: 18),
                label: Text(
                  controller.inputMode == InputMode.image && hasImage
                      ? AppActionCopy.chooseAnotherImage
                      : AppActionCopy.chooseImage,
                ),
              ),
            ),
            const Spacer(),
            if (controller.processState == ProcessState.error && hasImage)
              FilledButton.icon(
                onPressed: busy ? null : controller.analyze,
                icon: const Icon(Icons.refresh_rounded, size: 18),
                label: const Text(AppActionCopy.reanalyze),
              )
            else if (controller.inputMode == InputMode.image && hasImage)
              AppProgressActionButton(
                onPressed: busy ? null : controller.analyze,
                progressing: analyzing,
                progressLabel: AppActionCopy.analyzing,
                progressAnnouncement: AppActionCopy.analyzingAnnouncement,
                icon: const Icon(Icons.auto_awesome_outlined, size: 18),
                label: AppActionCopy.analyze,
              )
            else if (!controller.isCameraReady ||
                controller.hasActiveCameraIssue)
              FilledButton.icon(
                onPressed: busy || controller.cameraInitializing
                    ? null
                    : controller.reconnectCamera,
                icon: const Icon(Icons.videocam_outlined, size: 18),
                label: const Text(AppActionCopy.reconnect),
              )
            else
              AppProgressActionButton(
                focusNode: cameraPrimaryActionFocusNode,
                onPressed: busy ? null : controller.captureAndAnalyze,
                progressing: false,
                progressLabel: AppActionCopy.capturing,
                progressAnnouncement: AppActionCopy.capturingAnnouncement,
                icon: const Icon(Icons.camera_alt_outlined, size: 18),
                label: AppActionCopy.capture,
              ),
          ],
        ],
      ),
    );
  }
}

class _ReturnToCameraButton extends StatelessWidget {
  const _ReturnToCameraButton({required this.onPressed});

  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: AppActionCopy.returnToCamera,
      child: OutlinedButton.icon(
        onPressed: onPressed,
        icon: const Icon(Icons.videocam_outlined, size: 18),
        label: const Text(AppActionCopy.returnToCamera),
      ),
    );
  }
}

class _RecaptureLogButton extends StatelessWidget {
  const _RecaptureLogButton({required this.controller});

  final ScannerController controller;

  @override
  Widget build(BuildContext context) {
    final state = controller.recaptureLogSaveState;
    final saving = state == RecaptureLogSaveState.saving;
    final saved = state == RecaptureLogSaveState.saved;
    final failed = state == RecaptureLogSaveState.error;
    final label = saved
        ? AppActionCopy.recaptureLogSaved
        : failed
        ? AppActionCopy.retrySave
        : saving
        ? AppActionCopy.saving
        : AppActionCopy.saveRecaptureLog;
    final button = OutlinedButton.icon(
      key: const ValueKey('save-recapture-log'),
      onPressed: saving || saved ? null : controller.saveRecaptureLog,
      icon: saving
          ? AppProgressVisual(
              size: context.appTokens.inlineProgressSize,
              strokeWidth: 2,
              color: AppColors.muted,
            )
          : Icon(
              saved
                  ? Icons.check_circle_outline_rounded
                  : Icons.save_alt_rounded,
              size: 18,
            ),
      label: Text(label),
    );
    if (!saving) {
      return Tooltip(message: label, child: button);
    }
    return Semantics(
      container: true,
      excludeSemantics: true,
      liveRegion: true,
      button: true,
      enabled: false,
      label: AppActionCopy.savingAnnouncement,
      child: button,
    );
  }
}

class _ResultPanel extends StatefulWidget {
  const _ResultPanel({super.key, required this.controller});

  final ScannerController controller;

  @override
  State<_ResultPanel> createState() => _ResultPanelState();
}

class _ResultPanelState extends State<_ResultPanel> {
  final GlobalKey<_DetectionListState> _detectionListKey =
      GlobalKey<_DetectionListState>();
  final FocusNode _firstChoiceFocusNode = FocusNode(
    debugLabel: 'first-product-choice',
  );
  final FocusNode _finalActionFocusNode = FocusNode(
    debugLabel: 'final-confirmation-action',
  );
  final FocusNode _searchActionFocusNode = FocusNode(
    debugLabel: 'product-search-action',
  );

  ScannerController get controller => widget.controller;

  @override
  void dispose() {
    _firstChoiceFocusNode.dispose();
    _finalActionFocusNode.dispose();
    _searchActionFocusNode.dispose();
    super.dispose();
  }

  void _continueKeyboardReview() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final target = controller.allConfirmed
          ? _finalActionFocusNode
          : _firstChoiceFocusNode;
      if (target.canRequestFocus) target.requestFocus();
    });
  }

  void closeSearchFromKeyboard() {
    controller.hideSearch();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || !_searchActionFocusNode.canRequestFocus) return;
      _searchActionFocusNode.requestFocus();
    });
  }

  @override
  Widget build(BuildContext context) {
    final selectedDetection = controller.selectedDetection;
    final showReviewWorkspace =
        controller.processState == ProcessState.reviewing &&
        !controller.isRecapture &&
        !controller.hasActiveCameraIssue &&
        !controller.isCameraCheckActive &&
        selectedDetection != null;
    return ColoredBox(
      color: AppColors.surface,
      child: Column(
        children: [
          if (controller.processState == ProcessState.reviewing &&
              controller.hasResults &&
              controller.allConfirmed)
            AppStateAnnouncement(message: _reviewCompletionAnnouncement()),
          _ResultHeader(controller: controller),
          Expanded(
            child: showReviewWorkspace
                ? _ReviewWorkspace(
                    controller: controller,
                    detection: selectedDetection,
                    detectionListKey: _detectionListKey,
                    firstChoiceFocusNode: _firstChoiceFocusNode,
                    searchActionFocusNode: _searchActionFocusNode,
                    onKeyboardChoiceConfirmed: _continueKeyboardReview,
                    onKeyboardSearchClosed: closeSearchFromKeyboard,
                  )
                : _resultBody(),
          ),
          if ((controller.processState == ProcessState.reviewing ||
                  controller.processState == ProcessState.submitting) &&
              !controller.isRecapture &&
              !controller.hasActiveCameraIssue &&
              !controller.isCameraCheckActive &&
              controller.hasResults)
            _ReviewFooter(
              controller: controller,
              finalActionFocusNode: _finalActionFocusNode,
            ),
        ],
      ),
    );
  }

  String _reviewCompletionAnnouncement() {
    final total = controller.detections.length;
    return '검수 상태. $total개 상품 확인 완료. 최종 확정할 수 있어요.';
  }

  Widget _resultBody() {
    if (controller.processState == ProcessState.capturing) {
      return const _PanelMessage(
        icon: Icons.camera_alt_outlined,
        title: '촬영 중',
        detail: '카메라 촬영이 끝나면 자동으로 분석해요.',
      );
    }
    if (controller.processState == ProcessState.analyzing) {
      return const _PanelMessage(
        icon: Icons.hourglass_top_rounded,
        title: '상품을 찾고 있어요',
        detail: '이미지에서 상품 위치와 종류를 확인하고 있습니다.',
      );
    }
    if (controller.hasActiveCameraIssue) {
      return _PanelMessage(
        icon: Icons.videocam_off_outlined,
        title: controller.cameraIssueTitle,
        detail: controller.cameraMessage!,
        tone: AppColors.attention,
        announce: true,
      );
    }
    if (controller.isCameraCheckActive) {
      return const _PanelMessage(
        icon: Icons.sync_rounded,
        title: '카메라를 준비하고 있어요',
        detail: '연결 상태를 확인하고 있습니다.',
      );
    }
    if (controller.processState == ProcessState.error) {
      return _PanelMessage(
        icon: controller.errorRecovery == ScannerErrorRecovery.replaceInput
            ? Icons.broken_image_outlined
            : Icons.cloud_off_outlined,
        title: controller.errorRecovery == ScannerErrorRecovery.replaceInput
            ? '이미지를 분석할 수 없어요'
            : '분석하지 못했어요',
        detail: controller.errorMessage ?? '잠시 후 다시 분석해 주세요.',
        tone: AppColors.error,
        announce: true,
      );
    }
    if (controller.isRecapture) {
      final saveError = controller.recaptureLogError;
      return _PanelMessage(
        icon: Icons.center_focus_weak_rounded,
        title: controller.recaptureTitle,
        detail: saveError == null
            ? controller.recaptureDetail
            : '${controller.recaptureDetail}\n$saveError',
        tone: AppColors.attention,
        announce:
            saveError != null ||
            controller.recaptureLogSaveState == RecaptureLogSaveState.idle,
      );
    }
    if (!controller.hasResults) {
      if (controller.imageBytes != null) {
        return const _PanelMessage(
          icon: Icons.auto_awesome_outlined,
          title: '이미지 분석 준비가 됐어요',
          detail: '분석하기를 누르면 이미지에서 상품을 찾아요.',
        );
      }
      if (controller.inputMode == InputMode.image) {
        return const _PanelMessage(
          icon: Icons.image_outlined,
          title: '다음 이미지를 선택해 주세요',
          detail: '이미지를 선택하면 분석 준비 상태로 이동해요.',
        );
      }
      if (controller.isCameraReady) {
        return const _PanelMessage(
          icon: Icons.photo_camera_outlined,
          title: '상품을 촬영해 주세요',
          detail: '화면 안에 상품이 모두 보이면 촬영하기를 눌러주세요.',
        );
      }
      return const _PanelMessage(
        icon: Icons.videocam_off_outlined,
        title: '카메라를 연결해 주세요',
        detail: '다시 연결하거나 이미지 파일을 선택할 수 있어요.',
      );
    }
    return _DetectionList(controller: controller);
  }
}

class _ReviewWorkspace extends StatelessWidget {
  const _ReviewWorkspace({
    required this.controller,
    required this.detection,
    required this.detectionListKey,
    required this.firstChoiceFocusNode,
    required this.searchActionFocusNode,
    required this.onKeyboardChoiceConfirmed,
    required this.onKeyboardSearchClosed,
  });

  final ScannerController controller;
  final ReviewDetection detection;
  final GlobalKey<_DetectionListState> detectionListKey;
  final FocusNode firstChoiceFocusNode;
  final FocusNode searchActionFocusNode;
  final VoidCallback onKeyboardChoiceConfirmed;
  final VoidCallback onKeyboardSearchClosed;

  bool get _usesCandidatePicker =>
      controller.searchItemId != detection.source.itemId &&
      detection.source.top3.isNotEmpty;

  @override
  Widget build(BuildContext context) {
    final tokens = context.appTokens;
    return LayoutBuilder(
      builder: (context, constraints) {
        final dividerCount = (controller.detections.length - 1).clamp(
          0,
          controller.detections.length,
        );
        final desiredListHeight =
            controller.detections.length * tokens.rowHeight + dividerCount;
        final maxListHeight =
            (constraints.maxHeight - tokens.reviewInspectorReservedHeight)
                .clamp(tokens.rowHeight, constraints.maxHeight)
                .toDouble();
        final listHeight = desiredListHeight
            .clamp(tokens.rowHeight, maxListHeight)
            .toDouble();

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SizedBox(
              key: const ValueKey('review-object-list-frame'),
              height: listHeight,
              child: _DetectionList(
                key: detectionListKey,
                controller: controller,
                onExitForward: _usesCandidatePicker
                    ? firstChoiceFocusNode.requestFocus
                    : null,
              ),
            ),
            Flexible(
              fit: FlexFit.loose,
              child: _ReviewInspector(
                controller: controller,
                detection: detection,
                firstChoiceFocusNode: firstChoiceFocusNode,
                searchActionFocusNode: searchActionFocusNode,
                onCandidateExitBackward: () =>
                    detectionListKey.currentState?.requestSelectedRowFocus(),
                onKeyboardChoiceConfirmed: onKeyboardChoiceConfirmed,
                onKeyboardSearchClosed: onKeyboardSearchClosed,
              ),
            ),
          ],
        );
      },
    );
  }
}

class _ResultHeader extends StatelessWidget {
  const _ResultHeader({required this.controller});

  final ScannerController controller;

  @override
  Widget build(BuildContext context) {
    final title = controller.hasActiveCameraIssue
        ? '카메라 확인 필요'
        : controller.isCameraCheckActive
        ? '카메라 확인 중'
        : controller.hasResults
        ? controller.allConfirmed
              ? '검수 완료'
              : '상품 검수'
        : controller.isRecapture
        ? '재촬영 필요'
        : controller.processState == ProcessState.error
        ? '분석 오류'
        : controller.processState == ProcessState.capturing
        ? '촬영 중'
        : controller.processState == ProcessState.analyzing
        ? '분석 중'
        : controller.imageBytes != null
        ? '분석 준비'
        : controller.inputMode == InputMode.image
        ? '이미지 선택'
        : controller.isCameraReady
        ? '촬영 준비'
        : '입력 준비';
    final response = controller.response;
    final analysisTime = response == null || response.status == ScanStatus.error
        ? null
        : '분석 ${response.processingTimeMs.toStringAsFixed(1)} ms';
    final subtitle = controller.hasResults
        ? '${controller.confirmedCount}/${controller.detections.length}개 확인'
              '${analysisTime == null ? '' : ' · $analysisTime'}'
        : controller.isRecapture
        ? analysisTime
        : null;
    return AppPanelHeader(
      key: const ValueKey('scan-result-header'),
      title: title,
      subtitle: subtitle,
      trailing:
          controller.selectedIndex >= 0 && controller.detections.length > 1
          ? AppStepNavigator(
              current: controller.selectedIndex + 1,
              total: controller.detections.length,
              semanticUnit: '상품',
              previousTooltip: '이전 상품 (↑)',
              nextTooltip: '다음 상품 (↓)',
              onPrevious: controller.isBusy
                  ? null
                  : controller.selectPreviousDetection,
              onNext: controller.isBusy ? null : controller.selectNextDetection,
            )
          : null,
    );
  }
}

class _PanelMessage extends StatelessWidget {
  const _PanelMessage({
    required this.icon,
    required this.title,
    required this.detail,
    this.tone = AppColors.muted,
    this.announce = false,
  });

  final IconData icon;
  final String title;
  final String detail;
  final Color tone;
  final bool announce;

  @override
  Widget build(BuildContext context) {
    return AppEmptyState(
      icon: icon,
      title: title,
      detail: detail,
      tone: tone,
      announcement: announce ? '$title. $detail' : null,
    );
  }
}

class _DetectionList extends StatefulWidget {
  const _DetectionList({
    super.key,
    required this.controller,
    this.onExitForward,
  });

  final ScannerController controller;
  final VoidCallback? onExitForward;

  @override
  State<_DetectionList> createState() => _DetectionListState();
}

class _DetectionListState extends State<_DetectionList> {
  final ScrollController _scrollController = ScrollController();
  final Map<String, FocusNode> _rowFocusNodes = <String, FocusNode>{};
  String? _lastSelectedItemId;

  @override
  void initState() {
    super.initState();
    _lastSelectedItemId = widget.controller.selectedItemId;
    _revealSelectedRow();
  }

  @override
  void didUpdateWidget(covariant _DetectionList oldWidget) {
    super.didUpdateWidget(oldWidget);
    _disposeUnusedRowFocusNodes();
    _syncRowTraversalState();
    final selectedItemId = widget.controller.selectedItemId;
    if (_lastSelectedItemId == selectedItemId) return;
    _lastSelectedItemId = selectedItemId;
    _revealSelectedRow();
  }

  FocusNode _rowFocusNode(String itemId) {
    final focusNode = _rowFocusNodes.putIfAbsent(
      itemId,
      () => FocusNode(debugLabel: 'detection-row-$itemId'),
    );
    focusNode.skipTraversal =
        widget.controller.isBusy || itemId != widget.controller.selectedItemId;
    return focusNode;
  }

  void _syncRowTraversalState() {
    final selectedItemId = widget.controller.selectedItemId;
    for (final entry in _rowFocusNodes.entries) {
      entry.value.skipTraversal =
          widget.controller.isBusy || entry.key != selectedItemId;
    }
  }

  void _disposeUnusedRowFocusNodes() {
    final activeItemIds = widget.controller.detections
        .map((detection) => detection.source.itemId)
        .toSet();
    final removed = <FocusNode>[];
    _rowFocusNodes.removeWhere((itemId, focusNode) {
      if (activeItemIds.contains(itemId)) return false;
      removed.add(focusNode);
      return true;
    });
    if (removed.isEmpty) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      for (final focusNode in removed) {
        focusNode.dispose();
      }
    });
  }

  KeyEventResult _handleRowNavigation(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent && event is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    final rowFocused = _rowFocusNodes.values.any(
      (focusNode) => focusNode.hasFocus,
    );
    if (!rowFocused || widget.controller.isBusy) {
      return KeyEventResult.ignored;
    }
    if (event.logicalKey == LogicalKeyboardKey.tab &&
        !HardwareKeyboard.instance.isShiftPressed &&
        widget.onExitForward != null) {
      widget.onExitForward!();
      return KeyEventResult.handled;
    }
    if (event.logicalKey == LogicalKeyboardKey.arrowUp) {
      widget.controller.selectPreviousDetection();
    } else if (event.logicalKey == LogicalKeyboardKey.arrowDown) {
      widget.controller.selectNextDetection();
    } else {
      return KeyEventResult.ignored;
    }
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final selectedItemId = widget.controller.selectedItemId;
      if (selectedItemId == null) return;
      _rowFocusNode(selectedItemId).requestFocus();
    });
    return KeyEventResult.handled;
  }

  void requestSelectedRowFocus() {
    final selectedItemId = widget.controller.selectedItemId;
    if (selectedItemId == null) return;
    _rowFocusNode(selectedItemId).requestFocus();
    _revealSelectedRow();
  }

  void _revealSelectedRow() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || !_scrollController.hasClients) return;
      final selectedItemId = widget.controller.selectedItemId;
      final index = widget.controller.detections.indexWhere(
        (detection) => detection.source.itemId == selectedItemId,
      );
      if (index < 0) return;

      final position = _scrollController.position;
      final rowExtent = context.appTokens.rowHeight + 1;
      final centeredOffset =
          index * rowExtent -
          (position.viewportDimension - context.appTokens.rowHeight) / 2;
      final target = centeredOffset.clamp(
        position.minScrollExtent,
        position.maxScrollExtent,
      );
      if ((position.pixels - target).abs() < 1) return;

      if (MediaQuery.disableAnimationsOf(context)) {
        _scrollController.jumpTo(target);
      } else {
        _scrollController.animateTo(
          target,
          duration: context.appTokens.motionStandard,
          curve: AppMotion.interactionCurve,
        );
      }
    });
  }

  @override
  void dispose() {
    _scrollController.dispose();
    for (final focusNode in _rowFocusNodes.values) {
      focusNode.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Focus(
      canRequestFocus: false,
      skipTraversal: true,
      onKeyEvent: _handleRowNavigation,
      child: ListView.separated(
        key: const ValueKey('detection-list'),
        controller: _scrollController,
        padding: EdgeInsets.zero,
        itemCount: widget.controller.detections.length,
        separatorBuilder: (_, _) => const Divider(height: 1),
        itemBuilder: (context, index) {
          final detection = widget.controller.detections[index];
          return _DetectionRow(
            key: ValueKey('detection-row-${detection.source.itemId}'),
            controller: widget.controller,
            detection: detection,
            index: index + 1,
            selected:
                widget.controller.selectedItemId == detection.source.itemId,
            focusNode: _rowFocusNode(detection.source.itemId),
          );
        },
      ),
    );
  }
}

class _DetectionRow extends StatelessWidget {
  const _DetectionRow({
    super.key,
    required this.controller,
    required this.detection,
    required this.index,
    required this.selected,
    required this.focusNode,
  });

  final ScannerController controller;
  final ReviewDetection detection;
  final int index;
  final bool selected;
  final FocusNode focusNode;

  @override
  Widget build(BuildContext context) {
    final needsReview = !detection.isConfirmed;
    final tone = needsReview ? AppColors.attention : AppColors.success;
    final confidence =
        '${(detection.source.confidence * 100).toStringAsFixed(0)}%';
    final semanticLabel = needsReview
        ? '$index번 상품, 확인 필요'
        : '$index번 상품, ${detection.finalProduct!.displayName}, 확정, 신뢰도 $confidence';
    return AppSelectableSurface(
      selected: selected,
      selectedBackgroundColor: AppColors.surface,
      selectedBorder: Border.all(color: tone, width: 2),
      enabled: !controller.isBusy,
      inMutuallyExclusiveGroup: true,
      focusNode: focusNode,
      onTap: () => controller.selectDetection(detection.source.itemId),
      semanticLabel: semanticLabel,
      padding: const EdgeInsets.symmetric(
        horizontal: AppSpacing.x4,
        vertical: AppSpacing.x2,
      ),
      child: Row(
        children: [
          SizedBox(
            width: context.appTokens.compactVisualSize,
            child: Text(
              index.toString().padLeft(2, '0'),
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
          Icon(
            needsReview
                ? Icons.help_outline_rounded
                : Icons.check_circle_outline_rounded,
            size: 19,
            color: tone,
          ),
          const SizedBox(width: AppSpacing.x3),
          Expanded(
            child: Text(
              needsReview ? '상품 확인이 필요해요' : detection.finalProduct!.displayName,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(fontWeight: AppTypography.bold),
            ),
          ),
          Text(
            needsReview ? '확인 필요' : confidence,
            maxLines: 1,
            softWrap: false,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: needsReview ? tone : AppColors.muted,
              fontWeight: needsReview
                  ? AppTypography.bold
                  : AppTypography.medium,
            ),
          ),
          const SizedBox(width: AppSpacing.x2),
          Icon(
            selected
                ? Icons.arrow_forward_rounded
                : Icons.chevron_right_rounded,
            color: selected ? tone : AppColors.subtle,
            size: 18,
          ),
        ],
      ),
    );
  }
}

class _ReviewInspector extends StatefulWidget {
  const _ReviewInspector({
    required this.controller,
    required this.detection,
    required this.firstChoiceFocusNode,
    required this.searchActionFocusNode,
    required this.onCandidateExitBackward,
    required this.onKeyboardChoiceConfirmed,
    required this.onKeyboardSearchClosed,
  });

  final ScannerController controller;
  final ReviewDetection detection;
  final FocusNode firstChoiceFocusNode;
  final FocusNode searchActionFocusNode;
  final VoidCallback onCandidateExitBackward;
  final VoidCallback onKeyboardChoiceConfirmed;
  final VoidCallback onKeyboardSearchClosed;

  @override
  State<_ReviewInspector> createState() => _ReviewInspectorState();
}

class _ReviewInspectorState extends State<_ReviewInspector> {
  final ScrollController _scrollController = ScrollController();
  late String _lastContentSignature;

  String get _contentSignature =>
      '${widget.detection.source.itemId}|'
      '${widget.controller.searchItemId}|${widget.controller.searchQuery}';

  @override
  void initState() {
    super.initState();
    _lastContentSignature = _contentSignature;
  }

  @override
  void didUpdateWidget(covariant _ReviewInspector oldWidget) {
    super.didUpdateWidget(oldWidget);
    final contentSignature = _contentSignature;
    if (_lastContentSignature == contentSignature) return;
    _lastContentSignature = contentSignature;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted && _scrollController.hasClients) {
        _scrollController.jumpTo(0);
      }
    });
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      key: const ValueKey('review-inspector'),
      constraints: BoxConstraints(
        maxHeight: context.appTokens.reviewInspectorMaxHeight,
      ),
      decoration: const BoxDecoration(
        color: AppColors.elevated,
        border: Border(top: BorderSide(color: AppColors.divider)),
      ),
      child: Scrollbar(
        controller: _scrollController,
        thumbVisibility: true,
        child: SingleChildScrollView(
          controller: _scrollController,
          primary: false,
          padding: const EdgeInsets.fromLTRB(
            AppSpacing.x4,
            AppSpacing.x3,
            AppSpacing.x4,
            AppSpacing.x3,
          ),
          child:
              widget.controller.searchItemId == widget.detection.source.itemId
              ? _SearchProducts(
                  controller: widget.controller,
                  detection: widget.detection,
                  firstChoiceFocusNode: widget.firstChoiceFocusNode,
                  onKeyboardChoiceConfirmed: widget.onKeyboardChoiceConfirmed,
                  onKeyboardExit: widget.onKeyboardSearchClosed,
                )
              : widget.detection.source.top3.isNotEmpty
              ? _CandidatePicker(
                  controller: widget.controller,
                  detection: widget.detection,
                  firstChoiceFocusNode: widget.firstChoiceFocusNode,
                  searchActionFocusNode: widget.searchActionFocusNode,
                  onExitBackward: widget.onCandidateExitBackward,
                  onKeyboardChoiceConfirmed: widget.onKeyboardChoiceConfirmed,
                )
              : Row(
                  children: [
                    Expanded(
                      child: Text(
                        widget.detection.finalProduct?.displayName ?? '',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ),
                    TextButton(
                      focusNode: widget.searchActionFocusNode,
                      onPressed: widget.controller.isBusy
                          ? null
                          : () => widget.controller.showSearch(
                              widget.detection.source.itemId,
                            ),
                      child: const Text('상품 변경'),
                    ),
                  ],
                ),
        ),
      ),
    );
  }
}

class _CandidatePicker extends StatelessWidget {
  const _CandidatePicker({
    required this.controller,
    required this.detection,
    required this.firstChoiceFocusNode,
    required this.searchActionFocusNode,
    required this.onExitBackward,
    required this.onKeyboardChoiceConfirmed,
  });

  final ScannerController controller;
  final ReviewDetection detection;
  final FocusNode firstChoiceFocusNode;
  final FocusNode searchActionFocusNode;
  final VoidCallback onExitBackward;
  final VoidCallback onKeyboardChoiceConfirmed;

  @override
  Widget build(BuildContext context) {
    final itemNumber =
        controller.detections.indexWhere(
          (item) => item.source.itemId == detection.source.itemId,
        ) +
        1;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Semantics(
          header: true,
          liveRegion: true,
          child: Text(
            detection.isConfirmed
                ? '$itemNumber번 상품을 변경할까요?'
                : '$itemNumber번 상품을 확인해 주세요',
            style: Theme.of(context).textTheme.titleMedium,
          ),
        ),
        const SizedBox(height: AppSpacing.x1),
        Text(
          detection.isConfirmed ? '선택하면 상품이 바로 변경돼요.' : '선택하면 다음 확인 항목으로 이동해요.',
          style: Theme.of(
            context,
          ).textTheme.bodySmall?.copyWith(color: AppColors.muted),
        ),
        const SizedBox(height: AppSpacing.x3),
        _ProductChoiceGroup(
          groupId: 'candidate-${detection.source.itemId}',
          semanticLabel: '$itemNumber번 상품 후보',
          entries: [
            for (final rawCandidate in detection.source.top3)
              _ProductChoiceEntry(
                id: rawCandidate.classId,
                choiceKey: ValueKey(
                  'candidate-${detection.source.itemId}-${rawCandidate.classId}',
                ),
                product: controller.localizeCandidate(rawCandidate),
                selected:
                    detection.finalProduct?.classId == rawCandidate.classId,
                onTap: () => controller.confirmCandidate(
                  detection.source.itemId,
                  rawCandidate,
                ),
              ),
          ],
          enabled: !controller.isBusy,
          entryFocusNode: firstChoiceFocusNode,
          onExitBackward: onExitBackward,
          onKeyboardConfirmed: onKeyboardChoiceConfirmed,
        ),
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton(
            focusNode: searchActionFocusNode,
            onPressed: controller.isBusy
                ? null
                : () => controller.showSearch(detection.source.itemId),
            child: const Text('다른 상품 검색'),
          ),
        ),
      ],
    );
  }
}

@immutable
class _ProductChoiceEntry {
  const _ProductChoiceEntry({
    required this.id,
    required this.choiceKey,
    required this.product,
    required this.selected,
    required this.onTap,
  });

  final String id;
  final Key choiceKey;
  final Product product;
  final bool selected;
  final VoidCallback onTap;
}

class _ProductChoiceGroup extends StatefulWidget {
  const _ProductChoiceGroup({
    super.key,
    required this.groupId,
    required this.semanticLabel,
    required this.entries,
    required this.enabled,
    required this.entryFocusNode,
    required this.onKeyboardConfirmed,
    this.onExitBackward,
  });

  final String groupId;
  final String semanticLabel;
  final List<_ProductChoiceEntry> entries;
  final bool enabled;
  final FocusNode entryFocusNode;
  final VoidCallback onKeyboardConfirmed;
  final VoidCallback? onExitBackward;

  @override
  State<_ProductChoiceGroup> createState() => _ProductChoiceGroupState();
}

class _ProductChoiceGroupState extends State<_ProductChoiceGroup> {
  final Map<String, FocusNode> _ownedFocusNodes = <String, FocusNode>{};
  String? _rovingId;

  String? get _initialRovingId =>
      widget.entries.where((entry) => entry.selected).firstOrNull?.id ??
      widget.entries.firstOrNull?.id;

  @override
  void initState() {
    super.initState();
    _rovingId = _initialRovingId;
    _syncTraversalState();
  }

  @override
  void didUpdateWidget(covariant _ProductChoiceGroup oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.groupId != widget.groupId ||
        !widget.entries.any((entry) => entry.id == _rovingId)) {
      _rovingId = _initialRovingId;
    }
    _disposeUnusedFocusNodes();
    _syncTraversalState();
  }

  FocusNode _focusNodeFor(int index, String id) {
    if (index == 0) return widget.entryFocusNode;
    return _ownedFocusNodes.putIfAbsent(
      id,
      () => FocusNode(debugLabel: '${widget.groupId}-$id'),
    );
  }

  void _syncTraversalState() {
    for (final entry in widget.entries.indexed) {
      _focusNodeFor(entry.$1, entry.$2.id).skipTraversal =
          !widget.enabled || entry.$2.id != _rovingId;
    }
  }

  void _disposeUnusedFocusNodes() {
    final ownedIds = widget.entries.skip(1).map((entry) => entry.id).toSet();
    final removed = <FocusNode>[];
    _ownedFocusNodes.removeWhere((id, focusNode) {
      if (ownedIds.contains(id)) return false;
      removed.add(focusNode);
      return true;
    });
    if (removed.isEmpty) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      for (final focusNode in removed) {
        focusNode.dispose();
      }
    });
  }

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent && event is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    final focusedIndex = widget.entries.indexed
        .where((entry) => _focusNodeFor(entry.$1, entry.$2.id).hasFocus)
        .map((entry) => entry.$1)
        .firstOrNull;
    if (focusedIndex == null || widget.entries.isEmpty || !widget.enabled) {
      return KeyEventResult.ignored;
    }
    if (event.logicalKey == LogicalKeyboardKey.tab &&
        HardwareKeyboard.instance.isShiftPressed &&
        widget.onExitBackward != null) {
      widget.onExitBackward!();
      return KeyEventResult.handled;
    }
    final offset = switch (event.logicalKey) {
      LogicalKeyboardKey.arrowLeft || LogicalKeyboardKey.arrowUp => -1,
      LogicalKeyboardKey.arrowRight || LogicalKeyboardKey.arrowDown => 1,
      _ => 0,
    };
    if (offset == 0) return KeyEventResult.ignored;
    final nextIndex = (focusedIndex + offset) % widget.entries.length;
    final next = widget.entries[nextIndex];
    setState(() => _rovingId = next.id);
    _syncTraversalState();
    _requestFocusAndReveal(nextIndex, offset: offset);
    return KeyEventResult.handled;
  }

  void requestRovingFocus() {
    final rovingIndex = widget.entries.indexWhere(
      (entry) => entry.id == _rovingId,
    );
    if (rovingIndex < 0 || !widget.enabled) return;
    _requestFocusAndReveal(rovingIndex, offset: 1);
  }

  void _requestFocusAndReveal(int index, {required int offset}) {
    final entry = widget.entries[index];
    final nextFocusNode = _focusNodeFor(index, entry.id);
    nextFocusNode.requestFocus();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final targetContext = nextFocusNode.context;
      if (targetContext == null) return;
      Scrollable.ensureVisible(
        targetContext,
        duration: MediaQuery.disableAnimationsOf(context)
            ? Duration.zero
            : context.appTokens.motionFast,
        curve: AppMotion.interactionCurve,
        alignmentPolicy: offset > 0
            ? ScrollPositionAlignmentPolicy.keepVisibleAtEnd
            : ScrollPositionAlignmentPolicy.keepVisibleAtStart,
      );
    });
  }

  @override
  void dispose() {
    for (final focusNode in _ownedFocusNodes.values) {
      focusNode.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Semantics(
      container: true,
      explicitChildNodes: true,
      label: widget.semanticLabel,
      child: Focus(
        canRequestFocus: false,
        skipTraversal: true,
        onKeyEvent: _handleKeyEvent,
        child: Column(
          children: [
            for (final entry in widget.entries.indexed)
              _ProductChoice(
                key: entry.$2.choiceKey,
                product: entry.$2.product,
                selected: entry.$2.selected,
                enabled: widget.enabled,
                focusNode: _focusNodeFor(entry.$1, entry.$2.id),
                onTap: entry.$2.onTap,
                onKeyboardConfirmed: widget.onKeyboardConfirmed,
              ),
          ],
        ),
      ),
    );
  }
}

class _ProductChoice extends StatelessWidget {
  const _ProductChoice({
    super.key,
    required this.product,
    required this.selected,
    required this.onTap,
    this.enabled = true,
    required this.focusNode,
    this.onKeyboardConfirmed,
  });

  final Product product;
  final bool selected;
  final VoidCallback onTap;
  final bool enabled;
  final FocusNode focusNode;
  final VoidCallback? onKeyboardConfirmed;

  @override
  Widget build(BuildContext context) {
    final confidence = product is Candidate
        ? '${((product as Candidate).confidence * 100).toStringAsFixed(0)}%'
        : null;
    return AppSelectableSurface(
      selected: selected,
      enabled: enabled,
      inMutuallyExclusiveGroup: true,
      focusNode: focusNode,
      onTap: onTap,
      onKeyboardTap: onKeyboardConfirmed,
      semanticLabel: confidence == null
          ? product.displayName
          : '${product.displayName}, 신뢰도 $confidence',
      minHeight: context.appTokens.rowHeight,
      margin: const EdgeInsets.only(bottom: AppSpacing.x2),
      padding: const EdgeInsets.symmetric(horizontal: AppSpacing.x3),
      borderRadius: BorderRadius.circular(context.appTokens.selectionRadius),
      restingBorder: Border.all(color: AppColors.divider),
      child: Row(
        children: [
          Icon(
            selected
                ? Icons.radio_button_checked_rounded
                : Icons.radio_button_unchecked_rounded,
            color: selected ? AppColors.primary : AppColors.subtle,
            size: 19,
          ),
          const SizedBox(width: AppSpacing.x2),
          Expanded(
            child: Text(
              product.displayName,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                fontWeight: selected
                    ? AppTypography.bold
                    : AppTypography.semibold,
              ),
            ),
          ),
          Text(product.classId, style: Theme.of(context).textTheme.bodySmall),
          if (confidence != null) ...[
            const SizedBox(width: AppSpacing.x3),
            SizedBox(
              width: AppSpacing.x8 + AppSpacing.x4,
              child: Text(
                confidence,
                textAlign: TextAlign.right,
                maxLines: 1,
                softWrap: false,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  fontFeatures: const [FontFeature.tabularFigures()],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _SearchProducts extends StatefulWidget {
  const _SearchProducts({
    required this.controller,
    required this.detection,
    required this.firstChoiceFocusNode,
    required this.onKeyboardChoiceConfirmed,
    required this.onKeyboardExit,
  });

  final ScannerController controller;
  final ReviewDetection detection;
  final FocusNode firstChoiceFocusNode;
  final VoidCallback onKeyboardChoiceConfirmed;
  final VoidCallback onKeyboardExit;

  @override
  State<_SearchProducts> createState() => _SearchProductsState();
}

class _SearchProductsState extends State<_SearchProducts> {
  final GlobalKey<_ProductChoiceGroupState> _choiceGroupKey =
      GlobalKey<_ProductChoiceGroupState>();
  final FocusNode _backFocusNode = FocusNode(
    debugLabel: 'close-product-search',
  );

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent && event is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    if (event.logicalKey == LogicalKeyboardKey.escape ||
        (_backFocusNode.hasFocus &&
            (event.logicalKey == LogicalKeyboardKey.enter ||
                event.logicalKey == LogicalKeyboardKey.space))) {
      widget.onKeyboardExit();
      return KeyEventResult.handled;
    }
    if (event.logicalKey != LogicalKeyboardKey.tab ||
        HardwareKeyboard.instance.isShiftPressed ||
        !_isEditingText()) {
      return KeyEventResult.ignored;
    }
    final choiceGroup = _choiceGroupKey.currentState;
    if (choiceGroup == null) return KeyEventResult.ignored;
    choiceGroup.requestRovingFocus();
    return KeyEventResult.handled;
  }

  bool _isEditingText() {
    final context = FocusManager.instance.primaryFocus?.context;
    if (context == null) return false;
    return context.widget is EditableText ||
        context.findAncestorWidgetOfExactType<EditableText>() != null;
  }

  @override
  void dispose() {
    _backFocusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final detection = widget.detection;
    final itemNumber =
        controller.detections.indexWhere(
          (item) => item.source.itemId == detection.source.itemId,
        ) +
        1;
    final firstChoiceFocusNode = widget.firstChoiceFocusNode;
    final onKeyboardChoiceConfirmed = widget.onKeyboardChoiceConfirmed;
    final results = controller.searchResults;
    final visibleResults = results.take(8).toList(growable: false);
    final resultCountLabel = results.length > visibleResults.length
        ? '검색 결과 ${results.length}개 중 상위 ${visibleResults.length}개'
        : '검색 결과 ${visibleResults.length}개';
    return Focus(
      canRequestFocus: false,
      skipTraversal: true,
      onKeyEvent: _handleKeyEvent,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Semantics(
            header: true,
            child: Text(
              detection.isConfirmed
                  ? '$itemNumber번 상품 변경'
                  : '$itemNumber번 상품 검색',
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ),
          if (detection.finalProduct case final currentProduct?) ...[
            const SizedBox(height: AppSpacing.x1),
            Text(
              '현재 상품  ${currentProduct.displayName} · ${currentProduct.classId}',
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(color: AppColors.muted),
            ),
          ],
          const SizedBox(height: AppSpacing.x3),
          Row(
            children: [
              AppIconActionButton(
                semanticLabel: detection.source.top3.isNotEmpty
                    ? '후보로 돌아가기'
                    : '검색 닫기',
                tooltip: detection.source.top3.isNotEmpty
                    ? '후보로 돌아가기'
                    : '검색 닫기',
                focusNode: _backFocusNode,
                onPressed: controller.hideSearch,
                icon: const Icon(Icons.arrow_back_rounded, size: 17),
              ),
              Expanded(
                child: TextField(
                  autofocus: true,
                  enabled: !controller.isBusy,
                  onChanged: controller.updateSearch,
                  decoration: const InputDecoration(
                    hintText: '상품명 또는 class ID',
                    prefixIcon: Icon(Icons.search_rounded, size: 18),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: AppSpacing.x2),
          if (results.isEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: AppSpacing.x4),
              child: Semantics(
                container: true,
                liveRegion: true,
                label: '일치하는 상품이 없어요. 검색어를 바꾸거나 class ID를 입력해 보세요.',
                child: ExcludeSemantics(
                  child: Column(
                    children: [
                      const Icon(
                        Icons.search_off_rounded,
                        size: 24,
                        color: AppColors.subtle,
                      ),
                      const SizedBox(height: AppSpacing.x2),
                      Text(
                        '일치하는 상품이 없어요',
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          fontWeight: AppTypography.bold,
                        ),
                      ),
                      const SizedBox(height: AppSpacing.x1),
                      Text(
                        '검색어를 바꾸거나 class ID를 입력해 보세요.',
                        textAlign: TextAlign.center,
                        style: Theme.of(
                          context,
                        ).textTheme.bodySmall?.copyWith(color: AppColors.muted),
                      ),
                    ],
                  ),
                ),
              ),
            )
          else
            Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Semantics(
                  container: true,
                  header: true,
                  label: resultCountLabel,
                  child: ExcludeSemantics(
                    child: SizedBox(
                      key: const ValueKey('product-search-result-count'),
                      height: AppSpacing.x6,
                      child: Row(
                        children: [
                          Expanded(
                            child: Text(
                              '검색 결과',
                              style: Theme.of(context).textTheme.bodySmall
                                  ?.copyWith(fontWeight: AppTypography.bold),
                            ),
                          ),
                          Text(
                            results.length > visibleResults.length
                                ? '상위 ${visibleResults.length} / ${results.length}개'
                                : '${visibleResults.length}개',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: AppSpacing.x1),
                _ProductChoiceGroup(
                  key: _choiceGroupKey,
                  groupId:
                      'search-${detection.source.itemId}-${controller.searchQuery}',
                  semanticLabel: '상품 검색 결과',
                  entries: [
                    for (final product in visibleResults)
                      _ProductChoiceEntry(
                        id: product.classId,
                        choiceKey: ValueKey(
                          'search-product-${product.classId}',
                        ),
                        product: product,
                        selected:
                            detection.finalProduct?.classId == product.classId,
                        onTap: () => controller.confirmSearchProduct(
                          detection.source.itemId,
                          product,
                        ),
                      ),
                  ],
                  enabled: !controller.isBusy,
                  entryFocusNode: firstChoiceFocusNode,
                  onKeyboardConfirmed: onKeyboardChoiceConfirmed,
                ),
              ],
            ),
        ],
      ),
    );
  }
}

class _ReviewFooter extends StatelessWidget {
  const _ReviewFooter({
    required this.controller,
    required this.finalActionFocusNode,
  });

  final ScannerController controller;
  final FocusNode finalActionFocusNode;

  @override
  Widget build(BuildContext context) {
    final totalCount = controller.detections.length;
    final remainingCount = totalCount - controller.confirmedCount;
    final submitting = controller.processState == ProcessState.submitting;
    final feedbackState = controller.missedDetectionLogSaveState;
    final feedbackSaving = feedbackState == MissedDetectionLogSaveState.saving;
    final feedbackSaved = feedbackState == MissedDetectionLogSaveState.saved;
    final feedbackFailed = feedbackState == MissedDetectionLogSaveState.error;
    final visibleError =
        controller.errorMessage ?? controller.missedDetectionLogError;
    final feedbackLabel = feedbackSaved
        ? AppActionCopy.missedDetectionLogSaved
        : feedbackFailed
        ? AppActionCopy.retrySave
        : feedbackSaving
        ? AppActionCopy.saving
        : AppActionCopy.saveMissedDetectionLog;
    final feedbackIcon = feedbackSaving
        ? AppProgressVisual(
            size: context.appTokens.inlineProgressSize,
            strokeWidth: 2,
            color: AppColors.muted,
          )
        : Icon(
            feedbackSaved
                ? Icons.check_circle_outline_rounded
                : Icons.report_problem_outlined,
            size: 18,
          );
    final compactFeedbackAction =
        MediaQuery.textScalerOf(context).scale(1) > 1.25;
    return AppActionBar(
      child: Row(
        children: [
          if (visibleError != null) ...[
            const Icon(
              Icons.error_outline_rounded,
              color: AppColors.error,
              size: 17,
            ),
            const SizedBox(width: AppSpacing.x2),
          ],
          Expanded(
            child: Semantics(
              container: visibleError != null,
              excludeSemantics: visibleError != null,
              liveRegion: visibleError != null,
              label: visibleError,
              child: Text(
                visibleError ??
                    (controller.allConfirmed
                        ? '$totalCount개 상품 확인 완료'
                        : '${controller.confirmedCount} / $totalCount 상품 확인 완료'),
                maxLines: visibleError == null ? 1 : 2,
                overflow: TextOverflow.ellipsis,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  fontWeight: AppTypography.bold,
                  color: visibleError == null ? AppColors.ink : AppColors.error,
                ),
              ),
            ),
          ),
          const SizedBox(width: AppSpacing.x3),
          if (compactFeedbackAction)
            IconButton.outlined(
              key: const ValueKey('save-missed-detection-log'),
              tooltip: feedbackLabel,
              onPressed: controller.canSaveMissedDetectionLog
                  ? controller.saveMissedDetectionLog
                  : null,
              icon: feedbackIcon,
            )
          else
            OutlinedButton.icon(
              key: const ValueKey('save-missed-detection-log'),
              onPressed: controller.canSaveMissedDetectionLog
                  ? controller.saveMissedDetectionLog
                  : null,
              icon: feedbackIcon,
              label: Text(feedbackLabel),
            ),
          const SizedBox(width: AppSpacing.x2),
          AppProgressActionButton(
            focusNode: finalActionFocusNode,
            onPressed: controller.allConfirmed && !submitting
                ? controller.submit
                : null,
            progressing: submitting,
            progressLabel: AppActionCopy.saving,
            progressAnnouncement: AppActionCopy.savingAnnouncement,
            icon: Icon(
              controller.errorMessage != null
                  ? Icons.refresh_rounded
                  : controller.allConfirmed
                  ? Icons.check_rounded
                  : Icons.lock_outline_rounded,
              size: 19,
            ),
            label: controller.errorMessage != null
                ? AppActionCopy.retrySave
                : controller.allConfirmed
                ? '$totalCount개 상품 최종 확정'
                : '$remainingCount개 상품 확인 필요',
          ),
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
    return AppToast(
      message: message,
      icon: Icons.check_circle_rounded,
      iconColor: AppColors.primary,
    );
  }
}
