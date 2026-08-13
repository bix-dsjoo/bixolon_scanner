import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../activity/presentation/activity_screen.dart';
import '../../../core/design_system/components.dart';
import '../../../core/design_system/copy.dart';
import '../../../core/design_system/theme.dart';
import '../../../core/design_system/tokens.dart';
import '../../../shared/models/scan_models.dart';
import '../application/scanner_controller.dart';
import '../data/scanner_api.dart';

part 'top_navigation.dart';
part 'preview.dart';
part 'input_actions.dart';
part 'result_list.dart';
part 'review_inspector.dart';
part 'product_picker.dart';
part 'review_footer.dart';

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
