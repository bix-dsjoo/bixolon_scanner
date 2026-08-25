part of 'scanner_screen.dart';

enum _BoxEditorMode { edit, add }

enum _BoxHandle {
  topLeft,
  top,
  topRight,
  right,
  bottomRight,
  bottom,
  bottomLeft,
  left,
}

enum _CanvasAction { none, move, draw, pan, zoom }

Future<void> _showBoxEditorDialog(
  BuildContext context, {
  required ScannerController controller,
  ReviewDetection? detection,
}) async {
  if (controller.imageBytes == null || controller.imageSize == null) return;
  await showDialog<void>(
    context: context,
    barrierDismissible: false,
    builder: (context) => _BoxEditorDialog(
      controller: controller,
      mode: detection == null ? _BoxEditorMode.add : _BoxEditorMode.edit,
      detection: detection,
    ),
  );
}

class _BoxEditorDialog extends StatefulWidget {
  const _BoxEditorDialog({
    required this.controller,
    required this.mode,
    required this.detection,
  });

  final ScannerController controller;
  final _BoxEditorMode mode;
  final ReviewDetection? detection;

  @override
  State<_BoxEditorDialog> createState() => _BoxEditorDialogState();
}

class _BoxEditorDialogState extends State<_BoxEditorDialog> {
  static const double _minimumBoxSize = 8;
  static const double _minimumZoom = 1;
  static const double _maximumZoom = 8;

  final List<BoundingBox?> _undoStack = <BoundingBox?>[];
  final List<BoundingBox?> _redoStack = <BoundingBox?>[];
  final FocusNode _dialogFocusNode = FocusNode(debugLabel: 'box-editor');
  BoundingBox? _draft;
  Size? _viewportSize;
  double _zoom = 1;
  Offset _pan = Offset.zero;
  _CanvasAction _canvasAction = _CanvasAction.none;
  Offset? _drawStart;
  Offset? _zoomAnchorImage;
  double _gestureStartZoom = 1;

  Size get _imageSize => widget.controller.imageSize!;

  bool get _canApply {
    final bbox = _draft;
    return bbox != null &&
        bbox.width >= _minimumBoxSize &&
        bbox.height >= _minimumBoxSize &&
        bbox.isValid;
  }

  @override
  void initState() {
    super.initState();
    _draft = widget.detection?.finalBbox;
  }

  @override
  void dispose() {
    _dialogFocusNode.dispose();
    super.dispose();
  }

  void _initializeViewport(Size viewport) {
    if (_viewportSize == viewport) return;
    _viewportSize = viewport;
    final bbox = _draft;
    if (bbox == null || widget.mode == _BoxEditorMode.add) {
      _zoom = 1;
      _pan = Offset.zero;
      return;
    }
    final baseScale = _baseScale(viewport);
    final paddedWidth = bbox.width * 1.5;
    final paddedHeight = bbox.height * 1.5;
    final desiredScale = math.min(
      viewport.width / paddedWidth,
      viewport.height / paddedHeight,
    );
    _zoom = (desiredScale / baseScale).clamp(_minimumZoom, _maximumZoom);
    final scale = baseScale * _zoom;
    final origin = _centeredImageOrigin(viewport, scale);
    final bboxCenter = Offset(
      bbox.x + bbox.width / 2,
      bbox.y + bbox.height / 2,
    );
    _pan = viewport.center(Offset.zero) - origin - bboxCenter * scale;
  }

  double _baseScale(Size viewport) => math.min(
    viewport.width / _imageSize.width,
    viewport.height / _imageSize.height,
  );

  Offset _centeredImageOrigin(Size viewport, double scale) => Offset(
    (viewport.width - _imageSize.width * scale) / 2,
    (viewport.height - _imageSize.height * scale) / 2,
  );

  double _drawScale() => _baseScale(_viewportSize!) * _zoom;

  Offset _imageOrigin() =>
      _centeredImageOrigin(_viewportSize!, _drawScale()) + _pan;

  Rect _imageRect() => _imageOrigin() & (_imageSize * _drawScale());

  Rect _screenRect(BoundingBox bbox) {
    final scale = _drawScale();
    final origin = _imageOrigin();
    return Rect.fromLTWH(
      origin.dx + bbox.x * scale,
      origin.dy + bbox.y * scale,
      bbox.width * scale,
      bbox.height * scale,
    );
  }

  Offset _screenToImage(Offset point) {
    final scale = _drawScale();
    final origin = _imageOrigin();
    return Offset(
      ((point.dx - origin.dx) / scale).clamp(0, _imageSize.width),
      ((point.dy - origin.dy) / scale).clamp(0, _imageSize.height),
    );
  }

  void _setZoom(double value, {Offset? focalPoint}) {
    final viewport = _viewportSize;
    if (viewport == null) return;
    final focal = focalPoint ?? viewport.center(Offset.zero);
    final anchor = _screenToImage(focal);
    final nextZoom = value.clamp(_minimumZoom, _maximumZoom);
    final nextScale = _baseScale(viewport) * nextZoom;
    final nextOrigin = _centeredImageOrigin(viewport, nextScale);
    setState(() {
      _zoom = nextZoom;
      _pan = focal - nextOrigin - anchor * nextScale;
    });
  }

  void _resetView() {
    setState(() {
      _zoom = 1;
      _pan = Offset.zero;
    });
  }

  void _pushDraftHistory() {
    _undoStack.add(_draft);
    _redoStack.clear();
  }

  void _undo() {
    if (_undoStack.isEmpty) return;
    setState(() {
      _redoStack.add(_draft);
      _draft = _undoStack.removeLast();
    });
  }

  void _redo() {
    if (_redoStack.isEmpty) return;
    setState(() {
      _undoStack.add(_draft);
      _draft = _redoStack.removeLast();
    });
  }

  void _onScaleStart(ScaleStartDetails details) {
    _gestureStartZoom = _zoom;
    if (details.pointerCount >= 2) {
      _canvasAction = _CanvasAction.zoom;
      _zoomAnchorImage = _screenToImage(details.localFocalPoint);
      return;
    }
    final bbox = _draft;
    if (bbox != null && _screenRect(bbox).contains(details.localFocalPoint)) {
      _pushDraftHistory();
      _canvasAction = _CanvasAction.move;
      return;
    }
    if (widget.mode == _BoxEditorMode.add && bbox == null) {
      _pushDraftHistory();
      _canvasAction = _CanvasAction.draw;
      _drawStart = _screenToImage(details.localFocalPoint);
      return;
    }
    _canvasAction = _CanvasAction.pan;
  }

  void _onScaleUpdate(ScaleUpdateDetails details) {
    if (details.pointerCount >= 2 || _canvasAction == _CanvasAction.zoom) {
      final viewport = _viewportSize!;
      final anchor =
          _zoomAnchorImage ?? _screenToImage(details.localFocalPoint);
      final nextZoom = (_gestureStartZoom * details.scale).clamp(
        _minimumZoom,
        _maximumZoom,
      );
      final nextScale = _baseScale(viewport) * nextZoom;
      final nextOrigin = _centeredImageOrigin(viewport, nextScale);
      setState(() {
        _zoom = nextZoom;
        _pan = details.localFocalPoint - nextOrigin - anchor * nextScale;
        _canvasAction = _CanvasAction.zoom;
      });
      return;
    }
    switch (_canvasAction) {
      case _CanvasAction.move:
        final bbox = _draft;
        if (bbox == null) return;
        final scale = _drawScale();
        setState(() {
          _draft = _translateBox(
            bbox,
            details.focalPointDelta.dx / scale,
            details.focalPointDelta.dy / scale,
          );
        });
      case _CanvasAction.draw:
        final start = _drawStart;
        if (start == null) return;
        final end = _screenToImage(details.localFocalPoint);
        setState(() => _draft = _boxFromPoints(start, end));
      case _CanvasAction.pan:
        setState(() => _pan += details.focalPointDelta);
      case _CanvasAction.none || _CanvasAction.zoom:
        break;
    }
  }

  void _onScaleEnd(ScaleEndDetails details) {
    _canvasAction = _CanvasAction.none;
    _drawStart = null;
    _zoomAnchorImage = null;
  }

  void _resizeStart() => _pushDraftHistory();

  void _resizeUpdate(_BoxHandle handle, Offset screenDelta) {
    final bbox = _draft;
    if (bbox == null) return;
    final scale = _drawScale();
    setState(() {
      _draft = _resizeBox(
        bbox,
        handle,
        screenDelta.dx / scale,
        screenDelta.dy / scale,
      );
    });
  }

  void _keyboardResize(_BoxHandle handle, Offset imageDelta) {
    final bbox = _draft;
    if (bbox == null) return;
    _pushDraftHistory();
    setState(() {
      _draft = _resizeBox(bbox, handle, imageDelta.dx, imageDelta.dy);
    });
  }

  void _keyboardMove(Offset imageDelta) {
    final bbox = _draft;
    if (bbox == null) return;
    _pushDraftHistory();
    setState(() {
      _draft = _translateBox(bbox, imageDelta.dx, imageDelta.dy);
    });
  }

  KeyEventResult _handleDialogKey(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent) return KeyEventResult.ignored;
    final key = event.logicalKey;
    if (key == LogicalKeyboardKey.escape) {
      Navigator.of(context).pop();
      return KeyEventResult.handled;
    }
    if (HardwareKeyboard.instance.isControlPressed &&
        key == LogicalKeyboardKey.keyZ) {
      _undo();
      return KeyEventResult.handled;
    }
    if (HardwareKeyboard.instance.isControlPressed &&
        key == LogicalKeyboardKey.keyY) {
      _redo();
      return KeyEventResult.handled;
    }
    if (_editorTextInputActive()) return KeyEventResult.ignored;
    final amount = HardwareKeyboard.instance.isShiftPressed ? 10.0 : 1.0;
    final delta = switch (key) {
      LogicalKeyboardKey.arrowLeft => Offset(-amount, 0),
      LogicalKeyboardKey.arrowRight => Offset(amount, 0),
      LogicalKeyboardKey.arrowUp => Offset(0, -amount),
      LogicalKeyboardKey.arrowDown => Offset(0, amount),
      _ => null,
    };
    if (delta == null || _draft == null) return KeyEventResult.ignored;
    _keyboardMove(delta);
    return KeyEventResult.handled;
  }

  bool _editorTextInputActive() {
    final focusContext = FocusManager.instance.primaryFocus?.context;
    if (focusContext == null) return false;
    return focusContext.widget is EditableText ||
        focusContext.findAncestorWidgetOfExactType<EditableText>() != null;
  }

  void _apply() {
    final bbox = _draft;
    if (!_canApply || bbox == null) return;
    if (widget.mode == _BoxEditorMode.add) {
      widget.controller.addDetection(bbox);
    } else {
      widget.controller.updateSelectedBbox(bbox, recordUndo: true);
    }
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final media = MediaQuery.sizeOf(context);
    final fullScreen = media.width < AppBreakpoints.scanStacked;
    return Dialog(
      key: const ValueKey('box-editor-dialog'),
      insetPadding: fullScreen
          ? EdgeInsets.zero
          : const EdgeInsets.all(AppSpacing.x6),
      clipBehavior: Clip.hardEdge,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(
          fullScreen ? 0 : context.appTokens.panelRadius,
        ),
      ),
      child: Focus(
        autofocus: true,
        focusNode: _dialogFocusNode,
        onKeyEvent: _handleDialogKey,
        child: SizedBox(
          width: double.infinity,
          height: double.infinity,
          child: Column(
            children: [
              _buildHeader(context),
              _buildToolbar(context),
              Expanded(child: _buildCanvas(context)),
              _buildFooter(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    final editing = widget.mode == _BoxEditorMode.edit;
    return Container(
      constraints: BoxConstraints(minHeight: context.appTokens.headerHeight),
      padding: const EdgeInsets.only(left: AppSpacing.x6),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(bottom: BorderSide(color: AppColors.divider)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  editing ? '박스 수정' : '박스 추가',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                Text(
                  editing
                      ? '박스를 이동하거나 핸들을 드래그해 상품 영역에 맞추세요.'
                      : '상품의 한쪽 모서리에서 반대쪽 모서리까지 드래그하세요.',
                  style: Theme.of(
                    context,
                  ).textTheme.bodySmall?.copyWith(color: AppColors.muted),
                ),
              ],
            ),
          ),
          AppIconActionButton(
            semanticLabel: '박스 편집 취소',
            tooltip: '닫기 (Esc)',
            onPressed: () => Navigator.of(context).pop(),
            icon: const Icon(Icons.close_rounded, size: 20),
          ),
          const SizedBox(width: AppSpacing.x2),
        ],
      ),
    );
  }

  Widget _buildToolbar(BuildContext context) {
    return Container(
      constraints: BoxConstraints(minHeight: context.appTokens.actionHeight),
      padding: const EdgeInsets.symmetric(
        horizontal: AppSpacing.x4,
        vertical: AppSpacing.x2,
      ),
      decoration: const BoxDecoration(
        color: AppColors.elevated,
        border: Border(bottom: BorderSide(color: AppColors.divider)),
      ),
      child: Row(
        children: [
          OutlinedButton.icon(
            onPressed: _resetView,
            icon: const Icon(Icons.fit_screen_outlined, size: 18),
            label: const Text('전체 보기'),
          ),
          const SizedBox(width: AppSpacing.x2),
          AppIconActionButton(
            semanticLabel: '화면 축소',
            tooltip: '축소',
            onPressed: _zoom <= _minimumZoom
                ? null
                : () => _setZoom(_zoom - .25),
            icon: const Icon(Icons.remove_rounded, size: 20),
          ),
          SizedBox(
            width: 72,
            child: Text(
              '${(_zoom * 100).round()}%',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                fontWeight: AppTypography.bold,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ),
          AppIconActionButton(
            semanticLabel: '화면 확대',
            tooltip: '확대',
            onPressed: _zoom >= _maximumZoom
                ? null
                : () => _setZoom(_zoom + .25),
            icon: const Icon(Icons.add_rounded, size: 20),
          ),
          const SizedBox(width: AppSpacing.x4),
          const VerticalDivider(width: 1),
          const SizedBox(width: AppSpacing.x4),
          AppIconActionButton(
            semanticLabel: '박스 편집 실행 취소',
            tooltip: '실행 취소 (Ctrl+Z)',
            onPressed: _undoStack.isEmpty ? null : _undo,
            icon: const Icon(Icons.undo_rounded, size: 20),
          ),
          const SizedBox(width: AppSpacing.x1),
          AppIconActionButton(
            semanticLabel: '박스 편집 다시 실행',
            tooltip: '다시 실행 (Ctrl+Y)',
            onPressed: _redoStack.isEmpty ? null : _redo,
            icon: const Icon(Icons.redo_rounded, size: 20),
          ),
          const Spacer(),
          Text(
            '빈 공간 드래그: 이동 · 휠/두 손가락: 확대',
            style: Theme.of(
              context,
            ).textTheme.bodySmall?.copyWith(color: AppColors.muted),
          ),
        ],
      ),
    );
  }

  Widget _buildCanvas(BuildContext context) {
    return ColoredBox(
      color: AppColors.preview,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final viewport = Size(constraints.maxWidth, constraints.maxHeight);
          _initializeViewport(viewport);
          final imageRect = _imageRect();
          final draft = _draft;
          final draftRect = draft == null ? null : _screenRect(draft);
          final activeDetections = widget.controller.activeDetections;
          final selectedIndex = widget.detection == null
              ? activeDetections.length + 1
              : activeDetections.indexWhere(
                      (item) =>
                          item.source.itemId == widget.detection!.source.itemId,
                    ) +
                    1;
          return Listener(
            onPointerSignal: (event) {
              if (event is! PointerScrollEvent) return;
              final delta = event.scrollDelta.dy > 0 ? -.25 : .25;
              _setZoom(_zoom + delta, focalPoint: event.localPosition);
            },
            child: Stack(
              clipBehavior: Clip.hardEdge,
              children: [
                Positioned.fromRect(
                  rect: imageRect,
                  child: Image.memory(
                    widget.controller.imageBytes!,
                    fit: BoxFit.fill,
                    gaplessPlayback: true,
                    filterQuality: FilterQuality.medium,
                  ),
                ),
                Positioned.fill(
                  child: IgnorePointer(
                    child: CustomPaint(
                      painter: _BoxEditorContextPainter(
                        detections: activeDetections,
                        excludedItemId: widget.detection?.source.itemId,
                        imageOrigin: _imageOrigin(),
                        scale: _drawScale(),
                      ),
                    ),
                  ),
                ),
                Positioned.fill(
                  child: GestureDetector(
                    key: const ValueKey('box-editor-canvas'),
                    behavior: HitTestBehavior.opaque,
                    onScaleStart: _onScaleStart,
                    onScaleUpdate: _onScaleUpdate,
                    onScaleEnd: _onScaleEnd,
                  ),
                ),
                if (draftRect != null) ...[
                  Positioned.fromRect(
                    rect: draftRect,
                    child: IgnorePointer(
                      child: Container(
                        key: const ValueKey('box-editor-draft'),
                        decoration: BoxDecoration(
                          color: AppColors.primary.withValues(
                            alpha: AppOpacity.selectedStatusSurface,
                          ),
                          border: Border.all(
                            color: AppColors.primary,
                            width: 3,
                          ),
                        ),
                      ),
                    ),
                  ),
                  Positioned(
                    left: draftRect.left + AppSpacing.x3,
                    top: draftRect.top + AppSpacing.x3,
                    child: IgnorePointer(
                      child: _BoxEditorNumberLabel(
                        key: const ValueKey('box-editor-draft-label'),
                        index: selectedIndex,
                        color: AppColors.primary,
                      ),
                    ),
                  ),
                  ..._BoxHandle.values.map(
                    (handle) => _BoxEditorHandle(
                      handle: handle,
                      center: _handleCenter(draftRect, handle),
                      onDragStart: _resizeStart,
                      onDragUpdate: (delta) => _resizeUpdate(handle, delta),
                      onKeyboardResize: (delta) =>
                          _keyboardResize(handle, delta),
                    ),
                  ),
                ] else if (widget.mode == _BoxEditorMode.add)
                  const Center(
                    child: _CanvasInstruction(message: '드래그하여 상품 영역을 지정하세요'),
                  ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildFooter(BuildContext context) {
    final invalidDraft = _draft != null && !_canApply;
    return AppActionBar(
      child: Row(
        children: [
          Expanded(
            child: Text(
              invalidDraft
                  ? '박스를 조금 더 크게 지정해 주세요.'
                  : '방향키로 1px, Shift+방향키로 10px 이동할 수 있어요.',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: invalidDraft ? AppColors.error : AppColors.muted,
                fontWeight: invalidDraft
                    ? AppTypography.bold
                    : AppTypography.regular,
              ),
            ),
          ),
          OutlinedButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('취소'),
          ),
          const SizedBox(width: AppSpacing.x2),
          FilledButton.icon(
            key: const ValueKey('apply-box-edit'),
            onPressed: _canApply ? _apply : null,
            icon: const Icon(Icons.check_rounded, size: 19),
            label: const Text('적용'),
          ),
        ],
      ),
    );
  }

  BoundingBox _translateBox(BoundingBox bbox, double dx, double dy) {
    final x = (bbox.x + dx).round().clamp(
      0,
      (_imageSize.width - bbox.width).round(),
    );
    final y = (bbox.y + dy).round().clamp(
      0,
      (_imageSize.height - bbox.height).round(),
    );
    return BoundingBox(x: x, y: y, width: bbox.width, height: bbox.height);
  }

  BoundingBox _boxFromPoints(Offset a, Offset b) {
    final left = math.min(a.dx, b.dx).round();
    final top = math.min(a.dy, b.dy).round();
    final right = math.max(a.dx, b.dx).round();
    final bottom = math.max(a.dy, b.dy).round();
    return BoundingBox(
      x: left,
      y: top,
      width: math.max(1, right - left),
      height: math.max(1, bottom - top),
    );
  }

  BoundingBox _resizeBox(
    BoundingBox bbox,
    _BoxHandle handle,
    double dx,
    double dy,
  ) {
    var left = bbox.x.toDouble();
    var top = bbox.y.toDouble();
    var right = (bbox.x + bbox.width).toDouble();
    var bottom = (bbox.y + bbox.height).toDouble();
    if (_leftHandles.contains(handle)) {
      left = (left + dx).clamp(0, right - _minimumBoxSize);
    }
    if (_rightHandles.contains(handle)) {
      right = (right + dx).clamp(left + _minimumBoxSize, _imageSize.width);
    }
    if (_topHandles.contains(handle)) {
      top = (top + dy).clamp(0, bottom - _minimumBoxSize);
    }
    if (_bottomHandles.contains(handle)) {
      bottom = (bottom + dy).clamp(top + _minimumBoxSize, _imageSize.height);
    }
    return BoundingBox(
      x: left.round(),
      y: top.round(),
      width: (right - left).round(),
      height: (bottom - top).round(),
    );
  }
}

const _leftHandles = {
  _BoxHandle.topLeft,
  _BoxHandle.left,
  _BoxHandle.bottomLeft,
};
const _rightHandles = {
  _BoxHandle.topRight,
  _BoxHandle.right,
  _BoxHandle.bottomRight,
};
const _topHandles = {_BoxHandle.topLeft, _BoxHandle.top, _BoxHandle.topRight};
const _bottomHandles = {
  _BoxHandle.bottomLeft,
  _BoxHandle.bottom,
  _BoxHandle.bottomRight,
};

Offset _handleCenter(Rect rect, _BoxHandle handle) => switch (handle) {
  _BoxHandle.topLeft => rect.topLeft,
  _BoxHandle.top => rect.topCenter,
  _BoxHandle.topRight => rect.topRight,
  _BoxHandle.right => rect.centerRight,
  _BoxHandle.bottomRight => rect.bottomRight,
  _BoxHandle.bottom => rect.bottomCenter,
  _BoxHandle.bottomLeft => rect.bottomLeft,
  _BoxHandle.left => rect.centerLeft,
};

String _boxHandleLabel(_BoxHandle handle) => switch (handle) {
  _BoxHandle.topLeft => '왼쪽 위',
  _BoxHandle.top => '위쪽',
  _BoxHandle.topRight => '오른쪽 위',
  _BoxHandle.right => '오른쪽',
  _BoxHandle.bottomRight => '오른쪽 아래',
  _BoxHandle.bottom => '아래쪽',
  _BoxHandle.bottomLeft => '왼쪽 아래',
  _BoxHandle.left => '왼쪽',
};

class _BoxEditorHandle extends StatefulWidget {
  const _BoxEditorHandle({
    required this.handle,
    required this.center,
    required this.onDragStart,
    required this.onDragUpdate,
    required this.onKeyboardResize,
  });

  final _BoxHandle handle;
  final Offset center;
  final VoidCallback onDragStart;
  final ValueChanged<Offset> onDragUpdate;
  final ValueChanged<Offset> onKeyboardResize;

  @override
  State<_BoxEditorHandle> createState() => _BoxEditorHandleState();
}

class _BoxEditorHandleState extends State<_BoxEditorHandle> {
  final FocusNode _focusNode = FocusNode();
  bool _focused = false;

  @override
  void dispose() {
    _focusNode.dispose();
    super.dispose();
  }

  KeyEventResult _handleKey(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent) return KeyEventResult.ignored;
    final amount = HardwareKeyboard.instance.isShiftPressed ? 10.0 : 1.0;
    final delta = switch (event.logicalKey) {
      LogicalKeyboardKey.arrowLeft => Offset(-amount, 0),
      LogicalKeyboardKey.arrowRight => Offset(amount, 0),
      LogicalKeyboardKey.arrowUp => Offset(0, -amount),
      LogicalKeyboardKey.arrowDown => Offset(0, amount),
      _ => null,
    };
    if (delta == null) return KeyEventResult.ignored;
    widget.onKeyboardResize(delta);
    return KeyEventResult.handled;
  }

  @override
  Widget build(BuildContext context) {
    return Positioned(
      left: widget.center.dx - 22,
      top: widget.center.dy - 22,
      width: 44,
      height: 44,
      child: Semantics(
        container: true,
        focusable: true,
        focused: _focused,
        button: true,
        label: '${_boxHandleLabel(widget.handle)} 박스 크기 조절',
        hint: '드래그하거나 방향키로 조절합니다.',
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTap: _focusNode.requestFocus,
          onPanStart: (_) {
            _focusNode.requestFocus();
            widget.onDragStart();
          },
          onPanUpdate: (details) => widget.onDragUpdate(details.delta),
          child: Focus(
            focusNode: _focusNode,
            onKeyEvent: _handleKey,
            onFocusChange: (focused) {
              if (_focused != focused) setState(() => _focused = focused);
            },
            child: Center(
              child: Container(
                key: ValueKey('box-editor-handle-${widget.handle.name}'),
                width: 20,
                height: 20,
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  border: Border.all(
                    color: _focused
                        ? context.appComponents.focusRing
                        : AppColors.primary,
                    width: 3,
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

class _CanvasInstruction extends StatelessWidget {
  const _CanvasInstruction({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: AppSpacing.x4,
        vertical: AppSpacing.x3,
      ),
      decoration: BoxDecoration(
        color: context.appComponents.previewLabelSurface,
        border: Border.all(color: AppColors.surface),
        borderRadius: BorderRadius.circular(context.appTokens.controlRadius),
      ),
      child: Text(
        message,
        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
          color: context.appComponents.onPreview,
          fontWeight: AppTypography.bold,
        ),
      ),
    );
  }
}

class _BoxEditorContextPainter extends CustomPainter {
  const _BoxEditorContextPainter({
    required this.detections,
    required this.excludedItemId,
    required this.imageOrigin,
    required this.scale,
  });

  final List<ReviewDetection> detections;
  final String? excludedItemId;
  final Offset imageOrigin;
  final double scale;

  @override
  void paint(Canvas canvas, Size size) {
    for (var index = 0; index < detections.length; index += 1) {
      final detection = detections[index];
      if (detection.source.itemId == excludedItemId) continue;
      final bbox = detection.finalBbox;
      final rect = Rect.fromLTWH(
        imageOrigin.dx + bbox.x * scale,
        imageOrigin.dy + bbox.y * scale,
        bbox.width * scale,
        bbox.height * scale,
      );
      final line = Paint()
        ..color = _detectionStatusColor(detection)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;
      canvas.drawRect(rect, line);
      _paintBoxNumber(
        canvas,
        index: index + 1,
        origin: rect.topLeft,
        color: _detectionStatusColor(detection),
      );
    }
  }

  @override
  bool shouldRepaint(covariant _BoxEditorContextPainter oldDelegate) =>
      oldDelegate.detections != detections ||
      oldDelegate.excludedItemId != excludedItemId ||
      oldDelegate.imageOrigin != imageOrigin ||
      oldDelegate.scale != scale;
}

class _BoxEditorNumberLabel extends StatelessWidget {
  const _BoxEditorNumberLabel({
    super.key,
    required this.index,
    required this.color,
  });

  final int index;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final label = index.toString();
    return Container(
      width: _boxNumberWidth(label),
      height: AppSpacing.x6,
      alignment: Alignment.center,
      color: color,
      child: Text(
        label,
        maxLines: 1,
        style: TextStyle(
          color: context.appComponents.onPreview,
          fontSize: AppTypography.captionSize,
          height: 1,
          fontWeight: AppTypography.bold,
        ),
      ),
    );
  }
}

double _boxNumberWidth(String label) =>
    math.max(AppSpacing.x6, AppSpacing.x4 + label.length * AppSpacing.x2);

void _paintBoxNumber(
  Canvas canvas, {
  required int index,
  required Offset origin,
  required Color color,
}) {
  final label = index.toString();
  final size = Size(_boxNumberWidth(label), AppSpacing.x6);
  canvas.drawRect(origin & size, Paint()..color = color);
  final painter = TextPainter(
    text: TextSpan(
      text: label,
      style: const TextStyle(
        color: AppColors.surface,
        fontSize: AppTypography.captionSize,
        height: 1,
        fontWeight: AppTypography.semibold,
      ),
    ),
    textDirection: TextDirection.ltr,
    maxLines: 1,
  )..layout();
  painter.paint(
    canvas,
    origin +
        Offset(
          (size.width - painter.width) / 2,
          (size.height - painter.height) / 2,
        ),
  );
}
