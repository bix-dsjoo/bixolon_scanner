part of 'activity_screen.dart';

class _LogList extends StatefulWidget {
  const _LogList({
    super.key,
    required this.logs,
    required this.totalCount,
    required this.filtered,
    required this.query,
    required this.selectedId,
    required this.onSelected,
    this.onExitForward,
  });

  final List<ScanLogSummary> logs;
  final int totalCount;
  final bool filtered;
  final String query;
  final String? selectedId;
  final ValueChanged<ScanLogSummary> onSelected;
  final VoidCallback? onExitForward;

  @override
  State<_LogList> createState() => _LogListState();
}

class _LogListState extends State<_LogList> {
  final ScrollController _scrollController = ScrollController();
  final Map<String, FocusNode> _rowFocusNodes = {};
  String? _lastSelectedId;
  int _lastSelectedIndex = -1;

  @override
  void initState() {
    super.initState();
    _lastSelectedId = widget.selectedId;
    _lastSelectedIndex = _selectedIndex;
    _revealSelectedRow();
  }

  @override
  void didUpdateWidget(covariant _LogList oldWidget) {
    super.didUpdateWidget(oldWidget);
    _disposeUnusedRowFocusNodes();
    _syncRowTraversalState();
    final selectedIndex = _selectedIndex;
    if (_lastSelectedId == widget.selectedId &&
        _lastSelectedIndex == selectedIndex) {
      return;
    }
    _lastSelectedId = widget.selectedId;
    _lastSelectedIndex = selectedIndex;
    _revealSelectedRow();
  }

  int get _selectedIndex =>
      widget.logs.indexWhere((log) => log.scanId == widget.selectedId);

  String? get _traversalScanId {
    if (widget.logs.isEmpty) return null;
    final index = _selectedIndex;
    return index < 0 ? widget.logs.first.scanId : widget.logs[index].scanId;
  }

  FocusNode _rowFocusNode(String scanId) {
    final focusNode = _rowFocusNodes.putIfAbsent(
      scanId,
      () => FocusNode(debugLabel: 'activity-log-$scanId'),
    );
    focusNode.skipTraversal = scanId != _traversalScanId;
    return focusNode;
  }

  void _syncRowTraversalState() {
    final traversalScanId = _traversalScanId;
    for (final MapEntry(key: scanId, value: focusNode)
        in _rowFocusNodes.entries) {
      focusNode.skipTraversal = scanId != traversalScanId;
    }
  }

  void requestTraversalFocus() {
    final traversalScanId = _traversalScanId;
    if (traversalScanId == null) return;
    _rowFocusNode(traversalScanId).requestFocus();
    _revealSelectedRow();
  }

  void _disposeUnusedRowFocusNodes() {
    final activeIds = widget.logs.map((log) => log.scanId).toSet();
    final removed = <FocusNode>[];
    _rowFocusNodes.removeWhere((scanId, focusNode) {
      if (activeIds.contains(scanId)) return false;
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
    if (event.logicalKey == LogicalKeyboardKey.tab &&
        !HardwareKeyboard.instance.isShiftPressed &&
        widget.onExitForward != null) {
      widget.onExitForward!();
      return KeyEventResult.handled;
    }
    if (widget.logs.isEmpty) {
      return KeyEventResult.ignored;
    }
    final currentIndex = _selectedIndex < 0 ? 0 : _selectedIndex;
    final int? requestedIndex = switch (event.logicalKey) {
      LogicalKeyboardKey.arrowUp => currentIndex - 1,
      LogicalKeyboardKey.arrowDown => currentIndex + 1,
      LogicalKeyboardKey.home => 0,
      LogicalKeyboardKey.end => widget.logs.length - 1,
      LogicalKeyboardKey.pageUp => currentIndex - _pageStep,
      LogicalKeyboardKey.pageDown => currentIndex + _pageStep,
      _ => null,
    };
    if (requestedIndex == null) return KeyEventResult.ignored;
    final nextIndex = requestedIndex.clamp(0, widget.logs.length - 1);
    final nextLog = widget.logs[nextIndex];
    if (nextIndex != currentIndex) widget.onSelected(nextLog);
    _rowFocusNode(nextLog.scanId).requestFocus();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) _revealSelectedRow();
    });
    return KeyEventResult.handled;
  }

  int get _pageStep {
    if (!_scrollController.hasClients) return 1;
    final visibleRows =
        (_scrollController.position.viewportDimension /
                context.appTokens.rowHeight)
            .floor()
            .clamp(1, widget.logs.length);
    return visibleRows > 1 ? visibleRows - 1 : 1;
  }

  void _revealSelectedRow() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || !_scrollController.hasClients) return;
      final index = _selectedIndex;
      if (index < 0) return;

      final position = _scrollController.position;
      final rowHeight = context.appTokens.rowHeight;
      final visibleRowCount = (position.viewportDimension / rowHeight)
          .floor()
          .clamp(1, widget.logs.length);
      final maxFirstVisibleIndex = (widget.logs.length - visibleRowCount).clamp(
        0,
        widget.logs.length,
      );
      final firstVisibleIndex = (index - visibleRowCount ~/ 2).clamp(
        0,
        maxFirstVisibleIndex,
      );
      final alignedOffset = firstVisibleIndex * rowHeight;
      final target = alignedOffset.clamp(
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
    return ColoredBox(
      color: AppColors.surface,
      child: Column(
        children: [
          AppSectionLabel(
            label: widget.filtered ? '검색 결과' : '활동 목록',
            trailing: Text(
              widget.filtered
                  ? '${widget.logs.length} / ${widget.totalCount}건'
                  : '${widget.logs.length}건',
            ),
          ),
          const _LogTableHeader(),
          Expanded(
            child: LayoutBuilder(
              builder: (context, constraints) {
                final rowHeight = context.appTokens.rowHeight;
                final bottomBuffer = constraints.maxHeight % rowHeight;
                return Focus(
                  canRequestFocus: false,
                  skipTraversal: true,
                  onKeyEvent: _handleRowNavigation,
                  child: Scrollbar(
                    controller: _scrollController,
                    child: ListView.builder(
                      key: const ValueKey('activity-log-list'),
                      controller: _scrollController,
                      primary: false,
                      padding: EdgeInsets.only(bottom: bottomBuffer),
                      itemCount: widget.logs.length,
                      itemBuilder: (context, index) {
                        final log = widget.logs[index];
                        return _LogRow(
                          key: ValueKey('activity-log-${log.scanId}'),
                          log: log,
                          query: widget.query,
                          selected: log.scanId == widget.selectedId,
                          focusNode: _rowFocusNode(log.scanId),
                          onTap: () => widget.onSelected(log),
                        );
                      },
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

class _LogTableHeader extends StatelessWidget {
  const _LogTableHeader();

  @override
  Widget build(BuildContext context) {
    final style = Theme.of(context).textTheme.bodySmall?.copyWith(
      color: AppColors.muted,
      fontWeight: AppTypography.bold,
      letterSpacing: AppTypography.dataLabelTracking,
    );
    return SizedBox(
      key: const ValueKey('activity-table-header'),
      height: context.appTokens.sectionHeaderHeight,
      child: Row(
        children: [
          const SizedBox(width: AppSpacing.x4),
          Expanded(flex: 3, child: Text('기록 시각', style: style)),
          Expanded(flex: 4, child: Text('상품', style: style)),
          Expanded(flex: 2, child: Text('입력원', style: style)),
          Expanded(flex: 2, child: Text('검수 결과', style: style)),
          const SizedBox(width: AppSpacing.x8),
        ],
      ),
    );
  }
}

class _LogRow extends StatelessWidget {
  const _LogRow({
    super.key,
    required this.log,
    required this.query,
    required this.selected,
    required this.focusNode,
    required this.onTap,
  });

  final ScanLogSummary log;
  final String query;
  final bool selected;
  final FocusNode focusNode;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final contentLabel = activityLogContentLabel(log, query: query);
    final source = log.inputMode == InputMode.camera ? '카메라' : '이미지';
    final resultLabel = activityLogResultLabel(log);
    return AppSelectableSurface(
      selected: selected,
      inMutuallyExclusiveGroup: true,
      focusNode: focusNode,
      onTap: onTap,
      semanticLabel:
          '${_formatDate(log.recordedAt)} ${_formatTime(log.recordedAt)}, '
          '$contentLabel, '
          '${log.originalImagePath == null ? '저장 이미지 없음' : '저장 이미지 있음'}, '
          '$source, $resultLabel 활동 기록',
      restingBorder: const Border(top: BorderSide(color: AppColors.divider)),
      child: Row(
        children: [
          const SizedBox(width: AppSpacing.x4),
          Expanded(
            flex: 3,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  _formatDate(log.recordedAt),
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    fontWeight: AppTypography.bold,
                  ),
                ),
                Text(
                  _formatTime(log.recordedAt),
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
          Expanded(
            flex: 4,
            child: Row(
              children: [
                _ActivityLogImage(
                  imagePath: log.originalImagePath,
                  compact: true,
                ),
                const SizedBox(width: AppSpacing.x3),
                Expanded(
                  child: Text(
                    contentLabel,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      fontWeight: AppTypography.semibold,
                    ),
                  ),
                ),
              ],
            ),
          ),
          Expanded(flex: 2, child: _SourceLabel(mode: log.inputMode)),
          Expanded(flex: 2, child: _ReviewSummary(log: log)),
          SizedBox(
            width: context.appTokens.compactVisualSize,
            child: Icon(
              selected
                  ? Icons.arrow_forward_rounded
                  : Icons.chevron_right_rounded,
              size: 20,
              color: selected ? AppColors.primary : AppColors.subtle,
            ),
          ),
        ],
      ),
    );
  }
}

class _ReviewSummary extends StatelessWidget {
  const _ReviewSummary({required this.log});

  final ScanLogSummary log;

  @override
  Widget build(BuildContext context) {
    final modifiedCount = _modifiedItemCount(log.items);
    final modified = modifiedCount > 0;
    final recapture = log.isRecapture;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(
          recapture
              ? Icons.center_focus_weak_rounded
              : modified
              ? Icons.edit_outlined
              : Icons.check_circle_outline_rounded,
          size: 17,
          color: recapture || modified
              ? AppColors.attention
              : AppColors.success,
        ),
        const SizedBox(width: AppSpacing.x2),
        Flexible(
          child: Text(
            recapture
                ? '재촬영'
                : modified
                ? '$modifiedCount개 수정'
                : '자동 확정',
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: AppColors.ink,
              fontWeight: AppTypography.semibold,
            ),
          ),
        ),
      ],
    );
  }
}

class _SourceLabel extends StatelessWidget {
  const _SourceLabel({required this.mode});

  final InputMode mode;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(
          mode == InputMode.camera
              ? Icons.photo_camera_outlined
              : Icons.image_outlined,
          size: 17,
          color: AppColors.muted,
        ),
        const SizedBox(width: AppSpacing.x2),
        Text(
          mode == InputMode.camera ? '카메라' : '이미지',
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }
}
