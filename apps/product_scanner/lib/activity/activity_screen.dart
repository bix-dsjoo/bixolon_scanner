import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../models/scan_models.dart';
import '../services/scan_log_repository.dart';
import '../theme/app_copy.dart';
import '../theme/app_theme.dart';
import '../theme/app_tokens.dart';
import '../widgets/app_components.dart';
import 'activity_filters.dart';
import 'activity_presentation.dart';

class ActivityScreen extends StatefulWidget {
  const ActivityScreen({
    super.key,
    required this.loadLogs,
    required this.dataRevision,
    required this.latestSavedScanId,
    required this.active,
    required this.canChooseImageShortcut,
    required this.onChooseImageShortcut,
    required this.onNavigateToScan,
  });

  final Future<List<ScanLogSummary>> Function() loadLogs;
  final int dataRevision;
  final String? latestSavedScanId;
  final bool active;
  final bool canChooseImageShortcut;
  final VoidCallback onChooseImageShortcut;
  final VoidCallback onNavigateToScan;

  @override
  State<ActivityScreen> createState() => _ActivityScreenState();
}

class _ActivityScreenState extends State<ActivityScreen> {
  final GlobalKey<_LogListState> _logListKey = GlobalKey<_LogListState>();
  final FocusNode _shortcutFocusNode = FocusNode(
    debugLabel: 'activity-shortcuts',
  );
  final FocusNode _searchFocusNode = FocusNode(debugLabel: 'activity-search');
  final FocusNode _toolbarRefreshFocusNode = FocusNode(
    debugLabel: 'activity-toolbar-refresh',
  );
  final FocusNode _retryRefreshFocusNode = FocusNode(
    debugLabel: 'activity-retry-refresh',
  );
  final FocusNode _detailDisclosureFocusNode = FocusNode(
    debugLabel: 'activity-detail-disclosure',
  );
  final TextEditingController _searchController = TextEditingController();

  List<ScanLogSummary> _logs = const [];
  String? _selectedId;
  bool _loading = true;
  bool _restoreRefreshFocusOnCompletion = false;
  bool _announceRefreshProgress = true;
  String? _error;
  String? _errorDetail;
  String? _refreshError;
  String? _retryPreferredSelectionId;
  String _query = '';
  ActivityInputFilter _inputFilter = ActivityInputFilter.all;
  ActivityDateFilter _dateFilter = ActivityDateFilter.all;
  ActivitySortOrder _sortOrder = ActivitySortOrder.newest;
  int _loadedDataRevision = -1;

  @override
  void initState() {
    super.initState();
    final latestSavedScanId = widget.latestSavedScanId;
    _refresh(
      preferredSelectionId: latestSavedScanId,
      announceProgress: latestSavedScanId == null,
    );
    if (widget.active) _requestShortcutFocusAfterFrame();
  }

  @override
  void didUpdateWidget(covariant ActivityScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.active && !oldWidget.active) {
      _requestShortcutFocusAfterFrame();
    }
    if (widget.active &&
        _loading &&
        widget.dataRevision != oldWidget.dataRevision &&
        widget.latestSavedScanId != null) {
      _announceRefreshProgress = false;
    }
    if (widget.active &&
        _loadedDataRevision != widget.dataRevision &&
        !_loading) {
      _refresh(
        preferredSelectionId: widget.latestSavedScanId,
        announceProgress: widget.latestSavedScanId == null,
      );
    }
  }

  void _requestShortcutFocusAfterFrame() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted && widget.active) _shortcutFocusNode.requestFocus();
    });
  }

  void _refreshStaleDataAfterFrame() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted ||
          !widget.active ||
          _loading ||
          _loadedDataRevision == widget.dataRevision) {
        return;
      }
      _refresh(
        preferredSelectionId: widget.latestSavedScanId,
        announceProgress: false,
      );
    });
  }

  @override
  void dispose() {
    _shortcutFocusNode.dispose();
    _searchFocusNode.dispose();
    _toolbarRefreshFocusNode.dispose();
    _retryRefreshFocusNode.dispose();
    _detailDisclosureFocusNode.dispose();
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _refresh({
    String? preferredSelectionId,
    bool announceProgress = true,
  }) async {
    final requestedRevision = widget.dataRevision;
    final savedRecordSync = preferredSelectionId != null;
    _restoreRefreshFocusOnCompletion =
        _toolbarRefreshFocusNode.hasFocus || _retryRefreshFocusNode.hasFocus;
    setState(() {
      _loading = true;
      _announceRefreshProgress = announceProgress;
      if (!savedRecordSync) _retryPreferredSelectionId = null;
      _error = null;
      _errorDetail = null;
      _refreshError = null;
    });
    try {
      final logs = await widget.loadLogs();
      if (!mounted) return;
      final preferredStillExists =
          preferredSelectionId != null &&
          logs.any((log) => log.scanId == preferredSelectionId);
      final selectedStillExists = logs.any((log) => log.scanId == _selectedId);
      setState(() {
        _logs = logs;
        _loadedDataRevision = requestedRevision;
        _selectedId = preferredStillExists
            ? preferredSelectionId
            : selectedStillExists
            ? _selectedId
            : logs.isEmpty
            ? null
            : logs.first.scanId;
        _loading = false;
        _retryPreferredSelectionId = null;
        _refreshError = null;
      });
      _restoreRefreshFocusAfterFrame();
      if (_loadedDataRevision != widget.dataRevision) {
        _refreshStaleDataAfterFrame();
      }
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _retryPreferredSelectionId = preferredSelectionId;
        if (_logs.isEmpty) {
          _error = savedRecordSync ? '활동 화면을 갱신하지 못했어요' : '활동 기록을 불러오지 못했어요';
          _errorDetail = savedRecordSync
              ? '방금 확정한 기록은 저장됐어요. 잠시 후 새로고침해 주세요.'
              : '저장된 기록은 그대로 유지됩니다. 잠시 후 새로고침해 주세요.';
        } else {
          _error = null;
          _errorDetail = null;
          _refreshError = savedRecordSync
              ? '방금 확정한 기록은 저장됐어요. 활동 화면만 갱신하지 못했어요.'
              : '새로고침하지 못했어요. 기존 활동을 표시하고 있어요.';
        }
      });
      _restoreRefreshFocusAfterFrame();
    }
  }

  void _restoreRefreshFocusAfterFrame() {
    final requested = _restoreRefreshFocusOnCompletion;
    _restoreRefreshFocusOnCompletion = false;
    if (!requested) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || !widget.active) return;
      final target = _error != null || _refreshError != null
          ? _retryRefreshFocusNode
          : _toolbarRefreshFocusNode;
      if (target.canRequestFocus) {
        target.requestFocus();
      }
    });
  }

  Future<void> _retryRefresh() =>
      _refresh(preferredSelectionId: _retryPreferredSelectionId);

  bool get _hasSavedRecordSyncError =>
      _retryPreferredSelectionId != null &&
      (_error != null || _refreshError != null);

  List<ScanLogSummary> get _filteredLogs {
    return filterActivityLogs(
      logs: _logs,
      query: _query,
      inputFilter: _inputFilter,
      dateFilter: _dateFilter,
      sortOrder: _sortOrder,
    );
  }

  ScanLogSummary? _selectedLog(List<ScanLogSummary> filtered) {
    for (final log in filtered) {
      if (log.scanId == _selectedId) return log;
    }
    return filtered.isEmpty ? null : filtered.first;
  }

  bool get _hasActiveFilters =>
      _query.isNotEmpty ||
      _inputFilter != ActivityInputFilter.all ||
      _dateFilter != ActivityDateFilter.all ||
      _sortOrder != ActivitySortOrder.newest;

  bool get _hasResultFilters =>
      _query.isNotEmpty ||
      _inputFilter != ActivityInputFilter.all ||
      _dateFilter != ActivityDateFilter.all;

  void _clearSearch() {
    _searchController.clear();
    setState(() => _query = '');
    _searchFocusNode.requestFocus();
  }

  void _resetFilters() {
    _searchController.clear();
    setState(() {
      _query = '';
      _inputFilter = ActivityInputFilter.all;
      _dateFilter = ActivityDateFilter.all;
      _sortOrder = ActivitySortOrder.newest;
      _selectedId = _logs.isEmpty ? null : _logs.first.scanId;
    });
  }

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent) return KeyEventResult.ignored;
    if (!widget.active) return KeyEventResult.ignored;
    if (_loading &&
        event.logicalKey != LogicalKeyboardKey.enter &&
        event.logicalKey != LogicalKeyboardKey.space) {
      _restoreRefreshFocusOnCompletion = false;
    }
    if (HardwareKeyboard.instance.isControlPressed &&
        event.logicalKey == LogicalKeyboardKey.keyO) {
      if (!widget.canChooseImageShortcut) return KeyEventResult.ignored;
      widget.onChooseImageShortcut();
      return KeyEventResult.handled;
    }
    if (event.logicalKey == LogicalKeyboardKey.f5) {
      if (!_loading) _retryRefresh();
      return KeyEventResult.handled;
    }
    if (event.logicalKey == LogicalKeyboardKey.escape &&
        _searchFocusNode.hasFocus) {
      _searchFocusNode.unfocus();
      return KeyEventResult.handled;
    }
    if (event.logicalKey == LogicalKeyboardKey.slash && !_isEditingText()) {
      _searchFocusNode.requestFocus();
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

  @override
  Widget build(BuildContext context) {
    final filtered = _filteredLogs;
    return Listener(
      onPointerDown: (_) {
        if (_loading) _restoreRefreshFocusOnCompletion = false;
      },
      child: Focus(
        focusNode: _shortcutFocusNode,
        onKeyEvent: _handleKeyEvent,
        child: ColoredBox(
          color: AppColors.workspace,
          child: Column(
            children: [
              if (_error == null && _logs.isNotEmpty)
                _ActivityToolbar(
                  onRefresh: _loading ? null : _refresh,
                  refreshFocusNode: _toolbarRefreshFocusNode,
                  loading: _loading,
                  announceRefreshProgress: _announceRefreshProgress,
                  showRefresh: _refreshError == null,
                  searchController: _searchController,
                  searchFocusNode: _searchFocusNode,
                  query: _query,
                  onQueryChanged: (value) => setState(() => _query = value),
                  onClearQuery: _clearSearch,
                  inputFilter: _inputFilter,
                  onInputFilterChanged: (value) =>
                      setState(() => _inputFilter = value),
                  dateFilter: _dateFilter,
                  onDateFilterChanged: (value) =>
                      setState(() => _dateFilter = value),
                  sortOrder: _sortOrder,
                  onSortChanged: (value) => setState(() => _sortOrder = value),
                  hasActiveFilters: _hasActiveFilters && filtered.isNotEmpty,
                  onResetFilters: _resetFilters,
                )
              else
                const AppPanelHeader(title: '활동 기록'),
              if (_refreshError != null)
                AppInlineNotice(
                  message: _refreshError!,
                  icon: Icons.sync_problem_rounded,
                  tone: _hasSavedRecordSyncError
                      ? AppColors.attention
                      : AppColors.error,
                  backgroundColor: _hasSavedRecordSyncError
                      ? AppColors.attentionSoft
                      : AppColors.errorSoft,
                  action: TextButton(
                    focusNode: _retryRefreshFocusNode,
                    onPressed: _retryRefresh,
                    child: const Text(AppActionCopy.refresh),
                  ),
                ),
              Expanded(child: _body(filtered)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _body(List<ScanLogSummary> filtered) {
    if (_loading && _logs.isEmpty) {
      return AppLoadingState(
        message: '활동 기록을 불러오는 중이에요',
        announce: _announceRefreshProgress,
      );
    }
    if (_error != null) {
      return AppEmptyState(
        icon: _hasSavedRecordSyncError
            ? Icons.sync_problem_rounded
            : Icons.error_outline_rounded,
        title: _error!,
        detail: _errorDetail!,
        tone: _hasSavedRecordSyncError ? AppColors.attention : AppColors.error,
        announcement: '$_error. $_errorDetail',
        action: FilledButton.icon(
          focusNode: _retryRefreshFocusNode,
          onPressed: _retryRefresh,
          icon: const Icon(Icons.refresh_rounded, size: 18),
          label: const Text(AppActionCopy.refresh),
        ),
      );
    }
    if (filtered.isEmpty) {
      return AppEmptyState(
        icon: _logs.isEmpty
            ? Icons.history_toggle_off_rounded
            : Icons.search_off_rounded,
        title: _logs.isEmpty ? '저장된 활동이 없어요' : '조건에 맞는 기록이 없어요',
        detail: _logs.isEmpty
            ? '상품을 최종 확정하면 이곳에서 확인할 수 있어요.'
            : '검색어나 필터를 바꾸거나 모두 초기화해 보세요.',
        announcement: _logs.isEmpty
            ? '저장된 활동이 없어요. 상품을 최종 확정하면 이곳에서 확인할 수 있어요.'
            : '조건에 맞는 기록이 없어요. 검색어나 필터를 바꾸거나 모두 초기화해 보세요.',
        action: _logs.isEmpty
            ? FilledButton.icon(
                onPressed: widget.onNavigateToScan,
                icon: const Icon(Icons.center_focus_strong_rounded, size: 18),
                label: const Text('스캔 화면으로 이동'),
              )
            : _hasActiveFilters
            ? FilledButton.icon(
                onPressed: _resetFilters,
                icon: const Icon(Icons.restart_alt_rounded, size: 18),
                label: const Text(AppActionCopy.resetAll),
              )
            : null,
      );
    }
    final selected = _selectedLog(filtered)!;
    return LayoutBuilder(
      builder: (context, constraints) {
        if (constraints.maxWidth < AppBreakpoints.activityStacked) {
          return _LogList(
            key: _logListKey,
            logs: filtered,
            totalCount: _logs.length,
            filtered: _hasResultFilters,
            query: _query,
            selectedId: _selectedId,
            onSelected: _showCompactDetail,
          );
        }
        return Row(
          children: [
            Expanded(
              flex: 7,
              child: _LogList(
                key: _logListKey,
                logs: filtered,
                totalCount: _logs.length,
                filtered: _hasResultFilters,
                query: _query,
                selectedId: selected.scanId,
                onExitForward: _detailDisclosureFocusNode.requestFocus,
                onSelected: (log) => setState(() => _selectedId = log.scanId),
              ),
            ),
            const VerticalDivider(width: 1),
            Expanded(
              flex: 5,
              child: _LogDetail(
                log: selected,
                disclosureFocusNode: _detailDisclosureFocusNode,
                onExitBackward: () =>
                    _logListKey.currentState?.requestTraversalFocus(),
              ),
            ),
          ],
        );
      },
    );
  }

  Future<void> _showCompactDetail(ScanLogSummary log) async {
    setState(() => _selectedId = log.scanId);
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: AppColors.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(context.appTokens.panelRadius),
        ),
      ),
      builder: (context) =>
          FractionallySizedBox(heightFactor: .86, child: _LogDetail(log: log)),
    );
  }
}

class _ActivityToolbar extends StatelessWidget {
  const _ActivityToolbar({
    required this.onRefresh,
    required this.refreshFocusNode,
    required this.loading,
    required this.announceRefreshProgress,
    required this.showRefresh,
    required this.searchController,
    required this.searchFocusNode,
    required this.query,
    required this.onQueryChanged,
    required this.onClearQuery,
    required this.inputFilter,
    required this.onInputFilterChanged,
    required this.dateFilter,
    required this.onDateFilterChanged,
    required this.sortOrder,
    required this.onSortChanged,
    required this.hasActiveFilters,
    required this.onResetFilters,
  });

  final VoidCallback? onRefresh;
  final FocusNode refreshFocusNode;
  final bool loading;
  final bool announceRefreshProgress;
  final bool showRefresh;
  final TextEditingController searchController;
  final FocusNode searchFocusNode;
  final String query;
  final ValueChanged<String> onQueryChanged;
  final VoidCallback onClearQuery;
  final ActivityInputFilter inputFilter;
  final ValueChanged<ActivityInputFilter> onInputFilterChanged;
  final ActivityDateFilter dateFilter;
  final ValueChanged<ActivityDateFilter> onDateFilterChanged;
  final ActivitySortOrder sortOrder;
  final ValueChanged<ActivitySortOrder> onSortChanged;
  final bool hasActiveFilters;
  final VoidCallback onResetFilters;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(
        AppSpacing.x4,
        AppSpacing.x3,
        AppSpacing.x4,
        AppSpacing.x3,
      ),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(bottom: BorderSide(color: AppColors.divider)),
      ),
      child: FocusTraversalGroup(
        policy: OrderedTraversalPolicy(),
        child: Column(
          children: [
            Row(
              children: [
                Text('활동 기록', style: Theme.of(context).textTheme.titleLarge),
                const Spacer(),
                FocusTraversalOrder(
                  order: const NumericFocusOrder(1),
                  child: SizedBox(
                    width: context.appTokens.activitySearchWidth,
                    child: TextField(
                      controller: searchController,
                      focusNode: searchFocusNode,
                      onChanged: onQueryChanged,
                      decoration: InputDecoration(
                        hintText: '상품명 또는 Scan ID',
                        prefixIcon: const Icon(Icons.search_rounded, size: 19),
                        suffixIcon: query.isEmpty
                            ? const AppKeyboardShortcutHint(
                                shortcut: '/',
                                semanticLabel: '검색 단축키: /',
                              )
                            : AppIconActionButton(
                                semanticLabel: '검색어 지우기',
                                tooltip: '검색어 지우기',
                                onPressed: onClearQuery,
                                icon: const Icon(Icons.close_rounded, size: 18),
                              ),
                      ),
                    ),
                  ),
                ),
                if (showRefresh) ...[
                  const SizedBox(width: AppSpacing.x2),
                  AppIconActionButton(
                    focusNode: refreshFocusNode,
                    semanticLabel: '활동 기록 새로고침',
                    tooltip: '${AppActionCopy.refresh} (F5)',
                    onPressed: onRefresh,
                    progressing: loading,
                    announceProgress: announceRefreshProgress,
                    progressTooltip: AppActionCopy.refreshing,
                    progressAnnouncement: AppActionCopy.refreshingAnnouncement,
                    icon: const Icon(Icons.refresh_rounded, size: 20),
                  ),
                ],
              ],
            ),
            const SizedBox(height: AppSpacing.x3),
            Row(
              children: [
                AppFilterGroup<ActivityInputFilter>(
                  label: '입력원',
                  value: inputFilter,
                  options: const [
                    AppFilterOption(ActivityInputFilter.all, '전체'),
                    AppFilterOption(ActivityInputFilter.camera, '카메라'),
                    AppFilterOption(ActivityInputFilter.image, '이미지'),
                  ],
                  onChanged: onInputFilterChanged,
                ),
                const SizedBox(width: AppSpacing.x6),
                AppFilterGroup<ActivityDateFilter>(
                  label: '기간',
                  value: dateFilter,
                  options: const [
                    AppFilterOption(ActivityDateFilter.all, '전체'),
                    AppFilterOption(ActivityDateFilter.today, '오늘'),
                    AppFilterOption(ActivityDateFilter.sevenDays, '7일'),
                    AppFilterOption(ActivityDateFilter.thirtyDays, '30일'),
                  ],
                  onChanged: onDateFilterChanged,
                ),
                if (hasActiveFilters) ...[
                  const SizedBox(width: AppSpacing.x3),
                  TextButton.icon(
                    onPressed: onResetFilters,
                    icon: const Icon(Icons.restart_alt_rounded, size: 18),
                    label: const Text(AppActionCopy.resetAll),
                  ),
                ],
                const Spacer(),
                Text('정렬', style: Theme.of(context).textTheme.bodySmall),
                const SizedBox(width: AppSpacing.x2),
                AppDropdownControl<ActivitySortOrder>(
                  key: const ValueKey('activity-sort-control'),
                  value: sortOrder,
                  semanticLabel: '활동 정렬',
                  items: const [
                    DropdownMenuItem(
                      value: ActivitySortOrder.newest,
                      child: Text('최신순'),
                    ),
                    DropdownMenuItem(
                      value: ActivitySortOrder.oldest,
                      child: Text('오래된순'),
                    ),
                  ],
                  onChanged: (value) {
                    if (value != null) onSortChanged(value);
                  },
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

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
          Expanded(flex: 3, child: Text('확정 시각', style: style)),
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
    final products = summarizeActivityProducts(log.items, query: query);
    final source = log.inputMode == InputMode.camera ? '카메라' : '이미지';
    final reviewSummary = _reviewSummary(log.items);
    return AppSelectableSurface(
      selected: selected,
      inMutuallyExclusiveGroup: true,
      focusNode: focusNode,
      onTap: onTap,
      semanticLabel:
          '${_formatDate(log.confirmedAt)} ${_formatTime(log.confirmedAt)}, '
          '$products, $source, $reviewSummary 활동 기록',
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
                  _formatDate(log.confirmedAt),
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    fontWeight: AppTypography.bold,
                  ),
                ),
                Text(
                  _formatTime(log.confirmedAt),
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
          Expanded(
            flex: 4,
            child: Text(
              products,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                fontWeight: AppTypography.semibold,
              ),
            ),
          ),
          Expanded(flex: 2, child: _SourceLabel(mode: log.inputMode)),
          Expanded(flex: 2, child: _ReviewSummary(items: log.items)),
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
  const _ReviewSummary({required this.items});

  final List<ScanLogItemSummary> items;

  @override
  Widget build(BuildContext context) {
    final modifiedCount = _modifiedItemCount(items);
    final modified = modifiedCount > 0;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(
          modified ? Icons.edit_outlined : Icons.check_circle_outline_rounded,
          size: 17,
          color: modified ? AppColors.attention : AppColors.success,
        ),
        const SizedBox(width: AppSpacing.x2),
        Flexible(
          child: Text(
            modified ? '$modifiedCount개 수정' : '자동 확정',
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

class _LogDetail extends StatelessWidget {
  const _LogDetail({
    required this.log,
    this.disclosureFocusNode,
    this.onExitBackward,
  });

  final ScanLogSummary log;
  final FocusNode? disclosureFocusNode;
  final VoidCallback? onExitBackward;

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent && event is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    if (event.logicalKey == LogicalKeyboardKey.tab &&
        HardwareKeyboard.instance.isShiftPressed &&
        disclosureFocusNode?.hasFocus == true &&
        onExitBackward != null) {
      onExitBackward!();
      return KeyEventResult.handled;
    }
    return KeyEventResult.ignored;
  }

  @override
  Widget build(BuildContext context) {
    return Focus(
      canRequestFocus: false,
      skipTraversal: true,
      onKeyEvent: _handleKeyEvent,
      child: Material(
        color: AppColors.elevated,
        child: ListView(
          key: ValueKey('activity-detail-${log.scanId}'),
          primary: false,
          padding: EdgeInsets.zero,
          children: [
            AppPanelHeader(
              title: '확정 상품',
              subtitle: '${log.items.length}개',
              trailing: const AppStatusBadge(
                label: '저장됨',
                icon: Icons.check_circle_outline_rounded,
                color: AppColors.success,
                backgroundColor: AppColors.successSoft,
              ),
            ),
            ...log.items.map((item) => _LogItem(item: item)),
            AppDisclosure(
              title: '진단 정보',
              description: '스캔·모델·객체 판정 정보',
              icon: Icons.tune_rounded,
              focusNode: disclosureFocusNode,
              children: [
                _DetailLine(
                  label: 'Scan ID',
                  value: log.scanId,
                  selectable: true,
                ),
                _DetailLine(
                  label: '확정 시각',
                  value:
                      '${_formatDate(log.confirmedAt)}  ${_formatTime(log.confirmedAt)}',
                ),
                _DetailLine(
                  label: '입력원',
                  value: log.inputMode == InputMode.camera ? '카메라' : '이미지',
                ),
                _DetailLine(
                  label: '처리시간',
                  value: '${log.processingTimeMs.toStringAsFixed(1)} ms',
                ),
                _DetailLine(
                  label: '모델 버전',
                  value:
                      'Detector ${log.modelVersions.detector ?? '—'}  ·  Classifier ${log.modelVersions.classifier ?? '—'}',
                ),
                const _DiagnosticSectionTitle(label: '객체별 판정'),
                ...log.items.indexed.map(
                  (entry) =>
                      _DiagnosticItem(index: entry.$1 + 1, item: entry.$2),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _DiagnosticSectionTitle extends StatelessWidget {
  const _DiagnosticSectionTitle({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: AppSpacing.x4 + AppSpacing.x1),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: AppColors.ink,
                fontWeight: AppTypography.bold,
              ),
            ),
          ),
          const Expanded(child: Divider()),
        ],
      ),
    );
  }
}

class _DiagnosticItem extends StatelessWidget {
  const _DiagnosticItem({required this.index, required this.item});

  final int index;
  final ScanLogItemSummary item;

  @override
  Widget build(BuildContext context) {
    final confidence = '${(item.confidence * 100).toStringAsFixed(1)}%';
    return Padding(
      padding: const EdgeInsets.only(top: AppSpacing.x3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: context.appTokens.metadataLabelWidth,
            child: Text(
              '$index번 상품',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  activityProductLabel(item),
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    fontWeight: AppTypography.bold,
                  ),
                ),
                Text(
                  '${item.itemId}  ·  $confidence  ·  ${activityConfirmationMethodLabel(item.confirmationMethod)}',
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _DetailLine extends StatelessWidget {
  const _DetailLine({
    required this.label,
    required this.value,
    this.selectable = false,
  });

  final String label;
  final String value;
  final bool selectable;

  @override
  Widget build(BuildContext context) {
    final valueStyle = Theme.of(
      context,
    ).textTheme.bodyMedium?.copyWith(fontWeight: AppTypography.bold);
    return Padding(
      padding: const EdgeInsets.only(top: AppSpacing.x3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: context.appTokens.metadataLabelWidth,
            child: Text(label, style: Theme.of(context).textTheme.bodySmall),
          ),
          Expanded(
            child: selectable
                ? SelectableText(value, style: valueStyle)
                : Text(value, style: valueStyle),
          ),
        ],
      ),
    );
  }
}

class _LogItem extends StatelessWidget {
  const _LogItem({required this.item});

  final ScanLogItemSummary item;

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: BoxConstraints(minHeight: context.appTokens.rowHeight),
      padding: const EdgeInsets.symmetric(
        horizontal: AppSpacing.x4,
        vertical: AppSpacing.x2,
      ),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(bottom: BorderSide(color: AppColors.divider)),
      ),
      child: Row(
        children: [
          const Icon(
            Icons.check_circle_outline_rounded,
            color: AppColors.success,
            size: 19,
          ),
          const SizedBox(width: AppSpacing.x3),
          Expanded(
            child: Text(
              activityProductLabel(item),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ),
          if (item.userModified)
            const AppStatusBadge(
              label: '수정됨',
              icon: Icons.edit_outlined,
              color: AppColors.attention,
              backgroundColor: AppColors.attentionSoft,
            ),
        ],
      ),
    );
  }
}

String _formatDate(DateTime value) {
  final local = value.toLocal();
  return '${local.year}.${local.month.toString().padLeft(2, '0')}.${local.day.toString().padLeft(2, '0')}';
}

String _formatTime(DateTime value) {
  final local = value.toLocal();
  return '${local.hour.toString().padLeft(2, '0')}:${local.minute.toString().padLeft(2, '0')}:${local.second.toString().padLeft(2, '0')}';
}

int _modifiedItemCount(List<ScanLogItemSummary> items) =>
    items.where((item) => item.userModified).length;

String _reviewSummary(List<ScanLogItemSummary> items) {
  final modifiedCount = _modifiedItemCount(items);
  return modifiedCount > 0 ? '$modifiedCount개 수정' : '자동 확정';
}
