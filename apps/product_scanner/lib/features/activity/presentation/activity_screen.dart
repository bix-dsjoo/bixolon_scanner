import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../../core/design_system/components.dart';
import '../../../core/design_system/copy.dart';
import '../../../core/design_system/theme.dart';
import '../../../core/design_system/tokens.dart';
import '../../../shared/models/scan_models.dart';
import '../../../shared/logging/scan_log_repository.dart';
import '../domain/activity_filters.dart';
import 'activity_presentation.dart';

part 'toolbar.dart';
part 'log_list.dart';
part 'detail.dart';
part 'diagnostics.dart';

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
