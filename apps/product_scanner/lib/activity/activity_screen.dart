import 'package:flutter/material.dart';

import '../models/scan_models.dart';
import '../services/scan_log_repository.dart';
import '../theme/app_theme.dart';

class ActivityScreen extends StatefulWidget {
  const ActivityScreen({super.key, required this.loadLogs});

  final Future<List<ScanLogSummary>> Function() loadLogs;

  @override
  State<ActivityScreen> createState() => _ActivityScreenState();
}

class _ActivityScreenState extends State<ActivityScreen> {
  List<ScanLogSummary> _logs = const [];
  String? _selectedId;
  String _query = '';
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final logs = await widget.loadLogs();
      if (!mounted) return;
      setState(() {
        _logs = logs;
        _selectedId = logs.any((log) => log.scanId == _selectedId)
            ? _selectedId
            : logs.isEmpty
            ? null
            : logs.first.scanId;
        _loading = false;
      });
    } on Object {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = '로그를 불러오지 못했습니다.';
      });
    }
  }

  List<ScanLogSummary> get _filteredLogs {
    final normalized = _query.trim().toLowerCase();
    if (normalized.isEmpty) return _logs;
    return _logs
        .where((log) {
          return log.scanId.toLowerCase().contains(normalized) ||
              log.items.any(
                (item) => item.productName.toLowerCase().contains(normalized),
              );
        })
        .toList(growable: false);
  }

  ScanLogSummary? get _selectedLog {
    for (final log in _logs) {
      if (log.scanId == _selectedId) return log;
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: AppColors.workspace,
      child: Column(
        children: [
          _ActivityToolbar(
            count: _logs.length,
            onRefresh: _loading ? null : _refresh,
            query: _query,
            onChanged: (value) => setState(() => _query = value),
          ),
          Expanded(child: _body()),
        ],
      ),
    );
  }

  Widget _body() {
    if (_loading) {
      return const Center(
        child: SizedBox(
          width: 22,
          height: 22,
          child: CircularProgressIndicator(strokeWidth: 2),
        ),
      );
    }
    if (_error != null) {
      return _ActivityMessage(
        icon: Icons.error_outline_rounded,
        title: _error!,
        detail: '잠시 후 새로고침해 주세요.',
        action: OutlinedButton.icon(
          onPressed: _refresh,
          icon: const Icon(Icons.refresh_rounded, size: 18),
          label: const Text('새로고침'),
        ),
      );
    }
    final filtered = _filteredLogs;
    if (filtered.isEmpty) {
      return _ActivityMessage(
        icon: _logs.isEmpty
            ? Icons.history_toggle_off_rounded
            : Icons.search_off_rounded,
        title: _logs.isEmpty ? '저장된 활동이 없습니다' : '검색 결과가 없습니다',
        detail: _logs.isEmpty
            ? '상품을 최종 확정하면 이곳에 기록됩니다.'
            : '상품명이나 Scan ID를 다시 확인해 주세요.',
      );
    }
    return LayoutBuilder(
      builder: (context, constraints) {
        if (constraints.maxWidth < 920) {
          return _LogList(
            logs: filtered,
            selectedId: _selectedId,
            onSelected: (log) => _showMobileDetail(log),
          );
        }
        return Row(
          children: [
            Expanded(
              flex: 7,
              child: _LogList(
                logs: filtered,
                selectedId: _selectedId,
                onSelected: (log) => setState(() => _selectedId = log.scanId),
              ),
            ),
            const VerticalDivider(width: 1),
            Expanded(
              flex: 5,
              child: _LogDetail(log: _selectedLog ?? filtered.first),
            ),
          ],
        );
      },
    );
  }

  Future<void> _showMobileDetail(ScanLogSummary log) async {
    setState(() => _selectedId = log.scanId);
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: AppColors.surface,
      builder: (context) =>
          FractionallySizedBox(heightFactor: .82, child: _LogDetail(log: log)),
    );
  }
}

class _ActivityToolbar extends StatelessWidget {
  const _ActivityToolbar({
    required this.count,
    required this.onRefresh,
    required this.query,
    required this.onChanged,
  });

  final int count;
  final VoidCallback? onRefresh;
  final String query;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 52,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(bottom: BorderSide(color: AppColors.divider)),
      ),
      child: Row(
        children: [
          Text('Activity', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(width: 10),
          Text('확정된 스캔 $count건', style: Theme.of(context).textTheme.bodySmall),
          const Spacer(),
          SizedBox(
            width: 300,
            child: TextField(
              onChanged: onChanged,
              decoration: const InputDecoration(
                isDense: true,
                hintText: '상품명 또는 Scan ID 검색',
                prefixIcon: Icon(Icons.search_rounded, size: 17),
              ),
            ),
          ),
          const SizedBox(width: 6),
          IconButton(
            tooltip: '새로고침',
            onPressed: onRefresh,
            icon: const Icon(Icons.refresh_rounded, size: 18),
          ),
        ],
      ),
    );
  }
}

class _LogList extends StatelessWidget {
  const _LogList({
    required this.logs,
    required this.selectedId,
    required this.onSelected,
  });

  final List<ScanLogSummary> logs;
  final String? selectedId;
  final ValueChanged<ScanLogSummary> onSelected;

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: AppColors.surface,
      child: Column(
        children: [
          const _LogTableHeader(),
          Expanded(
            child: ListView.builder(
              primary: false,
              itemCount: logs.length,
              itemBuilder: (context, index) {
                final log = logs[index];
                return _LogRow(
                  log: log,
                  selected: log.scanId == selectedId,
                  onTap: () => onSelected(log),
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
    const style = TextStyle(
      color: AppColors.muted,
      fontSize: 11,
      fontWeight: FontWeight.w600,
      letterSpacing: .3,
    );
    return const SizedBox(
      height: 34,
      child: Row(
        children: [
          SizedBox(width: 18),
          Expanded(flex: 3, child: Text('TIME', style: style)),
          Expanded(flex: 4, child: Text('PRODUCTS', style: style)),
          Expanded(flex: 2, child: Text('SOURCE', style: style)),
          Expanded(flex: 2, child: Text('LATENCY', style: style)),
          SizedBox(width: 28),
        ],
      ),
    );
  }
}

class _LogRow extends StatelessWidget {
  const _LogRow({
    required this.log,
    required this.selected,
    required this.onTap,
  });

  final ScanLogSummary log;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final products = log.items.map((item) => item.productName).join(', ');
    return Material(
      color: selected ? AppColors.primarySoft : AppColors.surface,
      child: InkWell(
        onTap: onTap,
        child: Container(
          height: 52,
          decoration: BoxDecoration(
            border: Border(
              top: const BorderSide(color: AppColors.divider),
              left: BorderSide(
                color: selected ? AppColors.primary : Colors.transparent,
                width: 2,
              ),
            ),
          ),
          child: Row(
            children: [
              const SizedBox(width: 16),
              Expanded(
                flex: 3,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      _formatDate(log.confirmedAt),
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        fontWeight: FontWeight.w600,
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
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ),
              Expanded(flex: 2, child: _SourceLabel(mode: log.inputMode)),
              Expanded(
                flex: 2,
                child: Text(
                  '${log.processingTimeMs.toStringAsFixed(0)} ms',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ),
              const SizedBox(
                width: 28,
                child: Icon(
                  Icons.chevron_right_rounded,
                  size: 19,
                  color: AppColors.subtle,
                ),
              ),
            ],
          ),
        ),
      ),
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
          size: 15,
          color: AppColors.muted,
        ),
        const SizedBox(width: 6),
        Text(
          mode == InputMode.camera ? 'Camera' : 'Image',
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }
}

class _LogDetail extends StatelessWidget {
  const _LogDetail({required this.log});

  final ScanLogSummary log;

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: AppColors.elevated,
      child: ListView(
        primary: false,
        padding: EdgeInsets.zero,
        children: [
          Container(
            height: 48,
            padding: const EdgeInsets.symmetric(horizontal: 16),
            decoration: const BoxDecoration(
              color: AppColors.surface,
              border: Border(bottom: BorderSide(color: AppColors.divider)),
            ),
            child: Row(
              children: [
                const SizedBox(
                  width: 7,
                  height: 7,
                  child: DecoratedBox(
                    decoration: BoxDecoration(
                      color: AppColors.success,
                      shape: BoxShape.circle,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  'Confirmed',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 10),
            child: Column(
              children: [
                _DetailLine(
                  label: 'Scan ID',
                  value: log.scanId,
                  selectable: true,
                ),
                _DetailLine(
                  label: 'Confirmed',
                  value:
                      '${_formatDate(log.confirmedAt)}  ${_formatTime(log.confirmedAt)}',
                ),
                _DetailLine(
                  label: 'Input',
                  value: log.inputMode == InputMode.camera ? 'Camera' : 'Image',
                ),
                _DetailLine(
                  label: 'Processing',
                  value: '${log.processingTimeMs.toStringAsFixed(1)} ms',
                ),
                _DetailLine(
                  label: 'Models',
                  value:
                      'Detector ${log.modelVersions.detector ?? '—'}  ·  Classifier ${log.modelVersions.classifier ?? '—'}',
                ),
              ],
            ),
          ),
          Container(
            height: 34,
            padding: const EdgeInsets.symmetric(horizontal: 16),
            alignment: Alignment.centerLeft,
            decoration: const BoxDecoration(
              border: Border(
                top: BorderSide(color: AppColors.divider),
                bottom: BorderSide(color: AppColors.divider),
              ),
            ),
            child: Text(
              'Products',
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(fontWeight: FontWeight.w600),
            ),
          ),
          ...log.items.map((item) => _LogItem(item: item)),
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
    ).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600);
    return Padding(
      padding: const EdgeInsets.only(bottom: 9),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 88,
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
      height: 52,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(bottom: BorderSide(color: AppColors.divider)),
      ),
      child: Row(
        children: [
          const SizedBox(
            width: 7,
            height: 7,
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: AppColors.success,
                shape: BoxShape.circle,
              ),
            ),
          ),
          const SizedBox(width: 9),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  item.productName,
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                Text(
                  '${item.itemId}  ·  ${(item.confidence * 100).toStringAsFixed(1)}%  ·  ${_methodLabel(item.confirmationMethod)}',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
          if (item.userModified)
            const Text(
              'Edited',
              style: TextStyle(
                color: AppColors.attention,
                fontSize: 11,
                fontWeight: FontWeight.w600,
              ),
            ),
        ],
      ),
    );
  }
}

class _ActivityMessage extends StatelessWidget {
  const _ActivityMessage({
    required this.icon,
    required this.title,
    required this.detail,
    this.action,
  });

  final IconData icon;
  final String title;
  final String detail;
  final Widget? action;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 24, color: AppColors.subtle),
          const SizedBox(height: 10),
          Text(title, style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 3),
          Text(detail, style: Theme.of(context).textTheme.bodySmall),
          if (action != null) ...[const SizedBox(height: 12), action!],
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

String _methodLabel(String value) => switch (value) {
  'AUTO_APPROVED' => 'Auto approved',
  'TOP3_SELECTED' => 'Top-3 selected',
  'SEARCH_SELECTED' => 'Search selected',
  'USER_CORRECTED' => 'Corrected',
  _ => value,
};
