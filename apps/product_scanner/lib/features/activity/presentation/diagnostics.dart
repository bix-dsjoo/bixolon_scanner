part of 'activity_screen.dart';

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
    final reasonLabel = activityItemReasonLabel(item);
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
                if (reasonLabel.isNotEmpty)
                  Text(
                    '$reasonLabel · ${item.reasonCodes.join(', ')}',
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(
                      context,
                    ).textTheme.bodySmall?.copyWith(color: AppColors.muted),
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
          if (item.reasonCodes.contains('DETECTOR_CONTAINED_DUPLICATE'))
            const AppStatusBadge(
              label: '중복 검토',
              icon: Icons.copy_all_outlined,
              color: AppColors.attention,
              backgroundColor: AppColors.attentionSoft,
            )
          else if (item.userModified)
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
