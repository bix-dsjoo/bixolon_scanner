part of 'activity_screen.dart';

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
                        hintText: '상품명, Scan ID 또는 사유 코드',
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
