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
    required this.reviewFilter,
    required this.onReviewFilterChanged,
    required this.exporting,
    required this.onExportFiltered,
    required this.onExportAll,
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
  final ActivityReviewFilter reviewFilter;
  final ValueChanged<ActivityReviewFilter> onReviewFilterChanged;
  final bool exporting;
  final VoidCallback onExportFiltered;
  final VoidCallback onExportAll;
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
                PopupMenuButton<bool>(
                  key: const ValueKey('activity-export-menu'),
                  enabled: !exporting,
                  tooltip: '검수 기록 ZIP 내보내기',
                  onSelected: (filteredOnly) =>
                      filteredOnly ? onExportFiltered() : onExportAll(),
                  itemBuilder: (context) => const [
                    PopupMenuItem(value: true, child: Text('현재 표시 기록 내보내기')),
                    PopupMenuItem(value: false, child: Text('전체 기록 내보내기')),
                  ],
                  child: Semantics(
                    button: true,
                    enabled: !exporting,
                    label: exporting ? '검수 기록 내보내는 중' : '검수 기록 내보내기',
                    child: ExcludeSemantics(
                      child: Container(
                        constraints: BoxConstraints(
                          minHeight: context.appTokens.actionHeight,
                        ),
                        padding: const EdgeInsets.symmetric(
                          horizontal: AppSpacing.x4,
                        ),
                        decoration: BoxDecoration(
                          border: Border.all(color: AppColors.divider),
                          borderRadius: BorderRadius.circular(
                            context.appTokens.controlRadius,
                          ),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            if (exporting)
                              AppProgressVisual(
                                size: context.appTokens.inlineProgressSize,
                                color: AppColors.muted,
                                strokeWidth: 2,
                              )
                            else
                              const Icon(Icons.ios_share_outlined, size: 18),
                            const SizedBox(width: AppSpacing.x2),
                            Text(exporting ? '내보내는 중' : '내보내기'),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: AppSpacing.x3),
                FocusTraversalOrder(
                  order: const NumericFocusOrder(1),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      SizedBox(
                        width: context.appTokens.activitySearchWidth,
                        child: TextField(
                          controller: searchController,
                          focusNode: searchFocusNode,
                          onChanged: onQueryChanged,
                          decoration: InputDecoration(
                            hintText: '상품명, Scan ID, Worker·작업자 오류 코드',
                            prefixIcon: const Icon(
                              Icons.search_rounded,
                              size: 19,
                            ),
                            suffixIcon: query.isEmpty
                                ? null
                                : AppIconActionButton(
                                    semanticLabel: '검색어 지우기',
                                    tooltip: '검색어 지우기',
                                    onPressed: onClearQuery,
                                    icon: const Icon(
                                      Icons.close_rounded,
                                      size: 18,
                                    ),
                                  ),
                          ),
                        ),
                      ),
                      const SizedBox(width: AppSpacing.x2),
                      const IgnorePointer(
                        child: AppKeyboardShortcutHint(
                          shortcut: '/',
                          semanticLabel: '검색 단축키: /',
                        ),
                      ),
                    ],
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
            LayoutBuilder(
              builder: (context, constraints) {
                final inputAndDate = <Widget>[
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
                ];
                final reviewAndSort = <Widget>[
                  Text('저장 상태', style: Theme.of(context).textTheme.bodySmall),
                  const SizedBox(width: AppSpacing.x2),
                  AppDropdownControl<ActivityReviewFilter>(
                    key: const ValueKey('activity-review-filter'),
                    value: reviewFilter,
                    semanticLabel: '저장 상태 필터',
                    items: const [
                      DropdownMenuItem(
                        value: ActivityReviewFilter.all,
                        child: Text('전체'),
                      ),
                      DropdownMenuItem(
                        value: ActivityReviewFilter.accepted,
                        child: Text('그대로 저장'),
                      ),
                      DropdownMenuItem(
                        value: ActivityReviewFilter.corrected,
                        child: Text('수정 후 저장'),
                      ),
                      DropdownMenuItem(
                        value: ActivityReviewFilter.recaptureAgreed,
                        child: Text('재촬영 저장'),
                      ),
                      DropdownMenuItem(
                        value: ActivityReviewFilter.recaptureUnnecessary,
                        child: Text('재촬영 해제'),
                      ),
                      DropdownMenuItem(
                        value: ActivityReviewFilter.recaptureRequired,
                        child: Text('재촬영으로 변경'),
                      ),
                      DropdownMenuItem(
                        value: ActivityReviewFilter.legacy,
                        child: Text('기존 기록'),
                      ),
                    ],
                    onChanged: (value) {
                      if (value != null) onReviewFilterChanged(value);
                    },
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
                ];

                if (constraints.maxWidth < 1050) {
                  return Column(
                    children: [
                      Row(children: inputAndDate),
                      const SizedBox(height: AppSpacing.x2),
                      Row(children: reviewAndSort),
                    ],
                  );
                }
                return Row(
                  children: [
                    ...inputAndDate,
                    const SizedBox(width: AppSpacing.x6),
                    ...reviewAndSort,
                  ],
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}
