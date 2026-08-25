part of 'scanner_screen.dart';

class _ReviewFooter extends StatelessWidget {
  const _ReviewFooter({
    required this.controller,
    required this.finalActionFocusNode,
  });

  final ScannerController controller;
  final FocusNode finalActionFocusNode;

  @override
  Widget build(BuildContext context) {
    final submitting = controller.processState == ProcessState.submitting;
    final incomplete = controller.incompleteCount;
    final visibleError = controller.errorMessage;
    final status =
        visibleError ??
        (controller.operatorRequiresRecapture
            ? '재촬영 결과로 저장합니다'
            : incomplete == 0
            ? '${controller.confirmedCount}개 상품 선택 완료'
            : '$incomplete개 상품 선택 필요');
    return AppActionBar(
      child: Row(
        children: [
          if (visibleError != null) ...[
            const Icon(
              Icons.error_outline_rounded,
              color: AppColors.error,
              size: 18,
            ),
            const SizedBox(width: AppSpacing.x2),
          ],
          Expanded(
            child: Semantics(
              container: visibleError != null || incomplete > 0,
              liveRegion: visibleError != null,
              label: status,
              child: Text(
                status,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  fontWeight: AppTypography.bold,
                  color: visibleError == null ? AppColors.ink : AppColors.error,
                ),
              ),
            ),
          ),
          const SizedBox(width: AppSpacing.x3),
          AppProgressActionButton(
            key: const ValueKey('save-operator-review'),
            focusNode: finalActionFocusNode,
            onPressed: controller.canSaveReview ? controller.saveReview : null,
            progressing: submitting,
            progressLabel: AppActionCopy.saving,
            progressAnnouncement: AppActionCopy.savingAnnouncement,
            icon: Icon(
              visibleError != null
                  ? Icons.refresh_rounded
                  : incomplete > 0 && !controller.operatorRequiresRecapture
                  ? Icons.touch_app_outlined
                  : Icons.save_outlined,
              size: 19,
            ),
            label: visibleError != null
                ? AppActionCopy.retrySave
                : incomplete > 0 && !controller.operatorRequiresRecapture
                ? '$incomplete개 상품 선택 필요'
                : '결과 저장',
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
