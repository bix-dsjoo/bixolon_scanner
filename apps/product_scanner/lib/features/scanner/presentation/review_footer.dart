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
    final totalCount = controller.detections.length;
    final remainingCount = totalCount - controller.confirmedCount;
    final submitting = controller.processState == ProcessState.submitting;
    final feedbackState = controller.missedDetectionLogSaveState;
    final feedbackSaving = feedbackState == MissedDetectionLogSaveState.saving;
    final feedbackSaved = feedbackState == MissedDetectionLogSaveState.saved;
    final feedbackFailed = feedbackState == MissedDetectionLogSaveState.error;
    final visibleError =
        controller.errorMessage ?? controller.missedDetectionLogError;
    final feedbackLabel = feedbackSaved
        ? AppActionCopy.missedDetectionLogSaved
        : feedbackFailed
        ? AppActionCopy.retrySave
        : feedbackSaving
        ? AppActionCopy.saving
        : AppActionCopy.saveMissedDetectionLog;
    final feedbackIcon = feedbackSaving
        ? AppProgressVisual(
            size: context.appTokens.inlineProgressSize,
            strokeWidth: 2,
            color: AppColors.muted,
          )
        : Icon(
            feedbackSaved
                ? Icons.check_circle_outline_rounded
                : Icons.report_problem_outlined,
            size: 18,
          );
    final compactFeedbackAction =
        MediaQuery.textScalerOf(context).scale(1) > 1.25;
    return AppActionBar(
      child: Row(
        children: [
          if (visibleError != null) ...[
            const Icon(
              Icons.error_outline_rounded,
              color: AppColors.error,
              size: 17,
            ),
            const SizedBox(width: AppSpacing.x2),
          ],
          Expanded(
            child: Semantics(
              container: visibleError != null,
              excludeSemantics: visibleError != null,
              liveRegion: visibleError != null,
              label: visibleError,
              child: Text(
                visibleError ??
                    (controller.allConfirmed
                        ? '$totalCount개 상품 확인 완료'
                        : '${controller.confirmedCount} / $totalCount 상품 확인 완료'),
                maxLines: visibleError == null ? 1 : 2,
                overflow: TextOverflow.ellipsis,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  fontWeight: AppTypography.bold,
                  color: visibleError == null ? AppColors.ink : AppColors.error,
                ),
              ),
            ),
          ),
          const SizedBox(width: AppSpacing.x3),
          if (compactFeedbackAction)
            IconButton.outlined(
              key: const ValueKey('save-missed-detection-log'),
              tooltip: feedbackLabel,
              onPressed: controller.canSaveMissedDetectionLog
                  ? controller.saveMissedDetectionLog
                  : null,
              icon: feedbackIcon,
            )
          else
            OutlinedButton.icon(
              key: const ValueKey('save-missed-detection-log'),
              onPressed: controller.canSaveMissedDetectionLog
                  ? controller.saveMissedDetectionLog
                  : null,
              icon: feedbackIcon,
              label: Text(feedbackLabel),
            ),
          const SizedBox(width: AppSpacing.x2),
          AppProgressActionButton(
            focusNode: finalActionFocusNode,
            onPressed: controller.allConfirmed && !submitting
                ? controller.submit
                : null,
            progressing: submitting,
            progressLabel: AppActionCopy.saving,
            progressAnnouncement: AppActionCopy.savingAnnouncement,
            icon: Icon(
              controller.errorMessage != null
                  ? Icons.refresh_rounded
                  : controller.allConfirmed
                  ? Icons.check_rounded
                  : Icons.lock_outline_rounded,
              size: 19,
            ),
            label: controller.errorMessage != null
                ? AppActionCopy.retrySave
                : controller.allConfirmed
                ? '$totalCount개 상품 최종 확정'
                : '$remainingCount개 상품 확인 필요',
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
