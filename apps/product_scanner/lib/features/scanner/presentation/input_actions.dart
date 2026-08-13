part of 'scanner_screen.dart';

class _InputActionBar extends StatelessWidget {
  const _InputActionBar({
    required this.controller,
    required this.onReset,
    required this.onChooseImage,
    required this.onReturnToCamera,
    required this.onPrepareRecapture,
    required this.nextImageActionFocusNode,
    required this.cameraPrimaryActionFocusNode,
  });

  final ScannerController controller;
  final VoidCallback onReset;
  final VoidCallback onChooseImage;
  final VoidCallback onReturnToCamera;
  final VoidCallback onPrepareRecapture;
  final FocusNode nextImageActionFocusNode;
  final FocusNode cameraPrimaryActionFocusNode;

  @override
  Widget build(BuildContext context) {
    final busy = controller.isBusy;
    final hasImage = controller.imageBytes != null;
    final capturing = controller.processState == ProcessState.capturing;
    final analyzing = controller.processState == ProcessState.analyzing;
    final reviewingSuccess =
        controller.processState == ProcessState.reviewing &&
        !controller.isRecapture;
    if (controller.processState == ProcessState.submitting) {
      return AppActionBar(
        child: Row(
          children: [
            const Icon(
              Icons.lock_clock_outlined,
              size: 18,
              color: AppColors.muted,
            ),
            const SizedBox(width: AppSpacing.x2),
            Text(
              '저장이 끝나면 다음 이미지를 준비할 수 있어요',
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(color: AppColors.muted),
            ),
          ],
        ),
      );
    }
    if (capturing || analyzing) {
      return AppActionBar(
        child: Row(
          children: [
            const Spacer(),
            AppProgressActionButton(
              focusNode: capturing ? cameraPrimaryActionFocusNode : null,
              onPressed: null,
              progressing: true,
              progressLabel: capturing
                  ? AppActionCopy.capturing
                  : AppActionCopy.analyzing,
              progressAnnouncement: capturing
                  ? AppActionCopy.capturingAnnouncement
                  : AppActionCopy.analyzingAnnouncement,
              icon: Icon(
                controller.inputMode == InputMode.image
                    ? Icons.auto_awesome_outlined
                    : Icons.camera_alt_outlined,
                size: 18,
              ),
              label: capturing ? AppActionCopy.capture : AppActionCopy.analyze,
            ),
          ],
        ),
      );
    }
    return AppActionBar(
      child: Row(
        children: [
          if (controller.inputMode == InputMode.image && !hasImage) ...[
            _ReturnToCameraButton(
              onPressed: controller.canChooseImage ? onReturnToCamera : null,
            ),
            const Spacer(),
            Tooltip(
              message: '${AppActionCopy.chooseImage} (Ctrl+O)',
              child: FilledButton.icon(
                focusNode: nextImageActionFocusNode,
                onPressed: controller.canChooseImage ? onChooseImage : null,
                icon: const Icon(Icons.image_outlined, size: 18),
                label: const Text(AppActionCopy.chooseImage),
              ),
            ),
          ] else if (controller.hasActiveCameraIssue) ...[
            Tooltip(
              message: '${AppActionCopy.chooseImage} (Ctrl+O)',
              child: OutlinedButton.icon(
                onPressed: controller.canChooseImage ? onChooseImage : null,
                icon: const Icon(Icons.image_outlined, size: 18),
                label: const Text(AppActionCopy.chooseImage),
              ),
            ),
            const Spacer(),
            FilledButton.icon(
              onPressed: busy || controller.cameraInitializing
                  ? null
                  : controller.reconnectCamera,
              icon: const Icon(Icons.videocam_outlined, size: 18),
              label: const Text(AppActionCopy.reconnect),
            ),
          ] else if (controller.isCameraCheckActive) ...[
            Tooltip(
              message: '${AppActionCopy.chooseImage} (Ctrl+O)',
              child: OutlinedButton.icon(
                onPressed: controller.canChooseImage ? onChooseImage : null,
                icon: const Icon(Icons.image_outlined, size: 18),
                label: const Text(AppActionCopy.chooseImage),
              ),
            ),
            const Spacer(),
            AppProgressActionButton(
              onPressed: null,
              progressing: true,
              progressLabel: AppActionCopy.checkingConnection,
              progressAnnouncement:
                  AppActionCopy.checkingConnectionAnnouncement,
              icon: const Icon(Icons.videocam_outlined, size: 18),
              label: AppActionCopy.reconnect,
            ),
          ] else if (reviewingSuccess) ...[
            if (controller.inputMode == InputMode.image) ...[
              _ReturnToCameraButton(
                onPressed: controller.canChooseImage ? onReturnToCamera : null,
              ),
              const SizedBox(width: AppSpacing.x2),
            ],
            Tooltip(
              message: controller.inputMode == InputMode.image
                  ? '${AppActionCopy.chooseAnotherImage} (Ctrl+O)'
                  : '현재 검수를 버리고 ${AppActionCopy.recapture}',
              child: OutlinedButton.icon(
                onPressed: !controller.canChooseImage
                    ? null
                    : controller.inputMode == InputMode.image
                    ? onChooseImage
                    : onReset,
                icon: Icon(
                  controller.inputMode == InputMode.image
                      ? Icons.image_outlined
                      : Icons.refresh_rounded,
                  size: 18,
                ),
                label: Text(
                  controller.inputMode == InputMode.image
                      ? AppActionCopy.chooseAnotherImage
                      : AppActionCopy.recapture,
                ),
              ),
            ),
          ] else if (controller.isRecapture &&
              controller.inputMode == InputMode.camera) ...[
            Tooltip(
              message: '${AppActionCopy.chooseImage} (Ctrl+O)',
              child: OutlinedButton.icon(
                onPressed: controller.canChooseImage ? onChooseImage : null,
                icon: const Icon(Icons.image_outlined, size: 18),
                label: const Text(AppActionCopy.chooseImage),
              ),
            ),
            const Spacer(),
            _RecaptureLogButton(controller: controller),
            const SizedBox(width: AppSpacing.x2),
            FilledButton.icon(
              focusNode: cameraPrimaryActionFocusNode,
              onPressed: controller.canChooseImage ? onPrepareRecapture : null,
              icon: const Icon(Icons.videocam_outlined, size: 18),
              label: const Text(AppActionCopy.returnToCapture),
            ),
          ] else if (controller.isRecapture &&
              controller.inputMode == InputMode.image) ...[
            _ReturnToCameraButton(
              onPressed: controller.canChooseImage ? onReturnToCamera : null,
            ),
            const Spacer(),
            _RecaptureLogButton(controller: controller),
            const SizedBox(width: AppSpacing.x2),
            FilledButton.icon(
              onPressed: controller.canChooseImage ? onChooseImage : null,
              icon: const Icon(Icons.image_outlined, size: 18),
              label: const Text(AppActionCopy.chooseAnotherImage),
            ),
          ] else if (controller.processState == ProcessState.error &&
              controller.errorRecovery ==
                  ScannerErrorRecovery.replaceInput) ...[
            if (controller.inputMode == InputMode.image)
              _ReturnToCameraButton(
                onPressed: controller.canChooseImage ? onReturnToCamera : null,
              ),
            const Spacer(),
            FilledButton.icon(
              onPressed: !controller.canChooseImage
                  ? null
                  : controller.inputMode == InputMode.image
                  ? onChooseImage
                  : onPrepareRecapture,
              focusNode: controller.inputMode == InputMode.camera
                  ? cameraPrimaryActionFocusNode
                  : null,
              icon: Icon(
                controller.inputMode == InputMode.image
                    ? Icons.image_outlined
                    : Icons.videocam_outlined,
                size: 18,
              ),
              label: Text(
                controller.inputMode == InputMode.image
                    ? AppActionCopy.chooseAnotherImage
                    : AppActionCopy.returnToCapture,
              ),
            ),
          ] else ...[
            if (controller.inputMode == InputMode.image) ...[
              _ReturnToCameraButton(
                onPressed: controller.canChooseImage ? onReturnToCamera : null,
              ),
              const SizedBox(width: AppSpacing.x2),
            ],
            Tooltip(
              message: '${AppActionCopy.chooseImage} (Ctrl+O)',
              child: OutlinedButton.icon(
                onPressed: controller.canChooseImage ? onChooseImage : null,
                icon: const Icon(Icons.image_outlined, size: 18),
                label: Text(
                  controller.inputMode == InputMode.image && hasImage
                      ? AppActionCopy.chooseAnotherImage
                      : AppActionCopy.chooseImage,
                ),
              ),
            ),
            const Spacer(),
            if (controller.processState == ProcessState.error && hasImage)
              FilledButton.icon(
                onPressed: busy ? null : controller.analyze,
                icon: const Icon(Icons.refresh_rounded, size: 18),
                label: const Text(AppActionCopy.reanalyze),
              )
            else if (controller.inputMode == InputMode.image && hasImage)
              AppProgressActionButton(
                onPressed: busy ? null : controller.analyze,
                progressing: analyzing,
                progressLabel: AppActionCopy.analyzing,
                progressAnnouncement: AppActionCopy.analyzingAnnouncement,
                icon: const Icon(Icons.auto_awesome_outlined, size: 18),
                label: AppActionCopy.analyze,
              )
            else if (!controller.isCameraReady ||
                controller.hasActiveCameraIssue)
              FilledButton.icon(
                onPressed: busy || controller.cameraInitializing
                    ? null
                    : controller.reconnectCamera,
                icon: const Icon(Icons.videocam_outlined, size: 18),
                label: const Text(AppActionCopy.reconnect),
              )
            else
              AppProgressActionButton(
                focusNode: cameraPrimaryActionFocusNode,
                onPressed: busy ? null : controller.captureAndAnalyze,
                progressing: false,
                progressLabel: AppActionCopy.capturing,
                progressAnnouncement: AppActionCopy.capturingAnnouncement,
                icon: const Icon(Icons.camera_alt_outlined, size: 18),
                label: AppActionCopy.capture,
              ),
          ],
        ],
      ),
    );
  }
}

class _ReturnToCameraButton extends StatelessWidget {
  const _ReturnToCameraButton({required this.onPressed});

  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: AppActionCopy.returnToCamera,
      child: OutlinedButton.icon(
        onPressed: onPressed,
        icon: const Icon(Icons.videocam_outlined, size: 18),
        label: const Text(AppActionCopy.returnToCamera),
      ),
    );
  }
}

class _RecaptureLogButton extends StatelessWidget {
  const _RecaptureLogButton({required this.controller});

  final ScannerController controller;

  @override
  Widget build(BuildContext context) {
    final state = controller.recaptureLogSaveState;
    final saving = state == RecaptureLogSaveState.saving;
    final saved = state == RecaptureLogSaveState.saved;
    final failed = state == RecaptureLogSaveState.error;
    final label = saved
        ? AppActionCopy.recaptureLogSaved
        : failed
        ? AppActionCopy.retrySave
        : saving
        ? AppActionCopy.saving
        : AppActionCopy.saveRecaptureLog;
    final button = OutlinedButton.icon(
      key: const ValueKey('save-recapture-log'),
      onPressed: saving || saved ? null : controller.saveRecaptureLog,
      icon: saving
          ? AppProgressVisual(
              size: context.appTokens.inlineProgressSize,
              strokeWidth: 2,
              color: AppColors.muted,
            )
          : Icon(
              saved
                  ? Icons.check_circle_outline_rounded
                  : Icons.save_alt_rounded,
              size: 18,
            ),
      label: Text(label),
    );
    if (!saving) {
      return Tooltip(message: label, child: button);
    }
    return Semantics(
      container: true,
      excludeSemantics: true,
      liveRegion: true,
      button: true,
      enabled: false,
      label: AppActionCopy.savingAnnouncement,
      child: button,
    );
  }
}
