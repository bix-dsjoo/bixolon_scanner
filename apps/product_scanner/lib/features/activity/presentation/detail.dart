part of 'activity_screen.dart';

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
    final recapture = log.isRecapture;
    final contentLabel = activityLogContentLabel(log, query: '');
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
              title: recapture ? '재촬영 기록' : '확정 상품',
              subtitle: recapture ? contentLabel : '${log.items.length}개',
              trailing: recapture
                  ? const AppStatusBadge(
                      label: '재촬영',
                      icon: Icons.center_focus_weak_rounded,
                      color: AppColors.attention,
                      backgroundColor: AppColors.attentionSoft,
                    )
                  : const AppStatusBadge(
                      label: '저장됨',
                      icon: Icons.check_circle_outline_rounded,
                      color: AppColors.success,
                      backgroundColor: AppColors.successSoft,
                    ),
            ),
            _ActivityLogImage(imagePath: log.originalImagePath),
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
                  label: recapture ? '기록 시각' : '확정 시각',
                  value:
                      '${_formatDate(log.recordedAt)}  ${_formatTime(log.recordedAt)}',
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
                if (log.reasonCodes.isNotEmpty)
                  _DetailLine(
                    label: 'Reason code',
                    value: log.reasonCodes.join(', '),
                    selectable: true,
                  ),
                if (log.items.isNotEmpty) ...[
                  const _DiagnosticSectionTitle(label: '객체별 판정'),
                  ...log.items.indexed.map(
                    (entry) =>
                        _DiagnosticItem(index: entry.$1 + 1, item: entry.$2),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _ActivityLogImage extends StatelessWidget {
  const _ActivityLogImage({required this.imagePath, this.compact = false});

  final String? imagePath;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final imagePath = this.imagePath;
    final placeholder = _ActivityImagePlaceholder(compact: compact);
    final image = imagePath == null
        ? placeholder
        : Image.file(
            File(imagePath),
            key: ValueKey(
              'activity-image-$imagePath-${compact ? 'thumb' : 'detail'}',
            ),
            fit: compact ? BoxFit.cover : BoxFit.contain,
            cacheWidth: compact ? 132 : 960,
            filterQuality: FilterQuality.medium,
            errorBuilder: (context, error, stackTrace) => placeholder,
          );

    if (compact) {
      return Semantics(
        image: true,
        label: imagePath == null ? '저장 이미지 없음' : '저장 이미지 썸네일',
        child: ClipRRect(
          borderRadius: BorderRadius.circular(AppSpacing.x2),
          child: SizedBox.square(
            dimension: 44,
            child: ColoredBox(color: AppColors.elevated, child: image),
          ),
        ),
      );
    }

    return Semantics(
      image: true,
      label: imagePath == null ? '저장 이미지 없음' : '저장된 스캔 이미지',
      child: Container(
        key: const ValueKey('activity-detail-image'),
        width: double.infinity,
        height: 240,
        color: AppColors.preview,
        padding: const EdgeInsets.all(AppSpacing.x3),
        child: image,
      ),
    );
  }
}

class _ActivityImagePlaceholder extends StatelessWidget {
  const _ActivityImagePlaceholder({required this.compact});

  final bool compact;

  @override
  Widget build(BuildContext context) {
    if (compact) {
      return const ColoredBox(
        color: AppColors.elevated,
        child: Icon(
          Icons.image_not_supported_outlined,
          size: 18,
          color: AppColors.subtle,
        ),
      );
    }
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(
            Icons.image_not_supported_outlined,
            size: 28,
            color: AppColors.subtle,
          ),
          const SizedBox(height: AppSpacing.x2),
          Text(
            '저장 이미지를 불러올 수 없어요',
            style: Theme.of(
              context,
            ).textTheme.bodySmall?.copyWith(color: AppColors.subtle),
          ),
        ],
      ),
    );
  }
}
