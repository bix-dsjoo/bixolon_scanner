part of 'activity_screen.dart';

class _LogDetail extends StatelessWidget {
  const _LogDetail({
    required this.log,
    this.resultDisclosureFocusNode,
    this.disclosureFocusNode,
    this.performanceDisclosureFocusNode,
    this.onExitBackward,
  });

  final ScanLogSummary log;
  final FocusNode? resultDisclosureFocusNode;
  final FocusNode? disclosureFocusNode;
  final FocusNode? performanceDisclosureFocusNode;
  final VoidCallback? onExitBackward;

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent && event is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    final firstDisclosureHasFocus = resultDisclosureFocusNode?.hasFocus == true;
    if (event.logicalKey == LogicalKeyboardKey.tab &&
        HardwareKeyboard.instance.isShiftPressed &&
        firstDisclosureHasFocus &&
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
    final resultLabel = activityLogResultLabel(log);
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
              title: '저장 기록',
              subtitle:
                  '$resultLabel · ${recapture ? contentLabel : '${log.items.length}개'}',
              trailing: log.isLegacy
                  ? const AppStatusBadge(
                      label: '기존 기록',
                      icon: Icons.history_rounded,
                      color: AppColors.attention,
                      backgroundColor: AppColors.attentionSoft,
                    )
                  : recapture
                  ? AppStatusBadge(
                      label: resultLabel,
                      icon: Icons.center_focus_weak_rounded,
                      color: AppColors.attention,
                      backgroundColor: AppColors.attentionSoft,
                    )
                  : AppStatusBadge(
                      label: resultLabel,
                      icon: Icons.check_circle_outline_rounded,
                      color: AppColors.success,
                      backgroundColor: AppColors.successSoft,
                    ),
            ),
            _ActivityReviewImage(log: log),
            AppDisclosure(
              key: const ValueKey('activity-product-result'),
              title: recapture ? '저장 결과' : '상품 결과',
              description: recapture
                  ? resultLabel
                  : '${log.items.length}개 상품 · $resultLabel',
              icon: recapture
                  ? Icons.center_focus_weak_rounded
                  : Icons.inventory_2_outlined,
              initiallyExpanded: true,
              focusNode: resultDisclosureFocusNode,
              children: recapture
                  ? [_DetailLine(label: '저장 상태', value: resultLabel)]
                  : [
                      ...log.items.indexed.map(
                        (entry) =>
                            _LogItem(index: entry.$1 + 1, item: entry.$2),
                      ),
                    ],
            ),
            if (log.performance case final performance?)
              AppDisclosure(
                title: '성능 분석',
                description: '촬영부터 결과 화면까지 단계별 시간',
                icon: Icons.speed_rounded,
                focusNode: performanceDisclosureFocusNode,
                children: [_PerformanceDetail(performance: performance)],
              ),
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
                      'Worker ${log.modelVersions.worker ?? '—'}  ·  Detector ${log.modelVersions.detector ?? '—'}  ·  Classifier ${log.modelVersions.classifier ?? '—'}',
                ),
                if (log.reasonCodes.isNotEmpty)
                  _DetailLine(
                    label: 'Reason code',
                    value: log.reasonCodes.join(', '),
                    selectable: true,
                  ),
                if (log.operatorReview case final review?) ...[
                  _DetailLine(
                    label: '자동 오류 기록',
                    value: review.issueCodes.isEmpty
                        ? '없음'
                        : review.issueCodes
                              .map(operatorIssueCodeValue)
                              .join(', '),
                    selectable: true,
                  ),
                  if (review.note case final note? when note.isNotEmpty)
                    _DetailLine(label: '검수 메모', value: note),
                ],
                if (log.items.isNotEmpty) ...[
                  const _DiagnosticSectionTitle(label: '객체별 모델 진단'),
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

class _ActivityReviewImage extends StatelessWidget {
  const _ActivityReviewImage({required this.log});

  final ScanLogSummary log;

  @override
  Widget build(BuildContext context) {
    final imagePath = log.originalImagePath;
    final imageWidth = log.performance?.imageWidth ?? 0;
    final imageHeight = log.performance?.imageHeight ?? 0;
    return Semantics(
      image: true,
      label: imagePath == null ? '저장 이미지 없음' : '저장된 이미지와 최종 상품 박스',
      child: Container(
        key: const ValueKey('activity-detail-image'),
        width: double.infinity,
        height: 280,
        color: AppColors.preview,
        padding: const EdgeInsets.all(AppSpacing.x3),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final viewport = Size(constraints.maxWidth, constraints.maxHeight);
            final validSize = imageWidth > 0 && imageHeight > 0;
            final imageSize = validSize
                ? Size(imageWidth.toDouble(), imageHeight.toDouble())
                : viewport;
            final fitted = applyBoxFit(BoxFit.contain, imageSize, viewport);
            final imageRect = Alignment.center.inscribe(
              fitted.destination,
              Offset.zero & viewport,
            );
            final placeholder = const _ActivityImagePlaceholder(compact: false);
            return Stack(
              children: [
                Positioned.fill(
                  child: imagePath == null
                      ? placeholder
                      : Image.file(
                          File(imagePath),
                          fit: BoxFit.contain,
                          cacheWidth: 960,
                          filterQuality: FilterQuality.medium,
                          errorBuilder: (_, _, _) => placeholder,
                        ),
                ),
                if (validSize)
                  Positioned.fill(
                    child: IgnorePointer(
                      child: CustomPaint(
                        painter: _ActivityOverlayPainter(
                          items: log.items,
                          imageRect: imageRect,
                          imageSize: imageSize,
                        ),
                      ),
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _ActivityOverlayPainter extends CustomPainter {
  const _ActivityOverlayPainter({
    required this.items,
    required this.imageRect,
    required this.imageSize,
  });

  final List<ScanLogItemSummary> items;
  final Rect imageRect;
  final Size imageSize;

  @override
  void paint(Canvas canvas, Size size) {
    for (final entry in items.indexed) {
      final item = entry.$2;
      if (item.disposition != OperatorObjectDisposition.remove &&
          item.finalBbox != null) {
        final rect = _activityBboxRect(item.finalBbox!);
        final color = _activityItemStatusColor(item);
        canvas.drawRect(
          rect,
          Paint()
            ..color = color
            ..strokeWidth = 2
            ..style = PaintingStyle.stroke,
        );
        _drawLabel(canvas, rect, '${entry.$1 + 1}', color);
      }
    }
  }

  Rect _activityBboxRect(BoundingBox bbox) => Rect.fromLTWH(
    imageRect.left + bbox.x / imageSize.width * imageRect.width,
    imageRect.top + bbox.y / imageSize.height * imageRect.height,
    bbox.width / imageSize.width * imageRect.width,
    bbox.height / imageSize.height * imageRect.height,
  );

  void _drawLabel(Canvas canvas, Rect rect, String value, Color color) {
    final painter = TextPainter(
      text: TextSpan(
        text: value,
        style: TextStyle(
          color: color == AppColors.surface ? AppColors.ink : Colors.white,
          fontSize: AppTypography.captionSize,
          fontWeight: AppTypography.bold,
        ),
      ),
      maxLines: 1,
      ellipsis: '…',
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: rect.width.clamp(48, 180));
    final origin = Offset(
      rect.left,
      (rect.top - painter.height - 6).clamp(0, imageRect.bottom),
    );
    canvas.drawRect(
      origin & Size(painter.width + 10, painter.height + 6),
      Paint()..color = color,
    );
    painter.paint(canvas, origin + const Offset(5, 3));
  }

  @override
  bool shouldRepaint(covariant _ActivityOverlayPainter oldDelegate) =>
      oldDelegate.items != items || oldDelegate.imageRect != imageRect;
}

Color _activityItemStatusColor(ScanLogItemSummary item) {
  return switch (item.resultStatus) {
    ItemStatus.approved => AppColors.success,
    ItemStatus.unknown => AppColors.attention,
    ItemStatus.segmentRecapture => AppColors.error,
  };
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
