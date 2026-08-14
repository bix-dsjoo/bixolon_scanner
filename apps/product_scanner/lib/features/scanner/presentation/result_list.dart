part of 'scanner_screen.dart';

class _ResultPanel extends StatefulWidget {
  const _ResultPanel({super.key, required this.controller});

  final ScannerController controller;

  @override
  State<_ResultPanel> createState() => _ResultPanelState();
}

class _ResultPanelState extends State<_ResultPanel> {
  final GlobalKey<_DetectionListState> _detectionListKey =
      GlobalKey<_DetectionListState>();
  final FocusNode _firstChoiceFocusNode = FocusNode(
    debugLabel: 'first-product-choice',
  );
  final FocusNode _finalActionFocusNode = FocusNode(
    debugLabel: 'final-confirmation-action',
  );
  final FocusNode _searchActionFocusNode = FocusNode(
    debugLabel: 'product-search-action',
  );

  ScannerController get controller => widget.controller;

  @override
  void dispose() {
    _firstChoiceFocusNode.dispose();
    _finalActionFocusNode.dispose();
    _searchActionFocusNode.dispose();
    super.dispose();
  }

  void _continueKeyboardReview() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final target = controller.allConfirmed
          ? _finalActionFocusNode
          : _firstChoiceFocusNode;
      if (target.canRequestFocus) target.requestFocus();
    });
  }

  void closeSearchFromKeyboard() {
    controller.hideSearch();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || !_searchActionFocusNode.canRequestFocus) return;
      _searchActionFocusNode.requestFocus();
    });
  }

  @override
  Widget build(BuildContext context) {
    final selectedDetection = controller.selectedDetection;
    final showReviewWorkspace =
        controller.processState == ProcessState.reviewing &&
        !controller.isRecapture &&
        !controller.hasActiveCameraIssue &&
        !controller.isCameraCheckActive &&
        selectedDetection != null;
    return ColoredBox(
      color: AppColors.surface,
      child: Column(
        children: [
          if (controller.processState == ProcessState.reviewing &&
              controller.hasResults &&
              controller.allConfirmed)
            AppStateAnnouncement(message: _reviewCompletionAnnouncement()),
          _ResultHeader(controller: controller),
          Expanded(
            child: showReviewWorkspace
                ? _ReviewWorkspace(
                    controller: controller,
                    detection: selectedDetection,
                    detectionListKey: _detectionListKey,
                    firstChoiceFocusNode: _firstChoiceFocusNode,
                    searchActionFocusNode: _searchActionFocusNode,
                    onKeyboardChoiceConfirmed: _continueKeyboardReview,
                    onKeyboardSearchClosed: closeSearchFromKeyboard,
                  )
                : _resultBody(),
          ),
          if ((controller.processState == ProcessState.reviewing ||
                  controller.processState == ProcessState.submitting) &&
              !controller.isRecapture &&
              !controller.hasActiveCameraIssue &&
              !controller.isCameraCheckActive &&
              controller.hasResults)
            _ReviewFooter(
              controller: controller,
              finalActionFocusNode: _finalActionFocusNode,
            ),
        ],
      ),
    );
  }

  String _reviewCompletionAnnouncement() {
    final total = controller.detections.length;
    return '검수 상태. $total개 상품 확인 완료. 최종 확정할 수 있어요.';
  }

  Widget _resultBody() {
    if (controller.processState == ProcessState.capturing) {
      return const _PanelMessage(
        icon: Icons.camera_alt_outlined,
        title: '촬영 중',
        detail: '카메라 촬영이 끝나면 자동으로 분석해요.',
      );
    }
    if (controller.processState == ProcessState.analyzing) {
      return const _PanelMessage(
        icon: Icons.hourglass_top_rounded,
        title: '상품을 찾고 있어요',
        detail: '이미지에서 상품 위치와 종류를 확인하고 있습니다.',
      );
    }
    if (controller.hasActiveCameraIssue) {
      return _PanelMessage(
        icon: Icons.videocam_off_outlined,
        title: controller.cameraIssueTitle,
        detail: controller.cameraMessage!,
        tone: AppColors.attention,
        announce: true,
      );
    }
    if (controller.isCameraCheckActive) {
      return const _PanelMessage(
        icon: Icons.sync_rounded,
        title: '카메라를 준비하고 있어요',
        detail: '연결 상태를 확인하고 있습니다.',
      );
    }
    if (controller.processState == ProcessState.error) {
      return _PanelMessage(
        icon: controller.errorRecovery == ScannerErrorRecovery.replaceInput
            ? Icons.broken_image_outlined
            : Icons.cloud_off_outlined,
        title: controller.errorRecovery == ScannerErrorRecovery.replaceInput
            ? '이미지를 분석할 수 없어요'
            : '분석하지 못했어요',
        detail: controller.errorMessage ?? '잠시 후 다시 분석해 주세요.',
        tone: AppColors.error,
        announce: true,
      );
    }
    if (controller.isRecapture) {
      final saveError = controller.recaptureLogError;
      return _PanelMessage(
        icon: Icons.center_focus_weak_rounded,
        title: controller.recaptureTitle,
        detail: saveError == null
            ? controller.recaptureDetail
            : '${controller.recaptureDetail}\n$saveError',
        tone: AppColors.attention,
        announce:
            saveError != null ||
            controller.recaptureLogSaveState == RecaptureLogSaveState.idle,
      );
    }
    if (!controller.hasResults) {
      if (controller.imageBytes != null) {
        return const _PanelMessage(
          icon: Icons.auto_awesome_outlined,
          title: '이미지 분석 준비가 됐어요',
          detail: '분석하기를 누르면 이미지에서 상품을 찾아요.',
        );
      }
      if (controller.inputMode == InputMode.image) {
        return const _PanelMessage(
          icon: Icons.image_outlined,
          title: '다음 이미지를 선택해 주세요',
          detail: '이미지를 선택하면 분석 준비 상태로 이동해요.',
        );
      }
      if (controller.isCameraReady) {
        return const _PanelMessage(
          icon: Icons.photo_camera_outlined,
          title: '상품을 촬영해 주세요',
          detail: '화면 안에 상품이 모두 보이면 촬영하기를 눌러주세요.',
        );
      }
      return const _PanelMessage(
        icon: Icons.videocam_off_outlined,
        title: '카메라를 연결해 주세요',
        detail: '다시 연결하거나 이미지 파일을 선택할 수 있어요.',
      );
    }
    return _DetectionList(controller: controller);
  }
}

class _ReviewWorkspace extends StatelessWidget {
  const _ReviewWorkspace({
    required this.controller,
    required this.detection,
    required this.detectionListKey,
    required this.firstChoiceFocusNode,
    required this.searchActionFocusNode,
    required this.onKeyboardChoiceConfirmed,
    required this.onKeyboardSearchClosed,
  });

  final ScannerController controller;
  final ReviewDetection detection;
  final GlobalKey<_DetectionListState> detectionListKey;
  final FocusNode firstChoiceFocusNode;
  final FocusNode searchActionFocusNode;
  final VoidCallback onKeyboardChoiceConfirmed;
  final VoidCallback onKeyboardSearchClosed;

  bool get _usesCandidatePicker =>
      controller.searchItemId != detection.source.itemId &&
      detection.source.top3.isNotEmpty;

  @override
  Widget build(BuildContext context) {
    final tokens = context.appTokens;
    return LayoutBuilder(
      builder: (context, constraints) {
        final dividerCount = (controller.detections.length - 1).clamp(
          0,
          controller.detections.length,
        );
        final desiredListHeight =
            controller.detections.length * tokens.rowHeight + dividerCount;
        final maxListHeight =
            (constraints.maxHeight - tokens.reviewInspectorReservedHeight)
                .clamp(tokens.rowHeight, constraints.maxHeight)
                .toDouble();
        final listHeight = desiredListHeight
            .clamp(tokens.rowHeight, maxListHeight)
            .toDouble();

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SizedBox(
              key: const ValueKey('review-object-list-frame'),
              height: listHeight,
              child: _DetectionList(
                key: detectionListKey,
                controller: controller,
                onExitForward: _usesCandidatePicker
                    ? firstChoiceFocusNode.requestFocus
                    : null,
              ),
            ),
            Flexible(
              fit: FlexFit.loose,
              child: _ReviewInspector(
                controller: controller,
                detection: detection,
                firstChoiceFocusNode: firstChoiceFocusNode,
                searchActionFocusNode: searchActionFocusNode,
                onCandidateExitBackward: () =>
                    detectionListKey.currentState?.requestSelectedRowFocus(),
                onKeyboardChoiceConfirmed: onKeyboardChoiceConfirmed,
                onKeyboardSearchClosed: onKeyboardSearchClosed,
              ),
            ),
          ],
        );
      },
    );
  }
}

class _ResultHeader extends StatelessWidget {
  const _ResultHeader({required this.controller});

  final ScannerController controller;

  @override
  Widget build(BuildContext context) {
    final title = controller.hasActiveCameraIssue
        ? '카메라 확인 필요'
        : controller.isCameraCheckActive
        ? '카메라 확인 중'
        : controller.hasResults
        ? controller.allConfirmed
              ? '검수 완료'
              : '상품 검수'
        : controller.isRecapture
        ? '재촬영 필요'
        : controller.processState == ProcessState.error
        ? '분석 오류'
        : controller.processState == ProcessState.capturing
        ? '촬영 중'
        : controller.processState == ProcessState.analyzing
        ? '분석 중'
        : controller.imageBytes != null
        ? '분석 준비'
        : controller.inputMode == InputMode.image
        ? '이미지 선택'
        : controller.isCameraReady
        ? '촬영 준비'
        : '입력 준비';
    final response = controller.response;
    final analysisTime = response == null || response.status == ScanStatus.error
        ? null
        : '분석 ${response.processingTimeMs.toStringAsFixed(1)} ms';
    final subtitle = controller.hasResults
        ? '${controller.confirmedCount}/${controller.detections.length}개 확인'
              '${analysisTime == null ? '' : ' · $analysisTime'}'
        : controller.isRecapture
        ? analysisTime
        : null;
    return AppPanelHeader(
      key: const ValueKey('scan-result-header'),
      title: title,
      subtitle: subtitle,
      trailing:
          controller.selectedIndex >= 0 && controller.detections.length > 1
          ? AppStepNavigator(
              current: controller.selectedIndex + 1,
              total: controller.detections.length,
              semanticUnit: '상품',
              previousTooltip: '이전 상품 (↑)',
              nextTooltip: '다음 상품 (↓)',
              onPrevious: controller.isBusy
                  ? null
                  : controller.selectPreviousDetection,
              onNext: controller.isBusy ? null : controller.selectNextDetection,
            )
          : null,
    );
  }
}

class _PanelMessage extends StatelessWidget {
  const _PanelMessage({
    required this.icon,
    required this.title,
    required this.detail,
    this.tone = AppColors.muted,
    this.announce = false,
  });

  final IconData icon;
  final String title;
  final String detail;
  final Color tone;
  final bool announce;

  @override
  Widget build(BuildContext context) {
    return AppEmptyState(
      icon: icon,
      title: title,
      detail: detail,
      tone: tone,
      announcement: announce ? '$title. $detail' : null,
    );
  }
}

class _DetectionList extends StatefulWidget {
  const _DetectionList({
    super.key,
    required this.controller,
    this.onExitForward,
  });

  final ScannerController controller;
  final VoidCallback? onExitForward;

  @override
  State<_DetectionList> createState() => _DetectionListState();
}

class _DetectionListState extends State<_DetectionList> {
  final ScrollController _scrollController = ScrollController();
  final Map<String, FocusNode> _rowFocusNodes = <String, FocusNode>{};
  String? _lastSelectedItemId;

  @override
  void initState() {
    super.initState();
    _lastSelectedItemId = widget.controller.selectedItemId;
    _revealSelectedRow();
  }

  @override
  void didUpdateWidget(covariant _DetectionList oldWidget) {
    super.didUpdateWidget(oldWidget);
    _disposeUnusedRowFocusNodes();
    _syncRowTraversalState();
    final selectedItemId = widget.controller.selectedItemId;
    if (_lastSelectedItemId == selectedItemId) return;
    _lastSelectedItemId = selectedItemId;
    _revealSelectedRow();
  }

  FocusNode _rowFocusNode(String itemId) {
    final focusNode = _rowFocusNodes.putIfAbsent(
      itemId,
      () => FocusNode(debugLabel: 'detection-row-$itemId'),
    );
    focusNode.skipTraversal =
        widget.controller.isBusy || itemId != widget.controller.selectedItemId;
    return focusNode;
  }

  void _syncRowTraversalState() {
    final selectedItemId = widget.controller.selectedItemId;
    for (final entry in _rowFocusNodes.entries) {
      entry.value.skipTraversal =
          widget.controller.isBusy || entry.key != selectedItemId;
    }
  }

  void _disposeUnusedRowFocusNodes() {
    final activeItemIds = widget.controller.detections
        .map((detection) => detection.source.itemId)
        .toSet();
    final removed = <FocusNode>[];
    _rowFocusNodes.removeWhere((itemId, focusNode) {
      if (activeItemIds.contains(itemId)) return false;
      removed.add(focusNode);
      return true;
    });
    if (removed.isEmpty) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      for (final focusNode in removed) {
        focusNode.dispose();
      }
    });
  }

  KeyEventResult _handleRowNavigation(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent && event is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    final rowFocused = _rowFocusNodes.values.any(
      (focusNode) => focusNode.hasFocus,
    );
    if (!rowFocused || widget.controller.isBusy) {
      return KeyEventResult.ignored;
    }
    if (event.logicalKey == LogicalKeyboardKey.tab &&
        !HardwareKeyboard.instance.isShiftPressed &&
        widget.onExitForward != null) {
      widget.onExitForward!();
      return KeyEventResult.handled;
    }
    if (event.logicalKey == LogicalKeyboardKey.arrowUp) {
      widget.controller.selectPreviousDetection();
    } else if (event.logicalKey == LogicalKeyboardKey.arrowDown) {
      widget.controller.selectNextDetection();
    } else {
      return KeyEventResult.ignored;
    }
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final selectedItemId = widget.controller.selectedItemId;
      if (selectedItemId == null) return;
      _rowFocusNode(selectedItemId).requestFocus();
    });
    return KeyEventResult.handled;
  }

  void requestSelectedRowFocus() {
    final selectedItemId = widget.controller.selectedItemId;
    if (selectedItemId == null) return;
    _rowFocusNode(selectedItemId).requestFocus();
    _revealSelectedRow();
  }

  void _revealSelectedRow() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || !_scrollController.hasClients) return;
      final selectedItemId = widget.controller.selectedItemId;
      final index = widget.controller.detections.indexWhere(
        (detection) => detection.source.itemId == selectedItemId,
      );
      if (index < 0) return;

      final position = _scrollController.position;
      final rowExtent = context.appTokens.rowHeight + 1;
      final centeredOffset =
          index * rowExtent -
          (position.viewportDimension - context.appTokens.rowHeight) / 2;
      final target = centeredOffset.clamp(
        position.minScrollExtent,
        position.maxScrollExtent,
      );
      if ((position.pixels - target).abs() < 1) return;

      if (MediaQuery.disableAnimationsOf(context)) {
        _scrollController.jumpTo(target);
      } else {
        _scrollController.animateTo(
          target,
          duration: context.appTokens.motionStandard,
          curve: AppMotion.interactionCurve,
        );
      }
    });
  }

  @override
  void dispose() {
    _scrollController.dispose();
    for (final focusNode in _rowFocusNodes.values) {
      focusNode.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Focus(
      canRequestFocus: false,
      skipTraversal: true,
      onKeyEvent: _handleRowNavigation,
      child: ListView.separated(
        key: const ValueKey('detection-list'),
        controller: _scrollController,
        padding: EdgeInsets.zero,
        itemCount: widget.controller.detections.length,
        separatorBuilder: (_, _) => const Divider(height: 1),
        itemBuilder: (context, index) {
          final detection = widget.controller.detections[index];
          return _DetectionRow(
            key: ValueKey('detection-row-${detection.source.itemId}'),
            controller: widget.controller,
            detection: detection,
            index: index + 1,
            selected:
                widget.controller.selectedItemId == detection.source.itemId,
            focusNode: _rowFocusNode(detection.source.itemId),
          );
        },
      ),
    );
  }
}

class _DetectionRow extends StatelessWidget {
  const _DetectionRow({
    super.key,
    required this.controller,
    required this.detection,
    required this.index,
    required this.selected,
    required this.focusNode,
  });

  final ScannerController controller;
  final ReviewDetection detection;
  final int index;
  final bool selected;
  final FocusNode focusNode;

  @override
  Widget build(BuildContext context) {
    final needsReview = !detection.isConfirmed;
    final reviewPresentation = presentSegmentReview(detection.source);
    final tone = needsReview ? AppColors.attention : AppColors.success;
    final confidence =
        '${(detection.source.confidence * 100).toStringAsFixed(0)}%';
    final semanticLabel = needsReview
        ? '$index번 상품, ${reviewPresentation.shortLabel}'
        : '$index번 상품, ${detection.finalProduct!.displayName}, 확정, 신뢰도 $confidence';
    return AppSelectableSurface(
      selected: selected,
      selectedBackgroundColor: AppColors.surface,
      selectedBorder: Border.all(color: tone, width: 2),
      enabled: !controller.isBusy,
      inMutuallyExclusiveGroup: true,
      focusNode: focusNode,
      onTap: () => controller.selectDetection(detection.source.itemId),
      semanticLabel: semanticLabel,
      padding: const EdgeInsets.symmetric(
        horizontal: AppSpacing.x4,
        vertical: AppSpacing.x2,
      ),
      child: Row(
        children: [
          SizedBox(
            width: context.appTokens.compactVisualSize,
            child: Text(
              index.toString().padLeft(2, '0'),
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
          Icon(
            needsReview
                ? Icons.help_outline_rounded
                : Icons.check_circle_outline_rounded,
            size: 19,
            color: tone,
          ),
          const SizedBox(width: AppSpacing.x3),
          Expanded(
            child: Text(
              needsReview
                  ? reviewPresentation.rowTitle
                  : detection.finalProduct!.displayName,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(fontWeight: AppTypography.bold),
            ),
          ),
          Text(
            needsReview ? reviewPresentation.shortLabel : confidence,
            maxLines: 1,
            softWrap: false,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: needsReview ? tone : AppColors.muted,
              fontWeight: needsReview
                  ? AppTypography.bold
                  : AppTypography.medium,
            ),
          ),
          const SizedBox(width: AppSpacing.x2),
          Icon(
            selected
                ? Icons.arrow_forward_rounded
                : Icons.chevron_right_rounded,
            color: selected ? tone : AppColors.subtle,
            size: 18,
          ),
        ],
      ),
    );
  }
}
