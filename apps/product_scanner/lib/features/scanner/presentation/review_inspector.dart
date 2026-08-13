part of 'scanner_screen.dart';

class _ReviewInspector extends StatefulWidget {
  const _ReviewInspector({
    required this.controller,
    required this.detection,
    required this.firstChoiceFocusNode,
    required this.searchActionFocusNode,
    required this.onCandidateExitBackward,
    required this.onKeyboardChoiceConfirmed,
    required this.onKeyboardSearchClosed,
  });

  final ScannerController controller;
  final ReviewDetection detection;
  final FocusNode firstChoiceFocusNode;
  final FocusNode searchActionFocusNode;
  final VoidCallback onCandidateExitBackward;
  final VoidCallback onKeyboardChoiceConfirmed;
  final VoidCallback onKeyboardSearchClosed;

  @override
  State<_ReviewInspector> createState() => _ReviewInspectorState();
}

class _ReviewInspectorState extends State<_ReviewInspector> {
  final ScrollController _scrollController = ScrollController();
  late String _lastContentSignature;

  String get _contentSignature =>
      '${widget.detection.source.itemId}|'
      '${widget.controller.searchItemId}|${widget.controller.searchQuery}';

  @override
  void initState() {
    super.initState();
    _lastContentSignature = _contentSignature;
  }

  @override
  void didUpdateWidget(covariant _ReviewInspector oldWidget) {
    super.didUpdateWidget(oldWidget);
    final contentSignature = _contentSignature;
    if (_lastContentSignature == contentSignature) return;
    _lastContentSignature = contentSignature;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted && _scrollController.hasClients) {
        _scrollController.jumpTo(0);
      }
    });
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      key: const ValueKey('review-inspector'),
      constraints: BoxConstraints(
        maxHeight: context.appTokens.reviewInspectorMaxHeight,
      ),
      decoration: const BoxDecoration(
        color: AppColors.elevated,
        border: Border(top: BorderSide(color: AppColors.divider)),
      ),
      child: Scrollbar(
        controller: _scrollController,
        thumbVisibility: true,
        child: SingleChildScrollView(
          controller: _scrollController,
          primary: false,
          padding: const EdgeInsets.fromLTRB(
            AppSpacing.x4,
            AppSpacing.x3,
            AppSpacing.x4,
            AppSpacing.x3,
          ),
          child:
              widget.controller.searchItemId == widget.detection.source.itemId
              ? _SearchProducts(
                  controller: widget.controller,
                  detection: widget.detection,
                  firstChoiceFocusNode: widget.firstChoiceFocusNode,
                  onKeyboardChoiceConfirmed: widget.onKeyboardChoiceConfirmed,
                  onKeyboardExit: widget.onKeyboardSearchClosed,
                )
              : widget.detection.source.top3.isNotEmpty
              ? _CandidatePicker(
                  controller: widget.controller,
                  detection: widget.detection,
                  firstChoiceFocusNode: widget.firstChoiceFocusNode,
                  searchActionFocusNode: widget.searchActionFocusNode,
                  onExitBackward: widget.onCandidateExitBackward,
                  onKeyboardChoiceConfirmed: widget.onKeyboardChoiceConfirmed,
                )
              : Row(
                  children: [
                    Expanded(
                      child: Text(
                        widget.detection.finalProduct?.displayName ?? '',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ),
                    TextButton(
                      focusNode: widget.searchActionFocusNode,
                      onPressed: widget.controller.isBusy
                          ? null
                          : () => widget.controller.showSearch(
                              widget.detection.source.itemId,
                            ),
                      child: const Text('상품 변경'),
                    ),
                  ],
                ),
        ),
      ),
    );
  }
}

class _CandidatePicker extends StatelessWidget {
  const _CandidatePicker({
    required this.controller,
    required this.detection,
    required this.firstChoiceFocusNode,
    required this.searchActionFocusNode,
    required this.onExitBackward,
    required this.onKeyboardChoiceConfirmed,
  });

  final ScannerController controller;
  final ReviewDetection detection;
  final FocusNode firstChoiceFocusNode;
  final FocusNode searchActionFocusNode;
  final VoidCallback onExitBackward;
  final VoidCallback onKeyboardChoiceConfirmed;

  @override
  Widget build(BuildContext context) {
    final itemNumber =
        controller.detections.indexWhere(
          (item) => item.source.itemId == detection.source.itemId,
        ) +
        1;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Semantics(
          header: true,
          liveRegion: true,
          child: Text(
            detection.isConfirmed
                ? '$itemNumber번 상품을 변경할까요?'
                : '$itemNumber번 상품을 확인해 주세요',
            style: Theme.of(context).textTheme.titleMedium,
          ),
        ),
        const SizedBox(height: AppSpacing.x1),
        Text(
          detection.isConfirmed ? '선택하면 상품이 바로 변경돼요.' : '선택하면 다음 확인 항목으로 이동해요.',
          style: Theme.of(
            context,
          ).textTheme.bodySmall?.copyWith(color: AppColors.muted),
        ),
        const SizedBox(height: AppSpacing.x3),
        _ProductChoiceGroup(
          groupId: 'candidate-${detection.source.itemId}',
          semanticLabel: '$itemNumber번 상품 후보',
          entries: [
            for (final rawCandidate in detection.source.top3)
              _ProductChoiceEntry(
                id: rawCandidate.classId,
                choiceKey: ValueKey(
                  'candidate-${detection.source.itemId}-${rawCandidate.classId}',
                ),
                product: controller.localizeCandidate(rawCandidate),
                selected:
                    detection.finalProduct?.classId == rawCandidate.classId,
                onTap: () => controller.confirmCandidate(
                  detection.source.itemId,
                  rawCandidate,
                ),
              ),
          ],
          enabled: !controller.isBusy,
          entryFocusNode: firstChoiceFocusNode,
          onExitBackward: onExitBackward,
          onKeyboardConfirmed: onKeyboardChoiceConfirmed,
        ),
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton(
            focusNode: searchActionFocusNode,
            onPressed: controller.isBusy
                ? null
                : () => controller.showSearch(detection.source.itemId),
            child: const Text('다른 상품 검색'),
          ),
        ),
      ],
    );
  }
}
