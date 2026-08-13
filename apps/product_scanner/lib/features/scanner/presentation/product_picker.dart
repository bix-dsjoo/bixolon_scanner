part of 'scanner_screen.dart';

@immutable
class _ProductChoiceEntry {
  const _ProductChoiceEntry({
    required this.id,
    required this.choiceKey,
    required this.product,
    required this.selected,
    required this.onTap,
  });

  final String id;
  final Key choiceKey;
  final Product product;
  final bool selected;
  final VoidCallback onTap;
}

class _ProductChoiceGroup extends StatefulWidget {
  const _ProductChoiceGroup({
    super.key,
    required this.groupId,
    required this.semanticLabel,
    required this.entries,
    required this.enabled,
    required this.entryFocusNode,
    required this.onKeyboardConfirmed,
    this.onExitBackward,
  });

  final String groupId;
  final String semanticLabel;
  final List<_ProductChoiceEntry> entries;
  final bool enabled;
  final FocusNode entryFocusNode;
  final VoidCallback onKeyboardConfirmed;
  final VoidCallback? onExitBackward;

  @override
  State<_ProductChoiceGroup> createState() => _ProductChoiceGroupState();
}

class _ProductChoiceGroupState extends State<_ProductChoiceGroup> {
  final Map<String, FocusNode> _ownedFocusNodes = <String, FocusNode>{};
  String? _rovingId;

  String? get _initialRovingId =>
      widget.entries.where((entry) => entry.selected).firstOrNull?.id ??
      widget.entries.firstOrNull?.id;

  @override
  void initState() {
    super.initState();
    _rovingId = _initialRovingId;
    _syncTraversalState();
  }

  @override
  void didUpdateWidget(covariant _ProductChoiceGroup oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.groupId != widget.groupId ||
        !widget.entries.any((entry) => entry.id == _rovingId)) {
      _rovingId = _initialRovingId;
    }
    _disposeUnusedFocusNodes();
    _syncTraversalState();
  }

  FocusNode _focusNodeFor(int index, String id) {
    if (index == 0) return widget.entryFocusNode;
    return _ownedFocusNodes.putIfAbsent(
      id,
      () => FocusNode(debugLabel: '${widget.groupId}-$id'),
    );
  }

  void _syncTraversalState() {
    for (final entry in widget.entries.indexed) {
      _focusNodeFor(entry.$1, entry.$2.id).skipTraversal =
          !widget.enabled || entry.$2.id != _rovingId;
    }
  }

  void _disposeUnusedFocusNodes() {
    final ownedIds = widget.entries.skip(1).map((entry) => entry.id).toSet();
    final removed = <FocusNode>[];
    _ownedFocusNodes.removeWhere((id, focusNode) {
      if (ownedIds.contains(id)) return false;
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

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent && event is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    final focusedIndex = widget.entries.indexed
        .where((entry) => _focusNodeFor(entry.$1, entry.$2.id).hasFocus)
        .map((entry) => entry.$1)
        .firstOrNull;
    if (focusedIndex == null || widget.entries.isEmpty || !widget.enabled) {
      return KeyEventResult.ignored;
    }
    if (event.logicalKey == LogicalKeyboardKey.tab &&
        HardwareKeyboard.instance.isShiftPressed &&
        widget.onExitBackward != null) {
      widget.onExitBackward!();
      return KeyEventResult.handled;
    }
    final offset = switch (event.logicalKey) {
      LogicalKeyboardKey.arrowLeft || LogicalKeyboardKey.arrowUp => -1,
      LogicalKeyboardKey.arrowRight || LogicalKeyboardKey.arrowDown => 1,
      _ => 0,
    };
    if (offset == 0) return KeyEventResult.ignored;
    final nextIndex = (focusedIndex + offset) % widget.entries.length;
    final next = widget.entries[nextIndex];
    setState(() => _rovingId = next.id);
    _syncTraversalState();
    _requestFocusAndReveal(nextIndex, offset: offset);
    return KeyEventResult.handled;
  }

  void requestRovingFocus() {
    final rovingIndex = widget.entries.indexWhere(
      (entry) => entry.id == _rovingId,
    );
    if (rovingIndex < 0 || !widget.enabled) return;
    _requestFocusAndReveal(rovingIndex, offset: 1);
  }

  void _requestFocusAndReveal(int index, {required int offset}) {
    final entry = widget.entries[index];
    final nextFocusNode = _focusNodeFor(index, entry.id);
    nextFocusNode.requestFocus();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final targetContext = nextFocusNode.context;
      if (targetContext == null) return;
      Scrollable.ensureVisible(
        targetContext,
        duration: MediaQuery.disableAnimationsOf(context)
            ? Duration.zero
            : context.appTokens.motionFast,
        curve: AppMotion.interactionCurve,
        alignmentPolicy: offset > 0
            ? ScrollPositionAlignmentPolicy.keepVisibleAtEnd
            : ScrollPositionAlignmentPolicy.keepVisibleAtStart,
      );
    });
  }

  @override
  void dispose() {
    for (final focusNode in _ownedFocusNodes.values) {
      focusNode.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Semantics(
      container: true,
      explicitChildNodes: true,
      label: widget.semanticLabel,
      child: Focus(
        canRequestFocus: false,
        skipTraversal: true,
        onKeyEvent: _handleKeyEvent,
        child: Column(
          children: [
            for (final entry in widget.entries.indexed)
              _ProductChoice(
                key: entry.$2.choiceKey,
                product: entry.$2.product,
                selected: entry.$2.selected,
                enabled: widget.enabled,
                focusNode: _focusNodeFor(entry.$1, entry.$2.id),
                onTap: entry.$2.onTap,
                onKeyboardConfirmed: widget.onKeyboardConfirmed,
              ),
          ],
        ),
      ),
    );
  }
}

class _ProductChoice extends StatelessWidget {
  const _ProductChoice({
    super.key,
    required this.product,
    required this.selected,
    required this.onTap,
    this.enabled = true,
    required this.focusNode,
    this.onKeyboardConfirmed,
  });

  final Product product;
  final bool selected;
  final VoidCallback onTap;
  final bool enabled;
  final FocusNode focusNode;
  final VoidCallback? onKeyboardConfirmed;

  @override
  Widget build(BuildContext context) {
    final confidence = product is Candidate
        ? '${((product as Candidate).confidence * 100).toStringAsFixed(0)}%'
        : null;
    return AppSelectableSurface(
      selected: selected,
      enabled: enabled,
      inMutuallyExclusiveGroup: true,
      focusNode: focusNode,
      onTap: onTap,
      onKeyboardTap: onKeyboardConfirmed,
      semanticLabel: confidence == null
          ? product.displayName
          : '${product.displayName}, 신뢰도 $confidence',
      minHeight: context.appTokens.rowHeight,
      margin: const EdgeInsets.only(bottom: AppSpacing.x2),
      padding: const EdgeInsets.symmetric(horizontal: AppSpacing.x3),
      borderRadius: BorderRadius.circular(context.appTokens.selectionRadius),
      restingBorder: Border.all(color: AppColors.divider),
      child: Row(
        children: [
          Icon(
            selected
                ? Icons.radio_button_checked_rounded
                : Icons.radio_button_unchecked_rounded,
            color: selected ? AppColors.primary : AppColors.subtle,
            size: 19,
          ),
          const SizedBox(width: AppSpacing.x2),
          Expanded(
            child: Text(
              product.displayName,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                fontWeight: selected
                    ? AppTypography.bold
                    : AppTypography.semibold,
              ),
            ),
          ),
          Text(product.classId, style: Theme.of(context).textTheme.bodySmall),
          if (confidence != null) ...[
            const SizedBox(width: AppSpacing.x3),
            SizedBox(
              width: AppSpacing.x8 + AppSpacing.x4,
              child: Text(
                confidence,
                textAlign: TextAlign.right,
                maxLines: 1,
                softWrap: false,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  fontFeatures: const [FontFeature.tabularFigures()],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _SearchProducts extends StatefulWidget {
  const _SearchProducts({
    required this.controller,
    required this.detection,
    required this.firstChoiceFocusNode,
    required this.onKeyboardChoiceConfirmed,
    required this.onKeyboardExit,
  });

  final ScannerController controller;
  final ReviewDetection detection;
  final FocusNode firstChoiceFocusNode;
  final VoidCallback onKeyboardChoiceConfirmed;
  final VoidCallback onKeyboardExit;

  @override
  State<_SearchProducts> createState() => _SearchProductsState();
}

class _SearchProductsState extends State<_SearchProducts> {
  final GlobalKey<_ProductChoiceGroupState> _choiceGroupKey =
      GlobalKey<_ProductChoiceGroupState>();
  final FocusNode _backFocusNode = FocusNode(
    debugLabel: 'close-product-search',
  );

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent && event is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    if (event.logicalKey == LogicalKeyboardKey.escape ||
        (_backFocusNode.hasFocus &&
            (event.logicalKey == LogicalKeyboardKey.enter ||
                event.logicalKey == LogicalKeyboardKey.space))) {
      widget.onKeyboardExit();
      return KeyEventResult.handled;
    }
    if (event.logicalKey != LogicalKeyboardKey.tab ||
        HardwareKeyboard.instance.isShiftPressed ||
        !_isEditingText()) {
      return KeyEventResult.ignored;
    }
    final choiceGroup = _choiceGroupKey.currentState;
    if (choiceGroup == null) return KeyEventResult.ignored;
    choiceGroup.requestRovingFocus();
    return KeyEventResult.handled;
  }

  bool _isEditingText() {
    final context = FocusManager.instance.primaryFocus?.context;
    if (context == null) return false;
    return context.widget is EditableText ||
        context.findAncestorWidgetOfExactType<EditableText>() != null;
  }

  @override
  void dispose() {
    _backFocusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final detection = widget.detection;
    final itemNumber =
        controller.detections.indexWhere(
          (item) => item.source.itemId == detection.source.itemId,
        ) +
        1;
    final firstChoiceFocusNode = widget.firstChoiceFocusNode;
    final onKeyboardChoiceConfirmed = widget.onKeyboardChoiceConfirmed;
    final results = controller.searchResults;
    final visibleResults = results.take(8).toList(growable: false);
    final resultCountLabel = results.length > visibleResults.length
        ? '검색 결과 ${results.length}개 중 상위 ${visibleResults.length}개'
        : '검색 결과 ${visibleResults.length}개';
    return Focus(
      canRequestFocus: false,
      skipTraversal: true,
      onKeyEvent: _handleKeyEvent,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Semantics(
            header: true,
            child: Text(
              detection.isConfirmed
                  ? '$itemNumber번 상품 변경'
                  : '$itemNumber번 상품 검색',
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ),
          if (detection.finalProduct case final currentProduct?) ...[
            const SizedBox(height: AppSpacing.x1),
            Text(
              '현재 상품  ${currentProduct.displayName} · ${currentProduct.classId}',
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(color: AppColors.muted),
            ),
          ],
          const SizedBox(height: AppSpacing.x3),
          Row(
            children: [
              AppIconActionButton(
                semanticLabel: detection.source.top3.isNotEmpty
                    ? '후보로 돌아가기'
                    : '검색 닫기',
                tooltip: detection.source.top3.isNotEmpty
                    ? '후보로 돌아가기'
                    : '검색 닫기',
                focusNode: _backFocusNode,
                onPressed: controller.hideSearch,
                icon: const Icon(Icons.arrow_back_rounded, size: 17),
              ),
              Expanded(
                child: TextField(
                  autofocus: true,
                  enabled: !controller.isBusy,
                  onChanged: controller.updateSearch,
                  decoration: const InputDecoration(
                    hintText: '상품명 또는 class ID',
                    prefixIcon: Icon(Icons.search_rounded, size: 18),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: AppSpacing.x2),
          if (results.isEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: AppSpacing.x4),
              child: Semantics(
                container: true,
                liveRegion: true,
                label: '일치하는 상품이 없어요. 검색어를 바꾸거나 class ID를 입력해 보세요.',
                child: ExcludeSemantics(
                  child: Column(
                    children: [
                      const Icon(
                        Icons.search_off_rounded,
                        size: 24,
                        color: AppColors.subtle,
                      ),
                      const SizedBox(height: AppSpacing.x2),
                      Text(
                        '일치하는 상품이 없어요',
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          fontWeight: AppTypography.bold,
                        ),
                      ),
                      const SizedBox(height: AppSpacing.x1),
                      Text(
                        '검색어를 바꾸거나 class ID를 입력해 보세요.',
                        textAlign: TextAlign.center,
                        style: Theme.of(
                          context,
                        ).textTheme.bodySmall?.copyWith(color: AppColors.muted),
                      ),
                    ],
                  ),
                ),
              ),
            )
          else
            Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Semantics(
                  container: true,
                  header: true,
                  label: resultCountLabel,
                  child: ExcludeSemantics(
                    child: SizedBox(
                      key: const ValueKey('product-search-result-count'),
                      height: AppSpacing.x6,
                      child: Row(
                        children: [
                          Expanded(
                            child: Text(
                              '검색 결과',
                              style: Theme.of(context).textTheme.bodySmall
                                  ?.copyWith(fontWeight: AppTypography.bold),
                            ),
                          ),
                          Text(
                            results.length > visibleResults.length
                                ? '상위 ${visibleResults.length} / ${results.length}개'
                                : '${visibleResults.length}개',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: AppSpacing.x1),
                _ProductChoiceGroup(
                  key: _choiceGroupKey,
                  groupId:
                      'search-${detection.source.itemId}-${controller.searchQuery}',
                  semanticLabel: '상품 검색 결과',
                  entries: [
                    for (final product in visibleResults)
                      _ProductChoiceEntry(
                        id: product.classId,
                        choiceKey: ValueKey(
                          'search-product-${product.classId}',
                        ),
                        product: product,
                        selected:
                            detection.finalProduct?.classId == product.classId,
                        onTap: () => controller.confirmSearchProduct(
                          detection.source.itemId,
                          product,
                        ),
                      ),
                  ],
                  enabled: !controller.isBusy,
                  entryFocusNode: firstChoiceFocusNode,
                  onKeyboardConfirmed: onKeyboardChoiceConfirmed,
                ),
              ],
            ),
        ],
      ),
    );
  }
}
