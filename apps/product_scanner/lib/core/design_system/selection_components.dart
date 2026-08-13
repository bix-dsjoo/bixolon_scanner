part of 'components.dart';

@immutable
class AppFilterOption<T> {
  const AppFilterOption(this.value, this.label);

  final T value;
  final String label;
}

class AppFilterChip extends StatefulWidget {
  const AppFilterChip({
    super.key,
    required this.label,
    required this.semanticLabel,
    required this.selected,
    required this.onSelected,
    this.choiceKey,
    this.focusNode,
  });

  final String label;
  final String semanticLabel;
  final bool selected;
  final VoidCallback onSelected;
  final Key? choiceKey;
  final FocusNode? focusNode;

  @override
  State<AppFilterChip> createState() => _AppFilterChipState();
}

class _AppFilterChipState extends State<AppFilterChip> {
  late FocusNode _focusNode;
  late bool _ownsFocusNode;

  @override
  void initState() {
    super.initState();
    _attachFocusNode(widget.focusNode);
  }

  @override
  void didUpdateWidget(covariant AppFilterChip oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.focusNode == widget.focusNode) return;
    _focusNode.removeListener(_handleFocusChange);
    if (_ownsFocusNode) _focusNode.dispose();
    _attachFocusNode(widget.focusNode);
  }

  void _attachFocusNode(FocusNode? focusNode) {
    _ownsFocusNode = focusNode == null;
    _focusNode = focusNode ?? FocusNode(debugLabel: 'filter-${widget.label}');
    _focusNode.addListener(_handleFocusChange);
  }

  void _handleFocusChange() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _focusNode.removeListener(_handleFocusChange);
    if (_ownsFocusNode) _focusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final focused = _focusNode.hasFocus;
    return Semantics(
      container: true,
      excludeSemantics: true,
      button: true,
      selected: widget.selected,
      focused: focused,
      inMutuallyExclusiveGroup: true,
      label: widget.semanticLabel,
      onTap: widget.onSelected,
      child: ChoiceChip(
        key: widget.choiceKey,
        focusNode: _focusNode,
        label: Text(widget.label),
        selected: widget.selected,
        showCheckmark: false,
        side: BorderSide(
          color: focused
              ? context.appComponents.focusRing
              : widget.selected
              ? context.appComponents.selectionOutline
              : context.appColors.outline,
          width: focused
              ? context.appTokens.focusRingWidth
              : widget.selected
              ? context.appTokens.selectionOutlineWidth
              : 1,
        ),
        labelStyle: Theme.of(context).chipTheme.labelStyle?.copyWith(
          color: widget.selected
              ? context.appComponents.selectionContent
              : context.appColors.ink,
          fontWeight: widget.selected
              ? AppTypography.bold
              : AppTypography.semibold,
        ),
        onSelected: (_) => widget.onSelected(),
      ),
    );
  }
}

class AppFilterGroup<T> extends StatefulWidget {
  const AppFilterGroup({
    super.key,
    required this.label,
    required this.value,
    required this.options,
    required this.onChanged,
  });

  final String label;
  final T value;
  final List<AppFilterOption<T>> options;
  final ValueChanged<T> onChanged;

  @override
  State<AppFilterGroup<T>> createState() => _AppFilterGroupState<T>();
}

class _AppFilterGroupState<T> extends State<AppFilterGroup<T>> {
  final Map<T, FocusNode> _focusNodes = <T, FocusNode>{};

  @override
  void initState() {
    super.initState();
    _syncFocusNodes();
  }

  @override
  void didUpdateWidget(covariant AppFilterGroup<T> oldWidget) {
    super.didUpdateWidget(oldWidget);
    _syncFocusNodes();
    final activeValues = widget.options.map((option) => option.value).toSet();
    final removedNodes = <FocusNode>[];
    _focusNodes.removeWhere((value, focusNode) {
      if (activeValues.contains(value)) return false;
      removedNodes.add(focusNode);
      return true;
    });
    if (removedNodes.isEmpty) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      for (final focusNode in removedNodes) {
        focusNode.dispose();
      }
    });
  }

  void _syncFocusNodes() {
    for (final option in widget.options) {
      final focusNode = _focusNodes.putIfAbsent(
        option.value,
        () => FocusNode(debugLabel: 'filter-${widget.label}-${option.label}'),
      );
      focusNode.skipTraversal = option.value != widget.value;
    }
  }

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent && event is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    final focusedIndex = widget.options.indexWhere(
      (option) => _focusNodes[option.value]?.hasFocus ?? false,
    );
    if (focusedIndex < 0 || widget.options.isEmpty) {
      return KeyEventResult.ignored;
    }
    final offset = switch (event.logicalKey) {
      LogicalKeyboardKey.arrowLeft || LogicalKeyboardKey.arrowUp => -1,
      LogicalKeyboardKey.arrowRight || LogicalKeyboardKey.arrowDown => 1,
      _ => 0,
    };
    if (offset == 0) return KeyEventResult.ignored;
    final nextIndex = (focusedIndex + offset) % widget.options.length;
    final next = widget.options[nextIndex];
    widget.onChanged(next.value);
    _focusNodes[next.value]!.requestFocus();
    return KeyEventResult.handled;
  }

  @override
  void dispose() {
    for (final focusNode in _focusNodes.values) {
      focusNode.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Focus(
      canRequestFocus: false,
      onKeyEvent: _handleKeyEvent,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(widget.label, style: Theme.of(context).textTheme.bodySmall),
          const SizedBox(width: AppSpacing.x2),
          ...widget.options.map((option) {
            final selected = widget.value == option.value;
            return Padding(
              padding: const EdgeInsets.only(right: AppSpacing.x2),
              child: AppFilterChip(
                choiceKey: ValueKey('filter-${widget.label}-${option.label}'),
                label: option.label,
                semanticLabel: '${widget.label}, ${option.label}',
                selected: selected,
                focusNode: _focusNodes[option.value],
                onSelected: () => widget.onChanged(option.value),
              ),
            );
          }),
        ],
      ),
    );
  }
}

class AppDropdownControl<T> extends StatefulWidget {
  const AppDropdownControl({
    super.key,
    required this.value,
    required this.items,
    required this.onChanged,
    required this.semanticLabel,
    this.focusNode,
  });

  final T value;
  final List<DropdownMenuItem<T>> items;
  final ValueChanged<T?> onChanged;
  final String semanticLabel;
  final FocusNode? focusNode;

  @override
  State<AppDropdownControl<T>> createState() => _AppDropdownControlState<T>();
}

class _AppDropdownControlState<T> extends State<AppDropdownControl<T>> {
  late FocusNode _focusNode;
  late bool _ownsFocusNode;

  @override
  void initState() {
    super.initState();
    _attachFocusNode(widget.focusNode);
  }

  @override
  void didUpdateWidget(covariant AppDropdownControl<T> oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.focusNode == widget.focusNode) return;
    _focusNode.removeListener(_handleFocusChange);
    if (_ownsFocusNode) _focusNode.dispose();
    _attachFocusNode(widget.focusNode);
  }

  void _attachFocusNode(FocusNode? focusNode) {
    _ownsFocusNode = focusNode == null;
    _focusNode = focusNode ?? FocusNode(debugLabel: widget.semanticLabel);
    _focusNode.addListener(_handleFocusChange);
  }

  void _handleFocusChange() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _focusNode.removeListener(_handleFocusChange);
    if (_ownsFocusNode) _focusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final focused = _focusNode.hasFocus;
    return Semantics(
      container: true,
      label: widget.semanticLabel,
      child: AnimatedContainer(
        key: ValueKey('dropdown-surface-${widget.semanticLabel}'),
        duration: MediaQuery.disableAnimationsOf(context)
            ? Duration.zero
            : context.appTokens.motionFast,
        curve: AppMotion.interactionCurve,
        height: context.appTokens.controlHeight,
        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.x3),
        decoration: BoxDecoration(
          color: context.appColors.surface,
          border: Border.all(
            color: focused
                ? context.appComponents.focusRing
                : context.appColors.outline,
            width: focused ? context.appTokens.focusRingWidth : 1,
          ),
          borderRadius: BorderRadius.circular(context.appTokens.controlRadius),
        ),
        child: DropdownButtonHideUnderline(
          child: DropdownButton<T>(
            focusNode: _focusNode,
            value: widget.value,
            borderRadius: BorderRadius.circular(
              context.appTokens.controlRadius,
            ),
            focusColor: Colors.transparent,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              fontWeight: AppTypography.semibold,
            ),
            items: widget.items,
            onChanged: widget.onChanged,
          ),
        ),
      ),
    );
  }
}

/// Toss-style selection surface: the whole option changes state. A left-only
/// marker is intentionally avoided because it reads as navigation, not choice.
class AppSelectableSurface extends StatefulWidget {
  const AppSelectableSurface({
    super.key,
    required this.selected,
    required this.onTap,
    required this.child,
    this.minHeight,
    this.padding = EdgeInsets.zero,
    this.margin = EdgeInsets.zero,
    this.borderRadius,
    this.restingBorder,
    this.backgroundColor,
    this.selectedBackgroundColor,
    this.selectedBorder,
    this.semanticLabel,
    this.focusNode,
    this.autofocus = false,
    this.inMutuallyExclusiveGroup = false,
    this.onKeyboardTap,
    this.enabled = true,
  });

  final bool selected;
  final VoidCallback onTap;
  final Widget child;
  final double? minHeight;
  final EdgeInsetsGeometry padding;
  final EdgeInsetsGeometry margin;
  final BorderRadius? borderRadius;
  final BoxBorder? restingBorder;
  final Color? backgroundColor;
  final Color? selectedBackgroundColor;
  final BoxBorder? selectedBorder;
  final String? semanticLabel;
  final FocusNode? focusNode;
  final bool autofocus;
  final bool inMutuallyExclusiveGroup;
  final VoidCallback? onKeyboardTap;
  final bool enabled;

  @override
  State<AppSelectableSurface> createState() => _AppSelectableSurfaceState();
}

class _AppSelectableSurfaceState extends State<AppSelectableSurface> {
  bool _focused = false;
  bool _hovered = false;
  bool _pointerTapPending = false;

  @override
  void initState() {
    super.initState();
    _focused = widget.focusNode?.hasFocus ?? false;
  }

  @override
  void didUpdateWidget(covariant AppSelectableSurface oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.focusNode != widget.focusNode) {
      _focused = widget.focusNode?.hasFocus ?? false;
    }
    if (!widget.enabled) {
      _hovered = false;
      _focused = false;
    }
  }

  void _handleTap() {
    if (!widget.enabled) return;
    final keyboardActivation = !_pointerTapPending;
    _pointerTapPending = false;
    widget.onTap();
    if (keyboardActivation) widget.onKeyboardTap?.call();
  }

  @override
  Widget build(BuildContext context) {
    final component = context.appComponents;
    final tokens = context.appTokens;
    final radius = widget.borderRadius ?? BorderRadius.zero;
    return Semantics(
      container: true,
      excludeSemantics: true,
      button: true,
      enabled: widget.enabled,
      selected: widget.selected,
      focused: widget.enabled && _focused,
      inMutuallyExclusiveGroup: widget.inMutuallyExclusiveGroup,
      label: widget.semanticLabel,
      onTap: widget.enabled ? widget.onTap : null,
      child: Padding(
        padding: widget.margin,
        child: Material(
          color: Colors.transparent,
          borderRadius: radius,
          clipBehavior: Clip.antiAlias,
          child: InkWell(
            onTapDown: widget.enabled ? (_) => _pointerTapPending = true : null,
            onTapCancel: widget.enabled
                ? () => _pointerTapPending = false
                : null,
            onTap: widget.enabled ? _handleTap : null,
            excludeFromSemantics: true,
            focusNode: widget.focusNode,
            autofocus: widget.autofocus,
            mouseCursor: widget.enabled
                ? SystemMouseCursors.click
                : SystemMouseCursors.basic,
            onHover: (hovered) {
              final nextHovered = widget.enabled && hovered;
              if (_hovered != nextHovered) {
                setState(() => _hovered = nextHovered);
              }
            },
            onFocusChange: (focused) {
              if (_focused != focused) setState(() => _focused = focused);
            },
            borderRadius: radius,
            child: AnimatedContainer(
              duration: MediaQuery.disableAnimationsOf(context)
                  ? Duration.zero
                  : tokens.motionFast,
              curve: AppMotion.interactionCurve,
              constraints: BoxConstraints(
                minHeight: widget.minHeight ?? tokens.rowHeight,
              ),
              padding: widget.padding,
              decoration: BoxDecoration(
                color: widget.selected
                    ? widget.selectedBackgroundColor ??
                          component.selectionSurface
                    : !widget.enabled
                    ? component.disabledSurface
                    : _hovered
                    ? component.rowHover
                    : widget.backgroundColor ?? context.appColors.surface,
                borderRadius: radius,
                border: widget.enabled && _focused
                    ? Border.all(
                        color: component.focusRing,
                        width: tokens.focusRingWidth,
                      )
                    : widget.selected
                    ? widget.selectedBorder ??
                          Border.all(
                            color: component.selectionOutline,
                            width: tokens.selectionOutlineWidth,
                          )
                    : widget.restingBorder,
              ),
              child: Opacity(
                opacity: widget.enabled ? 1 : tokens.disabledContentOpacity,
                child: DefaultTextStyle.merge(
                  style: TextStyle(
                    color: widget.selected ? component.selectionContent : null,
                  ),
                  child: IconTheme.merge(
                    data: IconThemeData(
                      color: widget.selected
                          ? component.selectionContent
                          : null,
                    ),
                    child: widget.child,
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
