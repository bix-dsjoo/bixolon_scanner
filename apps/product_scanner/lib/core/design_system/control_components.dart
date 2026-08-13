part of 'components.dart';

class AppIconActionButton extends StatefulWidget {
  const AppIconActionButton({
    super.key,
    required this.semanticLabel,
    required this.tooltip,
    required this.icon,
    required this.onPressed,
    this.focusNode,
    this.progressing = false,
    this.progressTooltip,
    this.progressAnnouncement,
    this.announceProgress = true,
  }) : assert(
         !progressing ||
             (progressTooltip != null && progressAnnouncement != null),
       );

  final String semanticLabel;
  final String tooltip;
  final Widget icon;
  final VoidCallback? onPressed;
  final FocusNode? focusNode;
  final bool progressing;
  final String? progressTooltip;
  final String? progressAnnouncement;
  final bool announceProgress;

  @override
  State<AppIconActionButton> createState() => _AppIconActionButtonState();
}

class _AppIconActionButtonState extends State<AppIconActionButton> {
  late FocusNode _focusNode;
  late bool _ownsFocusNode;

  @override
  void initState() {
    super.initState();
    _attachFocusNode(widget.focusNode);
  }

  @override
  void didUpdateWidget(covariant AppIconActionButton oldWidget) {
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
    final enabled = !widget.progressing && widget.onPressed != null;
    final focused = enabled && _focusNode.hasFocus;
    final tooltip = widget.progressing
        ? widget.progressTooltip!
        : widget.tooltip;
    final label = widget.progressing
        ? widget.progressAnnouncement!
        : widget.semanticLabel;
    final tokens = context.appTokens;

    return Semantics(
      container: true,
      excludeSemantics: true,
      button: true,
      enabled: enabled,
      focused: focused,
      liveRegion: widget.progressing && widget.announceProgress,
      label: label,
      onTap: enabled ? widget.onPressed : null,
      child: AnimatedContainer(
        key: ValueKey('icon-action-surface-${widget.semanticLabel}'),
        duration: MediaQuery.disableAnimationsOf(context)
            ? Duration.zero
            : tokens.motionFast,
        curve: AppMotion.interactionCurve,
        width: tokens.actionHeight,
        height: tokens.actionHeight,
        decoration: BoxDecoration(
          border: focused
              ? Border.all(
                  color: context.appComponents.focusRing,
                  width: tokens.focusRingWidth,
                )
              : null,
          borderRadius: BorderRadius.circular(tokens.controlRadius),
        ),
        child: IconButton(
          focusNode: _focusNode,
          tooltip: tooltip,
          onPressed: enabled ? widget.onPressed : null,
          icon: widget.progressing
              ? AppProgressVisual(
                  size: tokens.inlineProgressSize,
                  strokeWidth: 2,
                  color: AppColors.muted,
                )
              : widget.icon,
        ),
      ),
    );
  }
}

class AppDisclosure extends StatefulWidget {
  const AppDisclosure({
    super.key,
    required this.title,
    required this.description,
    required this.icon,
    required this.children,
    this.initiallyExpanded = false,
    this.focusNode,
    this.onExpansionChanged,
  });

  final String title;
  final String description;
  final IconData icon;
  final List<Widget> children;
  final bool initiallyExpanded;
  final FocusNode? focusNode;
  final ValueChanged<bool>? onExpansionChanged;

  @override
  State<AppDisclosure> createState() => _AppDisclosureState();
}

class AppConfirmDialog extends StatelessWidget {
  const AppConfirmDialog({
    super.key,
    required this.title,
    required this.description,
    required this.confirmLabel,
    required this.onCancel,
    required this.onConfirm,
  });

  final String title;
  final String description;
  final String confirmLabel;
  final VoidCallback onCancel;
  final VoidCallback onConfirm;

  @override
  Widget build(BuildContext context) {
    final tokens = context.appTokens;
    return AlertDialog(
      constraints: BoxConstraints(
        minWidth: tokens.dialogWidth,
        maxWidth: tokens.dialogWidth,
      ),
      insetPadding: const EdgeInsets.all(AppSpacing.x6),
      titlePadding: const EdgeInsets.fromLTRB(
        AppSpacing.x6,
        AppSpacing.x6,
        AppSpacing.x6,
        AppSpacing.x2,
      ),
      contentPadding: const EdgeInsets.fromLTRB(
        AppSpacing.x6,
        0,
        AppSpacing.x6,
        AppSpacing.x6,
      ),
      actionsPadding: const EdgeInsets.fromLTRB(
        AppSpacing.x6,
        0,
        AppSpacing.x6,
        AppSpacing.x6,
      ),
      title: Text(title),
      content: Text(
        description,
        style: Theme.of(
          context,
        ).textTheme.bodyMedium?.copyWith(color: context.appColors.muted),
      ),
      actions: [
        Row(
          children: [
            Expanded(
              child: _AppDialogActionButton(
                label: '취소',
                onPressed: onCancel,
                autofocus: true,
              ),
            ),
            const SizedBox(width: AppSpacing.x2),
            Expanded(
              child: _AppDialogActionButton(
                label: confirmLabel,
                onPressed: onConfirm,
                primary: true,
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _AppDialogActionButton extends StatefulWidget {
  const _AppDialogActionButton({
    required this.label,
    required this.onPressed,
    this.primary = false,
    this.autofocus = false,
  });

  final String label;
  final VoidCallback onPressed;
  final bool primary;
  final bool autofocus;

  @override
  State<_AppDialogActionButton> createState() => _AppDialogActionButtonState();
}

class _AppDialogActionButtonState extends State<_AppDialogActionButton> {
  late final FocusNode _focusNode;

  @override
  void initState() {
    super.initState();
    _focusNode = FocusNode(debugLabel: 'dialog-action-${widget.label}');
  }

  @override
  void dispose() {
    _focusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final button = widget.primary
        ? FilledButton(
            focusNode: _focusNode,
            autofocus: widget.autofocus,
            onPressed: widget.onPressed,
            child: Text(widget.label),
          )
        : OutlinedButton(
            focusNode: _focusNode,
            autofocus: widget.autofocus,
            onPressed: widget.onPressed,
            child: Text(widget.label),
          );

    return SizedBox(height: context.appTokens.actionHeight, child: button);
  }
}

class _AppDisclosureState extends State<AppDisclosure> {
  late FocusNode _focusNode;
  late bool _ownsFocusNode;
  late bool _expanded;
  bool _hovered = false;

  @override
  void initState() {
    super.initState();
    _expanded = widget.initiallyExpanded;
    _attachFocusNode(widget.focusNode);
  }

  @override
  void didUpdateWidget(covariant AppDisclosure oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.focusNode == widget.focusNode) return;
    _focusNode.removeListener(_handleFocusChange);
    if (_ownsFocusNode) _focusNode.dispose();
    _attachFocusNode(widget.focusNode);
  }

  void _attachFocusNode(FocusNode? focusNode) {
    _ownsFocusNode = focusNode == null;
    _focusNode = focusNode ?? FocusNode(debugLabel: widget.title);
    _focusNode.addListener(_handleFocusChange);
  }

  void _handleFocusChange() {
    if (mounted) setState(() {});
  }

  void _toggle() {
    setState(() => _expanded = !_expanded);
    widget.onExpansionChanged?.call(_expanded);
  }

  @override
  void dispose() {
    _focusNode.removeListener(_handleFocusChange);
    if (_ownsFocusNode) _focusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final tokens = context.appTokens;
    final focused = _focusNode.hasFocus;
    final duration = MediaQuery.disableAnimationsOf(context)
        ? Duration.zero
        : tokens.motionStandard;
    final content = _expanded
        ? Padding(
            padding: const EdgeInsets.fromLTRB(
              AppSpacing.x4,
              0,
              AppSpacing.x4,
              AppSpacing.x4,
            ),
            child: Column(children: widget.children),
          )
        : const SizedBox(width: double.infinity);

    return Column(
      children: [
        Semantics(
          container: true,
          excludeSemantics: true,
          button: true,
          expanded: _expanded,
          focused: focused,
          label: widget.title,
          hint: _expanded ? '접기' : '펼치기',
          onTap: _toggle,
          onExpand: _expanded ? null : _toggle,
          onCollapse: _expanded ? _toggle : null,
          child: Material(
            color: Colors.transparent,
            child: InkWell(
              focusNode: _focusNode,
              excludeFromSemantics: true,
              onTap: _toggle,
              focusColor: Colors.transparent,
              hoverColor: Colors.transparent,
              onHover: (hovered) {
                if (_hovered != hovered) setState(() => _hovered = hovered);
              },
              child: AnimatedContainer(
                key: ValueKey('disclosure-surface-${widget.title}'),
                duration: MediaQuery.disableAnimationsOf(context)
                    ? Duration.zero
                    : tokens.motionFast,
                curve: AppMotion.interactionCurve,
                constraints: BoxConstraints(minHeight: tokens.headerHeight),
                padding: const EdgeInsets.symmetric(
                  horizontal: AppSpacing.x4,
                  vertical: AppSpacing.x2,
                ),
                decoration: BoxDecoration(
                  color: _hovered
                      ? context.appComponents.rowHover
                      : Colors.transparent,
                  border: focused
                      ? Border.all(
                          color: context.appComponents.focusRing,
                          width: tokens.focusRingWidth,
                        )
                      : null,
                ),
                child: Row(
                  children: [
                    Icon(widget.icon, size: 20, color: context.appColors.ink),
                    const SizedBox(width: AppSpacing.x4),
                    Expanded(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            widget.title,
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                          Text(
                            widget.description,
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ),
                    Icon(
                      _expanded
                          ? Icons.keyboard_arrow_up_rounded
                          : Icons.keyboard_arrow_down_rounded,
                      size: 22,
                      color: context.appColors.ink,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
        ClipRect(
          child: duration == Duration.zero
              ? SizedBox(
                  key: ValueKey('disclosure-content-${widget.title}'),
                  child: content,
                )
              : AnimatedSize(
                  key: ValueKey('disclosure-content-${widget.title}'),
                  duration: duration,
                  curve: AppMotion.interactionCurve,
                  alignment: Alignment.topCenter,
                  child: content,
                ),
        ),
      ],
    );
  }
}
