import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/app_theme.dart';
import '../theme/app_tokens.dart';

class AppBrandMark extends StatelessWidget {
  const AppBrandMark({super.key});

  @override
  Widget build(BuildContext context) {
    final tokens = context.appTokens;
    return Container(
      width: tokens.compactVisualSize,
      height: tokens.compactVisualSize,
      decoration: BoxDecoration(
        color: AppColors.primary,
        borderRadius: BorderRadius.circular(context.appTokens.controlRadius),
      ),
      alignment: Alignment.center,
      child: Icon(
        Icons.center_focus_strong_rounded,
        color: AppColors.ink,
        size: tokens.inlineProgressSize,
      ),
    );
  }
}

class AppStatusBadge extends StatelessWidget {
  const AppStatusBadge({
    super.key,
    required this.label,
    required this.icon,
    required this.color,
    this.backgroundColor,
    this.liveRegion = false,
  });

  final String label;
  final IconData icon;
  final Color color;
  final Color? backgroundColor;
  final bool liveRegion;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      container: true,
      excludeSemantics: true,
      liveRegion: liveRegion,
      label: label,
      child: Container(
        constraints: BoxConstraints(
          minHeight: context.appTokens.compactVisualSize,
        ),
        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.x3),
        decoration: BoxDecoration(
          color: backgroundColor ?? color.withValues(alpha: .08),
          borderRadius: BorderRadius.circular(context.appTokens.pillRadius),
          border: Border.all(color: color.withValues(alpha: .24)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 16, color: color),
            const SizedBox(width: AppSpacing.x2),
            Text(
              label,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: context.appColors.ink,
                fontWeight: AppTypography.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class AppPanelHeader extends StatelessWidget {
  const AppPanelHeader({
    super.key,
    required this.title,
    this.subtitle,
    this.trailing,
  });

  final String title;
  final String? subtitle;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: BoxConstraints(minHeight: context.appTokens.headerHeight),
      padding: const EdgeInsets.symmetric(horizontal: AppSpacing.x4),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(bottom: BorderSide(color: AppColors.divider)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: Theme.of(context).textTheme.titleMedium),
                if (subtitle != null)
                  Text(subtitle!, style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          ),
          ?trailing,
        ],
      ),
    );
  }
}

class AppStepNavigator extends StatelessWidget {
  const AppStepNavigator({
    super.key,
    required this.current,
    required this.total,
    required this.onPrevious,
    required this.onNext,
    required this.previousTooltip,
    required this.nextTooltip,
    this.semanticUnit = '항목',
  }) : assert(current > 0),
       assert(total > 0),
       assert(current <= total);

  final int current;
  final int total;
  final VoidCallback? onPrevious;
  final VoidCallback? onNext;
  final String previousTooltip;
  final String nextTooltip;
  final String semanticUnit;

  @override
  Widget build(BuildContext context) {
    final tokens = context.appTokens;
    final radius = BorderRadius.circular(tokens.controlRadius);
    return SizedBox(
      key: const ValueKey('step-navigator'),
      height: tokens.actionHeight,
      child: ClipRRect(
        borderRadius: radius,
        child: DecoratedBox(
          decoration: BoxDecoration(
            color: context.appColors.surface,
            border: Border.all(color: context.appColors.outline),
            borderRadius: radius,
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              AppIconActionButton(
                semanticLabel: previousTooltip,
                tooltip: previousTooltip,
                onPressed: onPrevious,
                icon: const Icon(Icons.keyboard_arrow_up_rounded),
              ),
              const VerticalDivider(
                width: 1,
                indent: AppSpacing.x2,
                endIndent: AppSpacing.x2,
              ),
              SizedBox(
                width: tokens.stepNavigatorLabelWidth,
                child: Semantics(
                  container: true,
                  excludeSemantics: true,
                  label: '현재 $current번 $semanticUnit, 전체 $total개',
                  child: Text(
                    '$current / $total',
                    textAlign: TextAlign.center,
                    maxLines: 1,
                    softWrap: false,
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      fontWeight: AppTypography.bold,
                    ),
                  ),
                ),
              ),
              const VerticalDivider(
                width: 1,
                indent: AppSpacing.x2,
                endIndent: AppSpacing.x2,
              ),
              AppIconActionButton(
                semanticLabel: nextTooltip,
                tooltip: nextTooltip,
                onPressed: onNext,
                icon: const Icon(Icons.keyboard_arrow_down_rounded),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class AppKeyboardShortcutHint extends StatelessWidget {
  const AppKeyboardShortcutHint({
    super.key,
    required this.shortcut,
    required this.semanticLabel,
  });

  final String shortcut;
  final String semanticLabel;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: context.appTokens.controlHeight,
      child: Tooltip(
        message: semanticLabel,
        excludeFromSemantics: true,
        child: Center(
          child: Semantics(
            container: true,
            excludeSemantics: true,
            label: semanticLabel,
            child: Container(
              key: ValueKey('shortcut-hint-$shortcut'),
              constraints: const BoxConstraints(minWidth: 24, minHeight: 24),
              padding: const EdgeInsets.symmetric(horizontal: AppSpacing.x2),
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: context.appColors.elevated,
                border: Border.all(color: context.appColors.outline),
                borderRadius: BorderRadius.circular(
                  context.appTokens.controlRadius,
                ),
              ),
              child: Text(
                shortcut,
                maxLines: 1,
                softWrap: false,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: context.appColors.ink,
                  fontWeight: AppTypography.bold,
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

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

class AppEmptyState extends StatelessWidget {
  const AppEmptyState({
    super.key,
    required this.icon,
    required this.title,
    required this.detail,
    this.tone = AppColors.muted,
    this.announcement,
    this.action,
  });

  final IconData icon;
  final String title;
  final String detail;
  final Color tone;
  final String? announcement;
  final Widget? action;

  @override
  Widget build(BuildContext context) {
    final tokens = context.appTokens;
    final message = Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: context.appTokens.actionHeight,
          height: context.appTokens.actionHeight,
          decoration: BoxDecoration(
            color: tone.withValues(alpha: .08),
            shape: BoxShape.circle,
          ),
          child: Icon(icon, color: tone, size: 24),
        ),
        const SizedBox(height: AppSpacing.x4),
        Text(
          title,
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: AppSpacing.x2),
        Text(
          detail,
          textAlign: TextAlign.center,
          style: Theme.of(
            context,
          ).textTheme.bodyMedium?.copyWith(color: AppColors.muted),
        ),
      ],
    );
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(AppSpacing.x6),
        child: ConstrainedBox(
          constraints: BoxConstraints(maxWidth: tokens.emptyStateMaxWidth),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (announcement case final announcement?)
                Semantics(
                  container: true,
                  excludeSemantics: true,
                  liveRegion: true,
                  label: announcement,
                  child: message,
                )
              else
                message,
              if (action != null) ...[
                const SizedBox(height: AppSpacing.x4),
                action!,
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class AppProgressVisual extends StatelessWidget {
  const AppProgressVisual({
    super.key,
    required this.size,
    required this.color,
    required this.strokeWidth,
  });

  final double size;
  final Color color;
  final double strokeWidth;

  @override
  Widget build(BuildContext context) {
    final reducedMotion = MediaQuery.disableAnimationsOf(context);
    return ExcludeSemantics(
      child: SizedBox.square(
        dimension: size,
        child: reducedMotion
            ? Icon(Icons.hourglass_top_rounded, size: size, color: color)
            : CircularProgressIndicator(strokeWidth: strokeWidth, color: color),
      ),
    );
  }
}

class AppStateAnnouncement extends StatelessWidget {
  const AppStateAnnouncement({super.key, required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      container: true,
      excludeSemantics: true,
      liveRegion: true,
      label: message,
      child: const SizedBox.shrink(),
    );
  }
}

class AppLoadingState extends StatelessWidget {
  const AppLoadingState({
    super.key,
    required this.message,
    this.announce = true,
  });

  final String message;
  final bool announce;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Semantics(
        container: true,
        excludeSemantics: true,
        liveRegion: announce,
        label: message,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            AppProgressVisual(
              size: context.appTokens.pageProgressSize,
              strokeWidth: 2.5,
              color: context.appColors.brand,
            ),
            const SizedBox(height: AppSpacing.x4),
            Text(message, style: Theme.of(context).textTheme.titleMedium),
          ],
        ),
      ),
    );
  }
}

class AppInlineNotice extends StatelessWidget {
  const AppInlineNotice({
    super.key,
    required this.message,
    required this.icon,
    required this.tone,
    required this.backgroundColor,
    this.action,
  });

  final String message;
  final IconData icon;
  final Color tone;
  final Color backgroundColor;
  final Widget? action;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      container: true,
      explicitChildNodes: true,
      child: Container(
        constraints: BoxConstraints(minHeight: context.appTokens.controlHeight),
        padding: const EdgeInsets.symmetric(
          horizontal: AppSpacing.x4,
          vertical: AppSpacing.x2,
        ),
        decoration: BoxDecoration(
          color: backgroundColor,
          border: Border(bottom: BorderSide(color: tone.withValues(alpha: .2))),
        ),
        child: Row(
          children: [
            ExcludeSemantics(child: Icon(icon, size: 18, color: tone)),
            const SizedBox(width: AppSpacing.x2),
            Expanded(
              child: Semantics(
                excludeSemantics: true,
                liveRegion: true,
                label: message,
                child: Text(
                  message,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: context.appColors.ink,
                    fontWeight: AppTypography.semibold,
                  ),
                ),
              ),
            ),
            if (action != null) ...[
              const SizedBox(width: AppSpacing.x3),
              action!,
            ],
          ],
        ),
      ),
    );
  }
}

class AppSectionLabel extends StatelessWidget {
  const AppSectionLabel({super.key, required this.label, this.trailing});

  final String label;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: BoxConstraints(
        minHeight: context.appTokens.sectionHeaderHeight,
      ),
      padding: const EdgeInsets.symmetric(horizontal: AppSpacing.x4),
      alignment: Alignment.centerLeft,
      decoration: const BoxDecoration(
        color: AppColors.elevated,
        border: Border(
          top: BorderSide(color: AppColors.divider),
          bottom: BorderSide(color: AppColors.divider),
        ),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: AppColors.ink,
                fontWeight: AppTypography.bold,
              ),
            ),
          ),
          ?trailing,
        ],
      ),
    );
  }
}

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
                    ? component.selectionSurface
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
                    ? Border.all(
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

class AppActionBar extends StatelessWidget {
  const AppActionBar({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: BoxConstraints(minHeight: context.appTokens.actionBarHeight),
      padding: const EdgeInsets.symmetric(
        horizontal: AppSpacing.x4,
        vertical: AppSpacing.x3,
      ),
      decoration: BoxDecoration(
        color: context.appComponents.actionBarSurface,
        border: Border(top: BorderSide(color: context.appColors.outline)),
      ),
      child: child,
    );
  }
}

class AppProgressActionButton extends StatelessWidget {
  const AppProgressActionButton({
    super.key,
    required this.label,
    required this.icon,
    required this.onPressed,
    required this.progressing,
    required this.progressLabel,
    required this.progressAnnouncement,
    this.focusNode,
  });

  final String label;
  final Widget icon;
  final VoidCallback? onPressed;
  final bool progressing;
  final String progressLabel;
  final String progressAnnouncement;
  final FocusNode? focusNode;

  @override
  Widget build(BuildContext context) {
    final tokens = context.appTokens;
    final button = FilledButton.icon(
      focusNode: focusNode,
      onPressed: progressing ? null : onPressed,
      icon: progressing
          ? AppProgressVisual(
              size: tokens.inlineProgressSize,
              strokeWidth: 2,
              color: AppColors.ink,
            )
          : icon,
      label: Text(progressing ? progressLabel : label),
    );
    if (!progressing) return button;

    return Semantics(
      container: true,
      excludeSemantics: true,
      liveRegion: true,
      button: true,
      enabled: false,
      label: progressAnnouncement,
      child: button,
    );
  }
}

class AppToast extends StatelessWidget {
  const AppToast({
    super.key,
    required this.message,
    required this.icon,
    this.iconColor,
  });

  final String message;
  final IconData icon;
  final Color? iconColor;

  @override
  Widget build(BuildContext context) {
    final component = context.appComponents;
    return Semantics(
      container: true,
      excludeSemantics: true,
      liveRegion: true,
      label: message,
      child: Material(
        color: component.toastSurface,
        elevation: 3,
        shadowColor: AppPalette.ink.withValues(alpha: .16),
        borderRadius: BorderRadius.circular(context.appTokens.panelRadius),
        child: Padding(
          padding: const EdgeInsets.symmetric(
            horizontal: AppSpacing.x4,
            vertical: AppSpacing.x3,
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, color: iconColor ?? context.appColors.brand, size: 21),
              const SizedBox(width: AppSpacing.x2),
              Text(
                message,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: component.onToast,
                  fontWeight: AppTypography.bold,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
