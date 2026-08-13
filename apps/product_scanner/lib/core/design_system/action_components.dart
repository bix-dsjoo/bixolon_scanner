part of 'components.dart';

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
