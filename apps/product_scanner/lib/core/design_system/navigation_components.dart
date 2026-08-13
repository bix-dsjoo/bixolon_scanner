part of 'components.dart';

class AppBrandMark extends StatelessWidget {
  const AppBrandMark({super.key});

  @override
  Widget build(BuildContext context) {
    final tokens = context.appTokens;
    return SizedBox(
      width: tokens.brandLogoWidth,
      height: tokens.brandLogoHeight,
      child: SvgPicture.asset(
        'assets/branding/bixolon_logo.svg',
        width: tokens.brandLogoWidth,
        height: tokens.brandLogoHeight,
        fit: BoxFit.contain,
        semanticsLabel: 'BIXOLON',
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
