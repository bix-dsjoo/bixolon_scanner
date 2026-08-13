part of 'components.dart';

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
