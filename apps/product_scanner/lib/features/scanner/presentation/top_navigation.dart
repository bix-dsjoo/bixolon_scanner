part of 'scanner_screen.dart';

class _TopBar extends StatelessWidget {
  const _TopBar({
    required this.controller,
    required this.section,
    required this.onSectionChanged,
  });

  final ScannerController controller;
  final _WorkspaceSection section;
  final ValueChanged<_WorkspaceSection> onSectionChanged;

  @override
  Widget build(BuildContext context) {
    final cameraReady = controller.isCameraReady;
    final imageModeActive =
        section == _WorkspaceSection.scan &&
        controller.inputMode == InputMode.image;
    final cameraAttentionRelevant =
        section == _WorkspaceSection.scan && !imageModeActive;
    final (
      statusColor,
      statusIcon,
      statusText,
    ) = controller.hasActiveCameraIssue && cameraAttentionRelevant
        ? (AppColors.attention, Icons.videocam_off_rounded, '카메라 확인 필요')
        : controller.isCameraCheckActive && cameraAttentionRelevant
        ? (AppColors.muted, Icons.sync_rounded, '카메라 확인 중')
        : switch ((
            cameraReady,
            controller.cameraInitializing,
            cameraAttentionRelevant,
          )) {
            (true, _, _) => (
              AppColors.success,
              Icons.videocam_rounded,
              '카메라 연결됨',
            ),
            (false, true, _) => (
              AppColors.muted,
              Icons.sync_rounded,
              '카메라 확인 중',
            ),
            (false, false, false) when imageModeActive => (
              AppColors.muted,
              Icons.image_outlined,
              '이미지 입력 · 카메라 미연결',
            ),
            (false, false, false) => (
              AppColors.muted,
              Icons.videocam_off_outlined,
              '카메라 미연결',
            ),
            _ => (AppColors.attention, Icons.videocam_off_rounded, '카메라 확인 필요'),
          };
    return Container(
      key: const ValueKey('app-top-bar'),
      height: context.appTokens.headerHeight,
      color: AppColors.workspace,
      child: Stack(
        children: [
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            height: context.appTokens.navigationIndicatorThickness,
            child: const ColoredBox(
              key: ValueKey('app-top-bar-accent'),
              color: AppColors.primary,
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: AppSpacing.x6),
            child: Row(
              children: [
                const AppBrandMark(),
                const SizedBox(width: AppSpacing.x8),
                _NavigationItem(
                  key: const ValueKey('navigation-scan'),
                  label: '스캔',
                  selected: section == _WorkspaceSection.scan,
                  onTap: () => onSectionChanged(_WorkspaceSection.scan),
                ),
                _NavigationItem(
                  key: const ValueKey('navigation-activity'),
                  label: '활동',
                  selected: section == _WorkspaceSection.activity,
                  onTap: () => onSectionChanged(_WorkspaceSection.activity),
                ),
                const Spacer(),
                AppStatusBadge(
                  label: statusText,
                  icon: statusIcon,
                  color: statusColor,
                  liveRegion: statusText == '카메라 연결됨',
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _NavigationItem extends StatefulWidget {
  const _NavigationItem({
    super.key,
    required this.label,
    required this.selected,
    required this.onTap,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;

  @override
  State<_NavigationItem> createState() => _NavigationItemState();
}

class _NavigationItemState extends State<_NavigationItem> {
  bool _focused = false;

  @override
  Widget build(BuildContext context) {
    final tokens = context.appTokens;
    final tabRadius = BorderRadius.only(
      topLeft: Radius.circular(tokens.controlRadius),
      topRight: Radius.circular(tokens.controlRadius),
    );
    final outlineColor = _focused
        ? context.appComponents.focusRing
        : widget.selected
        ? AppColors.primary
        : Colors.transparent;
    return Semantics(
      container: true,
      excludeSemantics: true,
      button: true,
      selected: widget.selected,
      focused: _focused,
      label: widget.label,
      onTap: widget.onTap,
      child: SizedBox(
        width: tokens.navigationItemWidth,
        height: tokens.headerHeight,
        child: Padding(
          padding: EdgeInsets.only(top: tokens.navigationItemTopInset),
          child: Material(
            color: Colors.transparent,
            borderRadius: tabRadius,
            clipBehavior: Clip.antiAlias,
            child: InkWell(
              onTap: widget.onTap,
              excludeFromSemantics: true,
              borderRadius: tabRadius,
              onFocusChange: (focused) {
                if (_focused != focused) setState(() => _focused = focused);
              },
              child: Stack(
                fit: StackFit.expand,
                children: [
                  AnimatedContainer(
                    key: ValueKey('navigation-tab-${widget.label}'),
                    duration: MediaQuery.disableAnimationsOf(context)
                        ? Duration.zero
                        : tokens.motionFast,
                    curve: AppMotion.interactionCurve,
                    decoration: BoxDecoration(
                      color: widget.selected
                          ? AppColors.surface
                          : Colors.transparent,
                      border: Border.all(
                        color: outlineColor,
                        width: tokens.navigationIndicatorThickness,
                      ),
                      borderRadius: tabRadius,
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      widget.label,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        fontWeight: widget.selected
                            ? AppTypography.bold
                            : AppTypography.semibold,
                        color: widget.selected
                            ? AppColors.ink
                            : AppColors.muted,
                      ),
                    ),
                  ),
                  if (widget.selected && !_focused)
                    Positioned(
                      left: tokens.navigationIndicatorThickness,
                      right: tokens.navigationIndicatorThickness,
                      bottom: 0,
                      height: tokens.navigationIndicatorThickness,
                      child: const ColoredBox(color: AppColors.surface),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
