part of 'activity_screen.dart';

class _PerformanceDetail extends StatelessWidget {
  const _PerformanceDetail({required this.performance});

  final ScanPerformanceMetrics performance;

  @override
  Widget build(BuildContext context) {
    final worker = performance.worker;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _DetailLine(
          label: '전체 체감시간',
          value: _formatMilliseconds(performance.endToEndMs),
        ),
        _DetailLine(
          label: '입력 이미지',
          value:
              '${performance.imageWidth} × ${performance.imageHeight}  ·  ${_formatBytes(performance.imageSizeBytes)}',
        ),
        if (performance.provider case final provider?)
          _DetailLine(label: '실행 장치', value: provider),
        const _DiagnosticSectionTitle(label: '앱 · 카메라'),
        if (performance.cameraCaptureMs > 0)
          _DetailLine(
            label: '카메라 촬영',
            value: _formatMilliseconds(performance.cameraCaptureMs),
          ),
        _DetailLine(
          label: '파일 읽기',
          value: _formatMilliseconds(performance.fileReadMs),
        ),
        _DetailLine(
          label: 'Flutter 디코딩',
          value: _formatMilliseconds(performance.flutterImageDecodeMs),
        ),
        _DetailLine(
          label: 'Worker 통신 전체',
          value: _formatMilliseconds(performance.apiTotalMs),
        ),
        _DetailLine(
          label: '준비 상태 확인',
          value: _formatMilliseconds(performance.readinessMs),
        ),
        _DetailLine(
          label: 'HTTP 요청 준비',
          value: _formatMilliseconds(performance.requestBuildMs),
        ),
        _DetailLine(
          label: 'HTTP 전송·응답',
          value: _formatMilliseconds(performance.httpRoundTripMs),
        ),
        _DetailLine(
          label: '응답 본문 읽기',
          value:
              '${_formatMilliseconds(performance.responseBodyReadMs)} (HTTP 포함)',
        ),
        _DetailLine(
          label: '응답 JSON 변환',
          value: _formatMilliseconds(performance.responseParseMs),
        ),
        _DetailLine(
          label: '결과 데이터 변환',
          value: _formatMilliseconds(performance.resultMappingMs),
        ),
        _DetailLine(
          label: '결과 첫 화면',
          value: _formatMilliseconds(performance.resultFirstFrameMs),
        ),
        if (worker != null) ...[
          const _DiagnosticSectionTitle(label: 'Worker 내부'),
          _DetailLine(
            label: 'Worker 전체',
            value: _formatMilliseconds(worker.requestTotalMs),
          ),
          _DetailLine(
            label: '업로드 읽기',
            value: _formatMilliseconds(worker.uploadReadMs),
          ),
          _DetailLine(
            label: '이미지 디코딩',
            value: _formatMilliseconds(worker.decodeMs),
          ),
          _DetailLine(
            label: '실행 대기',
            value: _formatMilliseconds(worker.queueWaitMs),
          ),
          _DetailLine(
            label: '분석 파이프라인',
            value: _formatMilliseconds(worker.pipelineMs),
          ),
          _DetailLine(
            label: 'Detector',
            value: _formatMilliseconds(worker.detectorMs),
          ),
          _DetailLine(
            label: 'Classifier',
            value: _formatMilliseconds(worker.classifierMs),
          ),
          _DetailLine(
            label: '판정·후처리',
            value: _formatMilliseconds(worker.decisionMs),
          ),
          _DetailLine(
            label: '서버 기타',
            value: _formatMilliseconds(worker.serverOverheadMs),
          ),
        ],
      ],
    );
  }
}

String _formatMilliseconds(double value) => '${value.toStringAsFixed(1)} ms';

String _formatBytes(int value) {
  if (value >= 1024 * 1024) {
    return '${(value / (1024 * 1024)).toStringAsFixed(1)} MB';
  }
  if (value >= 1024) return '${(value / 1024).toStringAsFixed(1)} KB';
  return '$value B';
}

class _DiagnosticSectionTitle extends StatelessWidget {
  const _DiagnosticSectionTitle({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: AppSpacing.x4 + AppSpacing.x1),
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
          const Expanded(child: Divider()),
        ],
      ),
    );
  }
}

class _DiagnosticItem extends StatelessWidget {
  const _DiagnosticItem({required this.index, required this.item});

  final int index;
  final ScanLogItemSummary item;

  @override
  Widget build(BuildContext context) {
    final confidence = '${(item.confidence * 100).toStringAsFixed(1)}%';
    final reasonLabel = activityItemReasonLabel(item);
    return Padding(
      padding: const EdgeInsets.only(top: AppSpacing.x3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: context.appTokens.metadataLabelWidth,
            child: Text(
              '$index번 진단',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${item.itemId}  ·  $confidence',
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    fontWeight: AppTypography.bold,
                  ),
                ),
                Text(
                  '확정 방식 · ${activityConfirmationMethodLabel(item.confirmationMethod)}',
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(
                    context,
                  ).textTheme.bodySmall?.copyWith(color: AppColors.muted),
                ),
                if (reasonLabel.isNotEmpty)
                  Text(
                    '$reasonLabel · ${item.reasonCodes.join(', ')}',
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(
                      context,
                    ).textTheme.bodySmall?.copyWith(color: AppColors.muted),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _DetailLine extends StatelessWidget {
  const _DetailLine({
    required this.label,
    required this.value,
    this.selectable = false,
  });

  final String label;
  final String value;
  final bool selectable;

  @override
  Widget build(BuildContext context) {
    final valueStyle = Theme.of(
      context,
    ).textTheme.bodyMedium?.copyWith(fontWeight: AppTypography.bold);
    return Padding(
      padding: const EdgeInsets.only(top: AppSpacing.x3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: context.appTokens.metadataLabelWidth,
            child: Text(label, style: Theme.of(context).textTheme.bodySmall),
          ),
          Expanded(
            child: selectable
                ? SelectableText(value, style: valueStyle)
                : Text(value, style: valueStyle),
          ),
        ],
      ),
    );
  }
}

class _LogItem extends StatelessWidget {
  const _LogItem({required this.index, required this.item});

  final int index;
  final ScanLogItemSummary item;

  @override
  Widget build(BuildContext context) {
    final removed = item.disposition == OperatorObjectDisposition.remove;
    final added = item.disposition == OperatorObjectDisposition.add;
    final modelName = item.modelProduct?.displayName ?? '상품 없음';
    final finalName = removed ? '삭제' : activityProductLabel(item);
    final productChanged =
        !removed &&
        !added &&
        item.modelProduct?.classId != null &&
        item.classId != null &&
        item.modelProduct!.classId != item.classId;
    final bboxChanged =
        !removed &&
        !added &&
        item.modelBbox != null &&
        item.finalBbox != null &&
        item.modelBbox != item.finalBbox;
    final changeLabel = removed
        ? '검출 삭제 · $modelName'
        : added
        ? '박스 추가'
        : productChanged
        ? '상품 변경 · $modelName → $finalName'
        : bboxChanged
        ? '박스 위치·크기 수정'
        : item.userModified
        ? '결과 수정'
        : '수정 없음';
    final issueLabels = item.issueCodes
        .map(activityOperatorIssueLabel)
        .join(', ');
    return Padding(
      padding: const EdgeInsets.only(top: AppSpacing.x3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: context.appTokens.metadataLabelWidth,
            child: Text(
              '$index번 상품',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  finalName,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    fontWeight: AppTypography.bold,
                  ),
                ),
                Text(
                  changeLabel,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: item.userModified
                        ? AppColors.attention
                        : AppColors.muted,
                  ),
                ),
                if (issueLabels.isNotEmpty)
                  Text(
                    issueLabels,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(
                      context,
                    ).textTheme.bodySmall?.copyWith(color: AppColors.muted),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

String _formatDate(DateTime value) {
  final local = value.toLocal();
  return '${local.year}.${local.month.toString().padLeft(2, '0')}.${local.day.toString().padLeft(2, '0')}';
}

String _formatTime(DateTime value) {
  final local = value.toLocal();
  return '${local.hour.toString().padLeft(2, '0')}:${local.minute.toString().padLeft(2, '0')}:${local.second.toString().padLeft(2, '0')}';
}

int _modifiedItemCount(List<ScanLogItemSummary> items) =>
    items.where((item) => item.userModified).length;
