import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'core/design_system/theme.dart';
import 'core/design_system/scan_status.dart';
import 'features/scanner/scan_controller.dart';
import 'features/scanner/result_image.dart';
import 'features/scanner/square_preview.dart';
import 'shared/result.dart';
import 'shared/run_log.dart';
import 'dart:typed_data';

class LiteApp extends StatelessWidget {
  const LiteApp({super.key, required this.controller});
  final ScanController controller;
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'BIXOLON Bakery AI Scanner Lite',
    debugShowCheckedModeBanner: false,
    theme: buildAppTheme(),
    home: LiteWorkspace(controller: controller),
  );
}

class LiteWorkspace extends StatefulWidget {
  const LiteWorkspace({super.key, required this.controller});
  final ScanController controller;
  @override
  State<LiteWorkspace> createState() => _LiteWorkspaceState();
}

class _LiteWorkspaceState extends State<LiteWorkspace> {
  int tab = 0;
  @override
  Widget build(BuildContext context) => AnimatedBuilder(
    animation: widget.controller,
    builder: (context, _) {
      final c = widget.controller;
      return Scaffold(
        backgroundColor: AppColors.surface,
        body: SafeArea(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 6,
                ),
                child: Row(
                  children: [
                    SvgPicture.asset(
                      'assets/branding/bixolon_logo.svg',
                      width: 88,
                      semanticsLabel: 'BIXOLON',
                    ),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Text(
                        'Bakery AI Scanner Lite',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                    TextButton(
                      onPressed: () => setState(() => tab = 0),
                      child: Text(
                        '스캔',
                        style: TextStyle(
                          fontWeight: tab == 0
                              ? FontWeight.w700
                              : FontWeight.w400,
                        ),
                      ),
                    ),
                    TextButton(
                      onPressed: () => setState(() => tab = 1),
                      child: Text(
                        '실행 로그',
                        style: TextStyle(
                          fontWeight: tab == 1
                              ? FontWeight.w700
                              : FontWeight.w400,
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    const Text(
                      '0.1.18',
                      style: TextStyle(fontSize: 12, color: AppColors.muted),
                    ),
                  ],
                ),
              ),
              const Divider(height: 1),
              if (c.storageWarning)
                const Notice(
                  '일부 로그 저장에 실패했습니다. 실행 로그에서 저장 상태를 확인하고 내보내 주세요.',
                  error: true,
                ),
              if (c.workerMessage != null)
                Notice(c.workerMessage!, error: true),
              Expanded(child: tab == 0 ? _scan(c) : _logs(c)),
            ],
          ),
        ),
      );
    },
  );

  Widget _toolbar(ScanController c) => Padding(
    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
    child: Wrap(
      spacing: 8,
      runSpacing: 8,
      crossAxisAlignment: WrapCrossAlignment.center,
      children: [
        SizedBox(
          width: 260,
          child: DropdownButtonFormField<CameraDescription>(
            key: ValueKey(c.selected?.name),
            initialValue: c.selected,
            isExpanded: true,
            decoration: const InputDecoration(
              labelText: '촬영 카메라',
              isDense: true,
            ),
            items: c.cameras
                .map(
                  (camera) => DropdownMenuItem(
                    value: camera,
                    child: Text(camera.name, overflow: TextOverflow.ellipsis),
                  ),
                )
                .toList(),
            onChanged: c.canInput
                ? (camera) {
                    if (camera != null) c.selectCamera(camera);
                  }
                : null,
          ),
        ),
        IconButton(
          tooltip: '카메라 새로고침',
          onPressed: c.canInput ? c.refreshCameras : null,
          icon: const Icon(Icons.refresh),
        ),
        FilledButton.icon(
          key: const Key('capture'),
          onPressed: c.canInput && c.cameraReady
              ? () => c.run(camera: true)
              : null,
          icon: const Icon(Icons.camera_alt_outlined),
          label: const Text('촬영하고 분석'),
        ),
        OutlinedButton.icon(
          key: const Key('pick'),
          onPressed: c.canInput ? () => c.run(camera: false) : null,
          icon: const Icon(Icons.image_outlined),
          label: const Text('이미지 파일 선택'),
        ),
        const Text(
          '촬영 2048 × 2048',
          style: TextStyle(fontSize: 12, color: AppColors.muted),
        ),
        StatusPill(
          c.workerMessage != null
              ? 'AI 시작 실패'
              : c.workerReady
              ? 'AI 준비 완료'
              : 'AI 준비 중',
          c.workerMessage != null
              ? AppColors.error
              : c.workerReady
              ? AppColors.success
              : AppColors.muted,
        ),
      ],
    ),
  );

  Widget _scan(ScanController c) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      _toolbar(c),
      if (c.busy) ...[
        const LinearProgressIndicator(minHeight: 2),
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Text(c.progress, style: const TextStyle(fontSize: 12)),
        ),
      ],
      Expanded(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 4, 12, 12),
          child: LayoutBuilder(
            builder: (context, constraints) {
              final camera = _camera(c);
              final result = _image(c);
              final details = _details(c);
              if (constraints.maxWidth < 1050) {
                return SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(child: camera),
                          const SizedBox(width: 12),
                          Expanded(child: result),
                        ],
                      ),
                      const Divider(height: 20),
                      details,
                    ],
                  ),
                );
              }
              return Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    flex: 4,
                    child: SingleChildScrollView(child: camera),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    flex: 4,
                    child: SingleChildScrollView(child: result),
                  ),
                  const VerticalDivider(width: 24),
                  Expanded(
                    flex: 3,
                    child: SingleChildScrollView(child: details),
                  ),
                ],
              );
            },
          ),
        ),
      ),
    ],
  );

  Widget _camera(ScanController c) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      const Padding(
        padding: EdgeInsets.only(bottom: 8),
        child: Text(
          '카메라',
          style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
        ),
      ),
      AspectRatio(
        aspectRatio: 1,
        child: ColoredBox(
          color: AppColors.preview,
          child: c.cameraBusy
              ? const Center(child: CircularProgressIndicator())
              : c.cameraReady && c.inputs.camera != null
              ? SquarePreview(controller: c.inputs.camera!)
              : Center(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Text(
                      c.cameraMessage ?? '카메라 미리보기',
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 13,
                      ),
                    ),
                  ),
                ),
        ),
      ),
    ],
  );

  Widget _image(ScanController c) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      const Padding(
        padding: EdgeInsets.only(bottom: 8),
        child: Text(
          '결과 이미지',
          style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
        ),
      ),
      if (c.image != null)
        ResultImage(bytes: c.image!, objects: c.result?.objects ?? const [])
      else
        const AspectRatio(
          aspectRatio: 1,
          child: ColoredBox(
            color: AppColors.elevated,
            child: Center(
              child: Padding(
                padding: EdgeInsets.all(12),
                child: Text(
                  '사진을 촬영하거나 이미지 파일을 선택하세요.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: AppColors.muted, fontSize: 13),
                ),
              ),
            ),
          ),
        ),
    ],
  );

  Widget _details(ScanController c) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      Row(
        children: [
          const Expanded(
            child: Text(
              '추론 결과',
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
            ),
          ),
          if (c.lastRecord != null) StoragePill(c.lastRecord!.storageState),
        ],
      ),
      const SizedBox(height: 12),
      if (c.result != null)
        ResultDetails(c.result!)
      else
        Text(
          c.busy ? '분석 중입니다.' : '입력 대기',
          style: const TextStyle(color: AppColors.muted, fontSize: 13),
        ),
    ],
  );

  Widget _logs(ScanController c) => Padding(
    padding: const EdgeInsets.all(12),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '실행 로그',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
                  ),
                  SizedBox(height: 6),
                  Text(
                    '최근 200건 · 이미지 30일 보관 · 메타데이터 유지',
                    style: TextStyle(color: AppColors.muted, fontSize: 13),
                  ),
                ],
              ),
            ),
            OutlinedButton.icon(
              onPressed: c.busy || c.exporting ? null : c.exportLogs,
              icon: const Icon(Icons.file_download_outlined),
              label: Text(c.exporting ? '내보내는 중' : '전체 로그 내보내기'),
            ),
          ],
        ),
        if (c.exportMessage != null)
          Padding(
            padding: const EdgeInsets.only(top: 12),
            child: Text(c.exportMessage!),
          ),
        const SizedBox(height: 18),
        const Divider(height: 1),
        Expanded(
          child: c.recent.isEmpty
              ? const Center(
                  child: Text(
                    '아직 실행 기록이 없습니다.',
                    style: TextStyle(color: AppColors.muted),
                  ),
                )
              : ListView.separated(
                  itemCount: c.recent.length,
                  separatorBuilder: (_, index) => const Divider(height: 1),
                  itemBuilder: (context, index) {
                    final record = c.recent[index];
                    return ExpansionTile(
                      tilePadding: const EdgeInsets.symmetric(
                        horizontal: 4,
                        vertical: 6,
                      ),
                      title: Wrap(
                        spacing: 12,
                        runSpacing: 6,
                        crossAxisAlignment: WrapCrossAlignment.center,
                        children: [
                          Text(
                            _time(record.occurredAt),
                            style: const TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          StatusPill(
                            record.result?.status ?? '입력 접수 · 결과 미완료',
                            scanStatusColor(record.result?.status),
                          ),
                          StoragePill(record.storageState),
                        ],
                      ),
                      subtitle: Padding(
                        padding: const EdgeInsets.only(top: 6),
                        child: Text(
                          '${record.inputMethod == 'camera' ? '카메라' : '파일'} · ${record.inputName}',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      children: [
                        Padding(
                          padding: const EdgeInsets.fromLTRB(12, 0, 12, 18),
                          child: LogRecordView(record: record, logs: c.logs),
                        ),
                      ],
                    );
                  },
                ),
        ),
      ],
    ),
  );
}

String _time(DateTime date) =>
    date.toLocal().toIso8601String().substring(0, 19).replaceFirst('T', ' ');

class LogRecordView extends StatefulWidget {
  const LogRecordView({super.key, required this.record, required this.logs});
  final RunRecord record;
  final RunLogs logs;
  @override
  State<LogRecordView> createState() => _LogRecordViewState();
}

class _LogRecordViewState extends State<LogRecordView> {
  late Future<Uint8List?> image;
  @override
  void initState() {
    super.initState();
    image = widget.logs.imageFor(widget.record);
  }

  @override
  void didUpdateWidget(LogRecordView old) {
    super.didUpdateWidget(old);
    if (old.record.attemptId != widget.record.attemptId ||
        old.record.imageReference != widget.record.imageReference) {
      image = widget.logs.imageFor(widget.record);
    }
  }

  @override
  Widget build(BuildContext context) => LayoutBuilder(
    builder: (context, constraints) {
      final photo = FutureBuilder<Uint8List?>(
        future: image,
        builder: (context, snapshot) {
          if (snapshot.hasData) {
            return ResultImage(
              bytes: snapshot.data!,
              objects: widget.record.result?.objects ?? const [],
            );
          }
          return Padding(
            padding: const EdgeInsets.all(12),
            child: Text(
              snapshot.connectionState == ConnectionState.waiting
                  ? '이미지를 불러오는 중입니다'
                  : widget.record.imageStorageStatus == 'failed'
                  ? '이미지 저장에 실패했습니다.'
                  : '보관된 이미지가 없습니다. 이전 기록 또는 30일 보관 기간이 지난 이미지입니다.',
              style: const TextStyle(fontSize: 13, color: AppColors.muted),
            ),
          );
        },
      );
      final details = widget.record.result != null
          ? ResultDetails(widget.record.result!)
          : const Text('입력은 기록됐지만 최종 결과가 기록되지 않았습니다.');
      if (constraints.maxWidth < 700) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [photo, const SizedBox(height: 12), details],
        );
      }
      return Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(width: constraints.maxWidth * .4, child: photo),
          const SizedBox(width: 16),
          Expanded(child: details),
        ],
      );
    },
  );
}

class StatusPill extends StatelessWidget {
  const StatusPill(this.label, this.color, {super.key});
  final String label;
  final Color color;
  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
    decoration: BoxDecoration(
      color: color.withValues(alpha: .08),
      borderRadius: BorderRadius.circular(6),
    ),
    child: Text(
      label,
      style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600),
    ),
  );
}

class StoragePill extends StatelessWidget {
  const StoragePill(this.state, {super.key});
  final String state;
  @override
  Widget build(BuildContext context) => StatusPill(
    switch (state) {
      'saved' => '로그 저장 완료',
      'failed' => '로그 저장 실패',
      _ => '로그 저장 중',
    },
    state == 'failed'
        ? AppColors.error
        : state == 'saved'
        ? AppColors.success
        : AppColors.muted,
  );
}

class Notice extends StatelessWidget {
  const Notice(this.message, {super.key, this.error = false});
  final String message;
  final bool error;
  @override
  Widget build(BuildContext context) => Container(
    width: double.infinity,
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: error ? AppColors.errorSoft : AppColors.attentionSoft,
      borderRadius: BorderRadius.circular(8),
    ),
    child: Text(
      message,
      style: TextStyle(
        color: error ? AppColors.error : AppColors.attention,
        fontSize: 14,
      ),
    ),
  );
}

class ResultDetails extends StatelessWidget {
  const ResultDetails(this.result, {super.key});
  final ScanResult result;
  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      Text(
        result.status,
        style: TextStyle(
          fontSize: 22,
          fontWeight: FontWeight.w700,
          color: scanStatusColor(result.status),
        ),
      ),
      const SizedBox(height: 8),
      if (result.status == 'IMAGE_RECAPTURE')
        const Notice('이미지 전체를 다시 촬영해 주세요.', error: true)
      else if (result.status == 'ERROR')
        Notice(
          result.origin == 'worker'
              ? '시스템 또는 입력 오류입니다. 촬영 품질에 대한 재촬영 판정이 아닙니다.'
              : '입력 또는 AI 연결 오류입니다. 모델 판정이 완료되지 않았습니다.',
          error: true,
        ),
      if (result.reasons.isNotEmpty)
        Padding(
          padding: const EdgeInsets.only(top: 10),
          child: Text(
            result.reasons.join(' · '),
            style: const TextStyle(color: AppColors.muted, fontSize: 12),
          ),
        ),
      if (result.objects.isNotEmpty) ...[
        const SizedBox(height: 18),
        for (var i = 0; i < result.objects.length; i++)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.only(top: 5),
                  child: Text(
                    '객체 ${i + 1}',
                    style: const TextStyle(
                      fontSize: 13,
                      color: AppColors.muted,
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      StatusPill(
                        result.objects[i].status,
                        scanStatusColor(result.objects[i].status),
                      ),
                      if (result.objects[i].top3.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(top: 6),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text(
                                'Top-3',
                                style: TextStyle(
                                  fontSize: 11,
                                  color: AppColors.muted,
                                ),
                              ),
                              for (
                                var rank = 0;
                                rank < result.objects[i].top3.length;
                                rank++
                              )
                                Text(
                                  '${rank + 1}. ${result.objects[i].top3[rank].className}',
                                  style: const TextStyle(fontSize: 13),
                                ),
                            ],
                          ),
                        ),
                      if (result.objects[i].reasons.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(top: 4),
                          child: Text(
                            result.objects[i].reasons.join(' · '),
                            style: const TextStyle(
                              fontSize: 12,
                              color: AppColors.muted,
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ),
          ),
      ],
      const SizedBox(height: 12),
      const Divider(),
      const SizedBox(height: 8),
      _metadata('요청 ID', result.requestId),
      _metadata(
        '처리 시간',
        result.processingMs == null
            ? '미실행'
            : '${result.processingMs!.toStringAsFixed(1)} ms',
      ),
      for (final entry in result.versions.entries)
        _metadata(switch (entry.key) {
          'worker_version' => 'Worker',
          'detector_version' => 'Detector',
          'classifier_version' => 'Classifier',
          'embedder_version' => 'Embedder',
          'detector_policy_version' => 'Detector policy',
          'classifier_policy_version' => 'Classifier policy',
          _ => 'Catalog',
        }, entry.value ?? '미실행'),
    ],
  );
  Widget _metadata(String label, String value) => Padding(
    padding: const EdgeInsets.only(bottom: 7),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 120,
          child: Text(
            label,
            style: const TextStyle(color: AppColors.muted, fontSize: 12),
          ),
        ),
        Expanded(child: Text(value, style: const TextStyle(fontSize: 12))),
      ],
    ),
  );
}
