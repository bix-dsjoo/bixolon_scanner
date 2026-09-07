import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import '../../shared/input.dart';
import '../../shared/camera_profile.dart';
import '../../shared/result.dart';
import '../../shared/run_log.dart';
import '../../shared/worker_client.dart';

class ScanController extends ChangeNotifier {
  ScanController(this.inputs, this.scanner, this.logs);
  final PhotoInputs inputs;
  final ScannerService scanner;
  final RunLogs logs;
  List<CameraDescription> cameras = [];
  CameraDescription? selected;
  Uint8List? image;
  ScanResult? result;
  RunRecord? lastRecord;
  List<RunRecord> recent = [];
  bool busy = false;
  bool cameraBusy = false;
  bool cameraReady = false;
  bool workerReady = false;
  bool initializing = true;
  bool exporting = false;
  bool _closed = false;
  String progress = '';
  String? cameraMessage;
  String? workerMessage;
  String? exportMessage;
  bool get canInput =>
      !_closed && !busy && !cameraBusy && !initializing && !exporting;
  bool get storageWarning =>
      logs.storageWarning || lastRecord?.storageState == 'failed';
  void _emit() {
    if (!_closed) notifyListeners();
  }

  Future<void> initialize() async {
    try {
      recent = await logs.recent();
    } catch (_) {
      /* Inputs remain available. */
    }
    initializing = false;
    _emit();
    await Future.wait([refreshCameras(), _startWorker()]);
  }

  Future<void> _startWorker() async {
    try {
      await scanner.start();
      workerReady = true;
    } catch (_) {
      workerMessage = 'AI 실행기를 시작하지 못했습니다. 앱을 다시 실행해 주세요.';
    }
    _emit();
  }

  Future<void> refreshCameras() async {
    if (busy || cameraBusy) return;
    cameraBusy = true;
    cameraReady = false;
    cameraMessage = null;
    _emit();
    try {
      cameras = await inputs.cameras();
      if (cameras.isEmpty) {
        await inputs.dispose();
        selected = null;
        cameraMessage = '연결된 카메라가 없습니다. 이미지 파일을 선택할 수 있습니다.';
      } else {
        selected = preferredCamera(cameras, selectedName: selected?.name);
        if (selected == null) {
          await inputs.dispose();
          cameraMessage = '설정된 ZHWY 카메라가 없습니다. 촬영 카메라를 직접 선택해 주세요.';
        } else {
          await inputs.select(selected!);
          cameraReady = true;
        }
      }
    } catch (_) {
      cameraMessage = '카메라를 연결하지 못했습니다. 연결과 권한을 확인해 주세요.';
    } finally {
      cameraBusy = false;
      _emit();
    }
  }

  Future<void> selectCamera(CameraDescription description) async {
    if (!canInput) return;
    cameraBusy = true;
    cameraReady = false;
    selected = description;
    cameraMessage = null;
    _emit();
    try {
      await inputs.select(description);
      cameraReady = true;
    } catch (_) {
      cameraMessage = '선택한 카메라를 열 수 없습니다.';
    } finally {
      cameraBusy = false;
      _emit();
    }
  }

  Future<void> run({required bool camera}) async {
    if (!canInput || (camera && !cameraReady)) return;
    busy = true;
    progress = camera ? '사진을 촬영하고 있습니다' : '이미지 파일을 선택해 주세요';
    exportMessage = null;
    _emit();
    final id = 'lite-${DateTime.now().microsecondsSinceEpoch}';
    final at = DateTime.now();
    var name = camera ? (selected?.name ?? '카메라') : '파일 입력';
    final method = camera ? 'camera' : 'file';
    ScanResult? outcome;
    try {
      final photo = camera ? await inputs.capture() : await inputs.pick();
      if (photo == null) return;
      if (!camera) name = photo.name;
      image = photo.bytes;
      result = null;
      progress = 'AI가 이미지를 분석하고 있습니다';
      _emit();
      final started = RunRecord(
        attemptId: id,
        occurredAt: at,
        inputMethod: method,
        inputName: name,
      );
      try {
        lastRecord = await logs.append(started, image: photo.bytes);
      } catch (_) {
        lastRecord = started.stored('failed', 'LOG_WRITE_FAILED');
      }
      _emit();
      try {
        outcome = await scanner.scan(photo.bytes, photo.name, id);
      } catch (_) {
        outcome = ScanResult.clientError(id, 'CLIENT_INFERENCE_FAILED');
      }
    } on InputReadFailure catch (failure) {
      name = failure.name;
      image = null;
      outcome = ScanResult.clientError(id, failure.reason);
    } catch (_) {
      image = null;
      outcome = ScanResult.clientError(
        id,
        camera ? 'CLIENT_CAMERA_CAPTURE_FAILED' : 'CLIENT_FILE_READ_FAILED',
      );
    } finally {
      // A cancelled file chooser is not an input; preserve the previous result and log.
      if (outcome != null) {
        result = outcome;
        await _complete(id, at, method, name);
      }
      busy = false;
      progress = '';
      _emit();
    }
  }

  Future<void> _complete(
    String id,
    DateTime at,
    String method,
    String name,
  ) async {
    _emit(); // A log failure must never prevent the result from being displayed.
    try {
      lastRecord = await logs.append(
        RunRecord(
          attemptId: id,
          occurredAt: at,
          inputMethod: method,
          inputName: name,
          result: result,
          imageReference: lastRecord?.attemptId == id
              ? lastRecord?.imageReference
              : null,
          imageStorageStatus: lastRecord?.attemptId == id
              ? lastRecord!.imageStorageStatus
              : 'not_saved',
        ),
      );
      recent = await logs.recent();
    } catch (_) {
      lastRecord = RunRecord(
        attemptId: id,
        occurredAt: at,
        inputMethod: method,
        inputName: name,
        result: result,
        storageState: 'failed',
        storageReason: 'LOG_WRITE_FAILED',
        imageReference: lastRecord?.attemptId == id
            ? lastRecord?.imageReference
            : null,
        imageStorageStatus: lastRecord?.attemptId == id
            ? lastRecord!.imageStorageStatus
            : 'not_saved',
      );
      recent = [lastRecord!, ...recent].take(200).toList();
    }
  }

  Future<void> exportLogs() async {
    if (exporting || busy) return;
    exporting = true;
    exportMessage = null;
    _emit();
    try {
      final path = await inputs.exportPath();
      if (path != null) {
        await logs.exportTo(path);
        exportMessage = '로그를 내보냈습니다.';
      }
    } catch (_) {
      exportMessage = '로그 내보내기에 실패했습니다. 저장 위치를 확인해 주세요.';
    } finally {
      exporting = false;
      _emit();
    }
  }

  Future<void> close() async {
    _closed = true;
    await inputs.dispose();
    await scanner.close();
  }
}
