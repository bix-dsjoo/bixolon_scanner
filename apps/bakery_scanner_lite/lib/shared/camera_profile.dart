import 'package:camera/camera.dart';

// Store camera selected by the user. Preserve the Windows device identifier.
const configuredCameraName =
    r'ZHWY Camera (\\?\usb#vid_0bda&pid_3038&mi_00#6&1eb0049&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global)';

String _deviceIdentity(String name) {
  // camera_windows uses "Friendly name <device ID>"; the shared config uses ().
  for (final pair in [(' <', '>'), (' (', ')')]) {
    final start = name.lastIndexOf(pair.$1);
    if (start >= 0 && name.endsWith(pair.$2)) {
      return name.substring(start + 2, name.length - 1).toLowerCase();
    }
  }
  return name.toLowerCase();
}

CameraDescription? preferredCamera(
  List<CameraDescription> available, {
  String? selectedName,
}) {
  for (final name in [selectedName, configuredCameraName]) {
    if (name == null) continue;
    for (final camera in available) {
      if (_deviceIdentity(camera.name) == _deviceIdentity(name)) return camera;
    }
  }
  return null;
}
