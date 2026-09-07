import 'package:camera/camera.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:bakery_scanner_lite/shared/camera_profile.dart';

void main() {
  test(
    'Parentheses and native angle brackets identify the same USB device',
    () {
      final native = CameraDescription(
        name: configuredCameraName
            .replaceFirst(' (', ' <')
            .replaceFirst(')', '>')
            .toUpperCase(),
        lensDirection: CameraLensDirection.external,
        sensorOrientation: 0,
      );
      expect(preferredCamera([native]), native);
    },
  );
  const other = CameraDescription(
    name: 'Other camera',
    lensDirection: CameraLensDirection.external,
    sensorOrientation: 0,
  );
  const configured = CameraDescription(
    name: configuredCameraName,
    lensDirection: CameraLensDirection.external,
    sensorOrientation: 0,
  );
  test('Configured Windows device wins regardless of enumeration order', () {
    expect(preferredCamera([other, configured]), configured);
    expect(
      preferredCamera([configured, other], selectedName: other.name),
      other,
    );
  });
  test(
    'A missing configured device does not silently open a different camera',
    () {
      expect(preferredCamera([other]), isNull);
      expect(preferredCamera([]), isNull);
    },
  );
}
