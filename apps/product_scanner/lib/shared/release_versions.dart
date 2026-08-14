/// Release composition values embedded in BIXOLON SCANNER 1.0.0+2.
abstract final class ReleaseVersions {
  static const app = String.fromEnvironment(
    'BIXOLON_APP_VERSION',
    defaultValue: '1.0.0+2',
  );
  static const worker = String.fromEnvironment(
    'BIXOLON_WORKER_VERSION',
    defaultValue: '1.0.0',
  );
  static const detector = String.fromEnvironment(
    'BIXOLON_DETECTOR_VERSION',
    defaultValue: '1.0.0',
  );
  static const classifier = String.fromEnvironment(
    'BIXOLON_CLASSIFIER_VERSION',
    defaultValue: '1.0.0',
  );
  static const dataset = String.fromEnvironment(
    'BIXOLON_DATASET_VERSION',
    defaultValue: 'bread-1.0-a52b4faa3e20',
  );
}
