/// Release composition values embedded in BIXOLON SCANNER 2.0.0+4.
abstract final class ReleaseVersions {
  static const app = String.fromEnvironment(
    'BIXOLON_APP_VERSION',
    defaultValue: '2.0.0+4',
  );
  static const worker = String.fromEnvironment(
    'BIXOLON_WORKER_VERSION',
    defaultValue: '2.0.0',
  );
  static const detector = String.fromEnvironment(
    'BIXOLON_DETECTOR_VERSION',
    defaultValue: '2.0.0',
  );
  static const classifier = String.fromEnvironment(
    'BIXOLON_CLASSIFIER_VERSION',
    defaultValue: '2.0.0',
  );
  static const dataset = String.fromEnvironment(
    'BIXOLON_DATASET_VERSION',
    defaultValue: 'bread-scanner-2.0.0-development-415-rc.8',
  );
}
