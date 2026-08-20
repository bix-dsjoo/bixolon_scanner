/// One product version embedded in every BIXOLON Scanner component.
abstract final class VersionInfo {
  static const current = String.fromEnvironment(
    'BIXOLON_VERSION',
    defaultValue: '0.0.1',
  );
}
