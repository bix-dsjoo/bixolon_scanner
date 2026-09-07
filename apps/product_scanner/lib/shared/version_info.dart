/// One product version embedded in every BIXOLON Bakery AI Scanner component.
abstract final class VersionInfo {
  static const current = String.fromEnvironment(
    'BIXOLON_VERSION',
    defaultValue: '0.1.16',
  );
}
