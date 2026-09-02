class BixolonScannerException implements Exception {
  const BixolonScannerException(
    this.message, {
    this.code = 'SCANNER_CLIENT_ERROR',
    this.cause,
  });

  final String code;
  final String message;
  final Object? cause;

  @override
  String toString() => '$code: $message';
}

class BixolonWorkerStartException extends BixolonScannerException {
  const BixolonWorkerStartException(
    super.message, {
    super.code = 'WORKER_START_FAILED',
    super.cause,
  });
}

class BixolonTransportException extends BixolonScannerException {
  const BixolonTransportException(
    super.message, {
    super.code = 'WORKER_TRANSPORT_ERROR',
    super.cause,
  });
}

class BixolonContractException extends BixolonScannerException {
  const BixolonContractException(
    super.message, {
    super.code = 'WORKER_CONTRACT_ERROR',
    super.cause,
  });
}
