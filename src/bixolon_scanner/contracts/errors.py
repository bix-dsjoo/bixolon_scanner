from __future__ import annotations


class ScannerError(Exception):
    reason_code = "WORKER_ERROR"
    http_status = 500


class InputError(ScannerError):
    http_status = 400


class MissingImageError(InputError):
    reason_code = "MISSING_IMAGE_FIELD"
    http_status = 422


class ImageTooLargeError(InputError):
    reason_code = "IMAGE_TOO_LARGE"
    http_status = 413


class UnsupportedImageFormatError(InputError):
    reason_code = "UNSUPPORTED_IMAGE_FORMAT"
    http_status = 415


class CorruptImageError(InputError):
    reason_code = "CORRUPT_IMAGE"
    http_status = 422


class PackageValidationError(ScannerError):
    reason_code = "MODEL_PACKAGE_INVALID"


class ProviderInitializationError(ScannerError):
    reason_code = "PROVIDER_INITIALIZATION_FAILED"
    http_status = 503


class ModelExecutionError(ScannerError):
    reason_code = "MODEL_EXECUTION_FAILED"
