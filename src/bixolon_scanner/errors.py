"""Compatibility imports for scanner errors."""

from .contracts.errors import (
    CorruptImageError,
    ImageTooLargeError,
    InputError,
    MissingImageError,
    ModelExecutionError,
    PackageValidationError,
    ProviderInitializationError,
    ScannerError,
    UnsupportedImageFormatError,
)

__all__ = [
    "CorruptImageError",
    "ImageTooLargeError",
    "InputError",
    "MissingImageError",
    "ModelExecutionError",
    "PackageValidationError",
    "ProviderInitializationError",
    "ScannerError",
    "UnsupportedImageFormatError",
]
