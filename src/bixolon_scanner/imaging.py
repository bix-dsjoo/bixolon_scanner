"""Compatibility alias for :mod:`bixolon_scanner.runtime.imaging`."""

import sys

from .runtime import imaging as _implementation

sys.modules[__name__] = _implementation
