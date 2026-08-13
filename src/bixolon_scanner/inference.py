"""Compatibility alias for :mod:`bixolon_scanner.runtime.onnx`."""

import sys

from .runtime import onnx as _implementation

sys.modules[__name__] = _implementation
