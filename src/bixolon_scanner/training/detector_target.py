"""Compatibility alias for the canonical target module."""

import sys

from ..experiments.detector import target as _implementation

sys.modules[__name__] = _implementation
