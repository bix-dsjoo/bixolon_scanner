"""Compatibility alias for the canonical aggregate_detector module."""

import sys

from ..evaluation import aggregate_detector as _implementation

sys.modules[__name__] = _implementation
