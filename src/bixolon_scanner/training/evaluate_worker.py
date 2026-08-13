"""Compatibility alias for the canonical worker module."""

import sys

from ..evaluation import worker as _implementation

sys.modules[__name__] = _implementation
