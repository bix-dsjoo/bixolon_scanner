"""Compatibility alias for the canonical operational module."""

import sys

from ..evaluation import operational as _implementation

sys.modules[__name__] = _implementation
