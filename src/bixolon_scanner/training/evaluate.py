"""Compatibility alias for the canonical classifier module."""

import sys

from ..evaluation import classifier as _implementation

sys.modules[__name__] = _implementation
