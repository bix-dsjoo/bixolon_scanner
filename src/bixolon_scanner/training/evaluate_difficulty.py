"""Compatibility alias for the canonical difficulty module."""

import sys

from ..evaluation import difficulty as _implementation

sys.modules[__name__] = _implementation
