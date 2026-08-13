"""Compatibility alias for the canonical parity module."""

import sys

from ..evaluation import parity as _implementation

sys.modules[__name__] = _implementation
