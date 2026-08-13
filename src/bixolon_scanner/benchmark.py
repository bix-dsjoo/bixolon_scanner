"""Compatibility alias for the canonical benchmark module."""

import sys

from .evaluation import benchmark as _implementation

sys.modules[__name__] = _implementation
