"""Compatibility alias for the canonical detector module."""

import sys

from ..evaluation import detector as _implementation

sys.modules[__name__] = _implementation
