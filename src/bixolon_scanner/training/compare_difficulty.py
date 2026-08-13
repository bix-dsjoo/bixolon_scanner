"""Compatibility alias for the canonical compare_difficulty module."""

import sys

from ..evaluation import compare_difficulty as _implementation

sys.modules[__name__] = _implementation
