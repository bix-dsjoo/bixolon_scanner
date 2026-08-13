"""Compatibility alias for the canonical selective module."""

import sys

from ..experiments.detector import selective as _implementation

sys.modules[__name__] = _implementation
