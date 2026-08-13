"""Compatibility alias for the canonical ten_shot module."""

import sys

from ..experiments.bread import ten_shot as _implementation

sys.modules[__name__] = _implementation
