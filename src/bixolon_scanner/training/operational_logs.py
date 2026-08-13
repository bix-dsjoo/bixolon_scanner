"""Compatibility alias for the canonical operational_logs module."""

import sys

from ..operations import operational_logs as _implementation

sys.modules[__name__] = _implementation
