"""Compatibility alias for the canonical threshold_sweep module."""

import sys

from ..experiments.rpc200 import threshold_sweep as _implementation

sys.modules[__name__] = _implementation
