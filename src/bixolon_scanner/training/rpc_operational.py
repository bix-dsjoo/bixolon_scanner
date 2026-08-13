"""Compatibility alias for the canonical operational module."""

import sys

from ..experiments.rpc200 import operational as _implementation

sys.modules[__name__] = _implementation
