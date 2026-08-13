"""Compatibility alias for the canonical worker_gate module."""

import sys

from ..experiments.rpc200 import worker_gate as _implementation

sys.modules[__name__] = _implementation
