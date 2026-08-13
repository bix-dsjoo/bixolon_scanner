"""Compatibility alias for the canonical small_object_recovery module."""

import sys

from ..experiments.rpc200 import small_object_recovery as _implementation

sys.modules[__name__] = _implementation
