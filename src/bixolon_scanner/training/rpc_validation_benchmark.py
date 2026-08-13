"""Compatibility alias for the canonical validation_benchmark module."""

import sys

from ..experiments.rpc200 import validation_benchmark as _implementation

sys.modules[__name__] = _implementation
