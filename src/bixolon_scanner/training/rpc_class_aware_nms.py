"""Compatibility alias for the canonical class_aware_nms module."""

import sys

from ..experiments.rpc200 import class_aware_nms as _implementation

sys.modules[__name__] = _implementation
