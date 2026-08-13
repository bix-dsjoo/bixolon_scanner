"""Compatibility alias for the canonical detector_nms_sweep module."""

import sys

from ..experiments.rpc200 import detector_nms_sweep as _implementation

sys.modules[__name__] = _implementation
