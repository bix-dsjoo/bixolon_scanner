"""Compatibility alias for the canonical detector_fp_audit module."""

import sys

from ..experiments.rpc200 import detector_fp_audit as _implementation

sys.modules[__name__] = _implementation
