"""Compatibility alias for the canonical data_scale module."""

import sys

from ..experiments.rpc200 import data_scale as _implementation

sys.modules[__name__] = _implementation
