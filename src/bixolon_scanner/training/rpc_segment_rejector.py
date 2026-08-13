"""Compatibility alias for the canonical segment_rejector module."""

import sys

from ..experiments.rpc200 import segment_rejector as _implementation

sys.modules[__name__] = _implementation
