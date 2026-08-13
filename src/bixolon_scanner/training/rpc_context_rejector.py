"""Compatibility alias for the canonical context_rejector module."""

import sys

from ..experiments.rpc200 import context_rejector as _implementation

sys.modules[__name__] = _implementation
