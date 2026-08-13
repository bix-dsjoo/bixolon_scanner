"""Compatibility alias for the canonical classifier_error_audit module."""

import sys

from ..experiments.rpc200 import classifier_error_audit as _implementation

sys.modules[__name__] = _implementation
