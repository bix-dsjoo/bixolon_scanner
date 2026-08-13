"""Compatibility alias for the canonical scan_log_review_export module."""

import sys

from ..operations import scan_log_review_export as _implementation

sys.modules[__name__] = _implementation
