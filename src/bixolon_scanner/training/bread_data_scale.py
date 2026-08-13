"""Compatibility alias for the canonical data_scale module."""

import sys

from ..experiments.bread import data_scale as _implementation

sys.modules[__name__] = _implementation
