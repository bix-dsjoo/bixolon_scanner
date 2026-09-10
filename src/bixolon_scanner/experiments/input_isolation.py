"""Compatibility alias for the shared training-input isolation guard."""

import sys

from ..training import input_isolation as _canonical

sys.modules[__name__] = _canonical
