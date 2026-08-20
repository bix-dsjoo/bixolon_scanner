"""Compatibility alias for the bread locked-candidate experiment evaluation."""

import sys

from ..experiments.bread import locked_candidate_evaluation as _implementation

sys.modules[__name__] = _implementation
