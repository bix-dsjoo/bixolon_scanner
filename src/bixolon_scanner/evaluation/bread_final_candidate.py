"""Compatibility alias for the bread final-candidate experiment evaluation."""

import sys

from ..experiments.bread import final_candidate_evaluation as _implementation

sys.modules[__name__] = _implementation
