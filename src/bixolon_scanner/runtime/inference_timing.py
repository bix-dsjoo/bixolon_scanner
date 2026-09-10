"""Opt-in, request-local inference timings without tensors or image information."""

from contextlib import contextmanager
from contextvars import ContextVar

_calls: ContextVar[list[dict] | None] = ContextVar("inference_calls", default=None)


@contextmanager
def collect_inference_timings():
    calls: list[dict] = []
    token = _calls.set(calls)
    try:
        yield calls
    finally:
        _calls.reset(token)


def active_inference_timings() -> list[dict] | None:
    return _calls.get()
