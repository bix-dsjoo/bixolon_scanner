from __future__ import annotations

import json
import logging
import os
import stat
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    _reserved = set(logging.makeLogRecord({}).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._reserved and key not in {"message", "asctime"}:
                payload[key] = value
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def _is_usable_log_stream(stream: object) -> bool:
    try:
        if stream is None or stream.closed:  # type: ignore[union-attr]
            return False
        descriptor = stream.fileno()  # type: ignore[union-attr]
        stream_stat = os.fstat(descriptor)
    except (AttributeError, OSError, ValueError):
        return False
    if os.name == "nt" and not stream.isatty():  # type: ignore[union-attr]
        return stat.S_ISREG(stream_stat.st_mode) or stat.S_ISFIFO(stream_stat.st_mode)
    return True


def configure_logging(level: str = "INFO", *, use_stderr: bool = True) -> None:
    stream = sys.stderr
    if not use_stderr or not _is_usable_log_stream(stream):
        handler: logging.Handler = logging.NullHandler()
    else:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
