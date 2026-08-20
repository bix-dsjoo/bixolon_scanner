from __future__ import annotations

import asyncio
import os
import sys

import uvicorn

from .api import create_app
from .logging import configure_logging
from .settings import WorkerSettings


def _windows_selector_loop_factory(
    use_subprocess: bool = False,
) -> asyncio.AbstractEventLoop:
    del use_subprocess
    return asyncio.SelectorEventLoop()


def serve() -> None:
    settings = WorkerSettings()
    use_stderr = os.environ.get("BIXOLON_LOG_TO_STDERR", "1").lower() not in {
        "0",
        "false",
    }
    configure_logging(settings.log_level, use_stderr=use_stderr)
    uvicorn.run(
        create_app(settings=settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
        ws="none",
        loop=_windows_selector_loop_factory if sys.platform == "win32" else "auto",
    )
